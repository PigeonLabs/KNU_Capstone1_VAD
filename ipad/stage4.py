"""Preregistered stage 4 experiments: temporal scores, memory budgets, process priors."""
import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import atomic_checkpoint, environment, write_json
from .phase_routing import digest, reserve, phase_probabilities
from .prototype import FeatureClips, PhaseHead, fit_banks
from .evaluate import write_csv, score_rows
from .metrics import binary_metrics, normalize, period_errors
from .data import label_path

SCENES=['R01','R02','R03','R04']


def read_csv(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def full_folder(scene,seed):
    return Path(f'runs/prototype/{scene}/seed0') if seed==0 else Path(f'runs/stage3/{scene}/seed{seed}/full')


def output(stage,scene,seed,extra=None):
    out=Path(f'runs/stage4/{stage}/{scene}/seed{seed}')
    if extra is not None:out=out/extra
    out.mkdir(parents=True,exist_ok=True)
    return out


def begin(out,stage,scene,seed,**kwargs):
    reserve()
    if (out/'completed.json').exists():return False
    if (out/'config.json').exists():raise RuntimeError(f'Partial result requires inspection: {out}')
    write_json(out/'config.json',{'stage':stage,'scene':scene,'seed':seed,'started_at':time.time(),
        'source_sha256':{str(p):digest(p) for p in Path('ipad').glob('*.py')},
        'protocol_sha256':digest('docs/stage4_protocol.md'),'environment':environment(),**kwargs})
    return True


def event(out,**kwargs):
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),**kwargs})+'\n')


def period(scene):
    return float(np.median([r['frames'] for r in FeatureClips(scene,'training').records]))


def original_rows(scene,seed):return read_csv(f'runs/stage3/{scene}/seed{seed}/test_raw.csv')


def groups(rows):
    return [[r for r in rows if r['video']==v] for v in sorted({r['video'] for r in rows},key=int)]


def attach_time(rows,normal_period):
    out=[]
    for video in groups(rows):
        frames=[int(r['frame']) for r in video];assert all(b-a==1 for a,b in zip(frames,frames[1:]))
        phase=[int(r['phase']) for r in video]
        t5=period_errors(phase,normal_period,window=5);t21=period_errors(phase,normal_period,window=21)
        for r,a,b in zip(video,t5,t21):
            if np.isfinite(a) and np.isfinite(b):out.append({**r,'time5':float(a),'time21':float(b)})
    return out


def evaluate_rows(rows,scene,components,combinations,policy='strict',shift=0):
    aligned=[];excluded=[]
    for video in groups(rows):
        labels=np.load(label_path('IPAD_dataset',scene,video[0]['video'])).reshape(-1)
        if policy=='strict' and len(labels)!=int(video[0]['video_length']):excluded.append(video[0]['video']);continue
        for r in video:
            index=int(r['frame'])+shift
            if 0<=index<len(labels):aligned.append({**r,'frame':int(r['frame']),'label':int(labels[index])})
    assert aligned
    values={k:normalize([float(r[k]) for r in aligned]) for k in components}
    for k,parts in combinations.items():values[k]=np.mean([values[p] for p in parts],axis=0)
    for i,r in enumerate(aligned):
        for k,v in values.items():r[k+'_score']=float(v[i])
    metrics={k:binary_metrics([r['label'] for r in aligned],v) for k,v in values.items()}
    return aligned,{'metrics':metrics,'frames':len(aligned),'excluded_videos':excluded,'shift':shift,
                    'policy':policy,'normalization':'offline scene-wide min-max','combinations':combinations}


def finish(out,scene,seed,rows,components,combinations,started,**extra):
    scored,summary=evaluate_rows(rows,scene,components,combinations)
    old={(r['video'],int(r['frame'])):int(r['label']) for r in read_csv(f'runs/stage3/{scene}/seed{seed}/scores.csv')}
    assert all(old[(r['video'],r['frame'])]==r['label'] for r in scored)
    # Exact support is common to 4-1 and 4-3 and determined only by the temporal window.
    expected,_=evaluate_rows(attach_time(original_rows(scene,seed),period(scene)),scene,['unconditional'],{})
    assert [(r['video'],r['frame'],r['label']) for r in scored]==[(r['video'],r['frame'],r['label']) for r in expected]
    write_csv(out/'raw.csv',rows);write_csv(out/'scores.csv',scored)
    summary.update(scene=scene,seed=seed,seconds=time.time()-started,**extra)
    write_json(out/'metrics.json',summary)
    if scene=='R02':write_json(out/'alignment_sensitivity.json',{str(s):evaluate_rows(rows,scene,components,combinations,'common',s)[1] for s in [-1,0,1]})
    write_json(out/'completed.json',summary);event(out,event='completed',seconds=time.time()-started)
    print(json.dumps({'scene':scene,'seed':seed,'output':str(out),'metrics':summary['metrics']}),flush=True)


def stage41(scene,seed):
    out=output('4-1',scene,seed);source=Path(f'runs/stage3/{scene}/seed{seed}/test_raw.csv')
    if not begin(out,'4-1',scene,seed,input_sha256={str(source):digest(source)}):return
    started=time.time();rows=attach_time(original_rows(scene,seed),period(scene))
    finish(out,scene,seed,rows,['unconditional','legacy_hard','top3_weighted','time5','time21'],
        {'appearance_time5':['unconditional','time5'],'appearance_time21':['unconditional','time21']},started,
        primary='appearance_time21',support='central window21; all baselines rescored on same frames')


def estimate_prior(sequences,alpha=1.):
    counts=np.full((20,20),alpha/20);durations=[[] for _ in range(20)]
    for q in sequences:
        assert q.ndim==2 and q.shape[1]==20
        if len(q)>1:counts+=q[:-1].T@q[1:]
        states=q.argmax(-1);bound=np.r_[0,np.flatnonzero(np.diff(states))+1,len(states)]
        # First and last runs are censored by video boundaries.
        for a,b in zip(bound[1:-2],bound[2:-1]):durations[int(states[a])].append(int(b-a))
    pooled=sorted(x for d in durations for x in d)
    if not pooled:raise ValueError('No complete normal runs for duration model')
    return {'transition':(counts/counts.sum(-1,keepdims=True)).tolist(),
            'soft_counts':counts.tolist(),'durations':[sorted(d) for d in durations],
            'pooled_durations':pooled,'dirichlet_total_per_row':alpha,'censored_edges_excluded':True}


def prior_scores(q,prior):
    q=np.maximum(np.asarray(q,dtype=np.float64),1e-12);q=q/q.sum(-1,keepdims=True)
    trans=np.array(prior['transition']);pred=np.maximum(q[:-1]@trans,1e-12)
    overlap=-np.log(np.maximum((pred*q[1:]).sum(-1),1e-12))
    kl=(q[1:]*(np.log(q[1:])-np.log(pred))).sum(-1).clip(0)
    entropy=-(q*np.log(q)).sum(-1);states=q.argmax(-1);age=0;duration=[]
    for i,state in enumerate(states):
        age=age+1 if i and state==states[i-1] else 1
        lengths=prior['durations'][state] or prior['pooled_durations']
        n=len(lengths);survival=(1+n-np.searchsorted(lengths,age,side='left'))/(n+1)
        duration.append(float(-np.log(survival)))
    return {'transition_overlap':np.r_[0.,overlap],'transition_kl':np.r_[0.,kl],
            'duration':np.array(duration),'phase_entropy':entropy}


def stage43(scene,seed):
    out=output('4-3',scene,seed);folder=full_folder(scene,seed)
    if not begin(out,'4-3',scene,seed,head_sha256=digest(folder/'phase_head.pt')):return
    started=time.time();model=PhaseHead().cuda().eval()
    model.load_state_dict(torch.load(folder/'phase_head.pt',map_location='cpu',weights_only=False)['model'])
    train=FeatureClips(scene,'training');record_rows=[];seqs={r['video']:[] for r in train.records}
    with torch.inference_mode():
        for step,b in enumerate(DataLoader(train,batch_size=128,num_workers=2)):
            reserve();prob=phase_probabilities(model(b['cls'].cuda())).cpu().numpy()
            for i,video in enumerate(b['video']):
                seqs[video].append(prob[i]);record_rows.append({'video':video,'frame':int(b['frame'][i]),**{f'p{j:02}':float(prob[i,j]) for j in range(20)}})
            if step%50==0:event(out,event='normal_posterior',step=step,rows=len(record_rows))
    write_csv(out/'normal_posteriors.csv',record_rows)
    prior=estimate_prior([np.array(s) for s in seqs.values()]);write_json(out/'normal_prior.json',prior)
    rows=[]
    for video in groups(original_rows(scene,seed)):
        q=np.array([[float(r[f'p{j:02}']) for j in range(20)] for r in video]);scores=prior_scores(q,prior)
        for i,r in enumerate(video):rows.append({**r,**{k:float(v[i]) for k,v in scores.items()}})
    rows=attach_time(rows,period(scene))
    components=['unconditional','time5','time21','transition_overlap','transition_kl','duration','phase_entropy']
    combinations={'process':['transition_kl','duration'],'appearance_transition':['unconditional','transition_kl'],
      'appearance_duration':['unconditional','duration'],'appearance_process':['unconditional','process'],
      'appearance_time21':['unconditional','time21'],'appearance_entropy':['unconditional','phase_entropy']}
    finish(out,scene,seed,rows,components,combinations,started,primary='appearance_process',
           warning='Prior learned from full-normal in-sample head predictions; no independent normal FPR or online claim')


def load_bank(banks):
    c=F.normalize(banks['conditional'].cuda().float(),dim=-1)
    u=F.normalize(banks['unconditional'].cuda().float(),dim=-1)
    g=banks['groups'].cuda();occupied=torch.unique(g)
    groupbank=torch.stack([c[:,g==v,:] for v in occupied])
    return c,u,g,occupied,groupbank


def budget_match(x,prob,legacy,bank,method=None):
    c,u,g,occupied,groupbank=bank;x=F.normalize(x.float(),dim=-1);result={}
    if method in (None,'unconditional'):
        result['unconditional']=(1-torch.einsum('bpc,pkc->bpk',x,u).max(-1).values).clamp_min(0).mean(-1)
    if method in (None,'legacy_hard'):
        d=(legacy[:,None]-occupied[None]).abs();index=torch.minimum(d,20-d).argmin(-1)
        selected=groupbank[index]
        result['legacy_hard']=(1-torch.einsum('bpc,bpkc->bpk',x,selected).max(-1).values).clamp_min(0).mean(-1)
    if method in (None,'top3_weighted'):
        values,index=prob[:,occupied].topk(3,-1);selected=groupbank[index]
        distance=(1-torch.einsum('bpc,bgpkc->bgpk',x,selected).max(-1).values).clamp_min(0).mean(-1)
        result['top3_weighted']=(distance*values/values.sum(-1,keepdim=True)).sum(-1)
    return result


def benchmark(scene,model,bank):
    from .features import DinoFeatures
    testing=FeatureClips(scene,'testing');rec=testing.records[0];n=min(256,rec['frames'])
    images=np.array(np.load(Path('cache/frames')/scene/'testing'/f"{rec['video']}.npy",mmap_mode='r')[:n])
    extractor=DinoFeatures().cuda().eval()
    def extract():
        cs=[];ps=[]
        for start in range(0,n,32):
            x=torch.from_numpy(images[start:start+32]).cuda().permute(0,3,1,2).float()/127.5-1
            f=extractor(x);cs.append(f['cls']);ps.append(f['patch12'])
        return torch.cat(cs),torch.cat(ps)
    def matching(cls,patch,method):
        for start in range(0,n-15,16):
            starts=torch.arange(start,min(start+16,n-15),device='cuda');x=patch[starts+8]
            if method=='unconditional':budget_match(x,None,None,bank,method)
            else:
                logits=model(cls[starts[:,None]+torch.arange(16,device='cuda')]);prob=phase_probabilities(logits)
                budget_match(x,prob,logits.argmax(-1)//10,bank,method)
    timing={}
    with torch.inference_mode():
        cls,patch=extract()
        for method in ['unconditional','legacy_hard','top3_weighted']:
            matching(cls,patch,method);cached=[];pipeline=[]
            for _ in range(10):
                torch.cuda.synchronize();t=time.perf_counter();matching(cls,patch,method);torch.cuda.synchronize();cached.append(time.perf_counter()-t)
            for _ in range(3):
                reserve();torch.cuda.synchronize();t=time.perf_counter();a,b=extract();matching(a,b,method);torch.cuda.synchronize();pipeline.append(time.perf_counter()-t)
            timing[method]={'cache_seconds':cached,'dino_pipeline_seconds':pipeline,
              'cache_fps':(n-15)/float(np.median(cached)), 'pipeline_output_fps':(n-15)/float(np.median(pipeline))}
    del extractor
    return {'video':rec['video'],'input_frames':n,'output_frames':n-15,'timings':timing,
       'scope':'warm decoded RAM frames -> RGB preprocessing -> DINO -> optional phase head -> candidate-only matching; excludes disk/camera I/O; centered offline clips'}


def stage42(scene,seed,k):
    out=output('4-2',scene,seed,f'k{k}');folder=full_folder(scene,seed)
    if not begin(out,'4-2',scene,seed,k=k,head_sha256=digest(folder/'phase_head.pt')):return
    started=time.time();torch.cuda.reset_peak_memory_stats();model=PhaseHead().cuda().eval()
    model.load_state_dict(torch.load(folder/'phase_head.pt',map_location='cpu',weights_only=False)['model'])
    if k==10:
        banks=torch.load(folder/'memory.pt',map_location='cpu',weights_only=False)['banks']
        write_json(out/'reused_bank.json',{'path':str(folder/'memory.pt'),'sha256':digest(folder/'memory.pt')})
    else:
        banks=fit_banks(FeatureClips(scene,'training'),k=k,seed=seed)
        atomic_checkpoint(out/'memory.pt',{'banks':banks})
    bank=load_bank(banks);test=FeatureClips(scene,'testing')
    maps=[np.load(r['path']/'patch12.npy',mmap_mode='r') for r in test.records];rows=[];compute=0.
    with torch.inference_mode():
        for step,b in enumerate(DataLoader(test,batch_size=32,num_workers=2)):
            reserve();x=torch.from_numpy(np.stack([maps[int(ri)][int(fr)] for ri,fr in zip(b['record'],b['frame'])])).cuda()
            torch.cuda.synchronize();t=time.perf_counter();logits=model(b['cls'].cuda());prob=phase_probabilities(logits);phase=logits.argmax(-1)
            values=budget_match(x,prob,phase//10,bank);torch.cuda.synchronize();compute+=time.perf_counter()-t
            values={key:v.cpu().tolist() for key,v in values.items()};phase=phase.cpu().tolist()
            for i,video in enumerate(b['video']):rows.append({'scene':scene,'video':video,'frame':int(b['frame'][i]),
                'video_length':int(b['length'][i]),'phase':phase[i],**{key:v[i] for key,v in values.items()}})
            if step%100==0:event(out,event='match',step=step,rows=len(rows))
    scored,summary=score_rows(rows,'IPAD_dataset',scene,period(scene),feature_keys=('unconditional','legacy_hard','top3_weighted'))
    old=read_csv(f'runs/stage3/{scene}/seed{seed}/scores.csv')
    assert [(r['video'],int(r['frame']),int(r['label'])) for r in old]==[(r['video'],r['frame'],r['label']) for r in scored]
    if k==10:
        errors={key:max(abs(float(a[key])-b[key]) for a,b in zip(old,scored)) for key in ['unconditional','legacy_hard','top3_weighted']}
        assert max(errors.values())<2e-6,errors;write_json(out/'baseline_equivalence.json',errors)
    write_csv(out/'raw.csv',rows);write_csv(out/'scores.csv',scored)
    if scene=='R02':write_json(out/'alignment_sensitivity.json',{str(s):score_rows(rows,'IPAD_dataset',scene,period(scene),'common',s,('unconditional','legacy_hard','top3_weighted'))[1] for s in [-1,0,1]})
    if seed==0:write_json(out/'benchmark.json',benchmark(scene,model,bank))
    summary.update(scene=scene,seed=seed,k=k,seconds=time.time()-started,all_arms_cache_seconds=compute,
       prototypes_per_position=banks['prototypes_per_position'],single_bank_bytes=banks['conditional'].numel()*banks['conditional'].element_size(),
       combined_experiment_peak_gb=torch.cuda.max_memory_allocated()/1e9,
       peak_note='Peak includes comparison banks and DINO benchmark; not per-method deployment peak')
    write_json(out/'metrics.json',summary);write_json(out/'completed.json',summary);event(out,event='completed',seconds=time.time()-started)
    print(json.dumps({'scene':scene,'seed':seed,'k':k,'seconds':summary['seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',required=True,choices=['4-1','4-2','4-3'])
    p.add_argument('--scene',required=True,choices=SCENES);p.add_argument('--seed',type=int,default=0);p.add_argument('--k',type=int,default=10)
    a=p.parse_args();torch.set_num_threads(8);reserve()
    if a.stage=='4-1':stage41(a.scene,a.seed)
    elif a.stage=='4-2':stage42(a.scene,a.seed,a.k)
    else:stage43(a.scene,a.seed)
