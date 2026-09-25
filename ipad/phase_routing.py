"""Stage 3: normal-only diagnostics and preregistered phase-routing comparisons."""
import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .common import atomic_checkpoint, environment, seed_everything, write_json
from .prototype import FeatureClips, PhaseHead, fit_banks
from .evaluate import score_rows, write_csv

METHODS=('unconditional','legacy_hard','posterior_hard','neighbor_nn','top3_nn',
         'top3_weighted','confidence_fallback','random3_nn','neighbor_random_nn','conditional_all')


def reserve():
    if Path('runs/disk_pause.json').exists() or shutil.disk_usage('.').free<=10*1024**3:
        raise RuntimeError('Disk safety pause: do not resume automatically')


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def event(out,kind,**kw):
    with (out/'events.jsonl').open('a') as f:
        f.write(json.dumps({'time':time.time(),'event':kind,**kw})+'\n')


def circular_error(a,b,bins=20):
    d=(np.asarray(a)-np.asarray(b))%bins
    return np.minimum(d,bins-d)


def phase_probabilities(logits):
    return logits.softmax(-1).reshape(-1,20,10).sum(-1)


def choose_threshold(rows):
    """Lowest confidence cutoff with >=90% within-one-bin accuracy and >=10% coverage."""
    confidence=np.array([r['confidence'] for r in rows])
    correct=circular_error([r['posterior_bin'] for r in rows],[r['reference_bin'] for r in rows])<=1
    for q in np.linspace(0,.9,19):
        cutoff=float(np.quantile(confidence,q));selected=confidence>=cutoff
        if selected.mean()>=.1 and correct[selected].mean()>=.9:
            return {'threshold':cutoff,'enabled':True,'coverage':float(selected.mean()),
                    'within_one_accuracy':float(correct[selected].mean()),'normal_quantile':float(q)}
    return {'threshold':1.1,'enabled':False,'coverage':0.,'within_one_accuracy':None,
            'reason':'No normal-validation cutoff met 90% accuracy and 10% coverage'}


def route_distances(per_bin,unconditional,posterior,legacy,occupied,random_order,threshold):
    """per_bin [B,P,O], preserving spatial positions until each candidate minimum."""
    bins=20;delta=(legacy[:,None]-occupied[None,:]).abs()
    resolved=torch.minimum(delta,bins-delta).argmin(-1)
    post=posterior[:,occupied];top=post.topk(min(3,len(occupied)),dim=-1).indices
    post_hard=post.argmax(-1)
    neighbor=torch.minimum(delta,bins-delta)<=1
    neighbor.scatter_(1,resolved[:,None],True)
    counts=neighbor.sum(-1)
    random3=random_order[:,:min(3,len(occupied))]
    random_neighbor=torch.zeros_like(neighbor)
    random_neighbor.scatter_(1,random_order,torch.arange(len(occupied),device=per_bin.device)[None,:]<counts[:,None])
    def gather(index):return per_bin.gather(2,index[:,None,:].expand(-1,per_bin.shape[1],-1))
    def masked(mask):return per_bin.masked_fill(~mask[:,None,:],torch.inf).min(-1).values.mean(-1)
    chosen=gather(top);weights=post.gather(1,top);weights=weights/weights.sum(-1,keepdim=True)
    weighted=(chosen.mean(1)*weights).sum(-1)
    values={'unconditional':unconditional,'legacy_hard':gather(resolved[:,None]).mean((1,2)),
            'posterior_hard':gather(post_hard[:,None]).mean((1,2)),
            'neighbor_nn':masked(neighbor),'top3_nn':chosen.min(-1).values.mean(-1),
            'top3_weighted':weighted,
            'confidence_fallback':torch.where(posterior.max(-1).values>=threshold,weighted,unconditional),
            'random3_nn':gather(random3).min(-1).values.mean(-1),
            'neighbor_random_nn':masked(random_neighbor),'conditional_all':per_bin.min(-1).values.mean(-1)}
    return values,counts


def fit_phase(train,val,out,seed,epochs=50):
    seed_everything(seed);model=PhaseHead().cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    loader=DataLoader(train,batch_size=128,shuffle=True,num_workers=2)
    with (out/'phase_batches.jsonl').open('w') as log:
        for epoch in range(epochs):
            model.train();total=n=0
            for step,row in enumerate(loader):
                reserve();x=row['cls'].cuda();y=row['phase'].cuda();opt.zero_grad(set_to_none=True)
                loss=F.cross_entropy(model(x),y);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'))
                if not torch.isfinite(loss) or not torch.isfinite(grad):raise ValueError('Nonfinite training')
                opt.step();n+=len(y);total+=loss.item()*len(y)
                log.write(json.dumps({'epoch':epoch+1,'step':step,'loss':loss.item(),'grad_norm':grad.item(),
                      'videos':list(row['video']),'frames':row['frame'].tolist(),'time':time.time()})+'\n')
            log.flush();event(out,'phase_epoch',epoch=epoch+1,ce=total/n)
    model.eval();atomic_checkpoint(out/'phase_head.pt',{'model':model.state_dict(),'seed':seed})
    return model


def predict(dataset,model,banks,out,name,seed,threshold=1.1,reference=False):
    model.eval();occupied=torch.unique(banks['groups']).cuda();groups=banks['groups'].cuda()
    conditional=F.normalize(banks['conditional'].cuda().float(),dim=-1)
    unconditional=F.normalize(banks['unconditional'].cuda().float(),dim=-1)
    maps=[np.load(r['path']/'patch12.npy',mmap_mode='r') for r in dataset.records]
    rng=np.random.default_rng(seed+187);rows=[];seconds=0.;counts=[]
    with torch.inference_mode():
        for step,b in enumerate(DataLoader(dataset,batch_size=32,num_workers=2)):
            reserve();x=torch.from_numpy(np.stack([maps[int(ri)][int(fr)] for ri,fr in zip(b['record'],b['frame'])])).cuda()
            x=F.normalize(x.float(),dim=-1);torch.cuda.synchronize();tick=time.perf_counter()
            logits=model(b['cls'].cuda());prob=phase_probabilities(logits);phase=logits.argmax(-1);legacy=phase//10
            sims=torch.einsum('bpc,pkc->bpk',x,conditional)
            per_bin=torch.stack([(1-sims[:,:,groups==o].max(-1).values).clamp_min(0) for o in occupied],-1)
            un=(1-torch.einsum('bpc,pkc->bpk',x,unconditional).max(-1).values).clamp_min(0).mean(-1)
            orders=torch.tensor(np.stack([rng.permutation(len(occupied)) for _ in range(len(x))]),device=x.device)
            scores,ncandidate=route_distances(per_bin,un,prob,legacy,occupied,orders,threshold)
            torch.cuda.synchronize();seconds+=time.perf_counter()-tick
            values={k:v.cpu().tolist() for k,v in scores.items()};posterior=prob.cpu().numpy();phases=phase.cpu().tolist()
            dist=per_bin.mean(1).cpu().numpy();oc=occupied.cpu().numpy()
            for i,video in enumerate(b['video']):
                r={'scene':dataset.records[0]['scene'],'video':video,
                   'frame':int(b['frame'][i]),'video_length':int(b['length'][i]),'phase':phases[i],
                   'legacy_bin':phases[i]//10,'posterior_bin':int(posterior[i].argmax()),
                   'confidence':float(posterior[i].max()),'neighbor_candidate_bins':int(ncandidate[i]),
                   **{k:values[k][i] for k in METHODS}}
                r.update({f'p{j:02}':float(posterior[i,j]) for j in range(20)})
                if reference:
                    # Reference is available only for normal development videos, never test videos.
                    target=int(b['phase'][i])//10;error=circular_error(oc,target);nearest=int(error.argmin())
                    r.update(reference_bin=target,reference_distance=float(dist[i,nearest]),
                             other_bin_distance=float(np.min(np.delete(dist[i],nearest))),
                             reference_bin_occupied=bool(target in oc))
                rows.append(r)
            if step%100==0:event(out,'matching',split=name,step=step,rows=len(rows))
    write_csv(out/f'{name}_raw.csv',rows)
    return rows,{'all_methods_gpu_seconds':seconds,'rows':len(rows),'occupied_bins':occupied.cpu().tolist(),
                 'note':'Combined cost of all comparison arms, not single-method streaming latency'}


def diagnostics(rows,banks):
    y=np.array([r['reference_bin'] for r in rows]);pred=np.array([r['posterior_bin'] for r in rows]);err=circular_error(y,pred)
    confusion=np.zeros((20,20),dtype=int);np.add.at(confusion,(y,pred),1)
    confidence=np.array([r['confidence'] for r in rows]);buckets=[]
    for lo,hi in zip(np.linspace(0,1,6)[:-1],np.linspace(0,1,6)[1:]):
        mask=(confidence>=lo)&(confidence<(hi if hi<1 else 1.01))
        buckets.append({'lo':float(lo),'hi':float(hi),'n':int(mask.sum()),
                        'accuracy':float((err[mask]==0).mean()) if mask.any() else None})
    return {'frames':len(rows),'bin20_accuracy':float((err==0).mean()),'within_one_accuracy':float((err<=1).mean()),
            'circular_mae_bins':float(err.mean()),'legacy_bin20_accuracy':float(np.mean(y==[r['legacy_bin'] for r in rows])),
            'confusion':confusion.tolist(),'confidence_buckets':buckets,
            'mean_distance':{k:float(np.mean([r[k] for r in rows])) for k in ['reference_distance','other_bin_distance',*METHODS]},
            'prototypes_per_bin':torch.bincount(banks['groups'],minlength=20).tolist(),
            'reference_vs_other_win_fraction':float(np.mean([r['reference_distance']<r['other_bin_distance'] for r in rows]))}


def verify_stage2(rows,scene):
    with open(f'runs/prototype/{scene}/seed0/raw_frames.csv') as f:old={(r['video'],int(r['frame'])):r for r in csv.DictReader(f)}
    assert len(old)==len(rows)
    errors={k:0. for k in ('legacy_hard','unconditional')}
    for r in rows:
        o=old[(r['video'],r['frame'])];assert r['phase']==int(o['phase'])
        for k,oldkey in [('legacy_hard','conditional_nn'),('unconditional','unconditional_nn')]:
            errors[k]=max(errors[k],abs(r[k]-float(o[oldkey])))
    assert max(errors.values())<2e-6,errors
    return errors


def run(scene,seed=0):
    torch.set_num_threads(8);reserve();seed_everything(seed)
    out=Path(f'runs/stage3/{scene}/seed{seed}');out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():print(f'Already complete: {out}');return
    if (out/'config.json').exists():raise RuntimeError('Partial run exists; inspect before resuming to preserve logs')
    started=time.time();torch.cuda.reset_peak_memory_stats()
    dev=FeatureClips(scene,'training',subset='train');val=FeatureClips(scene,'training',subset='val')
    train=FeatureClips(scene,'training');test=FeatureClips(scene,'testing')
    config={'scene':scene,'seed':seed,'methods':METHODS,'epochs':50,'bins':20,'k':10,'max_frames':512,
            'iterations':20,'normal_train_videos':[r['video'] for r in dev.records],
            'normal_val_videos':[r['video'] for r in val.records],
            'environment':environment(),'source_sha256':{str(p):digest(p) for p in Path('ipad').glob('*.py')},
            'protocol_sha256':digest('docs/stage3_protocol.md'),'started_at':started,
            'phase_reference':'clip-start position; target patch is start+8',
            'primary_candidate':'top3_weighted; other arms are prespecified ablations, no test-set selection',
            'calibration':'normal development model; cutoff frozen before loading test scores; distribution shift after full refit disclosed'}
    write_json(out/'config.json',config);event(out,'started',config=config)
    devout=out/'development';devout.mkdir();model=fit_phase(dev,val,devout,seed)
    banks=fit_banks(dev,seed=seed);atomic_checkpoint(devout/'memory.pt',{'banks':banks})
    vals,timing=predict(val,model,banks,out,'validation',seed,reference=True)
    diag=diagnostics(vals,banks);write_json(out/'diagnostics.json',diag)
    threshold=choose_threshold(vals);write_json(out/'calibration.json',threshold)
    write_json(out/'development_timing.json',timing)
    del model,banks
    event(out,'settings_frozen',threshold=threshold)
    # Full-normal refit for paper-comparison protocol, with no test labels consulted.
    if seed==0:
        folder=Path(f'runs/prototype/{scene}/seed0');model=PhaseHead().cuda()
        model.load_state_dict(torch.load(folder/'phase_head.pt',map_location='cpu',weights_only=False)['model'])
        banks=torch.load(folder/'memory.pt',map_location='cpu',weights_only=False)['banks']
        write_json(out/'reused_artifacts.json',{str(folder/name):digest(folder/name) for name in ['phase_head.pt','memory.pt']})
    else:
        full=out/'full';full.mkdir();model=fit_phase(train,None,full,seed)
        banks=fit_banks(train,seed=seed);atomic_checkpoint(full/'memory.pt',{'banks':banks})
    rows,timing=predict(test,model,banks,out,'test',seed,threshold['threshold'])
    if seed==0:write_json(out/'stage2_equivalence.json',verify_stage2(rows,scene))
    period=float(np.median([r['frames'] for r in train.records]));scored,summary=score_rows(rows,'IPAD_dataset',scene,period,feature_keys=METHODS)
    # All comparisons must cover exactly the stage-2 frame/label identities.
    with open(f'runs/prototype/{scene}/seed0/scores.csv') as f:
        identity=[(r['video'],int(r['frame']),int(r['label'])) for r in csv.DictReader(f)]
    assert identity==[(r['video'],r['frame'],r['label']) for r in scored]
    write_csv(out/'scores.csv',scored)
    summary.update(scene=scene,seed=seed,timing=timing,normal_validation=diag,
                   calibration=threshold,seconds=time.time()-started,peak_gb=torch.cuda.max_memory_allocated()/1e9,
                   protocol='offline centered clips/window, scene-wide test normalization; no online claim')
    write_json(out/'metrics.json',summary)
    if scene=='R02':
        sensitivity={str(s):score_rows(rows,'IPAD_dataset',scene,period,'common',s,METHODS)[1] for s in (-1,0,1)}
        write_json(out/'alignment_sensitivity.json',sensitivity)
    event(out,'completed',seconds=time.time()-started);write_json(out/'completed.json',{'finished_at':time.time(),**summary})
    print(json.dumps({'scene':scene,'seed':seed,'metrics':summary['metrics'],'diagnostics':diag['bin20_accuracy']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=['R01','R02','R03','R04'],required=True)
    p.add_argument('--seed',type=int,default=0);run(**vars(p.parse_args()))
