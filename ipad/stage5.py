"""Fixed normal calibration, causal DINO detectors, and batch-one paced replay."""
import argparse
from collections import deque
import csv
import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .common import atomic_checkpoint, environment, seed_everything, write_json
from .data import validation_videos, label_path, frames
from .evaluate import write_csv
from .features import DinoFeatures, DINO_COMMIT, install_source
from .metrics import binary_metrics, normalize, period_errors
from .phase_routing import digest, reserve
from .prototype import spherical_kmeans

SCENES=['R01','R02','R03','R04']


def read_csv(p):
    with Path(p).open() as f:return list(csv.DictReader(f))


def cache_root(backbone):return Path('cache/dino' if backbone=='B' else 'cache/dino_small')


def records(backbone,scene,split,subset='all'):
    hold=validation_videos('IPAD_dataset',scene);out=[]
    for p in sorted((cache_root(backbone)/scene/split).glob('*/meta.json'),key=lambda p:int(p.parent.name)):
        r=json.loads(p.read_text())
        if split=='training' and subset!='all' and ((r['video'] in hold)!=(subset=='val')):continue
        out.append({**r,'path':str(p.parent)})
    if not out:raise ValueError((backbone,scene,split,subset))
    return out


def event(out,kind,**kw):
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'event':kind,**kw})+'\n')


def begin(out,**kw):
    reserve();out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return False
    if (out/'config.json').exists():raise RuntimeError(f'Partial result needs inspection, not overwrite: {out}')
    write_json(out/'config.json',{'started_at':time.time(),'environment':environment(),
      'source_sha256':{str(p):digest(p) for p in Path('ipad').glob('*.py')},
      'protocol_sha256':digest('docs/stage5_protocol.md'),**kw})
    return True


class Extractor(DinoFeatures):
    def __init__(self,backbone):
        nn.Module.__init__(self);source=install_source();torch.hub.set_dir(str(Path('cache/torch_hub').resolve()))
        self.backbone=torch.hub.load(str(source.resolve()),'dinov2_vitb14' if backbone=='B' else 'dinov2_vits14',source='local',pretrained=True).eval()
        self.backbone.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None])
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None])


def cache_small(scene):
    out=cache_root('S')/scene
    if (out/'complete.json').exists():return
    out.mkdir(parents=True,exist_ok=True);start=time.time();model=Extractor('S').cuda().eval();rows=[]
    with torch.inference_mode():
        for split in ['training','testing']:
            for rec in records('B',scene,split):
                reserve();dest=out/split/rec['video'];dest.mkdir(parents=True,exist_ok=True)
                if (dest/'meta.json').exists():rows.append(json.loads((dest/'meta.json').read_text()));continue
                if list(dest.glob('*.npy')):raise RuntimeError(f'Partial feature cache: {dest}')
                n=rec['frames'];cls=np.lib.format.open_memmap(dest/'cls.tmp.npy',mode='w+',dtype=np.float16,shape=(n,384))
                patch=np.lib.format.open_memmap(dest/'patch12.tmp.npy',mode='w+',dtype=np.float16,shape=(n,324,384))
                inputs=np.load(Path('cache/frames')/scene/split/f"{rec['video']}.npy",mmap_mode='r');tick=time.time()
                for a in range(0,n,64):
                    reserve();x=torch.from_numpy(np.array(inputs[a:a+64])).cuda().permute(0,3,1,2).float()/127.5-1
                    f=model(x);cls[a:a+len(x)]=f['cls'].cpu().numpy().astype(np.float16);patch[a:a+len(x)]=f['patch12'].cpu().numpy().astype(np.float16)
                cls.flush();patch.flush();del cls,patch,inputs
                (dest/'cls.tmp.npy').replace(dest/'cls.npy');(dest/'patch12.tmp.npy').replace(dest/'patch12.npy')
                meta={**{k:v for k,v in rec.items() if k not in {'path','gpu_seconds','wall_seconds'}},'model':'dinov2_vits14','seconds':time.time()-tick,
                      'channels':384,'compute_dtype':'float32','cache_dtype':'float16','source_commit':DINO_COMMIT,
                      'sha256':{key:digest(dest/key) for key in ['cls.npy','patch12.npy']}}
                write_json(dest/'meta.json',meta);rows.append(meta);print(json.dumps({'cache':str(dest),'seconds':meta['seconds']}),flush=True)
    write_json(out/'complete.json',{'records':rows,'seconds':time.time()-start,'model':'dinov2_vits14','source_commit':DINO_COMMIT})


class Head(nn.Module):
    def __init__(self,c):
        super().__init__();self.layers=nn.Sequential(nn.Flatten(1),nn.Linear(16*c,512),nn.ReLU(),nn.Linear(512,200))
    def forward(self,x):return self.layers(F.normalize(x,dim=-1))


def training_arrays(recs,mode):
    xs=[];targets=[]
    for r in recs:
        cls=np.load(Path(r['path'])/'cls.npy',mmap_mode='r');t=np.arange(15,r['frames']-7)
        offsets=np.arange(-15,1) if mode=='causal' else np.arange(-8,8)
        xs.append(np.array(cls[t[:,None]+offsets],dtype=np.float32));targets.append(t*200//r['frames'])
    return torch.from_numpy(np.concatenate(xs)).cuda(),torch.from_numpy(np.concatenate(targets)).long().cuda()


def train_head(recs,mode,c,seed,out):
    seed_everything(seed);x,y=training_arrays(recs,mode);model=Head(c).cuda();opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    history=[];start=time.time()
    with (out/f'{mode}_batches.jsonl').open('w') as log:
        for epoch in range(50):
            reserve();model.train();order=torch.randperm(len(y),device='cuda');total=0.
            for step,a in enumerate(range(0,len(y),128)):
                reserve();idx=order[a:a+128];opt.zero_grad(set_to_none=True);loss=F.cross_entropy(model(x[idx]),y[idx]);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True)
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                opt.step();value=float(loss);total+=value*len(idx)
                log.write(json.dumps({'epoch':epoch+1,'step':step,'loss':value,'gradient_norm':float(grad),'time':time.time()})+'\n')
            row={'epoch':epoch+1,'train_ce':total/len(y),'seconds':time.time()-start};history.append(row);event(out,'epoch',mode=mode,**row);log.flush()
    model.eval();path=out/f'{mode}_head.pt';atomic_checkpoint(path,{'model':model.state_dict(),'mode':mode,'channels':c,'seed':seed})
    restored=Head(c).cuda().eval();restored.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    with torch.inference_mode():error=float((model(x[:8])-restored(x[:8])).abs().max())
    assert error==0.;assert history[-1]['train_ce']<history[0]['train_ce']
    write_json(out/f'{mode}_training.json',{'history':history,'restored_max_error':error,'samples':len(y),'seconds':time.time()-start,'sha256':digest(path)})
    del x,y,restored,opt;return model


def selected_frames(recs,seed):
    rng=np.random.default_rng(seed);selected=[];occupied=0
    for b in range(20):
        pool=[(ri,t) for ri,r in enumerate(recs) for t in range(15,r['frames']-7) if t*20//r['frames']==b]
        if not pool:continue
        occupied+=1;chosen=rng.choice(len(pool),min(512,len(pool)),replace=False)
        selected.extend(pool[i] for i in chosen)
    return selected,occupied


def fit_memory(recs,k,seed,out):
    selected,occupied=selected_frames(recs,seed);maps=[np.load(Path(r['path'])/'patch12.npy',mmap_mode='r') for r in recs];parts=[]
    for p in range(0,324,9):
        reserve();x=torch.from_numpy(np.stack([maps[ri][t,p:p+9] for ri,t in selected])).cuda().transpose(0,1)
        parts.append(spherical_kmeans(x,occupied*k,20,seed).cpu().half());event(out,'memory_fit',position=p)
    bank=torch.cat(parts);atomic_checkpoint(out/'memory.pt',{'bank':bank,'k':k,'occupied':occupied,'seed':seed})
    write_csv(out/'memory_selection.csv',[{'video':recs[ri]['video'],'frame':t} for ri,t in selected])
    write_json(out/'memory.json',{'prototypes':bank.shape[1],'bytes':bank.numel()*bank.element_size(),'shape':list(bank.shape),'k':k,
                                 'occupied':occupied,'pool_frames':len(selected),'sha256':digest(out/'memory.pt')})
    return F.normalize(bank.cuda().float(),dim=-1)


def match(x,bank):return (1-torch.einsum('bpc,pkc->bpk',F.normalize(x.float(),dim=-1),bank).max(-1).values).clamp_min(0).mean(-1)


def causal_time(phases,period,window=21):
    phases=np.asarray(phases,dtype=float);out=np.full(len(phases),np.nan)
    for t in range(window-1,len(phases)):
        ref=phases[t]+np.arange(1-window,1)*200/period
        out[t]=np.abs((phases[t-window+1:t+1]-ref+100)%200-100).mean()
    return out


def infer(recs,model,bank,mode,period):
    rows=[]
    with torch.inference_mode():
        for r in recs:
            cs=np.load(Path(r['path'])/'cls.npy',mmap_mode='r');ps=np.load(Path(r['path'])/'patch12.npy',mmap_mode='r');video=[]
            targets=np.arange(15,r['frames'] if mode=='causal' else r['frames']-7)
            offset=np.arange(-15,1) if mode=='causal' else np.arange(-8,8)
            for a in range(0,len(targets),64):
                reserve();t=targets[a:a+64];x=torch.from_numpy(np.array(cs[t[:,None]+offset],dtype=np.float32)).cuda()
                phase=model(x).argmax(-1).cpu().numpy();appearance=match(torch.from_numpy(np.array(ps[t])).cuda(),bank).cpu().numpy()
                for i,frame in enumerate(t):video.append({'video':r['video'],'frame':int(frame),'video_length':r['frames'],'phase':int(phase[i]),'appearance':float(appearance[i])})
            temporal=causal_time([v['phase'] for v in video],period) if mode=='causal' else period_errors([v['phase'] for v in video],period,window=21)
            for v,t in zip(video,temporal):
                if np.isfinite(t):rows.append({**v,'temporal':float(t)})
    return rows


def calibrate(rows):
    result={}
    for key in ['appearance','temporal']:
        q=np.quantile([r[key] for r in rows],[.5,.995]);result[key]={'median':float(q[0]),'q995':float(q[1]),'scale':float(max(q[1]-q[0],1e-8))}
    scores=apply_calibration(rows,result);result['threshold']=float(np.quantile([r['combined_score'] for r in scores],.995))
    result.update(quantile=.995,consecutive=3,calibration_frames=len(rows),calibration_videos=sorted({r['video'] for r in rows}))
    return result


def scaled(x,cal,key):return max(0.,(float(x)-cal[key]['median'])/cal[key]['scale'])


def apply_calibration(rows,cal):
    out=[]
    for r in rows:
        a=scaled(r['appearance'],cal,'appearance');t=scaled(r['temporal'],cal,'temporal')
        out.append({**r,'appearance_score':a,'temporal_score':t,'combined_score':(a+t)/2})
    return out


def aligned(rows,scene,common=True,policy='strict',shift=0):
    out=[]
    for vid in sorted({r['video'] for r in rows},key=int):
        group=[r for r in rows if r['video']==vid];labels=np.load(label_path('IPAD_dataset',scene,vid)).reshape(-1)
        if policy=='strict' and len(labels)!=group[0]['video_length']:continue
        for r in group:
            t=int(r['frame'])
            if common and not 35<=t<=r['video_length']-18:continue
            if 0<=t+shift<len(labels):out.append({**r,'label':int(labels[t+shift])})
    return out


def metric_rows(rows):return {k:binary_metrics([r['label'] for r in rows],[r[k] for r in rows]) for k in ['appearance_score','temporal_score','combined_score']}


def unit_path(backbone,scene,seed,k):
    return Path('runs/stage5')/('5-1' if backbone=='B' and k==10 else '5-2')/scene/f'seed{seed}'/f'{backbone}_k{k}'


def run_unit(scene,seed,backbone,k):
    out=unit_path(backbone,scene,seed,k);recs=records(backbone,scene,'training','train');val=records(backbone,scene,'training','val');test=records(backbone,scene,'testing')
    assert not ({r['video'] for r in recs}&{r['video'] for r in val})
    if not begin(out,scene=scene,seed=seed,backbone=backbone,k=k,train_videos=[r['video'] for r in recs],calibration_videos=[r['video'] for r in val],
                 cache_metadata_sha256={str(Path(r['path'])/'meta.json'):digest(Path(r['path'])/'meta.json') for r in recs+val+test}):return
    start=time.time();c=768 if backbone=='B' else 384;period=float(np.median([r['frames'] for r in recs]));base=unit_path(backbone,scene,seed,10)
    if k==10:head=train_head(recs,'causal',c,seed,out);head_path=out/'causal_head.pt'
    else:
        head_path=base/'causal_head.pt';head=Head(c).cuda().eval();head.load_state_dict(torch.load(head_path,map_location='cpu',weights_only=False)['model'])
        write_json(out/'reused_head.json',{'path':str(head_path),'sha256':digest(head_path)})
    bank=fit_memory(recs,k,seed,out);validation=infer(val,head,bank,'causal',period);cal=calibrate(validation)
    write_csv(out/'validation_raw.csv',validation);write_json(out/'calibration.json',cal)
    raw=infer(test,head,bank,'causal',period);scored=apply_calibration(raw,cal);common=aligned(scored,scene);full=aligned(scored,scene,common=False)
    write_csv(out/'causal_raw.csv',raw);write_csv(out/'scores.csv',common);write_csv(out/'online_scores.csv',full)
    metrics={'scene':scene,'seed':seed,'backbone':backbone,'k':k,'period':period,'common_frames':len(common),'online_frames':len(full),
             'metrics':metric_rows(common),'online_metrics':metric_rows(full),'threshold':cal['threshold'],'head_path':str(head_path),
             'head_sha256':digest(head_path),'head_parameters':sum(p.numel() for p in head.parameters()),'seconds':time.time()-start}
    if backbone=='B' and k==10:
        centered=train_head(recs,'centered',c,seed,out);v=infer(val,centered,bank,'centered',period);cc=calibrate(v)
        write_csv(out/'centered_validation_raw.csv',v);write_json(out/'centered_calibration.json',cc)
        cr=infer(test,centered,bank,'centered',period);write_csv(out/'centered_raw.csv',cr)
        cf=aligned(apply_calibration(cr,cc),scene);write_csv(out/'centered_fixed_scores.csv',cf)
        assert [(r['video'],r['frame'],r['label']) for r in cf]==[(r['video'],r['frame'],r['label']) for r in common]
        assert max(abs(a['appearance']-b['appearance']) for a,b in zip(common,cf))<1e-6
        # Offline comparator alone can use test-wide statistics; these never enter causal calibration.
        cn=[dict(r) for r in cf];aa=normalize([r['appearance'] for r in cn]);tt=normalize([r['temporal'] for r in cn])
        for i,r in enumerate(cn):r.update(appearance_score=float(aa[i]),temporal_score=float(tt[i]),combined_score=float((aa[i]+tt[i])/2))
        write_csv(out/'centered_testnorm_scores.csv',cn);metrics.update(centered_fixed=metric_rows(cf),centered_testnorm=metric_rows(cn))
    if scene=='R02':write_json(out/'alignment_sensitivity.json',{str(s):metric_rows(aligned(scored,scene,policy='common',shift=s)) for s in [-1,0,1]})
    metrics['seconds']=time.time()-start;write_json(out/'metrics.json',metrics);write_json(out/'completed.json',metrics);event(out,'completed',seconds=metrics['seconds'])
    print(json.dumps({'completed':str(out),'seconds':metrics['seconds']}),flush=True)


def alarm_flags(scores,threshold,consecutive=3):
    active=[];emitted=[];streak=0
    for s in scores:
        streak=streak+1 if s>threshold else 0
        active.append(streak>=consecutive);emitted.append(streak==consecutive)
    return np.array(active,dtype=bool),np.array(emitted,dtype=bool)


def label_segments(labels):
    labels=np.asarray(labels);change=np.diff(np.r_[0,labels,0]);return list(zip(np.flatnonzero(change==1),np.flatnonzero(change==-1)-1))


def operation_metrics(rows,scene,threshold):
    events=[];frames_out=[];normal=raw_fp=active_fp=false_alerts=0;valid=detected=preexisting=cold=0;delays=[]
    for vid in sorted({r['video'] for r in rows},key=int):
        group=[r for r in rows if r['video']==vid];ids=np.array([int(r['frame']) for r in group]);y=np.array([int(r['label']) for r in group]);score=np.array([float(r['combined_score']) for r in group])
        assert np.all(np.diff(ids)==1);active,emit=alarm_flags(score,threshold)
        normal+=int((y==0).sum());raw_fp+=int(((score>threshold)&(y==0)).sum());active_fp+=int((active&(y==0)).sum());false_alerts+=int((emit&(y==0)).sum())
        for i,r in enumerate(group):frames_out.append({**r,'threshold':threshold,'above_threshold':bool(score[i]>threshold),'alarm_active':bool(active[i]),'new_alarm':bool(emit[i])})
        labels=np.load(label_path('IPAD_dataset',scene,vid)).reshape(-1)
        for a,b in label_segments(labels):
            is_cold=a<ids[0];cold+=int(is_cold);index=np.flatnonzero((ids>=a)&(ids<=b)&active)
            delay=int(ids[index[0]]-a) if len(index) else None
            was_active=bool(a>ids[0] and active[a-ids[0]-1]);row={'video':vid,'start':int(a),'end':int(b),'cold_start':bool(is_cold),'detected':delay is not None,'delay_frames':delay,'alarm_preexisting_at_onset':was_active}
            events.append(row)
            if not is_cold:
                valid+=1;preexisting+=int(was_active)
                if delay is not None:detected+=1;delays.append(delay)
    summary={'eligible_segments':valid,'detected_segments':detected,'missed_segments':valid-detected,'cold_start_segments':cold,
      'preexisting_alarm_segments':preexisting,'segment_recall':detected/valid if valid else None,
      'delay_median_frames_detected_only':float(np.median(delays)) if delays else None,
      'delay_p95_frames_detected_only':float(np.quantile(delays,.95)) if delays else None,
      'normal_frames':normal,'raw_false_positive_frames':raw_fp,'alarm_false_positive_frames':active_fp,
      'frame_fpr':raw_fp/normal if normal else None,'active_alarm_fpr':active_fp/normal if normal else None,
      'false_alarm_episodes':false_alerts,'false_alarms_per_1000_normal_frames':false_alerts*1000/normal if normal else None,
      'false_alarms_per_minute_assuming_30fps':false_alerts*1800/normal if normal else None,
      'assumption':'contiguous label-positive segments; active alarm counts as detection; delays exclude misses and cold start, reported separately; actual source FPS unverified'}
    return summary,events,frames_out


class OnlineDetector:
    def __init__(self,extractor,head,bank,cal,period,quantize=False):
        self.extractor=extractor;self.head=head;self.bank=bank;self.cal=cal;self.period=period;self.quantize=quantize;self.reset()
    def reset(self):self.cls=deque(maxlen=16);self.phases=deque(maxlen=21);self.streak=0
    @torch.inference_mode()
    def step(self,bgr):
        image=cv2.resize(bgr,(256,256));x=torch.from_numpy(image).cuda().permute(2,0,1)[None].float()/127.5-1
        f=self.extractor(x);c=f['cls'][0];p=f['patch12']
        if self.quantize:c=c.half().float();p=p.half().float()
        self.cls.append(c)
        if len(self.cls)<16:return None
        phase=int(self.head(torch.stack(list(self.cls))[None]).argmax(-1));self.phases.append(phase)
        appearance=float(match(p,self.bank)[0])
        if len(self.phases)<21:return None
        temporal=float(causal_time(self.phases,self.period)[-1]);a=scaled(appearance,self.cal,'appearance');t=scaled(temporal,self.cal,'temporal');score=(a+t)/2
        self.streak=self.streak+1 if score>self.cal['threshold'] else 0
        return {'phase':phase,'appearance':appearance,'temporal':temporal,'combined_score':score,'alarm_active':self.streak>=3,'new_alarm':self.streak==3}


def benchmark_unit(scene,seed,backbone,k):
    source=unit_path(backbone,scene,seed,k);out=Path('runs/stage5/5-3')/scene/f'seed{seed}'/f'{backbone}_k{k}'
    if not begin(out,scene=scene,seed=seed,backbone=backbone,k=k,source_metrics_sha256=digest(source/'metrics.json')):return
    started=time.time();metrics=json.loads((source/'metrics.json').read_text());cal=json.loads((source/'calibration.json').read_text())
    rows=read_csv(source/'online_scores.csv');op,events,alarms=operation_metrics(rows,scene,cal['threshold'])
    write_json(out/'operation.json',op);write_csv(out/'segments.csv',events);write_csv(out/'alarms.csv',alarms)
    result={'scene':scene,'seed':seed,'backbone':backbone,'k':k,'operation':op}
    if seed==0:
        torch.cuda.empty_cache();extractor=Extractor(backbone).cuda().eval();head=Head(768 if backbone=='B' else 384).cuda().eval()
        head.load_state_dict(torch.load(metrics['head_path'],map_location='cpu',weights_only=False)['model'])
        saved=torch.load(source/'memory.pt',map_location='cpu',weights_only=False)['bank'];bank=F.normalize(saved.cuda().float(),dim=-1)
        detector=OnlineDetector(extractor,head,bank,cal,metrics['period']);video=rows[0]['video']
        files=frames(Path('IPAD_dataset')/scene/'testing/frames'/video)[:256]
        train=records(backbone,scene,'training','train')[0];warm=frames(Path('IPAD_dataset')/scene/'training/frames'/train['video'])[:36]
        for p in warm:reserve();detector.step(cv2.imread(str(p)))
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[];quality=[]
        # Independently rebuild cached-reference scores from exact raw frames in batch one.
        detector.quantize=True;detector.reset()
        reference={int(r['frame']):r for r in read_csv(source/'causal_raw.csv') if r['video']==video}
        for i,p in enumerate(files[:min(80,len(files))]):
            reserve();value=detector.step(cv2.imread(str(p)))
            if value is not None:
                old=reference[i];quality.append({'frame':i,'appearance_abs_error':abs(value['appearance']-float(old['appearance'])),
                  'phase_equal':value['phase']==int(old['phase']),'temporal_abs_error':abs(value['temporal']-float(old['temporal']))})
        write_json(out/'stream_cache_comparison.json',{'frames':quality,'note':'Single-frame FP32 extraction rounded to cache FP16 versus batched extraction cache; phase argmax may differ near ties'})
        detector.quantize=False
        for mode,repeats in [('capacity',3),('paced_30fps',1)]:
            for repeat in range(repeats):
                reserve();detector.reset();torch.cuda.synchronize();origin=time.perf_counter()
                for i,p in enumerate(files):
                    reserve();arrival=origin+i/30 if mode=='paced_30fps' else time.perf_counter()
                    if mode=='paced_30fps':time.sleep(max(0.,arrival-time.perf_counter()))
                    start=time.perf_counter();img=cv2.imread(str(p))
                    if img is None:raise ValueError(f'Unreadable {p}')
                    decoded=time.perf_counter();v=detector.step(img);torch.cuda.synchronize();end=time.perf_counter()
                    timings.append({'mode':mode,'repeat':repeat,'frame':i,'processing_ms':(end-start)*1000,
                      'read_decode_ms':(decoded-start)*1000,'queue_ms':max(0.,start-arrival)*1000,'end_to_end_ms':(end-arrival)*1000,
                      'elapsed_s':end-origin,'score':v['combined_score'] if v else '', 'alarm_active':v['alarm_active'] if v else '',
                      'valid_score':v is not None})
        write_csv(out/'latency_frames.csv',timings);summary={}
        for mode in ['capacity','paced_30fps']:
            group=[r for r in timings if r['mode']==mode];lat=np.array([r['processing_ms'] for r in group]);end=np.array([r['end_to_end_ms'] for r in group]);queue=np.array([r['queue_ms'] for r in group]);total=sum(max(r['elapsed_s'] for r in group if r['repeat']==rep) for rep in sorted({r['repeat'] for r in group}))
            summary[mode]={'input_frames':len(group),'elapsed_seconds':total,'input_fps':len(group)/total,'valid_score_fps':sum(r['valid_score'] for r in group)/total,
              'processing_mean_ms':float(lat.mean()),'processing_p50_ms':float(np.median(lat)),'processing_p95_ms':float(np.quantile(lat,.95)),
              'processing_max_ms':float(lat.max()),'end_to_end_p95_ms':float(np.quantile(end,.95)),
              'deadline_miss_fraction':float((end>1000/30).mean()),'max_queue_ms':float(queue.max()),'last_queue_ms':float(queue[-1])}
        result['benchmark']={'video':video,'fps_assumption':30,'batch_size':1,'precision':'FP32','capacity_repeats':3,'paced_repeats':1,
          'backbone_parameters':sum(p.numel() for p in extractor.backbone.parameters()),'head_parameters':sum(p.numel() for p in head.parameters()),
          'bank_storage_bytes':saved.numel()*saved.element_size(),'bank_compute_bytes':bank.numel()*bank.element_size(),
          'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
          'timings':summary,'scope':'warm model; original JPEG read/decode included, OS page cache may be warm; no camera/network; local GPU only'}
        write_json(out/'benchmark.json',result['benchmark'])
    result['seconds']=time.time()-started;write_json(out/'completed.json',result);event(out,'completed',seconds=result['seconds']);print(json.dumps({'completed':str(out),'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['cache','unit','benchmark'],required=True);p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--backbone',choices=['B','S'],default='B');p.add_argument('--k',type=int,choices=[5,10],default=10)
    a=p.parse_args();torch.set_num_threads(8);reserve()
    if a.action=='cache':cache_small(a.scene)
    elif a.action=='unit':run_unit(a.scene,a.seed,a.backbone,a.k)
    else:benchmark_unit(a.scene,a.seed,a.backbone,a.k)
