"""Bounded precision/budget frontier and normal-video-only calibration study."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from torch.nn import functional as F
from . import stage5 as s5
from .common import write_json,environment
from .phase_routing import reserve,digest
from .evaluate import write_csv
from .metrics import binary_metrics

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False

SCENES=s5.SCENES
VARIANTS=[(b,p,k) for b in ['B','S'] for p in ['fp32','bf16'] for k in [10,5,2]]
QGRID=[.95,.975,.99,.995,.999]


def key(b,p,k):return f'{b}_{p}_k{k}'

def folder(scene,seed,b,p,k):return Path('runs/stage6/6-1')/scene/f'seed{seed}'/key(b,p,k)

def load(p):return json.loads(Path(p).read_text())

def rows(p):
    values=s5.read_csv(p)
    for r in values:
        for k in ['frame','video_length','phase','label']:
            if k in r:r[k]=int(r[k])
        for k in ['appearance','temporal','appearance_score','temporal_score','combined_score']:
            if k in r:r[k]=float(r[k])
    return values


def begin(out,**kw):
    reserve();out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return False
    if (out/'config.json').exists():raise RuntimeError(f'Partial result must be inspected: {out}')
    write_json(out/'config.json',{'started_at':time.time(),'environment':environment(),'allow_bf16_reduced_precision_reduction':torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,'source_sha256':{str(p):digest(p) for p in Path('ipad').glob('*.py')},'protocol_sha256':digest('docs/stage6_protocol.md'),**kw});return True


def memory_path(scene,seed,b,k):return s5.unit_path(b,scene,seed,k)/'memory.pt' if k in [5,10] else folder(scene,seed,b,'fp32',2)/'memory.pt'


def head_path(scene,seed,b):return s5.unit_path(b,scene,seed,10)/'causal_head.pt'


def dtype(p):return torch.float32 if p=='fp32' else torch.bfloat16


class PrecisionExtractor(s5.Extractor):
    def __init__(self,b,p):
        super().__init__(b);self.precision=p;self.backbone.to(dtype=dtype(p))
    def forward(self,x):
        rgb=(x[:,[2,1,0]].float()+1)/2;rgb=F.interpolate(rgb,size=(252,252),mode='bilinear',align_corners=False)
        rgb=((rgb-self.mean)/self.std).to(dtype(self.precision))
        value=self.backbone.get_intermediate_layers(rgb,n=[11],return_class_token=True,norm=True)[0]
        return {'cls':F.normalize(value[1].float(),dim=-1),'patch12':F.normalize(value[0].float(),dim=-1)}


class PrecisionHead(s5.Head):
    def forward(self,x):return self.layers(F.normalize(x.float(),dim=-1).to(self.layers[1].weight.dtype))


def models(scene,seed,b,p,k):
    head=PrecisionHead(768 if b=='B' else 384).cuda().eval();head.load_state_dict(torch.load(head_path(scene,seed,b),map_location='cpu',weights_only=False)['model']);head.to(dtype(p));head.requires_grad_(False)
    bank=torch.load(memory_path(scene,seed,b,k),map_location='cpu',weights_only=False)['bank']
    return head,F.normalize(bank.cuda().float(),dim=-1).to(dtype(p))


def match(x,bank):
    x=F.normalize(x.float(),dim=-1).to(bank.dtype).transpose(0,1)
    if bank.dtype==torch.float32:sim=torch.bmm(x,bank.transpose(1,2))
    else:sim=torch.bmm(x,bank.transpose(1,2),out_dtype=torch.float32)
    return (1-sim.transpose(0,1).max(-1).values).clamp_min(0).mean(-1)


def finish_unit(scene,seed,b,p,k,out,validation,raw,started,**extra):
    cal=s5.calibrate(validation);scored=s5.apply_calibration(raw,cal);common=s5.aligned(scored,scene);full=s5.aligned(scored,scene,common=False)
    write_csv(out/'validation_raw.csv',validation);write_csv(out/'causal_raw.csv',raw);write_json(out/'calibration.json',cal)
    write_csv(out/'scores.csv',common);write_csv(out/'online_scores.csv',full)
    old=rows(s5.unit_path('B',scene,seed,10)/'scores.csv');assert [(r['video'],r['frame'],r['label']) for r in common]==[(r['video'],r['frame'],r['label']) for r in old]
    op,events,_=s5.operation_metrics(full,scene,cal['threshold']);write_json(out/'operation.json',op);write_csv(out/'segments.csv',events)
    bank_meta=torch.load(memory_path(scene,seed,b,k),map_location='cpu',weights_only=False)['bank']
    period=load(s5.unit_path(b,scene,seed,10)/'metrics.json')['period']
    m={'scene':scene,'seed':seed,'backbone':b,'precision':p,'k':k,'period':period,'frames':len(common),'online_frames':len(full),
       'metrics':s5.metric_rows(common),'online_metrics':s5.metric_rows(full),'operation':op,
       'memory_path':str(memory_path(scene,seed,b,k)),'memory_sha256':digest(memory_path(scene,seed,b,k)),
       'head_path':str(head_path(scene,seed,b)),'head_sha256':digest(head_path(scene,seed,b)),
       'bank_elements':bank_meta.numel(),'bank_runtime_bytes':bank_meta.numel()*(4 if p=='fp32' else 2),'seconds':time.time()-started,**extra}
    if scene=='R02':write_json(out/'alignment_sensitivity.json',{str(shift):s5.metric_rows(s5.aligned(scored,scene,policy='common',shift=shift)) for shift in [-1,0,1]})
    write_json(out/'metrics.json',m);write_json(out/'completed.json',m);s5.event(out,'completed',seconds=m['seconds'])


def fp32_unit(scene,seed,b,k):
    out=folder(scene,seed,b,'fp32',k)
    if not begin(out,scene=scene,seed=seed,backbone=b,precision='fp32',k=k):return
    started=time.time();base=s5.unit_path(b,scene,seed,10)
    if k in [5,10]:
        source=s5.unit_path(b,scene,seed,k);val=rows(source/'validation_raw.csv');raw=rows(source/'causal_raw.csv')
        finish_unit(scene,seed,b,'fp32',k,out,val,raw,started,reused_from=str(source),reused_metrics_sha256=digest(source/'metrics.json'))
        current=rows(out/'scores.csv');old=rows(source/'scores.csv');error=max(abs(a['combined_score']-o['combined_score']) for a,o in zip(current,old));assert error<1e-10
        write_json(out/'baseline_equivalence.json',{'score_max_error':error})
    else:
        recs=s5.records(b,scene,'training','train');bank=s5.fit_memory(recs,k,seed,out);head,_=models(scene,seed,b,'fp32',k);period=load(base/'metrics.json')['period']
        val=s5.infer(s5.records(b,scene,'training','val'),head,bank,'causal',period);raw=s5.infer(s5.records(b,scene,'testing'),head,bank,'causal',period)
        assert (out/'memory_selection.csv').read_text()==(base/'memory_selection.csv').read_text()
        finish_unit(scene,seed,b,'fp32',k,out,val,raw,started)


@torch.inference_mode()
def bf16_scene(scene,b):
    group=Path('runs/stage6/6-1/streams')/scene/b
    if not begin(group,scene=scene,backbone=b,precision='bf16',storage='No feature cache; normalized FP32 feature SHA256 only'):return
    started=time.time();extractor=PrecisionExtractor(b,'bf16').cuda().eval();heads={};banks={};outputs={};periods={}
    for seed in range(3):
        for k in [10,5,2]:
            out=folder(scene,seed,b,'bf16',k);assert begin(out,scene=scene,seed=seed,backbone=b,precision='bf16',k=k)
            head,bank=models(scene,seed,b,'bf16',k);heads[seed]=head;banks[seed,k]=bank;outputs[seed,k]={'validation':[],'testing':[]}
        periods[seed]=load(s5.unit_path(b,scene,seed,10)/'metrics.json')['period']
    feature_hashes=[]
    for part,recs in [('validation',s5.records(b,scene,'training','val')),('testing',s5.records(b,scene,'testing'))]:
        split='training' if part=='validation' else 'testing'
        for rec in recs:
            reserve();inputs=np.load(Path('cache/frames')/scene/split/f"{rec['video']}.npy",mmap_mode='r');cs=[];ps=[];h=hashlib.sha256()
            for a in range(0,len(inputs),64):
                reserve();x=torch.from_numpy(np.array(inputs[a:a+64])).cuda().permute(0,3,1,2).float()/127.5-1;f=extractor(x)
                for name in ['cls','patch12']:h.update(f[name].cpu().numpy().tobytes())
                cs.append(f['cls']);ps.append(f['patch12'])
            cls=torch.cat(cs);patch=torch.cat(ps);del cs,ps,inputs
            feature_hashes.append({'split':split,'video':rec['video'],'frames':rec['frames'],'sha256':h.hexdigest(),'hash_order':'batch64, CLS then patch12, normalized float32 little-endian'})
            for seed in range(3):
                temp={k:[] for k in [10,5,2]}
                for a in range(15,rec['frames'],64):
                    reserve();target=torch.arange(a,min(a+64,rec['frames']),device='cuda');phase=heads[seed](cls[target[:,None]+torch.arange(-15,1,device='cuda')]).argmax(-1).cpu().tolist()
                    distances={k:match(patch[target],banks[seed,k]).cpu().tolist() for k in [10,5,2]}
                    for i,t in enumerate(target.cpu().tolist()):
                        for k in [10,5,2]:temp[k].append({'video':rec['video'],'frame':t,'video_length':rec['frames'],'phase':phase[i],'appearance':distances[k][i]})
                for k in [10,5,2]:
                    temporal=s5.causal_time([r['phase'] for r in temp[k]],periods[seed])
                    outputs[seed,k][part].extend({**r,'temporal':float(t)} for r,t in zip(temp[k],temporal) if np.isfinite(t))
            del cls,patch;s5.event(group,'video',part=part,video=rec['video'],frames=rec['frames'])
    write_json(group/'feature_hashes.json',feature_hashes)
    for seed in range(3):
        for k in [10,5,2]:
            result=outputs[seed,k];finish_unit(scene,seed,b,'bf16',k,folder(scene,seed,b,'bf16',k),result['validation'],result['testing'],started,
                 feature_hashes_path=str(group/'feature_hashes.json'),feature_hashes_sha256=digest(group/'feature_hashes.json'))
    write_json(group/'completed.json',{'seconds':time.time()-started,'units':9,'backbone_dtype':str(next(extractor.backbone.parameters()).dtype)})


class Detector(s5.OnlineDetector):
    @torch.inference_mode()
    def step(self,bgr):
        image=cv2.resize(bgr,(256,256));x=torch.from_numpy(image).cuda().permute(2,0,1)[None].float()/127.5-1
        f=self.extractor(x);self.cls.append(f['cls'][0])
        if len(self.cls)<16:return None
        phase=int(self.head(torch.stack(list(self.cls))[None]).argmax(-1));self.phases.append(phase);appearance=float(match(f['patch12'],self.bank)[0])
        if len(self.phases)<21:return None
        temporal=float(s5.causal_time(self.phases,self.period)[-1]);score=(s5.scaled(appearance,self.cal,'appearance')+s5.scaled(temporal,self.cal,'temporal'))/2
        self.streak=self.streak+1 if score>self.cal['threshold'] else 0
        return {'phase':phase,'appearance':appearance,'temporal':temporal,'combined_score':score,'alarm_active':self.streak>=3}


def benchmark(scene,b,p,k):
    unit=folder(scene,0,b,p,k);out=unit/'benchmark'
    if not begin(out,scene=scene,seed=0,backbone=b,precision=p,k=k,source_metrics_sha256=digest(unit/'metrics.json')):return
    started=time.time();m=load(unit/'metrics.json');cal=load(unit/'calibration.json');head,bank=models(scene,0,b,p,k);extractor=PrecisionExtractor(b,p).cuda().eval()
    detector=Detector(extractor,head,bank,cal,m['period']);video=rows(unit/'scores.csv')[0]['video'];files=s5.frames(Path('IPAD_dataset')/scene/'testing/frames'/video)[:256]
    train=s5.records(b,scene,'training','train')[0];warm=s5.frames(Path('IPAD_dataset')/scene/'training/frames'/train['video'])[:36]
    for f in warm:reserve();detector.step(cv2.imread(str(f)))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[];quality=[];reference={r['frame']:r for r in rows(unit/'causal_raw.csv') if r['video']==video}
    for mode,repeats in [('capacity',3),('paced_30fps',1)]:
        for repeat in range(repeats):
            reserve();detector.reset();torch.cuda.synchronize();origin=time.perf_counter()
            for i,file in enumerate(files):
                reserve();arrival=origin+i/30 if mode=='paced_30fps' else time.perf_counter()
                if mode=='paced_30fps':time.sleep(max(0.,arrival-time.perf_counter()))
                start=time.perf_counter();img=cv2.imread(str(file));decoded=time.perf_counter()
                if img is None:raise ValueError(file)
                value=detector.step(img);torch.cuda.synchronize();end=time.perf_counter()
                timings.append({'mode':mode,'repeat':repeat,'frame':i,'processing_ms':(end-start)*1000,'read_decode_ms':(decoded-start)*1000,
                    'queue_ms':max(0.,start-arrival)*1000,'end_to_end_ms':(end-arrival)*1000,'elapsed_s':end-origin,'valid_score':value is not None,
                    'score':value['combined_score'] if value else '', 'alarm_active':value['alarm_active'] if value else ''})
                if mode=='capacity' and repeat==0 and value:
                    r=reference[i];quality.append({'frame':i,'phase_equal':value['phase']==r['phase'],'appearance_abs_error':abs(value['appearance']-r['appearance']),'temporal_abs_error':abs(value['temporal']-r['temporal'])})
    write_csv(out/'latency_frames.csv',timings);write_csv(out/'stream_batch_comparison.csv',quality);summary={}
    for mode in ['capacity','paced_30fps']:
        g=[r for r in timings if r['mode']==mode];lat=np.array([r['processing_ms'] for r in g]);steady=np.array([r['processing_ms'] for r in g if r['valid_score']]);e=np.array([r['end_to_end_ms'] for r in g]);q=np.array([r['queue_ms'] for r in g]);total=sum(max(r['elapsed_s'] for r in g if r['repeat']==rep) for rep in sorted({r['repeat'] for r in g}))
        summary[mode]={'frames':len(g),'input_fps':len(g)/total,'processing_p95_ms':float(np.quantile(lat,.95)),'steady_p95_ms':float(np.quantile(steady,.95)),
          'end_to_end_p95_ms':float(np.quantile(e,.95)),'deadline_miss_fraction':float((e>1000/30).mean()),'max_queue_ms':float(q.max()),'last_queue_ms':float(q[-1])}
    result={'scene':scene,'backbone':b,'precision':p,'k':k,'batch_size':1,'video':video,'timings':summary,
      'backbone_dtype':str(next(extractor.backbone.parameters()).dtype),'head_dtype':str(next(head.parameters()).dtype),'bank_dtype':str(bank.dtype),
      'parameter_bytes':sum(v.numel()*v.element_size() for v in list(extractor.parameters())+list(head.parameters())),
      'bank_bytes':bank.numel()*bank.element_size(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
      'phase_agreement':float(np.mean([r['phase_equal'] for r in quality])),'appearance_max_difference':max(r['appearance_abs_error'] for r in quality),
      'scope':'Current GPU, original JPEG decode, OS cache may be warm; 30FPS paced replay, no camera/network; warm model, video reset and35-frame coldstart', 'seconds':time.time()-started}
    write_json(out/'completed.json',result)


def weighted_quantile(values,videos,q):
    values=np.asarray(values,dtype=float);videos=np.asarray(videos);_,inv,counts=np.unique(videos,return_inverse=True,return_counts=True)
    weights=1/counts[inv];order=np.argsort(values,kind='stable');cumulative=np.cumsum(weights[order]);index=np.searchsorted(cumulative,q*cumulative[-1],side='left')
    return float(values[order[min(index,len(values)-1)]])


def fit_calibrator(data,kind):
    v=[r['video'] for r in data];result={'kind':kind,'videos':sorted(set(v)),'global':{},'phase':{}}
    for component in ['appearance','temporal']:
        values=[r[component] for r in data];result['global'][component]=[weighted_quantile(values,v,.5),weighted_quantile(values,v,.995)]
    if kind!='balanced':
        bins=np.array([r['phase']//10 for r in data])
        for b in range(20):
            delta=np.abs(bins-b);take=np.minimum(delta,20-delta)<=1;subset=[r for r,t in zip(data,take) if t];n=len(subset);video=[r['video'] for r in subset];stats={}
            eligible=n>=100 and len(set(video))>=2;weight=n/(n+200) if eligible else 0.
            for component in ['appearance','temporal']:
                global_q=np.array(result['global'][component]);local=np.array([weighted_quantile([r[component] for r in subset],video,q) for q in [.5,.995]]) if eligible else global_q
                stats[component]=((1-weight)*global_q+weight*local).tolist()
            result['phase'][str(b)]={'stats':stats,'frames':n,'videos':len(set(video)),'weight':weight}
    return result


def apply_calibrator(data,cal):
    out=[]
    for r in data:
        stats=cal['global'] if cal['kind']=='balanced' else cal['phase'][str(r['phase']//10)]['stats'];scores=[]
        for component in ['appearance','temporal']:
            lo,hi=stats[component];scores.append(max(0.,(r[component]-lo)/max(hi-lo,1e-8)))
        out.append({**r,'appearance_score':scores[0],'temporal_score':scores[1], 'combined_score':max(scores) if cal['kind']=='phase_max' else sum(scores)/2})
    return out


def choose_normal_quantile(data,kind):
    logs=[]
    for heldout in sorted({r['video'] for r in data}):
        train=[r for r in data if r['video']!=heldout];test=[r for r in data if r['video']==heldout];cal=fit_calibrator(train,kind)
        assert heldout not in cal['videos'];tr=apply_calibrator(train,cal);te=apply_calibrator(test,cal)
        for q in QGRID:
            threshold=weighted_quantile([r['combined_score'] for r in tr],[r['video'] for r in tr],q)
            active,_=s5.alarm_flags([r['combined_score'] for r in te],threshold)
            logs.append({'heldout_video':heldout,'fit_videos':cal['videos'],'q':q,'threshold':threshold,'active_fpr':float(active.mean()),'frames':len(active)})
    choice=QGRID[-1];met=False;grid=[]
    for q in QGRID:
        rates=[r['active_fpr'] for r in logs if r['q']==q];mean=float(np.mean(rates));worst=max(rates);ok=mean<=.01 and worst<=.05
        grid.append({'q':q,'mean_video_active_fpr':mean,'worst_video_active_fpr':worst,'constraint_met':ok})
        if ok and not met:choice=q;met=True
    return {'q':choice,'constraint_met':met,'grid':grid,'folds':logs,'policy':'lowest q with mean<=1% and worst<=5%; else .999 without coverage claim'}


def calibration_unit(scene,seed,b,k):
    root=Path('runs/stage6/6-2')/scene/f'seed{seed}'/f'{b}_k{k}';source=s5.unit_path(b,scene,seed,k)
    if not begin(root,scene=scene,seed=seed,backbone=b,k=k,source_metrics_sha256=digest(source/'metrics.json')):return
    started=time.time();validation=rows(source/'validation_raw.csv');test=rows(source/'causal_raw.csv');base_cal=load(source/'calibration.json');config=load(source/'config.json')
    assert not set(config['train_videos'])&set(config['calibration_videos'])
    choices={};calibrators={};outputs={'baseline':(s5.apply_calibration(test,base_cal),base_cal['threshold'])}
    for kind in ['balanced','phase_mean','phase_max']:
        reserve();choice=choose_normal_quantile(validation,kind);choices[kind]=choice;cal=fit_calibrator(validation,kind);calibrators[kind]=cal;fitted=apply_calibrator(validation,cal);scored=apply_calibrator(test,cal)
        for rule,q in [('fixed',.995),('cv',choice['q'])]:
            threshold=weighted_quantile([r['combined_score'] for r in fitted],[r['video'] for r in fitted],q);outputs[f'{kind}_{rule}']=(scored,threshold)
    write_json(root/'normal_cv.json',choices);write_json(root/'calibrators.json',calibrators);write_json(root/'normal_source.json',{'validation_sha256':digest(source/'validation_raw.csv'),'train_videos':config['train_videos'],'calibration_videos':config['calibration_videos']})
    result={'scene':scene,'seed':seed,'backbone':b,'k':k,'primary':'phase_mean_cv','methods':{}}
    for method,(scored,threshold) in outputs.items():
        dest=root/method;dest.mkdir(exist_ok=True);common=s5.aligned(scored,scene);full=s5.aligned(scored,scene,common=False)
        op,events,alarm=s5.operation_metrics(full,scene,threshold);write_csv(dest/'scores.csv',common);write_csv(dest/'alarms.csv',alarm);write_csv(dest/'segments.csv',events)
        m={'metrics':s5.metric_rows(common),'online_metrics':s5.metric_rows(full),'operation':op,'threshold':threshold,'frames':len(common)}
        if method=='baseline':
            old=rows(source/'scores.csv');assert max(abs(a['combined_score']-o['combined_score']) for a,o in zip(common,old))<1e-10
        if scene=='R02':write_json(dest/'alignment_sensitivity.json',{str(shift):s5.metric_rows(s5.aligned(scored,scene,policy='common',shift=shift)) for shift in [-1,0,1]})
        write_json(dest/'metrics.json',m);result['methods'][method]=m
    result['seconds']=time.time()-started;write_json(root/'completed.json',result)


def pareto(points,timing_tolerance=0.):
    winners=[]
    for i,a in enumerate(points):
        dominated=False
        for j,b in enumerate(points):
            if i==j:continue
            time_equal=abs(a['latency_ms']-b['latency_ms'])<=timing_tolerance*max(a['latency_ms'],b['latency_ms'])
            time_no_worse=time_equal or b['latency_ms']<=a['latency_ms'];time_better=not time_equal and b['latency_ms']<a['latency_ms']
            if b['auroc']>=a['auroc'] and b['memory_gib']<=a['memory_gib'] and time_no_worse and (b['auroc']>a['auroc'] or b['memory_gib']<a['memory_gib'] or time_better):dominated=True;break
        if not dominated:winners.append(a['variant'])
    return winners


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['fp32','bf16','benchmark','calibration'],required=True);p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--backbone',choices=['B','S'],default='B');p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--k',type=int,choices=[2,5,10],default=10)
    a=p.parse_args();torch.set_num_threads(8);reserve()
    if a.action=='fp32':fp32_unit(a.scene,a.seed,a.backbone,a.k)
    elif a.action=='bf16':bf16_scene(a.scene,a.backbone)
    elif a.action=='benchmark':benchmark(a.scene,a.backbone,a.precision,a.k)
    else:calibration_unit(a.scene,a.seed,a.backbone,a.k)
