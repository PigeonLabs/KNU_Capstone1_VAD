"""Controlled normal-holdout time transformations; not real anomaly-type labels."""
import csv
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from .common import write_json
from .prototype import FeatureClips, PhaseHead
from .phase_routing import reserve, phase_probabilities, digest
from .evaluate import write_csv
from .metrics import period_errors, binary_metrics


def transforms(n):
    base=np.arange(n);at=n//2;length=max(20,int(n*.12));end=min(n-16,at+length)
    if end<=at:raise ValueError('Video too short for controlled transform')
    return {
        'normal':(base,None),
        'pause':(np.concatenate([base[:at],np.full(end-at,at),base[at:]]),(at,end)),
        'reverse':(np.concatenate([base[:at],base[at:end][::-1],base[end:]]),(at,end)),
        'skip':(np.concatenate([base[:at],base[end:]]),(at,at+1)),
        'speed_0.9':(np.minimum((np.arange(0,n,.9)).astype(int),n-1),None),
        'speed_1.1':(np.minimum((np.arange(0,n,1.1)).astype(int),n-1),None)}


def run(scene):
    torch.set_num_threads(8);reserve();started=time.time()
    parent=Path(f'runs/stage3/{scene}/seed0');out=parent/'temporal';out.mkdir(exist_ok=True)
    if (out/'completed.json').exists():return
    if (out/'config.json').exists():raise RuntimeError('Partial temporal diagnostic exists; preserve before rerun')
    dev=FeatureClips(scene,'training',subset='train');val=FeatureClips(scene,'training',subset='val')
    period=float(np.median([r['frames'] for r in dev.records]))
    config={'scene':scene,'seed':0,'period_from_dev':period,'windows':[5,21],
            'types':['normal','pause','reverse','skip','speed_0.9','speed_1.1'],
            'amplitude':.12,'speed_control':'Assumed tolerated +/-10% nominal rate, not a factory specification',
            'positive_support':'transformed event interval expanded by clip support plus 21-frame phase window: 18 frames each side',
            'purpose':'Normal-only controlled diagnostic; no claim of real anomaly cause classification',
            'source_sha256':digest(__file__),'protocol_sha256':digest('docs/stage3_temporal_protocol.md')}
    write_json(out/'config.json',config)
    model=PhaseHead().cuda().eval();model.load_state_dict(torch.load(parent/'development/phase_head.pt',weights_only=False)['model'])
    banks=torch.load(parent/'development/memory.pt',map_location='cpu',weights_only=False)['banks']
    conditional=F.normalize(banks['conditional'].cuda().float(),dim=-1)
    unconditional=F.normalize(banks['unconditional'].cuda().float(),dim=-1)
    groups=banks['groups'].cuda();occupied=torch.unique(groups)
    allrows=[];manifest=[]
    with torch.inference_mode():
        for rec in val.records:
            cls=np.load(rec['path']/'cls.npy',mmap_mode='r');patch=np.load(rec['path']/'patch12.npy',mmap_mode='r')
            for kind,(indices,interval) in transforms(rec['frames']).items():
                current=[]
                manifest.append({'video':rec['video'],'variant':kind,'output_to_source_frame':indices.tolist(),
                                 'event_interval':interval,'original_length':rec['frames']})
                for start in range(0,len(indices)-15,64):
                    reserve();starts=np.arange(start,min(start+64,len(indices)-15))
                    clips=indices[starts[:,None]+np.arange(16)[None,:]]
                    feature=torch.from_numpy(np.array(cls[clips],dtype=np.float32)).cuda()
                    x=F.normalize(torch.from_numpy(np.array(patch[indices[starts+8]],dtype=np.float32)).cuda(),dim=-1)
                    logits=model(feature);prob=phase_probabilities(logits)[:,occupied];top=prob.topk(3,dim=-1)
                    sims=torch.einsum('bpc,pkc->bpk',x,conditional)
                    dist=torch.stack([(1-sims[:,:,groups==o].max(-1).values).clamp_min(0) for o in occupied],-1).mean(1)
                    weighted=(dist.gather(1,top.indices)*(top.values/top.values.sum(-1,keepdim=True))).sum(-1)
                    un=(1-torch.einsum('bpc,pkc->bpk',x,unconditional).max(-1).values).clamp_min(0).mean(-1)
                    values=torch.stack([logits.argmax(-1),un,weighted],-1).cpu().numpy()
                    for s,(phase,u,w) in zip(starts,values):
                        frame=int(s+8);label=int(interval is not None and interval[0]-18<=frame<interval[1]+18)
                        current.append({'scene':scene,'video':rec['video'],'variant':kind,'frame':frame,
                            'source_frame':int(indices[frame]),'phase':int(phase),'label':label,
                            'unconditional':float(u),'top3_weighted':float(w)})
                e5=period_errors([r['phase'] for r in current],period,window=5)
                e21=period_errors([r['phase'] for r in current],period,window=21)
                for r,a,b in zip(current,e5,e21):
                    if np.isfinite(a) and np.isfinite(b):allrows.append({**r,'time5':float(a),'time21':float(b)})
    write_json(out/'manifest.json',manifest)
    keys=['unconditional','top3_weighted','time5','time21']
    normal=[r for r in allrows if r['variant']=='normal']
    # Thresholds are descriptive on this same normal holdout, not independently validated.
    thresholds={k:float(np.quantile([r[k] for r in normal],.95)) for k in keys}
    for r in allrows:
        r['appearance_time_max']=max(r['top3_weighted']/max(thresholds['top3_weighted'],1e-12),r['time21']/max(thresholds['time21'],1e-12))
    keys.append('appearance_time_max');thresholds['appearance_time_max']=1.
    result={'config':config,'thresholds':thresholds,'variants':{},'seconds':time.time()-started,
            'threshold_note':'95th percentile of untransformed holdout; diagnostic calibration, not unbiased normal FPR estimation'}
    for kind in config['types']:
        rows=[r for r in allrows if r['variant']==kind];labels=[r['label'] for r in rows]
        result['variants'][kind]={k:{**binary_metrics(labels,[r[k] for r in rows]),
            'alert_fraction':float(np.mean([r[k]>thresholds[k] for r in rows])),
            'positive_recall_at_normal_p95':float(np.mean([r[k]>thresholds[k] for r in rows if r['label']])) if any(labels) else None}
            for k in keys}
    write_csv(out/'scores.csv',allrows);write_json(out/'completed.json',result)
    print(json.dumps({'scene':scene,'temporal_seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--scene',required=True,choices=['R01','R02','R03','R04'])
    run(p.parse_args().scene)
