"""Stage seven: diagnostic decomposition, causal alerts, and full batch-one replay."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from . import stage5 as s5
from . import stage6 as s6
from .common import environment,write_json
from .evaluate import write_csv as _write_csv
import csv
from .phase_routing import reserve,digest

SCENES=s5.SCENES
ROOT=Path('runs/stage7')
load=s6.load
rows=s6.rows


def begin(out,**kw):
    reserve();out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return False
    if (out/'config.json').exists():raise RuntimeError(f'Partial result requires inspection: {out}')
    write_json(out/'config.json',{'started_at':time.time(),'environment':environment(),'source_sha256':{str(p):digest(p) for p in Path('ipad').glob('*.py')},'protocol_sha256':digest('docs/stage7_protocol.md'),**kw});return True


def write_csv(path,data,fields=None):
    if data:return _write_csv(path,data)
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        if fields:csv.writer(f).writerow(fields)


def quantiles(values):
    x=np.asarray(values,dtype=float)
    return dict(zip(['median','q95','q995','max'],map(float,np.quantile(x,[.5,.95,.995,1])))) if len(x) else None


def spans(flags):
    v=np.asarray(flags,dtype=int);diff=np.diff(np.r_[0,v,0]);return list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)-1))


def diagnose(scene,seed,b,k):
    out=ROOT/'7-1'/scene/f'seed{seed}'/f'{b}_k{k}';source=s5.unit_path(b,scene,seed,k)
    if not begin(out,scene=scene,seed=seed,backbone=b,k=k,source=str(source),source_metrics_sha256=digest(source/'metrics.json')):return
    started=time.time();cal=load(source/'calibration.json');validation=rows(source/'validation_raw.csv');vscore=s5.apply_calibration(validation,cal);test=rows(source/'causal_raw.csv');scored=s5.apply_calibration(test,cal);full=s5.aligned(scored,scene,common=False);common=s5.aligned(scored,scene)
    cfg=load(source/'config.json');assert not set(cfg['train_videos'])&set(cfg['calibration_videos'])
    result={'scene':scene,'seed':seed,'backbone':b,'k':k,'components':{}}
    distributions=[]
    for split,data in [('normal_calibration',vscore),('test_normal',[r for r in full if r['label']==0]),('test_abnormal',[r for r in full if r['label']==1])]:
        for video in sorted({r['video'] for r in data}):
            group=[r for r in data if r['video']==video]
            for component in ['appearance_score','temporal_score','combined_score']:
                distributions.append({'split':split,'video':video,'component':component,'frames':len(group),**quantiles([r[component] for r in group])})
    write_csv(out/'score_distributions.csv',distributions)
    for component in ['appearance_score','temporal_score','combined_score']:
        threshold=float(np.quantile([r[component] for r in vscore],.995));evaluated=[{**r,'combined_score':r[component]} for r in full]
        op,events,alarms=s5.operation_metrics(evaluated,scene,threshold);write_csv(out/f'{component}_alarms.csv',alarms);write_csv(out/f'{component}_segments.csv',events)
        result['components'][component]={'threshold':threshold,'metrics':s5.binary_metrics([r['label'] for r in common],[r[component] for r in common]),'operation':op,
          'normal_calibration':quantiles([r[component] for r in vscore]),'test_normal':quantiles([r[component] for r in full if r['label']==0]),'test_abnormal':quantiles([r[component] for r in full if r['label']==1])}
        if component=='combined_score':
            stretches=[];misses=[]
            for video in sorted({r['video'] for r in alarms}):
                group=[r for r in alarms if r['video']==video]
                for a,z in spans([r['alarm_active'] and r['label']==0 for r in group]):
                    stretches.append({'video':video,'start':group[a]['frame'],'end':group[z]['frame'],'frames':z-a+1,'mean_appearance':float(np.mean([r['appearance_score'] for r in group[a:z+1]])),'mean_temporal':float(np.mean([r['temporal_score'] for r in group[a:z+1]]))})
                for e in events:
                    if e['video']!=video or e['detected'] or e['cold_start']:continue
                    g=[r for r in group if e['start']<=r['frame']<=e['end']]
                    misses.append({**e,'any_raw_threshold_crossing':any(r['above_threshold'] for r in g),'max_score':max(r['combined_score'] for r in g),'max_appearance':max(r['appearance_score'] for r in g),'max_temporal':max(r['temporal_score'] for r in g)})
            write_csv(out/'false_alarm_stretches.csv',stretches,['video','start','end','frames','mean_appearance','mean_temporal']);write_csv(out/'missed_segment_diagnostics.csv',misses,['video','start','end','cold_start','detected','delay_frames','alarm_preexisting_at_onset','any_raw_threshold_crossing','max_score','max_appearance','max_temporal'])
            result['false_alarm_stretches']={'count':len(stretches),'duration_frames':quantiles([r['frames'] for r in stretches]),'normal_alarm_frames_in_stretches_ge30':sum(r['frames'] for r in stretches if r['frames']>=30),'total_normal_alarm_frames':sum(r['frames'] for r in stretches)}
            result['misses_with_any_threshold_crossing']=sum(r['any_raw_threshold_crossing'] for r in misses)
    phase=[]
    for video in sorted({r['video'] for r in validation}):
        g=[r for r in validation if r['video']==video];p=np.array([r['phase'] for r in g]);target=np.array([r['frame']*200/r['video_length'] for r in g]);error=np.abs((p-target+100)%200-100);jumps=np.abs((np.diff(p)+100)%200-100)
        phase.append({'video':video,'circular_error_class_units':quantiles(error),'circular_step_class_units':quantiles(jumps),'note':'relative phase target only from normal calibration videos'})
    write_json(out/'normal_phase_diagnostics.json',phase)
    result['seconds']=time.time()-started;write_json(out/'completed.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['diagnose'],required=True);p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--backbone',choices=['B','S'],default='B');p.add_argument('--k',type=int,default=10)
    a=p.parse_args();torch.set_num_threads(8);reserve();diagnose(a.scene,a.seed,a.backbone,a.k)
