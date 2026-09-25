"""Normal-only cross-validation of fixed causal alarm policies."""
import argparse
from pathlib import Path
import time
import numpy as np
from . import stage5 as s5
from . import stage6 as s6
from .stage7 import ROOT,SCENES,begin,load,rows,write_csv
from .common import write_json
from .phase_routing import reserve,digest

RULES=['raw','ewma','hysteresis','ewma_hysteresis']
QGRID=[.95,.975,.99,.995,.999]
ALPHA=.2
OFF_RATIO=.7
CONSECUTIVE=3


def smooth(values,rule):
    x=np.asarray(values,dtype=float)
    if rule not in ['ewma','ewma_hysteresis'] or not len(x):return x.copy()
    y=np.empty_like(x);y[0]=x[0]
    for i in range(1,len(x)):y[i]=ALPHA*x[i]+(1-ALPHA)*y[i-1]
    return y


def flags(values,threshold,rule):
    x=smooth(values,rule)
    if rule not in ['hysteresis','ewma_hysteresis']:
        active,emit=s5.alarm_flags(x,threshold);return x,active,emit
    active=np.zeros(len(x),dtype=bool);emit=active.copy();on=False;hi=lo=0
    for i,value in enumerate(x):
        if not on:
            hi=hi+1 if value>threshold else 0
            if hi>=CONSECUTIVE:on=True;emit[i]=True;lo=0
        else:
            lo=lo+1 if value<=OFF_RATIO*threshold else 0
            if lo>=CONSECUTIVE:on=False;hi=0
        active[i]=on
    return x,active,emit


def transform(data,rule):
    output=[]
    for video in sorted({r['video'] for r in data},key=int):
        g=[r for r in data if r['video']==video];values=smooth([r['combined_score'] for r in g],rule)
        output.extend({**r,'unfiltered_score':r['combined_score'],'combined_score':float(v)} for r,v in zip(g,values))
    return output


def operation(data,scene,threshold,rule):
    normal=raw_fp=active_fp=false_alerts=valid=detected=preexisting=cold=0;delays=[];events=[];frames=[]
    for video in sorted({r['video'] for r in data},key=int):
        g=[r for r in data if r['video']==video];ids=np.array([r['frame'] for r in g]);assert np.all(np.diff(ids)==1);y=np.array([r['label'] for r in g]);x,active,emit=flags([r['combined_score'] for r in g],threshold,rule)
        normal+=int((y==0).sum());raw_fp+=int(((x>threshold)&(y==0)).sum());active_fp+=int((active&(y==0)).sum());false_alerts+=int((emit&(y==0)).sum())
        for i,r in enumerate(g):frames.append({**r,'unfiltered_score':r['combined_score'],'combined_score':float(x[i]),'threshold':threshold,'above_threshold':bool(x[i]>threshold),'alarm_active':bool(active[i]),'new_alarm':bool(emit[i])})
        labels=np.load(s5.label_path('IPAD_dataset',scene,video)).reshape(-1)
        for a,b in s5.label_segments(labels):
            is_cold=a<ids[0];cold+=int(is_cold);indices=np.flatnonzero((ids>=a)&(ids<=b)&active);delay=int(ids[indices[0]]-a) if len(indices) else None;was_active=bool(a>ids[0] and active[a-ids[0]-1])
            events.append({'video':video,'start':int(a),'end':int(b),'cold_start':bool(is_cold),'detected':delay is not None,'delay_frames':delay,'alarm_preexisting_at_onset':was_active})
            if not is_cold:
                valid+=1;preexisting+=int(was_active)
                if delay is not None:detected+=1;delays.append(delay)
    result={'eligible_segments':valid,'detected_segments':detected,'missed_segments':valid-detected,'cold_start_segments':cold,'preexisting_alarm_segments':preexisting,
      'segment_recall':detected/valid if valid else None,'delay_median_frames_detected_only':float(np.median(delays)) if delays else None,'delay_p95_frames_detected_only':float(np.quantile(delays,.95)) if delays else None,
      'normal_frames':normal,'raw_false_positive_frames':raw_fp,'alarm_false_positive_frames':active_fp,'frame_fpr':raw_fp/normal if normal else None,'active_alarm_fpr':active_fp/normal if normal else None,
      'false_alarm_episodes':false_alerts,'false_alarms_per_1000_normal_frames':false_alerts*1000/normal if normal else None,
      'false_alarms_per_minute_assuming_30fps':false_alerts*1800/normal if normal else None}
    return result,events,frames


def choose(validation,rule):
    folds=[]
    for heldout in sorted({r['video'] for r in validation},key=int):
        tr=[r for r in validation if r['video']!=heldout];te=[r for r in validation if r['video']==heldout];cal=s5.calibrate(tr);assert heldout not in cal['calibration_videos']
        fitted=transform(s5.apply_calibration(tr,cal),rule);test=s5.apply_calibration(te,cal)
        for q in QGRID:
            h=float(np.quantile([r['combined_score'] for r in fitted],q));_,a,e=flags([r['combined_score'] for r in test],h,rule)
            folds.append({'heldout_video':heldout,'fit_videos':cal['calibration_videos'],'q':q,'threshold':h,'frames':len(a),'active_fpr':float(a.mean()),'false_alarms_per1000':float(e.sum()*1000/len(a))})
    selected=QGRID[-1];met=False;grid=[]
    for q in QGRID:
        fs=[f for f in folds if f['q']==q];mean_fpr=float(np.mean([f['active_fpr'] for f in fs]));worst=max(f['active_fpr'] for f in fs);events=float(np.mean([f['false_alarms_per1000'] for f in fs]));ok=mean_fpr<=.01 and worst<=.05 and events<=1.
        grid.append({'q':q,'mean_active_fpr':mean_fpr,'max_active_fpr':worst,'mean_false_alarms_per1000':events,'constraint_met':ok})
        if ok and not met:selected=q;met=True
    return {'q':selected,'constraint_met':met,'grid':grid,'folds':folds,'fallback':'q=.999 if none; no false-alarm guarantee'}


def run(scene,seed,b,k):
    out=ROOT/'7-2'/scene/f'seed{seed}'/f'{b}_k{k}';source=s5.unit_path(b,scene,seed,k)
    if not begin(out,scene=scene,seed=seed,backbone=b,k=k,alpha=ALPHA,off_ratio=OFF_RATIO,on_consecutive=CONSECUTIVE,off_consecutive=CONSECUTIVE,qgrid=QGRID,source_metrics_sha256=digest(source/'metrics.json'),decision_sha256=digest(ROOT/'decision.json')):return
    started=time.time();validation=rows(source/'validation_raw.csv');raw=rows(source/'causal_raw.csv');cal=s5.calibrate(validation);scored=s5.apply_calibration(raw,cal);valid=s5.apply_calibration(validation,cal);normal_cfg=load(source/'config.json')
    assert not set(normal_cfg['train_videos'])&set(normal_cfg['calibration_videos']);assert set(cal['calibration_videos'])==set(normal_cfg['calibration_videos'])
    choices={r:choose(validation,r) for r in RULES};write_json(out/'normal_cv.json',choices);write_json(out/'calibration.json',cal);write_json(out/'source.json',{'source':str(source),'validation_sha256':digest(source/'validation_raw.csv'),'train_videos':normal_cfg['train_videos'],'calibration_videos':normal_cfg['calibration_videos']})
    methods={'baseline':('raw',cal['threshold'])}
    for rule in RULES:
        train=transform(valid,rule);methods[rule+'_cv']=(rule,float(np.quantile([r['combined_score'] for r in train],choices[rule]['q'])))
    result={'scene':scene,'seed':seed,'backbone':b,'k':k,'primary':'ewma_hysteresis_cv','methods':{}}
    for method,(rule,h) in methods.items():
        dest=out/method;dest.mkdir(exist_ok=True);common=s5.aligned(transform(scored,rule),scene);full=s5.aligned(scored,scene,common=False);op,events,alarms=operation(full,scene,h,rule)
        write_csv(dest/'scores.csv',common);write_csv(dest/'alarms.csv',alarms);write_csv(dest/'segments.csv',events)
        m={'rule':rule,'threshold':h,'metrics':s5.metric_rows(common),'operation':op,'frames':len(common)}
        if method=='baseline':
            old=rows(source/'scores.csv');assert [(r['video'],r['frame'],r['label']) for r in common]==[(r['video'],r['frame'],r['label']) for r in old]
            assert max(abs(a['combined_score']-b['combined_score']) for a,b in zip(common,old))<1e-10
            baseline_op,_,_=s5.operation_metrics(full,scene,h)
            for key,value in op.items():assert value==baseline_op[key]
        if scene=='R02':write_json(dest/'alignment_sensitivity.json',{str(shift):s5.metric_rows(s5.aligned(transform(scored,rule),scene,policy='common',shift=shift)) for shift in [-1,0,1]})
        write_json(dest/'metrics.json',m);result['methods'][method]=m
    result['seconds']=time.time()-started;write_json(out/'completed.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--backbone',choices=['B','S'],default='B');p.add_argument('--k',type=int,default=10);a=p.parse_args();reserve();run(a.scene,a.seed,a.backbone,a.k)
