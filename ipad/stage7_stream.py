"""Full JPEG batch-one accuracy and isolated single-model streaming measurements."""
import argparse
from collections import deque
import hashlib
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from . import stage5 as s5
from . import stage6 as s6
from .stage7 import ROOT,SCENES,begin,load,rows
from .common import write_json
from .evaluate import write_csv
from .phase_routing import reserve,digest

VARIANTS=['fp32','bf16','bf16_mixed']


def folder(scene,seed,variant):return ROOT/'7-3'/scene/f'seed{seed}'/f'B_k10_{variant}'


def models(scene,seed,variant):
    precision='bf16' if variant=='bf16' else 'fp32';head,bank=s6.models(scene,seed,'B',precision,10)
    if variant=='bf16_mixed':bank=bank.to(torch.bfloat16)
    return head,bank


def sensitivity(validation,raw,scene):
    results=[]
    for video in sorted({r['video'] for r in validation}):
        train=[r for r in validation if r['video']!=video];heldout=[r for r in validation if r['video']==video];cal=s5.calibrate(train)
        assert video not in cal['calibration_videos'];v=s5.apply_calibration(heldout,cal);active,emitted=s5.alarm_flags([r['combined_score'] for r in v],cal['threshold'])
        scored=s5.apply_calibration(raw,cal);common=s5.aligned(scored,scene);full=s5.aligned(scored,scene,common=False);op,_,_=s5.operation_metrics(full,scene,cal['threshold'])
        results.append({'heldout_video':video,'calibration':cal,'heldout_normal_frames':len(v),'heldout_active_fpr':float(active.mean()),'heldout_false_alarms_per1000':float(emitted.sum()*1000/len(v)),
          'test_metrics':s5.metric_rows(common),'test_operation':op,'selection':'none; all normal-calibration leave-one-video-out fits reported'})
    return results


def finish(scene,seed,variant,validation,raw,group,started):
    out=folder(scene,seed,variant);cal=s5.calibrate(validation);scored=s5.apply_calibration(raw,cal);common=s5.aligned(scored,scene);full=s5.aligned(scored,scene,common=False)
    write_csv(out/'validation_raw.csv',validation);write_csv(out/'causal_raw.csv',raw);write_json(out/'calibration.json',cal);write_csv(out/'scores.csv',common);write_csv(out/'online_scores.csv',full)
    op,events,alarms=s5.operation_metrics(full,scene,cal['threshold']);write_csv(out/'alarms.csv',alarms);write_csv(out/'segments.csv',events)
    precision='fp32' if variant=='fp32' else 'bf16';old=s6.folder(scene,seed,'B',precision,10);reference=rows(old/'scores.csv')
    assert [(r['video'],r['frame'],r['label']) for r in common]==[(r['video'],r['frame'],r['label']) for r in reference]
    old_raw={(r['video'],r['frame']):r for r in rows(old/'causal_raw.csv')};comparison=[]
    for r in raw:
        ref=old_raw[r['video'],r['frame']];comparison.append({'video':r['video'],'frame':r['frame'],'phase_equal':r['phase']==ref['phase'],'appearance_abs_error':abs(r['appearance']-ref['appearance']),'temporal_abs_error':abs(r['temporal']-ref['temporal'])})
    write_csv(out/'batch_comparison.csv',comparison);folds=sensitivity(validation,raw,scene);write_json(out/'normal_calibration_lovo.json',folds)
    cfg=load(s5.unit_path('B',scene,seed,10)/'config.json');assert set(cal['calibration_videos'])==set(cfg['calibration_videos']);assert not set(cfg['train_videos'])&set(cfg['calibration_videos'])
    m={'scene':scene,'seed':seed,'variant':variant,'metrics':s5.metric_rows(common),'operation':op,'frames':len(common),'online_frames':len(full),'threshold':cal['threshold'],
      'period':load(s5.unit_path('B',scene,seed,10)/'metrics.json')['period'],'source_model':str(s5.unit_path('B',scene,seed,10)),
      'head_sha256':digest(s6.head_path(scene,seed,'B')),'bank_sha256':digest(s6.memory_path(scene,seed,'B',10)),
      'training_history_sha256':digest(s5.unit_path('B',scene,seed,10)/'causal_training.json'),'new_training':False,
      'feature_hashes_path':str(group/'feature_hashes.json'),'feature_hashes_sha256':digest(group/'feature_hashes.json'),
      'batch_reference':str(old),'batch_metrics':load(old/'metrics.json')['metrics'],'batch_phase_agreement':float(np.mean([r['phase_equal'] for r in comparison])),
      'batch_appearance_max_difference':max(r['appearance_abs_error'] for r in comparison),'normal_calibration_folds':len(folds),
      'seconds':time.time()-started,'accuracy_pass_timing_is_not_single_model_benchmark':True}
    if scene=='R02':write_json(out/'alignment_sensitivity.json',{str(shift):s5.metric_rows(s5.aligned(scored,scene,policy='common',shift=shift)) for shift in [-1,0,1]})
    write_json(out/'completed.json',m)


@torch.inference_mode()
def accuracy(scene,precision):
    group=ROOT/'7-3/streams'/scene/precision
    if not begin(group,scene=scene,precision=precision,extraction_batch_size=1,input='original JPEG',storage='no full feature cache'):return
    started=time.time();extractor=s6.PrecisionExtractor('B',precision).cuda().eval();variants=['fp32'] if precision=='fp32' else ['bf16','bf16_mixed'];heads={};banks={};output={};period={};hashes=[]
    for seed in range(3):
        period[seed]=load(s5.unit_path('B',scene,seed,10)/'metrics.json')['period']
        for variant in variants:
            assert begin(folder(scene,seed,variant),scene=scene,seed=seed,variant=variant,extraction_batch_size=1,source_metrics_sha256=digest(s5.unit_path('B',scene,seed,10)/'metrics.json'))
            heads[seed,variant],banks[seed,variant]=models(scene,seed,variant);output[seed,variant]={'validation':[],'testing':[]}
    for part,recs in [('validation',s5.records('B',scene,'training','val')),('testing',s5.records('B',scene,'testing'))]:
        split='training' if part=='validation' else 'testing'
        for rec in recs:
            reserve();files=s5.frames(Path('IPAD_dataset')/scene/split/'frames'/rec['video']);assert len(files)==rec['frames'];cls=deque(maxlen=16);phase={key:deque(maxlen=21) for key in heads};h=hashlib.sha256()
            for t,file in enumerate(files):
                reserve();img=cv2.imread(str(file))
                if img is None:raise ValueError(file)
                image=cv2.resize(img,(256,256));x=torch.from_numpy(image).cuda().permute(2,0,1)[None].float()/127.5-1;features=extractor(x)
                for name in ['cls','patch12']:h.update(features[name].cpu().numpy().tobytes())
                cls.append(features['cls'][0])
                if len(cls)<16:continue
                history=torch.stack(list(cls))[None]
                for key,head in heads.items():
                    predicted=int(head(history).argmax(-1));phase[key].append(predicted)
                    if len(phase[key])<21:continue
                    appearance=float(s6.match(features['patch12'],banks[key])[0]);temporal=float(s5.causal_time(phase[key],period[key[0]])[-1])
                    output[key][part].append({'video':rec['video'],'frame':t,'video_length':rec['frames'],'phase':predicted,'appearance':appearance,'temporal':temporal})
            hashes.append({'split':split,'video':rec['video'],'frames':len(files),'sha256':h.hexdigest(),'order':'frame ascending; normalized float32 CLS then patch12; batch1'})
            s5.event(group,'video',split=split,video=rec['video'],frames=len(files),seconds=time.time()-started)
    write_json(group/'feature_hashes.json',hashes)
    for (seed,variant),data in output.items():finish(scene,seed,variant,data['validation'],data['testing'],group,started)
    write_json(group/'completed.json',{'scene':scene,'precision':precision,'units':len(output),'backbone_dtype':str(next(extractor.backbone.parameters()).dtype),'head_dtypes':{v:str(next(heads[0,v].parameters()).dtype) for v in variants},'seconds':time.time()-started})


def benchmark(scene,variant):
    unit=folder(scene,0,variant);out=unit/'benchmark'
    if not begin(out,scene=scene,seed=0,variant=variant,input='all eligible test JPEGs',batch_size=1,capacity_repeats=1,paced='longest eligible video at30FPS'):return
    started=time.time();m=load(unit/'completed.json');cal=load(unit/'calibration.json');precision='fp32' if variant=='fp32' else 'bf16';extractor=s6.PrecisionExtractor('B',precision).cuda().eval();head,bank=models(scene,0,variant);detector=s6.Detector(extractor,head,bank,cal,m['period'])
    reference=rows(unit/'online_scores.csv');reference_alarms={(r['video'],int(r['frame'])):r['alarm_active']=='True' for r in s5.read_csv(unit/'alarms.csv')};refs={(r['video'],r['frame']):r for r in reference};videos=sorted({r['video'] for r in reference},key=int);filemap={v:s5.frames(Path('IPAD_dataset')/scene/'testing/frames'/v) for v in videos};longest=max(videos,key=lambda v:len(filemap[v]))
    train=s5.records('B',scene,'training','train')[0];warm=s5.frames(Path('IPAD_dataset')/scene/'training/frames'/train['video'])[:36]
    for f in warm:reserve();detector.step(cv2.imread(str(f)))
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[];live=[];quality=[];video_summaries=[]
    for mode,vids in [('capacity',videos),('paced_30fps',[longest])]:
        for vid in vids:
            detector.reset();torch.cuda.synchronize();origin=time.perf_counter()
            for frame,file in enumerate(filemap[vid]):
                reserve();arrival=origin+frame/30 if mode=='paced_30fps' else time.perf_counter()
                if mode=='paced_30fps':time.sleep(max(0.,arrival-time.perf_counter()))
                tick=time.perf_counter();image=cv2.imread(str(file));decoded=time.perf_counter()
                if image is None:raise ValueError(file)
                value=detector.step(image);torch.cuda.synchronize();end=time.perf_counter()
                timings.append({'mode':mode,'video':vid,'frame':frame,'processing_ms':(end-tick)*1000,'read_decode_ms':(decoded-tick)*1000,'queue_ms':max(0.,tick-arrival)*1000,'end_to_end_ms':(end-arrival)*1000,'elapsed_s':end-origin,'valid_score':value is not None,'score':value['combined_score'] if value else '', 'alarm_active':value['alarm_active'] if value else ''})
                if value:
                    ref=refs[vid,frame];quality.append({'mode':mode,'video':vid,'frame':frame,'phase_equal':value['phase']==ref['phase'],'score_abs_error':abs(value['combined_score']-ref['combined_score']),'alarm_equal':value['alarm_active']==reference_alarms[vid,frame]})
                    if mode=='capacity':live.append({**ref,**{c:value[c] for c in ['phase','appearance','temporal','combined_score']},'appearance_score':s5.scaled(value['appearance'],cal,'appearance'),'temporal_score':s5.scaled(value['temporal'],cal,'temporal')})
            video_summaries.append({'mode':mode,'video':vid,'frames':len(filemap[vid]),'elapsed_s':end-origin});s5.event(out,'video',**video_summaries[-1])
    write_csv(out/'latency_frames.csv',timings);write_csv(out/'stream_accuracy_comparison.csv',quality);write_csv(out/'live_scores.csv',live);write_csv(out/'video_times.csv',video_summaries)
    op,events,alarms=s5.operation_metrics(live,scene,cal['threshold']);write_csv(out/'live_alarms.csv',alarms);write_csv(out/'live_segments.csv',events);summary={}
    for mode in ['capacity','paced_30fps']:
        g=[r for r in timings if r['mode']==mode];steady=[r['processing_ms'] for r in g if r['valid_score']];e=np.array([r['end_to_end_ms'] for r in g]);q=[r['queue_ms'] for r in g];elapsed=sum(v['elapsed_s'] for v in video_summaries if v['mode']==mode)
        summary[mode]={'frames':len(g),'videos':len({r['video'] for r in g}),'input_fps':len(g)/elapsed,'steady_p95_ms':float(np.quantile(steady,.95)),'end_to_end_p95_ms':float(np.quantile(e,.95)),'deadline_miss_fraction':float((e>1000/30).mean()),'max_queue_ms':max(q),'last_queue_ms':q[-1]}
    result={'scene':scene,'variant':variant,'seed':0,'timings':summary,'backbone_dtype':str(next(extractor.backbone.parameters()).dtype),'head_dtype':str(next(head.parameters()).dtype),'bank_dtype':str(bank.dtype),
      'phase_agreement':float(np.mean([r['phase_equal'] for r in quality])),'score_max_difference':max(r['score_abs_error'] for r in quality),'alarm_agreement':float(np.mean([r['alarm_equal'] for r in quality])),
      'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
      'metrics':s5.metric_rows(s5.aligned(live,scene)),'operation':op,'longest_video':longest,'seconds':time.time()-started,
      'scope':'one model, full eligible test capacity; longest eligible whole video paced30; JPEG/OS-cache environment, not physical camera'}
    write_json(out/'completed.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['accuracy','benchmark'],required=True);p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--precision',choices=['fp32','bf16'],default='fp32');p.add_argument('--variant',choices=VARIANTS,default='fp32');a=p.parse_args();torch.set_num_threads(8);reserve()
    if a.action=='accuracy':accuracy(a.scene,a.precision)
    else:benchmark(a.scene,a.variant)
