"""Stage-eight dedicated single-model JPEG replay; same measurement as stage seven."""
import argparse
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from . import stage5 as s5
from . import stage6 as s6
from .stage8 import begin, load, stream_dir, bf16_models, METHODS, SCENES
from .stage7 import write_csv
from .stage6 import rows
from .common import write_json
from .phase_routing import reserve


def benchmark(scene,variant):
    unit=stream_dir(scene,0,variant);out=unit/'benchmark'
    if not begin(out,scene=scene,seed=0,variant=variant,input='all eligible test JPEGs',batch_size=1,capacity_repeats=1,paced='longest eligible video at30FPS'):return
    started=time.time();m=load(unit/'completed.json');cal=load(unit/'calibration.json');extractor,head,bank=bf16_models(scene,0,variant);detector=s6.Detector(extractor,head,bank,cal,m['period'])
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
    p=argparse.ArgumentParser();p.add_argument('--scene',choices=SCENES,required=True);p.add_argument('--method',choices=METHODS,required=True);a=p.parse_args();torch.set_num_threads(8);reserve();benchmark(a.scene,a.method)
