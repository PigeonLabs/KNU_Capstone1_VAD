import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .common import PAPER_AUC, prepare_clip, write_json
from .data import Clips, label_path, videos
from .metrics import binary_metrics, normalize, period_errors
from .model import IPAD


def score_rows(rows, root, scene, period, policy='strict', shift=0, feature_keys=()):
    """All score variants share the same valid frames and scene-wide normalization."""
    output=[]
    excluded=[]
    observed={r['video'] for r in rows}
    expected={p.name for p in videos(root,scene,'testing')}
    missing=sorted(expected-observed,key=int)
    for video in sorted({r['video'] for r in rows}, key=int):
        current=sorted([dict(r) for r in rows if r['video']==video],key=lambda r:r['frame'])
        frame_ids=[r['frame'] for r in current]
        if any(b-a!=1 for a,b in zip(frame_ids,frame_ids[1:])):
            raise ValueError(f'Duplicate/noncontiguous prediction frames: {video}')
        labels=np.load(label_path(root,scene,video)).reshape(-1)
        full_length=current[0]['video_length']
        if policy=='strict' and len(labels)!=full_length:
            excluded.append(video)
            continue
        errors=period_errors([r['phase'] for r in current],period)
        for row,error in zip(current,errors):
            index=row['frame']+shift
            if not np.isfinite(error) or not 0<=index<len(labels): continue
            row.update(label=int(labels[index]),phase_error=float(error))
            output.append(row)
    if not output: raise ValueError('No aligned evaluation frames')
    components=['phase_error',*feature_keys]
    if 'mse' in output[0]:
        for r in output: r['negative_psnr']=float(10*np.log10(max(r['mse'],1e-12)))
        components.insert(0,'negative_psnr')
    for key in components:
        normalized=normalize([r[key] for r in output])
        for row,value in zip(output,normalized): row[key+'_normalized']=float(value)
    scores={key:[r[key+'_normalized'] for r in output] for key in components}
    for key in components:
        if key=='phase_error': continue
        combined=(np.array(scores[key])+np.array(scores['phase_error']))/2
        scores[key+'_with_phase']=combined
        for row,value in zip(output,combined): row[key+'_with_phase']=float(value)
    metrics={key:binary_metrics([r['label'] for r in output],value) for key,value in scores.items()}
    return output,{'metrics':metrics,'excluded_videos':excluded,'missing_videos':missing,'label_shift':shift,'policy':policy,
                   'frames':len(output),'paper_auc':PAPER_AUC[scene],
                   'paper_comparable_label_coverage':not excluded and not missing and shift==0 and all(
                       len(np.load(label_path(root,scene,v)).reshape(-1))==next(r['video_length'] for r in rows if r['video']==v)
                       for v in {r['video'] for r in rows})}


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate(checkpoint,root='IPAD_dataset',batch_size=4,workers=4,dino=False):
    torch.set_num_threads(8)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=state['config']; scene=config['scene']; trained_epoch=state['epoch']
    model=IPAD(**config['model_options']).cuda().eval()
    model.load_state_dict(state['model']); del state
    ds=Clips(root,scene,'testing',label_policy='common')
    lengths={r['video']:r['length'] for r in ds.records}
    loader=DataLoader(ds,batch_size=batch_size,num_workers=workers,pin_memory=True)
    extractor=None
    if dino:
        from .features import DinoFeatures, reconstruction_distances
        extractor=DinoFeatures().cuda().eval()
    rows=[]; inference_seconds=0.
    torch.cuda.reset_peak_memory_stats()
    begin=time.perf_counter()
    with torch.inference_mode():
        for step,batch in enumerate(loader):
            x=prepare_clip(batch['clip'],'cuda')
            torch.cuda.synchronize(); start=time.perf_counter()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=config['precision']=='bf16'):
                output=model(x)
            mse=(output['reconstruction'][:,:,8].float()-x[:,:,8]).square().mean((1,2,3)).cpu().tolist()
            phase=output['phase_logits'].argmax(-1).cpu().tolist()
            features={}
            if extractor:
                features=reconstruction_distances(extractor,x[:,:,8],output['reconstruction'][:,:,8].float())
            torch.cuda.synchronize(); inference_seconds+=time.perf_counter()-start
            for i,video in enumerate(batch['video']):
                rows.append({'scene':scene,'video':video,'frame':int(batch['frame'][i]),
                             'video_length':lengths[video],'mse':mse[i],'phase':phase[i],
                             **{k:float(v[i]) for k,v in features.items()}})
            if step%100==0: print(json.dumps({'scene':scene,'evaluation_step':step,'steps':len(loader)}),flush=True)
    out=Path(checkpoint).parent/('evaluation_dino' if dino else 'evaluation')
    out.mkdir(exist_ok=True)
    write_csv(out/'raw_frames.csv',rows)
    keys=tuple(features)
    scored,summary=score_rows(rows,root,scene,config['median_train_period'],feature_keys=keys)
    write_csv(out/'scores.csv',scored)
    summary.update(scene=scene,checkpoint=str(checkpoint),checkpoint_epoch=trained_epoch,
                   complete_training_protocol=config['complete_protocol'],
                   seconds=time.perf_counter()-begin,inference_seconds=inference_seconds,
                   clips=len(ds),inference_clips_per_second=len(ds)/inference_seconds,
                   peak_gb=torch.cuda.max_memory_allocated()/1e9,
                   protocol='offline scene-wide normalization; centered clip and window; no test-phase oracle')
    write_json(out/'metrics.json',summary)
    if scene=='R02':
        sensitivity={}
        for shift in (-1,0,1):
            _,sensitivity[str(shift)]=score_rows(rows,root,scene,config['median_train_period'],'common',shift,keys)
        write_json(out/'alignment_sensitivity.json',sensitivity)
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--root',default='IPAD_dataset')
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--dino',action='store_true')
    evaluate(**vars(p.parse_args()))
