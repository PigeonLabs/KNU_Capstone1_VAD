"""Independent artifact/support/calibration/metrics/stream equivalence audit."""
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ipad import stage5 as s5
from ipad import stage6 as s6
from ipad.stage8 import ROOT as RUNS,SCENES,METHODS,unit,stream_dir,adapter_dir,load
from ipad.common import write_json
from ipad.phase_routing import digest


def close(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:close(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):close(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):assert np.isclose(a,b,atol=1e-9,rtol=1e-9),(a,b)
    else: assert a==b,(a,b)


def verify(stage):
    smoke=load(RUNS/'8-1/smoke/completed.json');assert smoke['base_unchanged'] and smoke['base_gradients_none']
    assert smoke['initial_max_error']==smoke['restored_max_error']==0 and smoke['merged_max_error']<2e-5
    assert smoke['last5_loss']<smoke['first5_loss'];count=adapters=benchmarks=0
    pairs=[('R01',0)] if stage=='8-1' else [(scene,seed) for seed in ([0] if stage=='8-2' else range(3)) for scene in SCENES]
    for scene,seed in pairs:
        train=s5.records('B',scene,'training','train');val=s5.records('B',scene,'training','val')
        assert not {r['video'] for r in train}&{r['video'] for r in val}
        for method in METHODS[1:]:
            path=adapter_dir(scene,seed,method);m=load(path/'completed.json');cfg=load(path/'config.json');samples=s5.read_csv(path/'training_samples.csv')
            assert set(cfg['train_videos'])=={r['video'] for r in train};assert {r['video'] for r in samples}==set(cfg['train_videos'])
            assert m['adapter_sha256']==digest(path/'adapter.pt');assert m['merge_error']<2e-5 and m['restore_error']==0
            assert len(load(path/'history.json'))==10 and m['trainable_parameters']==49152;adapters+=1
        if stage=='8-1':continue
        for method in METHODS:
            model=unit(scene,seed,method)
            for path in ([model] if stage=='8-2' else [model,stream_dir(scene,seed,method)]):
                m=load(path/'completed.json');cal=load(path/'calibration.json');validation=s6.rows(path/'validation_raw.csv');raw=s6.rows(path/'causal_raw.csv')
                close(cal,s5.calibrate(validation));assert set(cal['calibration_videos'])=={r['video'] for r in val}
                common=s5.aligned(s5.apply_calibration(raw,cal),scene);online=s5.aligned(s5.apply_calibration(raw,cal),scene,common=False)
                close(s5.metric_rows(common),m['metrics']);op,_,_=s5.operation_metrics(online,scene,cal['threshold']);close(op,m['operation'])
                previous=s6.rows(s5.unit_path('B',scene,seed,10)/'scores.csv')
                assert [(r['video'],r['frame'],r['label']) for r in common]==[(r['video'],r['frame'],r['label']) for r in previous]
                for video in {r['video'] for r in raw}:
                    group=[r for r in raw if r['video']==video];assert [r['frame'] for r in group]==list(range(35,group[0]['video_length']))
                    assert all(0<=r['phase']<200 for r in group)
                assert m['head_sha256']==digest(model/'causal_head.pt') and m['memory_sha256']==digest(model/'memory.pt')
                assert len(common)==m['common_frames'] and len(online)==m['online_frames'];count+=1
            if stage=='8-3' and seed==0:
                path=stream_dir(scene,seed,method)/'benchmark';m=load(path/'completed.json')
                assert m['phase_agreement']==m['alarm_agreement']==1 and m['score_max_difference']<1e-6
                timings=s5.read_csv(path/'latency_frames.csv');normal=s6.rows(stream_dir(scene,seed,method)/'online_scores.csv');vids={r['video'] for r in normal}
                assert {r['video'] for r in timings if r['mode']=='capacity'}==vids
                assert sum(r['mode']=='paced_30fps' for r in timings)==len(s5.frames(Path('IPAD_dataset')/scene/'testing/frames'/m['longest_video']))
                benchmarks+=1
    result={'stage':stage,'passed':True,'audited_accuracy_units':count,'adapters':adapters,'benchmarks':benchmarks,'protocol_sha256':digest('docs/stage8_protocol.md')}
    write_json(RUNS/stage/'verification.json',result)
    if stage=='8-3':write_json(RUNS/'final_verification.json',result)
    print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['8-1','8-2','8-3'],required=True);verify(p.parse_args().stage)
