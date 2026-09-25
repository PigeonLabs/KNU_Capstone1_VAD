"""Audit completed stage 3 identities, probability ranges, splits and provenance."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json


def main():
    root=ROOT/'runs/stage3';checks=[];hashes=set()
    for scene in ['R01','R02','R03','R04']:
        with (ROOT/f'runs/prototype/{scene}/seed0/scores.csv').open() as f:
            expected=[(r['video'],int(r['frame']),int(r['label'])) for r in csv.DictReader(f)]
        for seed in range(3):
            out=root/scene/f'seed{seed}';assert (out/'completed.json').exists()
            config=json.loads((out/'config.json').read_text());assert not set(config['normal_train_videos'])&set(config['normal_val_videos'])
            hashes.add(config['source_sha256']['ipad/phase_routing.py'])
            with (out/'scores.csv').open() as f:rows=list(csv.DictReader(f))
            assert expected==[(r['video'],int(r['frame']),int(r['label'])) for r in rows]
            prob=np.array([[float(r[f'p{j:02}']) for j in range(20)] for r in rows]);assert np.isfinite(prob).all()
            assert prob.min()>=0 and prob.max()<=1
            assert np.max(abs(prob.sum(-1)-1))<1e-5
            assert all(0<=int(r['phase'])<200 for r in rows)
            assert all(abs(float(r['confidence'])-max(float(r[f'p{j:02}']) for j in range(20)))<1e-7 for r in rows)
            checks.append({'scene':scene,'seed':seed,'frames':len(rows),'identity_match':True,'normal_split_disjoint':True})
        temporal=root/scene/'seed0/temporal';assert (temporal/'completed.json').exists()
        with (temporal/'scores.csv').open() as f:diagnostic_rows=list(csv.DictReader(f))
        original={(r['video'],r['frame']):r for r in diagnostic_rows if r['variant']=='normal'}
        paired={}
        for kind in ['pause','reverse','skip','speed_0.9','speed_1.1']:
            pairs=[(r,original[(r['video'],r['source_frame'])]) for r in diagnostic_rows
                   if r['variant']==kind and (r['video'],r['source_frame']) in original]
            assert pairs
            delta={k:np.array([float(r[k])-float(o[k]) for r,o in pairs])
                   for k in ['unconditional','top3_weighted','time5','time21']}
            positive=np.array([int(r['label']) for r,o in pairs],dtype=bool)
            maximum=float(np.max(np.abs(delta['unconditional'])))
            assert maximum<2e-6,'Image-only score changed under frame reorder'
            paired[kind]={'matched_frames':len(pairs),'unconditional_max_abs_change':maximum,
                'positive_mean_change':{k:float(v[positive].mean()) if positive.any() else None for k,v in delta.items()},
                'negative_mean_change':{k:float(v[~positive].mean()) if (~positive).any() else None for k,v in delta.items()}}
        write_json(temporal/'paired_source_frame_audit.json',{'variants':paired,
            'interpretation':'Image-only scores are invariant for the same source frame. Their diagnostic AUROC can arise from which phases were selected/repeated, not detection of temporal disorder.'})
        manifest=json.loads((temporal/'manifest.json').read_text())
        for item in manifest:
            assert 0<=min(item['output_to_source_frame'])<=max(item['output_to_source_frame'])<item['original_length']
            assert item['video'] in config['normal_val_videos']
    assert len(hashes)==1,'Routing algorithm source changed between repeats'
    inputs={}
    for stage in ['stage1_reproduction','stage2_dinov2']:
        p=ROOT/'experiments'/stage/'artifacts.jsonl'
        with p.open('rb') as f:inputs[str(p.relative_to(ROOT))]=hashlib.file_digest(f,'sha256').hexdigest()
    value={'passed':True,'checks':checks,'routing_source_hash':list(hashes)[0],
           'input_inventory_references':inputs,'checked_at':time.time(),
           'note':'Prior-stage binary inventories are referenced, not republished; diagnostic transformed indices reference original cached normal features.'}
    write_json(root/'final_verification.json',value);print(json.dumps(value,indent=2))


if __name__=='__main__':main()
