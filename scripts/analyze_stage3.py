"""Paired video-cluster bootstrap; no independent-frame resampling."""
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ipad.common import write_json


def auc_blocks(labels,scores,video_ids):
    ids=sorted(set(video_ids));labels=np.asarray(labels);scores=np.asarray(scores);video_ids=np.asarray(video_ids)
    pos=[scores[(video_ids==v)&(labels==1)] for v in ids]
    neg=[np.sort(scores[(video_ids==v)&(labels==0)]) for v in ids]
    credit=np.zeros((len(ids),len(ids)))
    for i,p in enumerate(pos):
        for j,n in enumerate(neg):
            credit[i,j]=(np.searchsorted(n,p,'left')+np.searchsorted(n,p,'right')).sum()/2
    return credit,np.array([len(x) for x in pos]),np.array([len(x) for x in neg])


def draw_auc(block,counts):
    credit,pos,neg=block;den=(counts@pos)*(counts@neg)
    return np.divide(np.einsum('bi,ij,bj->b',counts,credit,counts),den,out=np.full(len(counts),np.nan),where=den>0)*100


def main():
    comparisons=[('top3_weighted','unconditional'),('legacy_hard','unconditional'),
                 ('top3_nn','random3_nn'),('neighbor_nn','neighbor_random_nn'),
                 ('conditional_all','unconditional')]
    result={'unit':'video','draws':2000,'seed':1729,'note':'Exploratory fixed-test-set uncertainty; not independent external validation',
            'scenes':{},'seed_summary':{}}
    samples={};rng=np.random.default_rng(1729)
    for scene in ['R01','R02','R03','R04']:
        path=ROOT/f'runs/stage3/{scene}/seed0/scores.csv'
        if not path.exists():continue
        with path.open() as f:rows=list(csv.DictReader(f))
        y=np.array([int(r['label']) for r in rows]);v=np.array([r['video'] for r in rows]);n=len(set(v))
        counts=rng.multinomial(n,np.full(n,1/n),size=2000)
        keys=set(k for pair in comparisons for k in pair)
        auc={k:draw_auc(auc_blocks(y,[float(r[k]) for r in rows],v),counts) for k in keys}
        result['scenes'][scene]={}
        for a,b in comparisons:
            name=a+' - '+b;diff=auc[a]-auc[b];valid=diff[np.isfinite(diff)]
            result['scenes'][scene][name]={'delta_auroc_pp_ci95':np.quantile(valid,[.025,.975]).tolist(),
                                         'valid_draws':len(valid),'videos':n}
            samples.setdefault(name,[]).append(diff)
    result['macro_delta_ci95']={name:np.nanquantile(np.nanmean(np.stack(d),axis=0),[.025,.975]).tolist()
                                 for name,d in samples.items() if len(d)==4}
    for path in sorted((ROOT/'runs/stage3').glob('R*/seed*/metrics.json')):
        d=json.loads(path.read_text());seed=str(d['seed'])
        for k,m in d['metrics'].items():result['seed_summary'].setdefault(seed,{}).setdefault(k,[]).append(m['auroc'])
    result['seed_summary']={seed:{k:float(np.mean(v)) for k,v in vals.items() if len(v)==4} for seed,vals in result['seed_summary'].items()}
    write_json(ROOT/'runs/stage3/bootstrap.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
