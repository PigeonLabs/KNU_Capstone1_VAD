import csv
import json
import numpy as np
from ipad.stage7 import spans,write_csv


def test_empty_and_boundary_events(tmp_path):
    assert spans([])==[]
    assert spans([True,True,False,True])==[(0,1),(3,3)]
    assert json.loads(json.dumps(spans([True,True])))==[[0,1]]
    p=tmp_path/'empty.csv';write_csv(p,[],['video','start'])
    with p.open() as f:
        r=csv.DictReader(f);assert r.fieldnames==['video','start'];assert list(r)==[]

from ipad import stage5 as s5
from ipad.stage7_alerts import flags,smooth,transform,choose


def test_raw_alarm_equivalence_and_prefix_causality():
    rng=np.random.default_rng(0);x=rng.random(100)
    for rule in ['raw','ewma','hysteresis','ewma_hysteresis']:
        first=flags(x[:60],.4,rule);whole=flags(x,.4,rule)
        for a,b in zip(first,whole):np.testing.assert_array_equal(a,b[:60])
    _,a,e=flags(x,.4,'raw');a0,e0=s5.alarm_flags(x,.4)
    np.testing.assert_array_equal(a,a0);np.testing.assert_array_equal(e,e0)


def test_hysteresis_zero_threshold_and_duration():
    _,active,emit=flags([1,1,1,0,0,0],0.,'hysteresis')
    assert active.tolist()==[False,False,True,True,True,False]
    assert emit.sum()==1
    _,a,e=flags([2,2,2,.8,.8,.8,.6,.6,.6],1.,'hysteresis')
    assert a.tolist()==[False,False,True,True,True,True,True,True,False]


def test_smoothing_resets_at_video_boundary():
    data=[{'video':'01','combined_score':10.},{'video':'01','combined_score':0.},{'video':'02','combined_score':0.}]
    out=transform(data,'ewma');assert [r['combined_score'] for r in out]==[10.,8.,0.]


def test_normal_cv_disjoint_and_minimum_feasible_q():
    data=[{'video':str(v),'appearance':float(t%11)/11,'temporal':float(t%7)/7} for v in range(4) for t in range(100)]
    cv=choose(data,'ewma_hysteresis')
    for fold in cv['folds']:assert fold['heldout_video'] not in fold['fit_videos']
    feasible=[r['q'] for r in cv['grid'] if r['constraint_met']]
    assert cv['q']==(min(feasible) if feasible else .999)
    assert cv['constraint_met']==bool(feasible)
