import numpy as np
import torch
from ipad.stage4 import estimate_prior,prior_scores,budget_match,attach_time


def test_complete_runs_exclude_censored_boundaries_and_keep_video_boundaries():
    q=np.eye(20)[[0,0,1,1,1,2,2]]
    prior=estimate_prior([q,q])
    assert prior['durations'][1]==[3,3]
    assert prior['durations'][0]==[] and prior['durations'][2]==[]
    a=np.array(prior['transition']);np.testing.assert_allclose(a.sum(-1),1)
    assert a[0,1]>a[0,2] and np.all(a>0)
    # No fabricated transition from last state 2 back to first state 0.
    assert prior['soft_counts'][2][0]==.05


def test_duration_increases_during_long_stay_and_resets():
    prior=estimate_prior([np.eye(20)[[0,1,1,2]]])
    score=prior_scores(np.eye(20)[[1]*8+[2]],prior)
    assert score['duration'][7]>score['duration'][1]
    assert score['duration'][8]==0
    assert all(np.isfinite(v).all() for v in score.values())


def test_actual_candidate_gather_matches_exhaustive_reference():
    from ipad.phase_routing import route_distances
    gen=torch.Generator().manual_seed(12)
    c=torch.nn.functional.normalize(torch.randn(4,12,5,generator=gen),dim=-1)
    x=torch.nn.functional.normalize(torch.randn(2,4,5,generator=gen),dim=-1)
    g=torch.arange(4).repeat_interleave(3);oc=torch.arange(4)
    groupbank=torch.stack([c[:,g==v] for v in oc]);prob=torch.zeros(2,20);prob[:,:4]=torch.tensor([.4,.3,.2,.1])
    legacy=torch.tensor([0,19]);bank=(c,c,g,oc,groupbank)
    actual=budget_match(x,prob,legacy,bank)
    sim=torch.einsum('bpc,pkc->bpk',x,c)
    distances=torch.stack([(1-sim[:,:,g==o].max(-1).values).clamp_min(0) for o in oc],-1)
    expected,_=route_distances(distances,(1-sim.max(-1).values).mean(-1),prob,legacy,oc,torch.arange(4).expand(2,-1),1.1)
    for k,v in actual.items():torch.testing.assert_close(v,expected[k])


def test_time_support_never_crosses_video_boundaries():
    rows=[{'video':str(v),'frame':str(i+8),'phase':str(i)} for v in [1,2] for i in range(30)]
    result=attach_time(rows,200)
    assert len(result)==20
    assert {int(r['frame']) for r in result}==set(range(18,28))
