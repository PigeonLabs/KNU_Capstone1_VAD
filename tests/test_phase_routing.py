import numpy as np
import torch
from ipad.phase_routing import circular_error, phase_probabilities, route_distances, choose_threshold


def test_phase_aggregation_and_wrap():
    p=phase_probabilities(torch.zeros(2,200))
    torch.testing.assert_close(p,torch.full((2,20),.05))
    np.testing.assert_equal(circular_error([19,0,10],[0,19,0]),[1,1,10])


def test_routing_preserves_patch_minimum_and_candidate_controls():
    dist=torch.tensor([[[0.,1.,.6],[1.,0.,.6]],[[.4,.5,.6],[.4,.5,.6]]])
    occupied=torch.tensor([0,1,19]);prob=torch.zeros(2,20);prob[:,0]=.6;prob[:,1]=.3;prob[:,19]=.1
    order=torch.tensor([[0,1,2],[2,1,0]])
    scores,counts=route_distances(dist,torch.tensor([.2,.2]),prob,torch.tensor([0,19]),occupied,order,.7)
    torch.testing.assert_close(scores['top3_nn'],torch.tensor([0.,.4]))
    torch.testing.assert_close(scores['legacy_hard'],torch.tensor([.5,.6]))
    torch.testing.assert_close(scores['confidence_fallback'],torch.tensor([.2,.2]))
    torch.testing.assert_close(scores['top3_weighted'],torch.tensor([.51,.45]))
    assert counts.tolist()==[3,2]
    assert torch.isfinite(torch.stack(list(scores.values()))).all()


def test_normal_calibration_has_explicit_disable_case():
    rows=[{'confidence':.6,'posterior_bin':5,'reference_bin':0} for _ in range(20)]
    assert choose_threshold(rows)['enabled'] is False
    for r in rows:r['posterior_bin']=19
    assert choose_threshold(rows)['enabled'] is True
