import torch
from ipad.prototype import match_prototypes, spherical_kmeans, PhaseHead


def test_prototypes_preserve_positions_and_phase_restriction():
    centers=torch.tensor([[[1.,0.],[0.,1.]],[[0.,1.],[1.,0.]]])
    x=torch.tensor([[[1.,0.],[0.,1.]]])
    nn,soft=match_prototypes(x,centers,torch.tensor([0,1]),torch.tensor([0]),bins=2)
    torch.testing.assert_close(nn,torch.zeros(1));torch.testing.assert_close(soft,torch.zeros(1))
    wrong,_=match_prototypes(x,centers,torch.tensor([0,1]),torch.tensor([1]),bins=2)
    torch.testing.assert_close(wrong,torch.ones(1))
    empty,_=match_prototypes(x,centers,torch.tensor([0,0]),torch.tensor([1]),bins=3)
    assert torch.isfinite(empty).all()


def test_kmeans_retains_distinct_position_clusters():
    p=torch.tensor([[[1.,0.],[1.,0.],[0.,1.],[0.,1.]],
                    [[-1.,0.],[-1.,0.],[0.,-1.],[0.,-1.]]])
    centers=spherical_kmeans(p,4,iterations=3)
    assert centers.shape==(2,4,2)
    distances=1-(p@centers.transpose(1,2)).max(-1).values
    torch.testing.assert_close(distances,torch.zeros_like(distances))


def test_head_uses_temporal_order_and_shape():
    model=PhaseHead()
    x=torch.randn(2,16,768)
    assert model(x).shape==(2,200)
    assert not torch.equal(model(x),model(x.flip(1)))
