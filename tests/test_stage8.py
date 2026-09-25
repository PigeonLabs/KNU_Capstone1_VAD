import torch
from ipad.lora import QVLoRA,photometric,cosine_loss
from ipad.stage8 import sampled_frames


def test_qv_only_update_and_merge():
    torch.manual_seed(0);base=torch.nn.Linear(8,24);layer=QVLoRA(base,rank=2,alpha=4);x=torch.randn(3,5,8)
    assert torch.equal(layer(x),base(x))
    loss=layer(x).square().mean();loss.backward()
    assert all(p.grad is None for p in base.parameters())
    assert layer.q_b.grad.abs().sum()>0 and layer.v_b.grad.abs().sum()>0
    with torch.no_grad():layer.q_b.add_(.02);layer.v_b.sub_(.03)
    assert torch.equal(layer(x)[...,8:16],base(x)[...,8:16])
    torch.testing.assert_close(layer(x),layer.merged()(x),atol=1e-6,rtol=1e-5)
    layer.enabled=False;assert torch.equal(layer(x),base(x))


def test_photometric_keeps_spatial_support_and_range():
    x=torch.zeros(2,3,8,8);x[:,:,2:4,2:4]=.5
    y=photometric(x,torch.Generator().manual_seed(9))
    assert y.shape==x.shape and y.min()>=-1 and y.max()<=1
    assert torch.all(y[:,:,2:4,2:4]>y[:,:,0:2,0:2])
    assert torch.equal(y[:,:,0,0],y[:,:,7,7])


def test_loss_gradients_and_sampling():
    x=torch.randn(2,6,8,requires_grad=True);y=torch.randn(2,6,8,requires_grad=True)
    cosine_loss(x,y).backward();assert x.grad is not None and y.grad is not None
    recs=[{'frames':50},{'frames':250}];samples=sampled_frames(recs,0)
    assert samples==sampled_frames(recs,0) and len(samples)==114
    assert len(set(samples))==len(samples)
    assert all(0<=t<recs[i]['frames'] for i,t in samples)
