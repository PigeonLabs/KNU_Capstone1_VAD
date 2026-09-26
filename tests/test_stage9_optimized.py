import pytest
import torch
from ipad.stage9_optimized import PackedInt4Linear,EagerGraph

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA-specific inference')

@torch.inference_mode()
def test_int4_unpack_matches_independent_group_reconstruction():
    torch.manual_seed(0)
    layer=torch.nn.Linear(768,96).cuda().to(torch.bfloat16).eval()
    packed=PackedInt4Linear(layer)
    q=torch.empty((96,768),device='cuda',dtype=torch.float32)
    q[:,::2]=(packed.packed>>4).float();q[:,1::2]=(packed.packed&15).float()
    expected=((q-8)*packed.scale.float().repeat_interleave(128,dim=1)+packed.zero.float().repeat_interleave(128,dim=1)).to(torch.bfloat16)
    assert torch.equal(packed.dequantized_weight(),expected)
    assert packed.packed.numel()*2==layer.weight.numel()
    x=torch.randn(1,325,768,device='cuda',dtype=torch.bfloat16)
    assert torch.equal(packed(x),torch.matmul(x,expected.t())+layer.bias)

@torch.inference_mode()
def test_graph_replay_uses_new_input_and_matches_eager():
    layer=torch.nn.Linear(128,64).cuda().to(torch.bfloat16).eval()
    x=torch.randn(1,17,128,device='cuda',dtype=torch.bfloat16)
    graph=EagerGraph(layer,x)
    first=graph(x).clone();y=x*.5
    second=graph(y).clone()
    assert torch.equal(first,layer(x))
    assert torch.equal(second,layer(y))
    assert not torch.equal(first,second)
