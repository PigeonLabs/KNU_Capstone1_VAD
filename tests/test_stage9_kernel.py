import pytest
import torch
from ipad.stage9_optimized import PackedInt4Linear
from ipad.stage9_kernel import fused_linear,CONFIGS

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA only')
@torch.inference_mode()
def test_fused_packed_int4_uses_same_values_with_tail_masks():
    torch.manual_seed(0);torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    layer=PackedInt4Linear(torch.nn.Linear(768,96).cuda().bfloat16().eval())
    for m in [1,325]:
        x=torch.randn(1,m,768,device='cuda',dtype=torch.bfloat16);ref=layer(x)
        for tile in CONFIGS:
            actual=fused_linear(x,layer,tile)
            relative=torch.linalg.vector_norm(actual.float()-ref.float())/torch.linalg.vector_norm(ref.float())
            assert relative<.01
            assert torch.isfinite(actual).all()
