"""Read-only hardware-counter permission probe on actual R01 qkv activation."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stage9_quant_compile import samples,image_input,torch,PrecisionExtractor,disk_check

@torch.inference_mode()
def main():
    disk_check();torch.set_num_threads(8)
    model=PrecisionExtractor('B','bf16').cuda().eval();layer=model.backbone.blocks[0].attn.qkv;captured=[]
    h=layer.register_forward_pre_hook(lambda module,args:captured.append(args[0].detach().clone()))
    model(image_input(samples()[0][3]));h.remove();x=captured[0]
    for _ in range(10):y=layer(x)
    torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStart()
    y=layer(x);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
    print('R01 qkv shape',tuple(x.shape),'finite',bool(torch.isfinite(y).all()))

if __name__=='__main__':main()
