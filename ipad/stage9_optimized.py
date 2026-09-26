"""Bounded stage 9-1 optimizations; no global installed-package mutations."""
import inspect
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'cache/quantization/torchao017'))
import torch
from torch import nn
from torch.nn import functional as F
from torchao.quantization import quantize_,Int8WeightOnlyConfig,Int8DynamicActivationInt8WeightConfig,Int4WeightOnlyConfig
from torchao.quantization.quant_primitives import MappingType,_choose_qparams_affine_tinygemm,_quantize_affine_tinygemm


class BackboneOnly(nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone
    def forward(self,rgb):
        value=self.backbone.get_intermediate_layers(rgb,n=[11],return_class_token=True,norm=True)[0]
        return {'cls':F.normalize(value[1].float(),dim=-1),'patch12':F.normalize(value[0].float(),dim=-1)}


def preprocess(model,x):
    """Always eager FP32: same arithmetic/input bytes for every runtime."""
    rgb=(x[:,[2,1,0]].float()+1)/2
    rgb=F.interpolate(rgb,size=(252,252),mode='bilinear',align_corners=False)
    return (rgb-model.mean)/model.std


def patch_int8_repr():
    import torchao.kernel.intmm as im
    from torch._subclasses.fake_tensor import FakeTensor
    original=im.safe_int_mm;source=inspect.getsource(original)
    needle='"FakeTensor" in input.__repr__()'
    if source.count(needle)!=1:raise RuntimeError('Unexpected TorchAO source; review patch')
    source=source.replace(needle,'isinstance(input, FakeTensor)')
    namespace={**original.__globals__,'FakeTensor':FakeTensor}
    exec(compile(source,'<stage9_type_check>','exec'),namespace)
    im.safe_int_mm=namespace['safe_int_mm']
    return source


class PackedInt4Linear(nn.Module):
    """Two nibbles/byte; transient BF16 dequant + GEMM, NOT INT4 compute.

    Same tinygemm group128 quantization values, excluding artificial K padding.
    Buffers stay packed. No persistent dequantized full-model weight copy.
    """
    def __init__(self,linear,group_size=128):
        super().__init__();w=linear.weight.detach();n,k=w.shape
        assert w.dtype==torch.bfloat16 and k%group_size==0 and k%2==0
        self.n=n;self.k=k;self.group_size=group_size
        scale,zero=_choose_qparams_affine_tinygemm(w,mapping_type=MappingType.ASYMMETRIC,
            block_size=(1,group_size),target_dtype=torch.int32,quant_min=0,quant_max=15,
            scale_dtype=w.dtype,zero_point_dtype=w.dtype)
        q=_quantize_affine_tinygemm(w,(1,group_size),scale,zero,torch.int32,quant_min=0,quant_max=15)
        self.register_buffer('packed',((q[:,::2]<<4)|q[:,1::2]).to(torch.uint8).contiguous())
        self.register_buffer('scale',scale.reshape(n,-1).contiguous())
        self.register_buffer('zero',zero.reshape(n,-1).contiguous())
        self.register_buffer('bias',linear.bias.detach().clone() if linear.bias is not None else None)
        # Compare q/parameters to the original padded tinygemm recipe at conversion.
        pad_k=((k+1023)//1024)*1024;wp=F.pad(w,(0,pad_k-k))
        ps,pz=_choose_qparams_affine_tinygemm(wp,mapping_type=MappingType.ASYMMETRIC,
            block_size=(1,group_size),target_dtype=torch.int32,quant_min=0,quant_max=15,
            scale_dtype=w.dtype,zero_point_dtype=w.dtype)
        pq=_quantize_affine_tinygemm(wp,(1,group_size),ps,pz,torch.int32,quant_min=0,quant_max=15)
        assert torch.equal(q,pq[:,:k])
        assert torch.equal(self.scale,ps.reshape(n,-1)[:,:k//group_size])
        assert torch.equal(self.zero,pz.reshape(n,-1)[:,:k//group_size])
    def dequantized_weight(self):
        q=torch.stack((self.packed>>4,self.packed&15),dim=-1).reshape(self.n,self.k)
        # float32 reconstruction then one BF16 cast; compiler preserves this boundary.
        w=((q.reshape(self.n,-1,self.group_size).float()-8)*self.scale.float().unsqueeze(-1)+self.zero.float().unsqueeze(-1))
        return w.reshape(self.n,self.k).to(torch.bfloat16)
    def forward(self,x):
        # Explicit matmul then bias preserves original output-rounding boundary.
        y=torch.matmul(x,self.dequantized_weight().t())
        return y+self.bias if self.bias is not None else y


def configure(backbone,variant):
    details={}
    if variant=='w8a16':
        quantize_(backbone,Int8WeightOnlyConfig(version=2,set_inductor_config=False))
        import torch._inductor.fx_passes.quantization as q
        q._register_woq_lowerings=lambda:None
        details['process_local_woq_lowering_bypass']=True
    elif variant=='w8a8':
        details['process_local_safe_int_mm_source']=patch_int8_repr()
        quantize_(backbone,Int8DynamicActivationInt8WeightConfig(version=2,set_inductor_config=False))
    elif variant=='w4_native':
        quantize_(backbone,Int4WeightOnlyConfig(group_size=128,int4_packing_format='tile_packed_to_4d',set_inductor_config=False))
    elif variant=='w4_packed':
        replaced=[]
        for name,m in list(backbone.named_modules()):
            if not isinstance(m,nn.Linear):continue
            parent,_,leaf=name.rpartition('.')
            getattr(backbone,'get_submodule')(parent).__setattr__(leaf,PackedInt4Linear(m))
            replaced.append(name)
        assert len(replaced)==48
        details['same_padded_quantization_values_verified']=len(replaced)
        details['math']='packed INT4 storage -> transient BF16 dequantization -> BF16 GEMM'
    elif variant!='bf16':raise ValueError(variant)
    return details


class EagerGraph:
    """One static input/output; caller consumes output before the next replay."""
    def __init__(self,model,example):
        self.input=example.clone();stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(4):model(self.input)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.output=model(self.input)
    def __call__(self,x):self.input.copy_(x);self.graph.replay();return self.output


def build_runtime(model,example,mode):
    if mode=='eager':return model
    if mode=='graph':return EagerGraph(model,example)
    options={'triton.cudagraphs':True,'emulate_precision_casts':True}
    if mode=='autotune_graph':options.update({'max_autotune':True,'max_autotune_gemm_backends':'ATEN,TRITON'})
    elif mode!='compile_graph':raise ValueError(mode)
    compiled=torch.compile(model,backend='inductor',fullgraph=True,dynamic=False,options=options)
    def call(x):torch.compiler.cudagraph_mark_step_begin();return compiled(x)
    return call
