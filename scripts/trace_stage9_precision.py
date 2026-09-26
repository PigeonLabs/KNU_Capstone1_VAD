"""Normal-only layerwise FP32/BF16/compiler diagnostic and fixed-position control."""
import json
import os
from pathlib import Path
import sys
import time
import types
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stage9_quant_compile import samples,image_input,disk_check,torch,PrecisionExtractor
from ipad.stage9_optimized import BackboneOnly,preprocess
from scripts.diagnose_stage9 import diff
from ipad.common import write_json,seed_everything
from ipad.phase_routing import digest
from ipad.stage7 import write_csv
from torch.nn import functional as F


def cache_position(backbone):
    dummy=torch.empty((1,325,768),device=backbone.pos_embed.device,dtype=backbone.pos_embed.dtype)
    cached=backbone.interpolate_pos_encoding(dummy,252,252).detach().clone()
    backbone.register_buffer('fixed_position_252',cached)
    def lookup(self,x,w,h):
        assert w==252 and h==252 and x.shape[1]==325
        return self.fixed_position_252
    backbone.interpolate_pos_encoding=types.MethodType(lookup,backbone)


class Trace(torch.nn.Module):
    def __init__(self,backbone):super().__init__();self.backbone=backbone
    def forward(self,x):
        b=self.backbone;t=b.prepare_tokens_with_masks(x);values=[F.normalize(t.float(),dim=-1)]
        for block in b.blocks:
            t=block(t);values.append(F.normalize(t.float(),dim=-1))
        values.append(F.normalize(b.norm(t).float(),dim=-1));return tuple(values)


@torch.inference_mode()
def main():
    out=ROOT/'runs/stage9/9-1/optimized/layer_diagnostic';out.mkdir(parents=True,exist_ok=True);disk_check()
    assert not (out/'config.json').exists()
    seed_everything(0);torch.set_num_threads(8);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    items=[next(r for r in samples() if r[0]==s) for s in ['R01','R02','R03','R04']]
    write_json(out/'config.json',{'source_sha256':digest(__file__),'command':sys.argv,'seed':0,'test_labels_used':False,'samples':[{'scene':s,'video':v,'frame':t,'sha256':digest(p)} for s,v,t,p in items],'cache':os.environ.get('TORCHINDUCTOR_CACHE_DIR'),'started_at':time.time()})
    fp=PrecisionExtractor('B','fp32').cuda().eval();bf=PrecisionExtractor('B','bf16').cuda().eval()
    traces=Trace(bf.backbone).eval();compiled=torch.compile(traces,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
    data=[];original=[]
    for scene,video,frame,path in items:
        x=preprocess(fp,image_input(path));f32=[v.cpu() for v in Trace(fp.backbone)(x)];eager=[v.cpu() for v in traces(x.to(torch.bfloat16))];comp=[v.cpu() for v in compiled(x.to(torch.bfloat16))]
        original.append(eager)
        for layer,(a,b,c) in enumerate(zip(f32,eager,comp)):
            row={'scene':scene,'video':video,'frame':frame,'layer':layer}
            for name,p,q in [('bf16_fp32',b,a),('compile_fp32',c,a),('compile_bf16',c,b)]:
                for key,value in diff(p,q).items():row[name+'_'+key]=value
            data.append(row)
    write_csv(out/'layer_differences.csv',data)
    cache_position(bf.backbone);fixed=BackboneOnly(bf.backbone).eval()
    cfix=torch.compile(fixed,fullgraph=True,dynamic=False,options={'triton.cudagraphs':True,'emulate_precision_casts':True})
    controls=[]
    for (scene,video,frame,path),prior in zip(items,original):
        x=preprocess(fp,image_input(path)).to(torch.bfloat16)
        a=fixed(x);saved={k:v.cpu() for k,v in a.items()};del a
        torch.compiler.cudagraph_mark_step_begin();b={k:v.cpu() for k,v in cfix(x).items()}
        controls.append({'scene':scene,'fixed_position_eager_change':diff(saved['patch12'],prior[-1][:,1:]),'compiled_vs_same_eager':diff(b['patch12'],saved['patch12'])})
    write_json(out/'completed.json',{'state':'completed','fixed_position_controls':controls,'trace_note':'Returning all intermediate values can affect compiler fusion; this is diagnostic, not production timing','time':time.time()})
    print(json.dumps(controls),flush=True)


if __name__=='__main__':main()
