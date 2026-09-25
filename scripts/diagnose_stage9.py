"""Stage 9-1 follow-up: bounded, normal-only causal checks; originals immutable."""
import argparse
import inspect
import os
from pathlib import Path
import sys
import time
import json
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.stage9_quant_compile import (torch, np, samples, image_input, disk_check,
    PrecisionExtractor, quantize_, Int8DynamicActivationInt8WeightConfig, Int8WeightOnlyConfig)
from ipad.common import write_json, environment, seed_everything
from ipad.phase_routing import digest
from ipad.stage5 import read_csv
from ipad.stage7 import write_csv
from torch.nn import functional as F


def diff(a,b):
    delta=(a.float()-b.float()).abs()
    return {'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),
            'cosine_mean':float((1-F.cosine_similarity(a.float(),b.float(),dim=-1)).clamp_min(0).mean())}


def bench(fn,inputs):
    for x in inputs[:8]:fn(x)
    torch.cuda.synchronize();rows=[]
    for repeat in range(3):
        for index,x in enumerate(inputs):
            disk_check();torch.cuda.synchronize();start=time.perf_counter();y=fn(x);torch.cuda.synchronize()
            rows.append({'repeat':repeat,'index':index,'ms':(time.perf_counter()-start)*1000});del y
    return rows


def profile(fn,x):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as p:
        for _ in range(3):y=fn(x);torch.cuda.synchronize();del y
    return [{'name':e.key,'count':e.count,'device_type':str(e.device_type),'self_cpu_us':e.self_cpu_time_total,'self_device_us':e.self_device_time_total} for e in p.key_averages()]


def prep(model,x):
    rgb=(x[:,[2,1,0]].float()+1)/2
    rgb=F.interpolate(rgb,size=(252,252),mode='bilinear',align_corners=False)
    return ((rgb-model.mean)/model.std).to(torch.bfloat16)


class Body(torch.nn.Module):
    def __init__(self,model):super().__init__();self.backbone=model.backbone
    def forward(self,x):
        v=self.backbone.get_intermediate_layers(x,n=[11],return_class_token=True,norm=True)[0]
        return {'cls':F.normalize(v[1].float(),dim=-1),'patch12':F.normalize(v[0].float(),dim=-1)}


@torch.inference_mode()
def run(kind):
    out=ROOT/'runs/stage9/9-1/audit'/kind
    out.mkdir(parents=True,exist_ok=True)
    if (out/'config.json').exists():raise RuntimeError('Preserve prior diagnosis: '+str(out))
    disk_check();seed_everything(0);torch.set_num_threads(8)
    all_items=samples()
    if kind=='numeric':
        old=read_csv(ROOT/'runs/stage9/9-1/bf16_reduce-overhead-precise/feature_differences.csv')
        top=sorted(old,key=lambda r:float(r['same_eager_patch12_max_abs']),reverse=True)[:8]
        keys={(r['scene'],r['video'],int(r['frame'])) for r in top}
        items=[x for x in all_items if x[:3] in keys]
        items+= [all_items[i] for i in np.linspace(0,len(all_items)-1,8,dtype=int) if all_items[i] not in items]
    else:items=[all_items[i] for i in np.linspace(0,len(all_items)-1,16,dtype=int)]
    write_json(out/'config.json',{'state':'diagnostic','kind':kind,'command':sys.argv,'script_sha256':digest(__file__),
        'environment':environment(),'started_at':time.time(),'test_labels_used':False,
        'sample_rule':'top8 original BF16 precise patch errors plus8 evenly spaced normal frames' if kind=='numeric' else '16 evenly spaced from original384 normal frames',
        'samples':[{'scene':s,'video':v,'frame':t,'sha256':digest(p)} for s,v,t,p in items],
        'timing_scope':'preloaded GPU input, model only, synchronized wall time; do not compare directly with JPEG-inclusive original table',
        'cache':os.environ.get('TORCHINDUCTOR_CACHE_DIR')})
    write_json(out/'progress.json',{'step':'loading'})
    inputs=[image_input(p) for *_,p in items]
    model=PrecisionExtractor('B','bf16').cuda().eval()
    result={}
    if kind=='repr':
        import torchao.kernel.intmm as im
        quantize_(model.backbone,Int8DynamicActivationInt8WeightConfig(version=2,set_inductor_config=False))
        original=im.safe_int_mm;source=inspect.getsource(original)
        needle='"FakeTensor" in input.__repr__()'
        assert source.count(needle)==1
        source=source.replace(needle,'isinstance(input, FakeTensor)')
        from torch._subclasses.fake_tensor import FakeTensor
        namespace={**original.__globals__,'FakeTensor':FakeTensor}
        exec(compile(source,'<stage9_safe_int_mm_type_check>','exec'),namespace)
        fixed=namespace['safe_int_mm']
        (out/'process_only_patch.txt').write_text(source)
        result['source_sha256']=digest(inspect.getsourcefile(original))
        refs=None
        try:
            for label,fn in [('original_before',original),('type_check_only',fixed),('original_restored',original)]:
                im.safe_int_mm=fn;write_json(out/'progress.json',{'step':label})
                vals=[{k:v.cpu() for k,v in model(x).items()} for x in inputs]
                if refs is None:refs=vals
                result[label]={'differences':[{k:diff(v,refs[i][k]) for k,v in item.items()} for i,item in enumerate(vals)]}
                timing=bench(model,inputs);write_csv(out/(label+'_timings.csv'),timing)
                result[label]['mean_ms']=float(np.mean([x['ms'] for x in timing]))
                write_csv(out/(label+'_profiler.csv'),profile(model,inputs[0]))
        finally:im.safe_int_mm=original
    elif kind=='numeric':
        body=Body(model).eval()
        compiled=torch.compile(model,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        cprep=torch.compile(lambda x:prep(model,x),fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        cbody=torch.compile(body,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        rows=[]
        for index,((scene,video,frame,_),x) in enumerate(zip(items,inputs)):
            disk_check();write_json(out/'progress.json',{'step':'numeric','index':index})
            ref={k:v.cpu() for k,v in model(x).items()};again={k:v.cpu() for k,v in model(x).items()}
            p=prep(model,x);cp=cprep(x).clone()
            whole={k:v.cpu() for k,v in compiled(x).items()}
            common={k:v.cpu() for k,v in cbody(p).items()}
            prep_only={k:v.cpu() for k,v in body(cp).items()}
            split={k:v.cpu() for k,v in body(p).items()}
            row={'scene':scene,'video':video,'frame':frame,'prep_max_abs':float((p.float()-cp.float()).abs().max()),'prep_changed_fraction':float((p!=cp).float().mean())}
            for name,values in [('eager_repeat',again),('eager_split',split),('whole_compile',whole),('body_compile_same_input',common),('prep_compile_only',prep_only)]:
                for k,v in values.items():
                    for metric,value in diff(v,ref[k]).items():row[f'{name}_{k}_{metric}']=value
            rows.append(row)
        write_csv(out/'differences.csv',rows);result['rows']=rows
    elif kind in ['woq_original','woq_bypass']:
        quantize_(model.backbone,Int8WeightOnlyConfig(version=2,set_inductor_config=False))
        if kind=='woq_bypass':
            import torch._inductor.fx_passes.quantization as q
            result['lowering_source_sha256']=digest(inspect.getsourcefile(q._register_woq_lowerings))
            # A diagnostic-only process-local bypass; no installed file mutation.
            q._register_woq_lowerings=lambda:None
        refs=[{k:v.cpu() for k,v in model(x).items()} for x in inputs]
        compiled=torch.compile(model,fullgraph=True,dynamic=False,options={'emulate_precision_casts':True})
        write_json(out/'progress.json',{'step':'compile'})
        vals=[{k:v.cpu() for k,v in compiled(x).items()} for x in inputs]
        result['differences']=[{k:diff(v,refs[i][k]) for k,v in item.items()} for i,item in enumerate(vals)]
        timings=bench(compiled,inputs);write_csv(out/'timings.csv',timings)
        result['mean_ms']=float(np.mean([x['ms'] for x in timings]))
        write_csv(out/'profiler.csv',profile(compiled,inputs[0]))
    result['state']='completed';result['finished_at']=time.time()
    write_json(out/'completed.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','differences']}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['repr','numeric','woq_original','woq_bypass'],required=True);a=p.parse_args()
    try:run(a.kind)
    except Exception:
        write_json(ROOT/'runs/stage9/9-1/audit'/a.kind/'failure.json',{'traceback':traceback.format_exc()});raise
