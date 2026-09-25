"""Stage 9-1: real integer GEMM and equally compiled DINOv2 baselines."""
import argparse
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/quantization/torchao017'))
import cv2
import numpy as np
import torch
import torchao
from torchao.quantization import quantize_,Int8WeightOnlyConfig,Int4WeightOnlyConfig,Int8DynamicActivationInt8WeightConfig
from ipad.stage6 import PrecisionExtractor
from ipad.stage5 import records,frames
from ipad.common import write_json,environment,seed_everything
from ipad.stage7 import write_csv
from ipad.phase_routing import digest
from scripts.probe_quantization import tensor_payload,image_input,disk_check

VARIANTS=['bf16','w8a16','w8a8','w4a16']
MODES=['eager','default','reduce-overhead','reduce-overhead-precise']


def samples():
    result=[]
    for scene in ['R01','R02','R03','R04']:
        for r in records('B',scene,'training','train')[:3]:
            files=frames(ROOT/'IPAD_dataset'/scene/'training/frames'/r['video'])
            for t in np.linspace(0,len(files)-1,32,dtype=int):result.append((scene,r['video'],int(t),files[t]))
    assert len(result)==384
    return result


def summarize_times(data):
    keys=['input_ms','model_wall_ms','gpu_interval_ms','total_ms'];result={}
    for key in keys:
        a=np.array([r[key] for r in data]);result[key]={'mean':float(a.mean()),'p50':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}
    return result


def compare(a,b):
    return {'cosine':float((1-(a.float()*b.float()).sum(-1)).clamp_min(0).mean()),'max_abs':float((a.float()-b.float()).abs().max())}


def cache_inventory(out):
    rows=[]
    for env in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR']:
        root=Path(os.environ[env])
        for p in sorted(root.rglob('*')):
            if p.is_file():rows.append({'cache':env,'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':digest(p)})
    write_json(out/'generated_cache_hashes.json',rows)
    # Small kernel-call excerpts retain dtype/dispatch evidence without publishing generated binaries.
    excerpts=[]
    for r in rows:
        p=ROOT/r['path']
        if p.suffix!='.py':continue
        lines=p.read_text(errors='replace').splitlines()
        hits=[(i,l) for i,l in enumerate(lines,1) if any(k in l for k in ['_int_mm','int_scaled_matmul','_weight_int4pack_mm','extern_kernels.mm','tl.dot(','s8','torch.int8','triton_poi_fused'])]
        if hits:excerpts.append({'path':r['path'],'lines':[{'line':i,'text':l[:500]} for i,l in hits[:100]]})
    write_json(out/'generated_kernel_evidence.json',excerpts)
    return sum(r['bytes'] for r in rows)


@torch.inference_mode()
def run(variant,mode):
    out=ROOT/'runs/stage9/9-1'/f'{variant}_{mode}';disk_check();out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    if (out/'config.json').exists():raise RuntimeError(f'Partial result must be inspected: {out}')
    for name,leaf in [('TORCHINDUCTOR_CACHE_DIR','inductor'),('TRITON_CACHE_DIR','triton')]:
        os.environ.setdefault(name,str(ROOT/'cache/stage9_compile'/f'{variant}_{mode}'/leaf))
    seed_everything(0);torch.set_num_threads(8);torch._dynamo.config.suppress_errors=False
    items=samples();started=time.time();config={'variant':variant,'mode':mode,'seed':0,'batch_size':1,'test_labels_used':False,
        'environment':environment(),'torchao_version':torchao.__version__,'torchao_source_sha256':digest(Path(torchao.__file__).parent/'quantization/quant_api.py'),
        'script_sha256':digest(__file__),'protocol_sha256':digest(ROOT/'docs/stage9_protocol.md'),'pretrained_sha256':digest(ROOT/'cache/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth'),
        'samples':[{'scene':s,'video':v,'frame':t,'sha256':digest(p)} for s,v,t,p in items],
        'compile':{'backend':'inductor','mode':mode,'fullgraph':True,'dynamic':False,'set_inductor_config':False,'suppress_errors':False,'emulate_precision_casts':mode=='reduce-overhead-precise'},
        'env':{k:os.environ.get(k) for k in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_FREEZING','TORCH_LOGS']},'command':sys.argv,'started_at':started}
    write_json(out/'config.json',config)
    def progress(step,**kw):write_json(out/'progress.json',{'step':step,'time':time.time(),**kw})
    try:
        progress('baseline_reference');model=PrecisionExtractor('B','bf16').cuda().eval();baseline=[]
        for scene,v,t,p in items:
            disk_check();f=model(image_input(p));baseline.append({k:x.cpu() for k,x in f.items()})
        del f
        if variant=='w8a16':quantize_(model.backbone,Int8WeightOnlyConfig(version=2,set_inductor_config=False))
        elif variant=='w8a8':quantize_(model.backbone,Int8DynamicActivationInt8WeightConfig(version=2,set_inductor_config=False))
        elif variant=='w4a16':quantize_(model.backbone,Int4WeightOnlyConfig(group_size=128,int4_packing_format='tile_packed_to_4d',set_inductor_config=False))
        changed=[n for n,m in model.backbone.named_modules() if isinstance(m,torch.nn.Linear) and type(m.weight) not in (torch.Tensor,torch.nn.Parameter)]
        assert len(changed)==(0 if variant=='bf16' else 48)
        payload,types=tensor_payload(model.backbone);write_json(out/'tensor_types.json',types)
        buf=io.BytesIO();torch.save(model.backbone.state_dict(),buf);serialized=buf.tell();state_sha=hashlib.sha256(buf.getbuffer()).hexdigest();del buf
        progress('same_precision_eager_reference');reference=[]
        for scene,v,t,p in items:
            disk_check();f=model(image_input(p));reference.append({k:x.cpu() for k,x in f.items()})
        del f
        from torch._dynamo.utils import counters
        counters.clear();x=image_input(items[0][3]);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();progress('first_compile_call')
        start=time.perf_counter()
        if mode=='eager':run_model=model
        elif mode=='reduce-overhead-precise':run_model=torch.compile(model,backend='inductor',fullgraph=True,dynamic=False,options={'triton.cudagraphs':True,'emulate_precision_casts':True})
        else:run_model=torch.compile(model,backend='inductor',fullgraph=True,dynamic=False,mode=mode)
        def call(inp):
            if mode.startswith('reduce-overhead'):torch.compiler.cudagraph_mark_step_begin()
            return run_model(inp)
        z=call(x);torch.cuda.synchronize();first_call=time.perf_counter()-start;del z
        progress('warmup',first_call_seconds=first_call);warmstart=time.perf_counter()
        for _ in range(12):disk_check();z=call(x);del z
        torch.cuda.synchronize();warmup=time.perf_counter()-warmstart;cold_peak=torch.cuda.max_memory_allocated()/2**20
        progress('numeric_validation');differences=[]
        for (scene,v,t,p),base,ref in zip(items,baseline,reference):
            disk_check();f=call(image_input(p));cpu={k:x.cpu() for k,x in f.items()};del f
            row={'scene':scene,'video':v,'frame':t}
            for k,value in cpu.items():
                assert torch.isfinite(value).all()
                for kind,other in [('bf16',base),('same_eager',ref)]:
                    for metric,val in compare(value,other[k]).items():row[f'{kind}_{k}_{metric}']=val
            differences.append(row)
        del baseline,reference,cpu;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        progress('timing');torch.cuda.reset_peak_memory_stats();timings=[];event_a=torch.cuda.Event(enable_timing=True);event_b=torch.cuda.Event(enable_timing=True)
        for repeat in range(3):
            for scene,v,t,p in items:
                disk_check();torch.cuda.synchronize();start=time.perf_counter();inp=image_input(p);prepared=time.perf_counter();event_a.record();f=call(inp);event_b.record();torch.cuda.synchronize();end=time.perf_counter()
                timings.append({'repeat':repeat,'scene':scene,'video':v,'frame':t,'input_ms':(prepared-start)*1000,'model_wall_ms':(end-prepared)*1000,'gpu_interval_ms':event_a.elapsed_time(event_b),'total_ms':(end-start)*1000});del f
        peak_allocated=torch.cuda.max_memory_allocated()/2**20;peak_reserved=torch.cuda.max_memory_reserved()/2**20
        progress('profiler');profdata=[]
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            for _,_,_,p in items[:5]:inp=image_input(p);f=call(inp);torch.cuda.synchronize();del f
        for e in prof.key_averages():
            profdata.append({'name':e.key,'count':e.count,'device_type':str(e.device_type),'self_cpu_us':e.self_cpu_time_total,'self_device_us':e.self_device_time_total,'device_total_us':e.device_time_total})
        write_csv(out/'profiler.csv',profdata)
        operators=[e['name'] for e in profdata];write_json(out/'operators.json',operators)
        compiler_stats={k:dict(v) for k,v in counters.items() if v};write_json(out/'compiler_stats.json',compiler_stats)
        write_csv(out/'feature_differences.csv',differences);write_csv(out/'timings.csv',timings)
        numeric={}
        for kind in ['bf16','same_eager']:
            for feature in ['cls','patch12']:
                numeric[f'{kind}_{feature}_cosine_mean']=float(np.mean([r[f'{kind}_{feature}_cosine'] for r in differences]))
                numeric[f'{kind}_{feature}_max_abs']=max(r[f'{kind}_{feature}_max_abs'] for r in differences)
        numeric_pass=all(numeric[f'same_eager_{k}_cosine_mean']<=1e-3 and numeric[f'same_eager_{k}_max_abs']<=.05 for k in ['cls','patch12'])
        evidence={'int_mm_operator':any('_int_mm' in name or 'int_scaled_matmul' in name for name in operators),
            'integer_kernel_names':[name for name in operators if any(token in name.lower() for token in ['i168','s8','int8','int_scaled','_int_mm'])],
            'packed_int4_operator':any('_weight_int4pack_mm' in name or 'tinygemm' in name for name in operators),
            'cuda_graph_events':[name for name in operators if 'graph' in name.lower()]}
        cache_bytes=cache_inventory(out)
        result={'state':'completed','variant':variant,'mode':mode,'samples':len(items),'repeats':3,'timed_calls':len(timings),'quantized_layers':len(changed),
            'backbone_payload_bytes':payload,'serialized_bytes':serialized,'state_sha256':state_sha,'first_call_seconds':first_call,'warmup_seconds':warmup,
            'cold_peak_allocated_mib':cold_peak,'peak_allocated_mib':peak_allocated,'peak_reserved_mib':peak_reserved,'timings':summarize_times(timings),
            'per_scene':{scene:summarize_times([r for r in timings if r['scene']==scene]) for scene in ['R01','R02','R03','R04']},
            'numeric':numeric,'numeric_pass':numeric_pass,'kernel_evidence':evidence,'compiler_stats':compiler_stats,'generated_cache_bytes':cache_bytes,
            'seconds':time.time()-started,'free_gib':shutil.disk_usage(ROOT).free/2**30,'scope':'normal-only backbone timing including JPEG decode; not full VAD accuracy/streaming; GPU interval includes scheduling gaps'}
        write_json(out/'completed.json',result);progress('completed',numeric_pass=numeric_pass);print(json.dumps(result),flush=True)
    except Exception as e:
        write_json(out/'failure.json',{'state':'failed','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--mode',choices=MODES,required=True);a=p.parse_args();run(a.variant,a.mode)
