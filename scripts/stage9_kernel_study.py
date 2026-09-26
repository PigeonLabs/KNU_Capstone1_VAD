"""Real-activation linear-kernel and warm/evicted-cache diagnostic."""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stage9_quant_compile import torch,np,samples,image_input,PrecisionExtractor,disk_check,cache_inventory
from ipad.stage9_optimized import PackedInt4Linear,quantize_,Int4WeightOnlyConfig
from ipad.stage9_kernel import fused_linear,CONFIGS
from ipad.common import write_json,environment,seed_everything
from ipad.phase_routing import digest
from ipad.stage7 import write_csv
from scripts.diagnose_stage9 import profile
from torch.nn import functional as F
OUT=ROOT/'runs/stage9/9-1/kernel_study'
LAYERS={'qkv':'blocks.0.attn.qkv','proj':'blocks.0.attn.proj','fc1':'blocks.0.mlp.fc1','fc2':'blocks.0.mlp.fc2'}


def setup():
    seed_everything(0);torch.set_num_threads(8);torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch._dynamo.config.recompile_limit=256;torch._dynamo.config.accumulated_recompile_limit=512
    model=PrecisionExtractor('B','bf16').cuda().eval();captured={};hooks=[]
    for name,path in LAYERS.items():
        def hook(module,args,name=name):captured[name]=args[0].detach().clone().contiguous()
        hooks.append(model.backbone.get_submodule(path).register_forward_pre_hook(hook))
    inputs={};items=[]
    for scene in ['R01','R02','R03','R04']:
        item=next(x for x in samples() if x[0]==scene);items.append(item)
        model(image_input(item[3]));inputs[scene]=dict(captured)
    for h in hooks:h.remove()
    return model,inputs,items


def error(value,ref):
    v=value.float();r=ref.float()
    rel=float(torch.linalg.vector_norm(v-r)/torch.linalg.vector_norm(r).clamp_min(1e-12))
    cosine=float((1-F.cosine_similarity(v,r,dim=-1)).clamp_min(0).mean())
    return {'relative_l2':rel,'cosine':cosine,'max_abs':float((v-r).abs().max()),'pass':bool(torch.isfinite(v).all()) and rel<=.01 and cosine<=1e-3}


def compiled(fn):
    return torch.compile(fn,fullgraph=True,dynamic=False,options={'max_autotune':True,'max_autotune_gemm_backends':'ATEN,TRITON','triton.cudagraphs':False,'emulate_precision_casts':True})


def graph(fn,x,repeats):
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(4):y=fn(x)
    torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(repeats):y=fn(x)
    return g,y


def measure(g,repeats,flush=None):
    if flush is not None:flush.zero_()
    a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
    a.record();g.replay();b.record();b.synchronize();return a.elapsed_time(b)*1000/repeats


def choose_fused(xs,packed,out):
    candidates=[];refs={s:packed(x) for s,x in xs.items()}
    for config in CONFIGS:
        disk_check();fn=lambda x,c=config:fused_linear(x,packed,c)
        checks={s:error(fn(x),refs[s]) for s,x in xs.items()}
        g,y=graph(fn,xs['R01'],32);times=[measure(g,32) for _ in range(5)]
        candidates.append({'tile':config,'checks':checks,'warm_us':float(np.median(times)),'all_pass':all(r['pass'] for r in checks.values())});del g,y
    write_json(out/'fused_tuning.json',candidates)
    eligible=[r for r in candidates if r['all_pass']]
    if not eligible:return None
    return min(eligible,key=lambda r:r['warm_us'])['tile']


def variants(layer,packed,native,tile):
    weight=packed.dequantized_weight()
    funcs={'bf16':compiled(lambda x:F.linear(x,layer.weight,layer.bias)),
        'native_int4':compiled(lambda x:native(x)),
        'packed_int4':compiled(lambda x:packed(x)),
        'dequant_only':compiled(lambda x:packed.dequantized_weight()),
        'predequant_gemm':compiled(lambda x:torch.matmul(x,weight.t())+packed.bias)}
    if tile:funcs['fused_int4']=lambda x:fused_linear(x,packed,tile)
    return funcs


@torch.inference_mode()
def study():
    OUT.mkdir(parents=True,exist_ok=True);disk_check();assert not (OUT/'config.json').exists()
    started=time.time();model,inputs,items=setup()
    write_json(OUT/'config.json',{'seed':0,'environment':environment(),'source_sha256':digest(__file__),'kernel_source_sha256':digest(ROOT/'ipad/stage9_kernel.py'),'protocol_sha256':digest(ROOT/'docs/stage9_kernel_protocol.md'),'weights_sha256':digest(ROOT/'cache/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth'),'command':sys.argv,'test_labels_used':False,'started_at':started,'L2_bytes':torch.cuda.get_device_properties(0).L2_cache_size,'flush_bytes':256*2**20,'samples':[{'scene':s,'video':v,'frame':t,'sha256':digest(p)} for s,v,t,p in items],'input_hashes':{s:{n:hashlib.sha256(x.view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n,x in d.items()} for s,d in inputs.items()}})
    flush=torch.empty(256*2**20,device='cuda',dtype=torch.uint8);all_rows=[];numeric=[];choices={};failed=[]
    for name,path in LAYERS.items():
        layer=model.backbone.get_submodule(path);packed=PackedInt4Linear(layer)
        native=copy.deepcopy(layer);quantize_(native,Int4WeightOnlyConfig(group_size=128,int4_packing_format='tile_packed_to_4d',set_inductor_config=False))
        for m in [1,16,64,325]:
            disk_check();out=OUT/'cases'/f'{name}_m{m}';out.mkdir(parents=True,exist_ok=True)
            write_json(OUT/'progress.json',{'step':str(out.relative_to(OUT)),'time':time.time()})
            xs={s:d[name][:,:m,:].contiguous() for s,d in inputs.items()};tile=choose_fused(xs,packed,out);choices[f'{name}_m{m}']=tile
            funcs=variants(layer,packed,native,tile);graphs={};case_rows=[];case_numeric=[];compile_times={}
            for kind,fn in funcs.items():
                try:
                    start=time.perf_counter();v=fn(xs['R01']);torch.cuda.synchronize();compile_times[kind]=time.perf_counter()-start;del v
                    for scene,x in xs.items():
                        ref=layer(x) if kind=='bf16' else (packed.dequantized_weight() if kind=='dequant_only' else (native(x) if kind=='native_int4' else packed(x)))
                        info=error(fn(x),ref);row={'layer':name,'M':m,'N':layer.out_features,'K':layer.in_features,'kind':kind,'scene':scene,**info};case_numeric.append(row)
                    graphs[kind]=(graph(fn,xs['R01'],32),graph(fn,xs['R01'],1))
                except Exception:
                    failure={'layer':name,'M':m,'kind':kind,'traceback':traceback.format_exc()};failed.append(failure);write_json(out/(kind+'_failure.json'),failure)
            order=list(graphs);rng=random.Random(0)
            for regime,rounds,index,reps in [('warm',7,0,32),('eviction_attempt',21,1,1)]:
                for repeat in range(rounds):
                    rng.shuffle(order)
                    for kind in order:
                        disk_check();duration=measure(graphs[kind][index][0],reps,flush if regime=='eviction_attempt' else None)
                        row={'layer':name,'M':m,'N':layer.out_features,'K':layer.in_features,'kind':kind,'regime':regime,'repeat':repeat,'us':duration,'calls_per_event':reps};case_rows.append(row)
            if m in [1,325]:
                for kind in ['bf16','packed_int4','fused_int4','dequant_only','predequant_gemm']:
                    if kind in graphs:write_csv(out/(kind+'_profiler.csv'),profile(funcs[kind],xs['R01']))
            write_csv(out/'timings.csv',case_rows);write_csv(out/'numerics.csv',case_numeric);write_json(out/'compile_seconds.json',compile_times)
            all_rows+=case_rows;numeric+=case_numeric
            write_json(out/'completed.json',{'state':'completed','fused_tile':tile,'measurements':len(case_rows)})
            del graphs,funcs;gc.collect();torch.cuda.empty_cache()
    write_csv(OUT/'timings.csv',all_rows);write_csv(OUT/'numerics.csv',numeric);write_json(OUT/'fused_choices.json',choices);write_json(OUT/'failures.json',failed)
    cache_inventory(OUT)
    write_json(OUT/'completed.json',{'state':'completed' if not failed else 'completed_with_failures','seconds':time.time()-started,'cases':16,'timing_rows':len(all_rows),'numeric_rows':len(numeric),'test_labels_used':False,'hardware_counters':'requires separate administrator capture'})


@torch.inference_mode()
def counters():
    # Separate process: no hardware profiling overlaps the benchmark.
    assert (OUT/'completed.json').exists(),'Finish ordinary timing before counters'
    model,inputs,_=setup();choices=json.loads((OUT/'fused_choices.json').read_text());flush=torch.empty(256*2**20,device='cuda',dtype=torch.uint8)
    for name,path in LAYERS.items():
        layer=model.backbone.get_submodule(path);packed=PackedInt4Linear(layer);native=copy.deepcopy(layer)
        quantize_(native,Int4WeightOnlyConfig(group_size=128,int4_packing_format='tile_packed_to_4d',set_inductor_config=False))
        for m in [1,325]:
            x=inputs['R01'][name][:,:m,:].contiguous();funcs=variants(layer,packed,native,choices[f'{name}_m{m}'])
            for kind in ['bf16','packed_int4','fused_int4']:
                if kind not in funcs:continue
                fn=funcs[kind]
                for _ in range(5):y=fn(x)
                torch.cuda.synchronize()
                for regime in ['warm','eviction_attempt']:
                    disk_check()
                    if regime=='eviction_attempt':flush.zero_();torch.cuda.synchronize()
                    label=f'{name}_M{m}_{kind}_{regime}';torch.cuda.nvtx.range_push(label)
                    torch.cuda.cudart().cudaProfilerStart();y=fn(x);torch.cuda.synchronize();torch.cuda.cudart().cudaProfilerStop()
                    torch.cuda.nvtx.range_pop();print('COUNTER_RANGE',label,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--counters',action='store_true');a=p.parse_args()
    if a.counters:counters()
    else:
        OUT.mkdir(parents=True,exist_ok=True);log=(OUT/'disk_guard.log').open('a');guard=subprocess.Popen([sys.executable,'scripts/disk_guard.py'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        try:study()
        except Exception:write_json(OUT/'failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
        finally:
            if guard.poll() is None:guard.terminate();guard.wait()
            log.close()
