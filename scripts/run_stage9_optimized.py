"""Fixed-input, equally optimized low-bit inference; sequential auditable suite."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.stage9_quant_compile import samples,image_input,disk_check,torch,np,PrecisionExtractor,cache_inventory
from ipad.stage9_optimized import BackboneOnly,preprocess,configure,build_runtime
from scripts.probe_quantization import tensor_payload
from scripts.diagnose_stage9 import diff,profile
from ipad.common import write_json,environment,seed_everything
from ipad.phase_routing import digest
from ipad.stage7 import write_csv
BASE=ROOT/'runs/stage9/9-1/optimized'
VARIANTS=['bf16','w8a16','w8a8','w4_native','w4_packed']
MODES=['eager','graph','compile_graph','autotune_graph']


def progress(out,step,**extra):write_json(out/'progress.json',{'step':step,'time':time.time(),**extra})


def refs(model,items,dtype):
    values=[]
    body=BackboneOnly(model.backbone).eval()
    for *_,p in items:
        disk_check();x=preprocess(model,image_input(p)).to(dtype)
        values.append({k:v.cpu() for k,v in body(x).items()})
    return values


def stats(rows,key):
    a=np.array([r[key] for r in rows]);return {'mean':float(a.mean()),'p50':float(np.median(a)),'p95':float(np.quantile(a,.95)),'max':float(a.max())}


@torch.inference_mode()
def unit(phase,variant,mode):
    out=BASE/phase/f'{variant}_{mode}';out.mkdir(parents=True,exist_ok=True);disk_check()
    if (out/'config.json').exists():raise RuntimeError('Refusing overwrite: '+str(out))
    seed_everything(0);torch.set_num_threads(8);torch._dynamo.config.suppress_errors=False
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    items=samples()
    if phase=='pilot':items=[items[i] for i in np.linspace(0,len(items)-1,16,dtype=int)]
    started=time.time()
    config={'phase':phase,'variant':variant,'mode':mode,'seed':0,'batch_size':1,'test_labels_used':False,'environment':environment(),
        'script_sha256':digest(__file__),'module_sha256':digest(ROOT/'ipad/stage9_optimized.py'),'protocol_sha256':digest(ROOT/'docs/stage9_optimization_protocol.md'),
        'weights_sha256':digest(ROOT/'cache/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth'),
        'command':sys.argv,'started_at':started,'eager_fp32_preprocessing':True,'fp32_tf32':False,
        'samples':[{'scene':s,'video':v,'frame':t,'sha256':digest(p)} for s,v,t,p in items],
        'compile_env':{k:os.environ.get(k) for k in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_FREEZING']}}
    write_json(out/'config.json',config)
    try:
        progress(out,'fp32_reference');fp=PrecisionExtractor('B','fp32').cuda().eval();fp_ref=refs(fp,items,torch.float32);del fp;gc.collect();torch.cuda.empty_cache()
        progress(out,'bf16_reference');model=PrecisionExtractor('B','bf16').cuda().eval();bf_ref=refs(model,items,torch.bfloat16)
        details=configure(model.backbone,variant);write_json(out/'optimization.json',details)
        body=BackboneOnly(model.backbone).cuda().eval();payload,_=tensor_payload(body)
        progress(out,'same_eager_reference');same_ref=refs(model,items,torch.bfloat16)
        example=preprocess(model,image_input(items[0][3])).to(torch.bfloat16)
        from torch._dynamo.utils import counters
        counters.clear();progress(out,'compile_or_capture');start=time.perf_counter()
        runtime=build_runtime(body,example,mode);z=runtime(example);torch.cuda.synchronize();del z
        startup=time.perf_counter()-start
        for _ in range(12):disk_check();z=runtime(example);del z
        torch.cuda.synchronize();progress(out,'numeric_validation');numeric=[]
        for idx,((scene,video,frame,p),a,b,c) in enumerate(zip(items,fp_ref,bf_ref,same_ref)):
            disk_check();x=preprocess(model,image_input(p)).to(torch.bfloat16)
            pre_hash=hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
            f=runtime(x);cpu={k:v.cpu() for k,v in f.items()};del f
            row={'scene':scene,'video':video,'frame':frame,'input_bf16_sha256':pre_hash}
            for feature,value in cpu.items():
                assert torch.isfinite(value).all()
                for name,ref in [('fp32',a),('bf16_eager',b),('same_eager',c)]:
                    for metric,val in diff(value,ref[feature]).items():row[f'{name}_{feature}_{metric}']=val
            numeric.append(row)
        write_csv(out/'feature_differences.csv',numeric)
        del fp_ref,bf_ref,same_ref,cpu;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
        progress(out,'timing');torch.cuda.reset_peak_memory_stats();timings=[]
        event_a=torch.cuda.Event(enable_timing=True);event_b=torch.cuda.Event(enable_timing=True)
        for repeat in range(3):
            for scene,video,frame,p in items:
                disk_check();torch.cuda.synchronize();start=time.perf_counter()
                x=preprocess(model,image_input(p)).to(torch.bfloat16)
                # Account GPU preprocessing fully before the backbone interval.
                torch.cuda.synchronize();prepared=time.perf_counter()
                event_a.record();f=runtime(x);event_b.record();torch.cuda.synchronize();end=time.perf_counter();del f
                timings.append({'repeat':repeat,'scene':scene,'video':video,'frame':frame,'preprocess_ms':(prepared-start)*1000,'model_ms':(end-prepared)*1000,'total_ms':(end-start)*1000,'gpu_ms':event_a.elapsed_time(event_b)})
        allocated=torch.cuda.max_memory_allocated()/2**20;reserved=torch.cuda.max_memory_reserved()/2**20
        write_csv(out/'timings.csv',timings);progress(out,'profiler');write_csv(out/'profiler.csv',profile(runtime,example))
        summary={}
        for ref in ['fp32','bf16_eager','same_eager']:
            for feature in ['cls','patch12']:
                summary[f'{ref}_{feature}_cosine_mean']=float(np.mean([r[f'{ref}_{feature}_cosine_mean'] for r in numeric]))
                summary[f'{ref}_{feature}_max_abs']=max(r[f'{ref}_{feature}_max_abs'] for r in numeric)
        passed=all(summary[f'same_eager_{k}_cosine_mean']<=1e-3 and summary[f'same_eager_{k}_max_abs']<=.05 for k in ['cls','patch12'])
        write_json(out/'compiler_stats.json',{k:dict(v) for k,v in counters.items() if v})
        cache_bytes=cache_inventory(out)
        result={'state':'completed','phase':phase,'variant':variant,'mode':mode,'samples':len(items),'timed_calls':len(timings),'timings':{k:stats(timings,k) for k in ['preprocess_ms','model_ms','total_ms','gpu_ms']},
            'numeric':summary,'numeric_pass':passed,'peak_allocated_mib':allocated,'peak_reserved_mib':reserved,'payload_bytes':payload,'startup_seconds':startup,'cache_bytes':cache_bytes,'seconds':time.time()-started,'free_gib':shutil.disk_usage(ROOT).free/2**30}
        write_json(out/'completed.json',result);progress(out,'completed',numeric_pass=passed);print(json.dumps(result),flush=True)
    except Exception:
        write_json(out/'failure.json',{'state':'failed','traceback':traceback.format_exc(),'time':time.time()});raise


def suite():
    BASE.mkdir(parents=True,exist_ok=True);disk_check()
    if (BASE/'status.json').exists():raise RuntimeError('Review existing suite; no automatic restart')
    state={'state':'running','started_at':time.time(),'completed':[],'failed':[]}
    guardlog=(BASE/'disk_guard.log').open('a')
    guard=subprocess.Popen([sys.executable,'scripts/disk_guard.py'],cwd=ROOT,stdout=guardlog,stderr=subprocess.STDOUT)
    def update(**kw):state.update(kw);write_json(BASE/'status.json',state)
    def execute(phase,v,m):
        disk_check();key=f'{phase}/{v}_{m}';out=BASE/phase/f'{v}_{m}';cache=ROOT/'cache/stage9_optimized'/phase/f'{v}_{m}'
        env=os.environ.copy();env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_FREEZING='0',TORCH_LOGS='recompiles,graph_breaks')
        cmd=[sys.executable,'scripts/run_stage9_optimized.py','--phase',phase,'--variant',v,'--mode',m]
        with (BASE/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':cmd,'env':{k:env[k] for k in ['TORCHINDUCTOR_CACHE_DIR','TRITON_CACHE_DIR','TORCHINDUCTOR_COMPILE_THREADS','TORCHINDUCTOR_FREEZING']}})+'\n')
        update(step=key)
        with (BASE/f'{phase}_{v}_{m}.log').open('a') as f:p=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
        (state['completed'] if p.returncode==0 else state['failed']).append(key);update()
        if (ROOT/'runs/disk_pause.json').exists():raise RuntimeError('Disk pause; never resume automatically')
        return json.loads((out/'completed.json').read_text()) if p.returncode==0 else None
    try:
        update();pilots={}
        for v in VARIANTS:
            pilots[v]=[]
            for m in MODES:
                r=execute('pilot',v,m)
                if r:pilots[v].append(r)
        chosen={}
        for v,results in pilots.items():
            eligible=[r for r in results if r['numeric_pass']]
            if not eligible:chosen[v]=None;continue
            chosen[v]=min(eligible,key=lambda r:r['timings']['model_ms']['mean'])['mode']
        write_json(BASE/'frozen_selection.json',{'rule':'fastest pilot model mean among same-eager numerical-gate passing conditions; normal images only','chosen':chosen,'time':time.time(),'pilot_configs_sha256':{str(p.relative_to(BASE)):digest(p) for p in (BASE/'pilot').glob('*/config.json')}})
        for v,m in chosen.items():
            if m is None:continue
            r=execute('full',v,m)
            if (not r or not r['numeric_pass']) and m!='graph':execute('full',v,'graph')
        update(state='completed' if not state['failed'] else 'completed_with_failures',finished_at=time.time())
    except Exception as e:update(state='paused_low_disk' if (ROOT/'runs/disk_pause.json').exists() else 'failed',error=str(e));raise
    finally:
        if guard.poll() is None:guard.terminate();guard.wait()
        guardlog.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',action='store_true');p.add_argument('--phase',choices=['pilot','full']);p.add_argument('--variant',choices=VARIANTS);p.add_argument('--mode',choices=MODES);a=p.parse_args()
    if a.suite:suite()
    else:unit(a.phase,a.variant,a.mode)
