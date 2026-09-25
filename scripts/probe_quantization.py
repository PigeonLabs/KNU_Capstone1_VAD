"""Normal-only DINOv2 weight quantization feasibility; not final VAD evaluation."""
import argparse
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'cache/quantization/torchao017'))
import cv2
import numpy as np
import torch
import torchao
from torchao.quantization import quantize_,Int8WeightOnlyConfig,Int4WeightOnlyConfig
from ipad.stage6 import PrecisionExtractor
from ipad.stage5 import records,frames
from ipad.common import write_json,environment,seed_everything
from ipad.stage7 import write_csv
from ipad.phase_routing import reserve,digest


def disk_check():
    if shutil.disk_usage(ROOT).free<=10*2**30 and not (ROOT/'runs/disk_pause.json').exists():
        from scripts.disk_guard import pause
        pause(shutil.disk_usage(ROOT).free)
    reserve()


def samples():
    result=[]
    for r in records('B','R01','training','train')[:3]:
        files=frames(ROOT/'IPAD_dataset/R01/training/frames'/r['video'])
        for t in np.linspace(0,len(files)-1,32,dtype=int):result.append((r['video'],int(t),files[t]))
    return result


def tensor_payload(model):
    storages={};types={}
    def visit(t):
        if type(t) not in (torch.Tensor,torch.nn.Parameter) and hasattr(t,'__tensor_flatten__'):
            names,_=t.__tensor_flatten__()
            for name in names: visit(getattr(t,name))
        else:
            storage=t.untyped_storage();key=(str(t.device),storage.data_ptr());storages[key]=storage.nbytes()
    for n,p in model.state_dict().items():
        types[n]=str(type(p));visit(p)
    return sum(storages.values()),types


def image_input(path):
    image=cv2.imread(str(path))
    if image is None:raise ValueError(path)
    image=cv2.resize(image,(256,256))
    return torch.from_numpy(image).cuda().permute(2,0,1)[None].float()/127.5-1


@torch.inference_mode()
def run(variant):
    out=ROOT/'runs/quantization_probe'/variant
    disk_check();out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists():return
    if (out/'config.json').exists():raise RuntimeError(f'Partial result: {out}; do not overwrite')
    started=time.time();seed_everything(0);torch.set_num_threads(8);items=samples()
    config={'variant':variant,'state':'diagnostic','environment':environment(),'torchao':torchao.__version__,
        'source_sha256':digest(__file__),'protocol_sha256':digest(ROOT/'docs/quantization_probe.md'),
        'pretrained_sha256':digest(ROOT/'cache/torch_hub/checkpoints/dinov2_vitb14_pretrain.pth'),
        'samples':[{'video':v,'frame':t,'sha256':digest(f)} for v,t,f in items],
        'command':sys.argv,'started_at':started,'test_labels_used':False,'compile':False}
    write_json(out/'config.json',config)
    try:
        model=PrecisionExtractor('B','bf16').cuda().eval();reference=[]
        for v,t,file in items:
            disk_check();features=model(image_input(file));reference.append({k:x.cpu() for k,x in features.items()})
        del features
        total_linear=sum(isinstance(m,torch.nn.Linear) for m in model.backbone.modules())
        if variant=='int8':quantize_(model.backbone,Int8WeightOnlyConfig(version=2))
        elif variant=='int4':quantize_(model.backbone,Int4WeightOnlyConfig(group_size=128,int4_packing_format='tile_packed_to_4d'))
        changed=[n for n,m in model.backbone.named_modules() if isinstance(m,torch.nn.Linear) and type(m.weight) not in (torch.Tensor,torch.nn.Parameter)]
        if variant!='bf16':assert len(changed)==total_linear==48,(len(changed),total_linear)
        payload,types=tensor_payload(model.backbone);buffer=io.BytesIO();torch.save(model.backbone.state_dict(),buffer)
        serialized_bytes=buffer.tell();state_hash=hashlib.sha256(buffer.getbuffer()).hexdigest();del buffer
        differences=[]
        for (v,t,file),ref in zip(items,reference):
            disk_check();f=model(image_input(file));row={'video':v,'frame':t}
            for k,x in f.items():
                assert torch.isfinite(x).all()
                old=ref[k].cuda();row[k+'_cosine_distance']=float((1-(x*old).sum(-1)).clamp_min(0).mean());row[k+'_max_abs_difference']=float((x-old).abs().max());del old
            differences.append(row)
        del reference,f;gc.collect();torch.cuda.empty_cache()
        x=image_input(items[0][2])
        for _ in range(8):model(x)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();timings=[]
        for repeat in range(3):
            for v,t,file in items:
                disk_check();torch.cuda.synchronize();start=time.perf_counter();inp=image_input(file);f=model(inp);torch.cuda.synchronize();duration=(time.perf_counter()-start)*1000
                timings.append({'repeat':repeat,'video':v,'frame':t,'processing_ms':duration})
        peak_allocated=torch.cuda.max_memory_allocated()/2**20;peak_reserved=torch.cuda.max_memory_reserved()/2**20
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:model(x);torch.cuda.synchronize()
        operators=sorted({e.key for e in prof.key_averages()});write_json(out/'profiler_operators.json',operators)
        write_json(out/'tensor_types.json',types);write_csv(out/'feature_differences.csv',differences);write_csv(out/'timings.csv',timings)
        times=np.array([r['processing_ms'] for r in timings])
        result={'variant':variant,'state':'diagnostic_complete','samples':len(items),'repeats':3,'batch_size':1,'quantized_linear_layers':len(changed),'total_linear_layers':total_linear,
            'backbone_tensor_payload_bytes':payload,'backbone_serialized_bytes':serialized_bytes,'state_sha256':state_hash,
            'peak_allocated_mib':peak_allocated,'peak_reserved_mib':peak_reserved,'mean_ms':float(times.mean()),'p50_ms':float(np.median(times)),'p95_ms':float(np.quantile(times,.95)),
            'patch_cosine_distance_mean':float(np.mean([r['patch12_cosine_distance'] for r in differences])),
            'cls_cosine_distance_mean':float(np.mean([r['cls_cosine_distance'] for r in differences])),
            'seconds':time.time()-started,'scope':'normal-only backbone probe; not VAD accuracy/full streaming; CPU JPEG + preprocess + backbone + GPU sync; no torch.compile',
            'free_gib':shutil.disk_usage(ROOT).free/2**30}
        write_json(out/'completed.json',result);print(json.dumps(result),flush=True)
    except Exception as e:
        write_json(out/'failure.json',{'state':'failed','error':repr(e),'traceback':traceback.format_exc(),'time':time.time()});raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['bf16','int8','int4'],required=True);run(p.parse_args().variant)
