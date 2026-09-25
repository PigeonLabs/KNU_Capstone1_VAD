"""Normal-only LoRA adaptation, bounded RAM banks, and aligned causal evaluation."""
import argparse
from collections import deque
import hashlib
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
from . import stage5 as s5
from . import stage6 as s6
from .common import atomic_checkpoint, environment, seed_everything, write_json
from .phase_routing import reserve, digest
from .prototype import spherical_kmeans
from .stage7 import write_csv
from .lora import LoRAExtractor, photometric, cosine_loss

ROOT = Path('runs/stage8')
METHODS = ['frozen','consistency','anchored']
SCENES = s5.SCENES
load = s6.load


def begin(out, **kw):
    reserve(); out.mkdir(parents=True,exist_ok=True)
    if (out/'completed.json').exists(): return False
    if (out/'config.json').exists(): raise RuntimeError(f'Inspect partial run before restarting: {out}')
    write_json(out/'config.json',{'started_at':time.time(),'environment':environment(),
        'source_sha256':{str(p):digest(p) for p in Path('ipad').glob('*.py')},
        'protocol_sha256':digest('docs/stage8_protocol.md'),**kw})
    return True


def unit(scene,seed,method):
    return ROOT/('8-2' if seed==0 else '8-3')/scene/f'seed{seed}'/method


def adapter_dir(scene,seed,method):
    stage = '8-1' if scene=='R01' and seed==0 else ('8-2' if seed==0 else '8-3')
    return ROOT/stage/'adapters'/scene/f'seed{seed}'/method


def sampled_frames(recs,seed):
    rng=np.random.default_rng(seed);samples=[]
    for i,r in enumerate(recs):
        edges=np.linspace(0,r['frames'],min(64,r['frames'])+1,dtype=int)
        samples.extend((i,int(rng.integers(a,b))) for a,b in zip(edges[:-1],edges[1:]))
    return samples


class Images:
    def __init__(self,scene,recs,split='training'):
        self.maps=[np.load(Path('cache/frames')/scene/split/f"{r['video']}.npy",mmap_mode='r') for r in recs]
    def batch(self,samples):
        return torch.from_numpy(np.stack([self.maps[i][t] for i,t in samples])).cuda().permute(0,3,1,2).float()/127.5-1


def model_for(scene,seed,method,precision='fp32'):
    model=LoRAExtractor(method!='frozen').cuda().eval()
    if method!='frozen':
        source=adapter_dir(scene,seed,method)/'adapter.pt'
        assert (source.parent/'completed.json').exists()
        model.load_adapter(torch.load(source,map_location='cpu',weights_only=False)['adapter'])
        model.merge()
    model.requires_grad_(False)
    if precision=='bf16': model.backbone.to(torch.bfloat16)
    return model


def one_loss(model,x,method,generator=None):
    aug=photometric(x,generator)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        original=model(x)['patch12'];changed=model(aug)['patch12']
        with model.teacher(): teacher=model(x)['patch12']
    consistency=cosine_loss(original,changed)
    anchor=cosine_loss(original,teacher)
    loss=consistency+(anchor if method=='anchored' else 0)
    return loss,consistency,anchor,original


def smoke():
    out=ROOT/'8-1/smoke'
    if not begin(out,scene='R01',seed=0,diagnostic=True): return
    seed_everything(0);started=time.time();recs=s5.records('B','R01','training','train');images=Images('R01',recs)
    samples=sampled_frames(recs,0)[:8];x=images.batch(samples)
    model=LoRAExtractor().cuda().eval();params=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in params)==49152
    with torch.no_grad():
        baseline=model(x)
        with model.teacher(): ref=model(x)
        initial=max(float((baseline[k]-ref[k]).abs().max()) for k in baseline)
    assert initial==0
    base_hash={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters() if not p.requires_grad}
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);history=[]
    torch.cuda.reset_peak_memory_stats()
    # Identical augmentation every step isolates the ability to optimize a fixed task.
    for step in range(30):
        reserve();g=torch.Generator(device='cuda').manual_seed(401)
        opt.zero_grad(set_to_none=True);loss,cons,anchor,_=one_loss(model,x,'anchored',g)
        loss.backward();grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        assert torch.isfinite(loss);assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
        opt.step();history.append({'step':step,'loss':float(loss),'consistency':float(cons),'anchor':float(anchor),'grad':float(grad)})
    assert np.mean([r['loss'] for r in history[-5:]]) < np.mean([r['loss'] for r in history[:5]])
    assert all(hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()==base_hash[n] for n,p in model.named_parameters() if not p.requires_grad)
    state=model.adapter_state();atomic_checkpoint(out/'adapter.pt',{'adapter':state})
    with torch.no_grad(): before=model(x)
    restored=LoRAExtractor().cuda().eval();restored.load_adapter(torch.load(out/'adapter.pt',weights_only=False)['adapter'])
    with torch.no_grad(): after=restored(x)
    restored_error=max(float((before[k]-after[k]).abs().max()) for k in before)
    del restored
    model.merge()
    with torch.no_grad(): merged=model(x)
    merge_error=max(float((before[k]-merged[k]).abs().max()) for k in before)
    assert restored_error==0;assert merge_error<2e-5
    write_csv(out/'history.csv',history)
    result={'initial_max_error':initial,'restored_max_error':restored_error,'merged_max_error':merge_error,
        'trainable_parameters':49152,'base_unchanged':True,'base_gradients_none':True,'finite_loss_gradient':True,
        'first5_loss':float(np.mean([r['loss'] for r in history[:5]])),'last5_loss':float(np.mean([r['loss'] for r in history[-5:]])),
        'seconds':time.time()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    write_json(out/'completed.json',result);print(json.dumps(result),flush=True)


def train_adapter(scene,seed,method):
    if method=='frozen': return
    out=adapter_dir(scene,seed,method);recs=s5.records('B',scene,'training','train');val=s5.records('B',scene,'training','val')
    if not begin(out,scene=scene,seed=seed,method=method,train_videos=[r['video'] for r in recs],calibration_videos=[r['video'] for r in val],epochs=10,lr=1e-4,batch_size=32): return
    assert not {r['video'] for r in recs}&{r['video'] for r in val}
    seed_everything(seed);started=time.time();model=LoRAExtractor().cuda().eval();params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);samples=sampled_frames(recs,seed);images=Images(scene,recs)
    write_csv(out/'training_samples.csv',[{'video':recs[i]['video'],'frame':t} for i,t in samples]);history=[]
    torch.cuda.reset_peak_memory_stats()
    with (out/'batches.jsonl').open('w') as log:
        for epoch in range(10):
            order=torch.randperm(len(samples)).tolist();total=np.zeros(3);epoch_start=time.time()
            for step,a in enumerate(range(0,len(samples),32)):
                reserve();indices=order[a:a+32];x=images.batch([samples[i] for i in indices]);opt.zero_grad(set_to_none=True)
                loss,cons,anchor,features=one_loss(model,x,method);loss.backward()
                grad=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
                if not torch.isfinite(loss): raise ValueError('Nonfinite adapter loss')
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                opt.step();values=np.array([float(loss),float(cons),float(anchor)]);total+=values*len(indices)
                log.write(json.dumps({'epoch':epoch+1,'step':step,'indices':indices,'loss':values[0],'consistency':values[1],'anchor':values[2],
                    'gradient_norm':float(grad),'patch_variance':float(features.detach().var((0,1)).mean()),'time':time.time()})+'\n')
            means=total/len(samples);row={'epoch':epoch+1,'loss':means[0],'consistency':means[1],'anchor':means[2],'seconds':time.time()-epoch_start};history.append(row)
            log.flush();reserve();atomic_checkpoint(out/f'epoch{epoch+1:02d}.pt',{'adapter':model.adapter_state(),'optimizer':opt.state_dict(),'epoch':epoch+1,'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all()})
            s5.event(out,'epoch',**row);print(json.dumps({'adapter':str(out),**row}),flush=True)
    state=model.adapter_state();atomic_checkpoint(out/'adapter.pt',{'adapter':state,'method':method,'seed':seed,'epochs':10})
    x=images.batch(samples[:8])
    with torch.no_grad(): unmerged=model(x)
    model.load_adapter(torch.load(out/'adapter.pt',map_location='cpu',weights_only=False)['adapter'])
    with torch.no_grad(): restored=model(x)
    restore_error=max(float((unmerged[k]-restored[k]).abs().max()) for k in unmerged)
    model.merge()
    with torch.no_grad(): merged=model(x)
    error=max(float((unmerged[k]-merged[k]).abs().max()) for k in unmerged)
    assert restore_error==0 and error<2e-5
    write_json(out/'history.json',history);write_json(out/'completed.json',{'scene':scene,'seed':seed,'method':method,'samples':len(samples),'epochs':10,
        'trainable_parameters':49152,'restore_error':restore_error,'merge_error':error,'adapter_sha256':digest(out/'adapter.pt'),
        'first_loss':history[0]['loss'],'last_loss':history[-1]['loss'],'seconds':time.time()-started,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'training_data_only':True,'checkpoint_selection':'fixed final epoch; no test/calibration selection'})


@torch.inference_mode()
def extract_training(model,recs,scene,seed,out):
    selected,occupied=s5.selected_frames(recs,seed);index={v:i for i,v in enumerate(selected)}
    pool=np.empty((len(selected),324,768),dtype=np.float16);clsrecs=[];hashes=[];images=Images(scene,recs)
    for ri,r in enumerate(recs):
        dest=out/'cls'/r['video'];dest.mkdir(parents=True,exist_ok=True);cs=np.empty((r['frames'],768),dtype=np.float16);h=hashlib.sha256()
        for a in range(0,r['frames'],32):
            reserve();ids=list(range(a,min(a+32,r['frames'])));f=model(images.batch([(ri,t) for t in ids]));c=f['cls'].cpu().numpy().astype(np.float16);p=f['patch12'].cpu().numpy().astype(np.float16)
            cs[a:a+len(ids)]=c;h.update(c.tobytes());h.update(p.tobytes())
            for j,t in enumerate(ids):
                if (ri,t) in index: pool[index[ri,t]]=p[j]
        np.save(dest/'cls.npy',cs);hashes.append({'video':r['video'],'frames':r['frames'],'cls_sha256':digest(dest/'cls.npy'),'features_sha256':h.hexdigest()})
        clsrecs.append({**r,'path':str(dest)});s5.event(out,'train_features',video=r['video'])
    write_json(out/'training_feature_hashes.json',hashes)
    return clsrecs,pool,selected,occupied


def fit_memory(pool,selected,occupied,recs,seed,out):
    parts=[]
    for p in range(0,324,9):
        reserve();x=torch.from_numpy(np.array(pool[:,p:p+9])).cuda().transpose(0,1)
        parts.append(spherical_kmeans(x,occupied*10,20,seed).cpu().half());s5.event(out,'memory_fit',position=p)
    bank=torch.cat(parts);atomic_checkpoint(out/'memory.pt',{'bank':bank,'seed':seed,'k':10})
    write_csv(out/'memory_selection.csv',[{'video':recs[i]['video'],'frame':t} for i,t in selected])
    assert (out/'memory_selection.csv').read_text()==(s5.unit_path('B',out.parent.parent.name,seed,10)/'memory_selection.csv').read_text()
    return torch.nn.functional.normalize(bank.cuda().float(),dim=-1)


@torch.inference_mode()
def infer(model,head,bank,recs,scene,split,period,batch_size=32,jpeg=False):
    result=[];hashes=[]
    for r in recs:
        inputs=None if jpeg else np.load(Path('cache/frames')/scene/split/f"{r['video']}.npy",mmap_mode='r')
        files=s5.frames(Path('IPAD_dataset')/scene/split/'frames'/r['video']) if jpeg else None
        if files is not None: assert len(files)==r['frames']
        cs=deque(maxlen=16);phases=deque(maxlen=21);h=hashlib.sha256()
        for a in range(0,r['frames'],batch_size):
            reserve();ids=range(a,min(a+batch_size,r['frames']))
            if jpeg:
                values=[]
                for t in ids:
                    image=cv2.imread(str(files[t]))
                    if image is None: raise ValueError(files[t])
                    values.append(cv2.resize(image,(256,256)))
                x=np.stack(values)
            else: x=np.array(inputs[a:a+batch_size])
            f=model(torch.from_numpy(x).cuda().permute(0,3,1,2).float()/127.5-1)
            app=s6.match(f['patch12'],bank).cpu().numpy()
            for key in ['cls','patch12']: h.update(f[key].cpu().numpy().tobytes())
            for j,t in enumerate(ids):
                cs.append(f['cls'][j])
                if len(cs)<16: continue
                phase=int(head(torch.stack(list(cs))[None]).argmax(-1));phases.append(phase)
                if len(phases)<21: continue
                result.append({'video':r['video'],'frame':t,'video_length':r['frames'],'phase':phase,'appearance':float(app[j]),'temporal':float(s5.causal_time(phases,period)[-1])})
        hashes.append({'video':r['video'],'frames':r['frames'],'sha256':h.hexdigest(),'batch_size':batch_size,'split':split})
    return result,hashes


def finish(out,scene,seed,method,period,validation,raw,hashes,started,precision):
    cal=s5.calibrate(validation);scored=s5.apply_calibration(raw,cal);common=s5.aligned(scored,scene);full=s5.aligned(scored,scene,common=False)
    for name,data in [('validation_raw',validation),('causal_raw',raw),('scores',common),('online_scores',full)]: write_csv(out/f'{name}.csv',data)
    old=s6.rows(s5.unit_path('B',scene,seed,10)/'scores.csv')
    assert [(r['video'],r['frame'],r['label']) for r in common]==[(r['video'],r['frame'],r['label']) for r in old]
    op,events,alarms=s5.operation_metrics(full,scene,cal['threshold'])
    write_csv(out/'segments.csv',events);write_csv(out/'alarms.csv',alarms);write_json(out/'calibration.json',cal);write_json(out/'feature_hashes.json',hashes)
    assert set(cal['calibration_videos'])=={r['video'] for r in s5.records('B',scene,'training','val')}
    m={'scene':scene,'seed':seed,'method':method,'precision':precision,'period':period,'metrics':s5.metric_rows(common),'operation':op,'common_frames':len(common),'online_frames':len(full),
        'seconds':time.time()-started,'threshold':cal['threshold'],'head_sha256':digest(unit(scene,seed,method)/'causal_head.pt'),'memory_sha256':digest(unit(scene,seed,method)/'memory.pt')}
    if method!='frozen':m['adapter_sha256']=digest(adapter_dir(scene,seed,method)/'adapter.pt')
    if scene=='R02':write_json(out/'alignment_sensitivity.json',{str(shift):s5.metric_rows(s5.aligned(scored,scene,policy='common',shift=shift)) for shift in [-1,0,1]})
    write_json(out/'completed.json',m);print(json.dumps({'completed':str(out),'seconds':m['seconds']}),flush=True)


def run_unit(scene,seed,method):
    out=unit(scene,seed,method);recs=s5.records('B',scene,'training','train');val=s5.records('B',scene,'training','val')
    assert (ROOT/'freeze.json').exists()
    if not begin(out,scene=scene,seed=seed,method=method,precision='fp32',train_videos=[r['video'] for r in recs],calibration_videos=[r['video'] for r in val],freeze_sha256=digest(ROOT/'freeze.json')): return
    started=time.time();model=model_for(scene,seed,method);clsrecs,pool,selected,occupied=extract_training(model,recs,scene,seed,out)
    head=s5.train_head(clsrecs,'causal',768,seed,out)
    bank=fit_memory(pool,selected,occupied,recs,seed,out);del pool;period=float(np.median([r['frames'] for r in recs]))
    validation,vh=infer(model,head,bank,val,scene,'training',period)
    raw,th=infer(model,head,bank,s5.records('B',scene,'testing'),scene,'testing',period)
    finish(out,scene,seed,method,period,validation,raw,vh+th,started,'fp32_batch32')


def bf16_models(scene,seed,method):
    source=unit(scene,seed,method);model=model_for(scene,seed,method,'bf16')
    head=s6.PrecisionHead(768).cuda().eval();head.load_state_dict(torch.load(source/'causal_head.pt',map_location='cpu',weights_only=False)['model']);head.to(torch.bfloat16).requires_grad_(False)
    bank=torch.load(source/'memory.pt',map_location='cpu',weights_only=False)['bank']
    return model,head,torch.nn.functional.normalize(bank.cuda().float(),dim=-1).to(torch.bfloat16)


def stream_dir(scene,seed,method): return ROOT/'8-3/streams'/scene/f'seed{seed}'/method


def accuracy(scene,seed,method):
    out=stream_dir(scene,seed,method)
    if not begin(out,scene=scene,seed=seed,method=method,precision='bf16',batch_size=1,jpeg=True): return
    started=time.time();model,head,bank=bf16_models(scene,seed,method);period=load(unit(scene,seed,method)/'completed.json')['period']
    validation,vh=infer(model,head,bank,s5.records('B',scene,'training','val'),scene,'training',period,1,True)
    raw,th=infer(model,head,bank,s5.records('B',scene,'testing'),scene,'testing',period,1,True)
    finish(out,scene,seed,method,period,validation,raw,vh+th,started,'bf16_batch1')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['smoke','train','unit','accuracy'],required=True);p.add_argument('--scene',choices=SCENES,default='R01');p.add_argument('--seed',type=int,default=0);p.add_argument('--method',choices=METHODS,default='anchored');a=p.parse_args()
    torch.set_num_threads(8);reserve()
    if a.action=='smoke': smoke()
    elif a.action=='train': train_adapter(a.scene,a.seed,a.method)
    elif a.action=='unit': run_unit(a.scene,a.seed,a.method)
    else: accuracy(a.scene,a.seed,a.method)
