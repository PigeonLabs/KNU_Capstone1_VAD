"""Decoder-free DINOv2 phase-conditioned and budget-matched prototype baselines."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from .common import SCENES, atomic_checkpoint, seed_everything, write_json
from .data import validation_videos
from .evaluate import score_rows, write_csv


class FeatureClips(Dataset):
    def __init__(self,scene,split,root='IPAD_dataset',subset='all',diagnostic=False):
        base=Path('cache/dino')/scene
        if not diagnostic and not (base/'complete.json').exists():
            raise ValueError('Full feature extraction must complete before a reportable experiment')
        self.split=split; self.records=[]; self.samples=[]; self.maps={}
        heldout=validation_videos(root,scene)
        for p in sorted((base/split).glob('*/meta.json'),key=lambda p:int(p.parent.name)):
            info=json.loads(p.read_text());video=info['video']
            if split=='training' and subset!='all' and ((video in heldout)!=(subset=='val')):continue
            ri=len(self.records)
            self.records.append({**info,'path':p.parent})
            self.samples.extend((ri,start) for start in range(info['frames']-15))
        if not self.samples:raise ValueError('No cached feature clips for requested split')

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        ri,start=self.samples[index];r=self.records[ri]
        if ri not in self.maps:self.maps[ri]=np.load(r['path']/'cls.npy',mmap_mode='r')
        row={'cls':np.array(self.maps[ri][start:start+16],dtype=np.float32),
             'video':r['video'],'frame':start+8,'length':r['frames'],'record':ri}
        if self.split=='training':row['phase']=min(199,start*200//r['frames'])
        return row


class PhaseHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(nn.Flatten(1),nn.Linear(16*768,512),nn.ReLU(),nn.Linear(512,200))

    def forward(self,x):return self.layers(F.normalize(x,dim=-1))


def learn_phase(train,val,epochs,seed):
    seed_everything(seed)
    model=PhaseHead().cuda()
    opt=torch.optim.Adam(model.parameters(),lr=1e-4)
    loader=DataLoader(train,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)
    history=[]
    for epoch in range(epochs):
        model.train();total=0.;n=0
        for row in loader:
            logits=model(row['cls'].cuda(non_blocking=True))
            loss=F.cross_entropy(logits,row['phase'].cuda())
            opt.zero_grad(set_to_none=True);loss.backward();opt.step()
            total+=loss.item()*len(logits);n+=len(logits)
        result={'epoch':epoch+1,'train_ce':total/n}
        if val and (epoch==epochs-1 or (epoch+1)%10==0):
            model.eval();losses=[];correct=0;count=0
            with torch.inference_mode():
                for row in DataLoader(val,batch_size=128):
                    logits=model(row['cls'].cuda());y=row['phase'].cuda()
                    losses.append(F.cross_entropy(logits,y,reduction='sum').item())
                    correct+=(logits.argmax(-1)==y).sum().item();count+=len(y)
            result.update(val_ce=sum(losses)/count,val_accuracy=correct/count)
        history.append(result)
    return model.eval(),history


def spherical_kmeans(points,k,iterations=20,seed=0):
    """Independent spatial locations: [positions, observations, features]."""
    points=F.normalize(points.float(),dim=-1)
    p,n,c=points.shape;k=min(k,n)
    gen=torch.Generator(device=points.device).manual_seed(seed)
    index=torch.randperm(n,generator=gen,device=points.device)[:k]
    centers=points[:,index].clone()
    for _ in range(iterations):
        assignment=(points@centers.transpose(1,2)).argmax(-1)
        sums=torch.zeros_like(centers)
        sums.scatter_add_(1,assignment[:,:,None].expand(p,n,c),points)
        counts=torch.zeros(p,k,device=points.device)
        counts.scatter_add_(1,assignment,torch.ones_like(assignment,dtype=torch.float32))
        centers=torch.where((counts>0)[:,:,None],F.normalize(sums,dim=-1),centers)
    return centers


def fit_banks(dataset,bins=20,k=10,max_frames=512,iterations=20,seed=0):
    rng=np.random.default_rng(seed)
    groups=[[] for _ in range(bins)]
    for ri,start in dataset.samples:
        rec=dataset.records[ri]
        phase=min(bins-1,start*bins//rec['frames'])
        groups[phase].append((ri,start+8))
    selected=[];banks=[];group_ids=[]
    mapped=[np.load(r['path']/'patch12.npy',mmap_mode='r') for r in dataset.records]
    for phase,candidates in enumerate(groups):
        if not candidates:continue
        chosen=rng.choice(len(candidates),min(len(candidates),max_frames),replace=False)
        entries=[candidates[i] for i in chosen];selected.extend(entries)
        patches=np.stack([mapped[ri][frame] for ri,frame in entries])
        centers=[]
        for pos in range(0,324,18):
            points=torch.from_numpy(np.array(patches[:,pos:pos+18],copy=True)).cuda().transpose(0,1)
            centers.append(spherical_kmeans(points,k,iterations,seed+phase).cpu())
        bank=torch.cat(centers,dim=0)
        banks.append(bank);group_ids.extend([phase]*bank.shape[1])
        print(json.dumps({'fit_phase':phase,'samples':len(entries),'prototypes_per_position':bank.shape[1]}),flush=True)
    conditioned=torch.cat(banks,dim=1)
    # Unconditioned gets exactly the same observations and number of prototypes.
    total=conditioned.shape[1];unconditional=[]
    for pos in range(0,324,9):
        patches=np.stack([mapped[ri][frame,pos:pos+9] for ri,frame in selected])
        points=torch.from_numpy(patches).cuda().transpose(0,1)
        unconditional.append(spherical_kmeans(points,total,iterations,seed).cpu())
    unconditional=torch.cat(unconditional,dim=0)
    assert unconditional.shape==conditioned.shape
    return {'conditional':conditioned.half(),'unconditional':unconditional.half(),
            'groups':torch.tensor(group_ids),'bins':bins,'samples':len(selected),
            'prototypes_per_position':total,'observations_per_phase_cap':max_frames,
            'kmeans_iterations':iterations,'seed':seed}


def match_prototypes(features,centers,groups=None,phase_bins=None,bins=20,temperature=.1):
    """Cosine NN and normalized convex-projection residual; preserve patch coordinates."""
    features=F.normalize(features.float(),dim=-1)
    centers=F.normalize(centers.float(),dim=-1)
    similarity=torch.einsum('bpc,pkc->bpk',features,centers)
    if groups is not None:
        # Sparse/unseen phases use nearest occupied circular bin, explicitly logged.
        occupied=torch.unique(groups)
        delta=(phase_bins[:,None]-occupied[None,:]).abs()
        closest=occupied[torch.minimum(delta,bins-delta).argmin(-1)]
        allowed=groups[None,:]==closest[:,None]
        similarity=similarity.masked_fill(~allowed[:,None,:],-torch.inf)
    nearest=(1-similarity.max(-1).values).clamp_min(0).mean(-1)
    weights=(similarity/temperature).softmax(-1)
    projection=F.normalize(torch.einsum('bpk,pkc->bpc',weights,centers),dim=-1)
    residual=(1-(projection*features).sum(-1)).clamp_min(0).mean(-1)
    return nearest,residual


def run(scene,output,root='IPAD_dataset',seed=0,epochs=50,bins=20,k=10,max_fit_frames=512,
        kmeans_iterations=20,temperature=.1,diagnostic=False):
    if not (1<=bins<=200 and k>0 and max_fit_frames>0 and kmeans_iterations>0 and epochs>0 and temperature>0):
        raise ValueError('Invalid prototype hyperparameters')
    torch.set_num_threads(8);seed_everything(seed)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    training=FeatureClips(scene,'training',root,diagnostic=diagnostic)
    testing=FeatureClips(scene,'testing',root,diagnostic=diagnostic)
    config=dict(scene=scene,seed=seed,epochs=epochs,bins=bins,k=k,max_fit_frames=max_fit_frames,
                kmeans_iterations=kmeans_iterations,temperature=temperature,diagnostic=diagnostic,
                fallback='nearest occupied circular phase bin',cache_dtype='float16')
    write_json(out/'config.json',config)
    # Development fit is separate; validation clips never build this model.
    if not diagnostic:
        dev=FeatureClips(scene,'training',root,subset='train')
        val=FeatureClips(scene,'training',root,subset='val')
        dev_model,dev_history=learn_phase(dev,val,epochs,seed)
        write_json(out/'phase_validation.json',dev_history)
        del dev_model
    phase_model,history=learn_phase(training,None,epochs,seed)
    write_json(out/'phase_training.json',history)
    memory_file=out/'memory.pt'
    if memory_file.exists():
        saved=torch.load(memory_file,map_location='cpu',weights_only=False)
        if saved['config']!=config:raise ValueError('Existing memory configuration differs')
        banks=saved['banks']
    else:
        banks=fit_banks(training,bins,k,max_fit_frames,kmeans_iterations,seed)
        atomic_checkpoint(memory_file,{'banks':banks,'config':config})
    atomic_checkpoint(out/'phase_head.pt',{'model':phase_model.state_dict(),'config':config})
    conditional=banks['conditional'].cuda();unconditional=banks['unconditional'].cuda();groups=banks['groups'].cuda()
    patches=[np.load(r['path']/'patch12.npy',mmap_mode='r') for r in testing.records]
    rows=[];compute_seconds=0.;begin=time.perf_counter()
    with torch.inference_mode():
        for batch in DataLoader(testing,batch_size=32,num_workers=2,pin_memory=True):
            images=np.stack([patches[int(ri)][int(frame)] for ri,frame in zip(batch['record'],batch['frame'])])
            features=torch.from_numpy(images).cuda()
            torch.cuda.synchronize();tick=time.perf_counter()
            phases=phase_model(batch['cls'].cuda()).argmax(-1)
            phase_bins=torch.div(phases*bins,200,rounding_mode='floor')
            nn_c,soft_c=match_prototypes(features,conditional,groups,phase_bins,bins,temperature)
            nn_u,soft_u=match_prototypes(features,unconditional,temperature=temperature)
            torch.cuda.synchronize();compute_seconds+=time.perf_counter()-tick
            values=torch.stack([phases,nn_c,soft_c,nn_u,soft_u],-1).cpu().numpy()
            for i,video in enumerate(batch['video']):
                phase,cn,cs,un,us=values[i]
                rows.append({'scene':scene,'video':video,'frame':int(batch['frame'][i]),'video_length':int(batch['length'][i]),
                             'phase':int(phase),'conditional_nn':float(cn),'conditional_soft':float(cs),
                             'unconditional_nn':float(un),'unconditional_soft':float(us)})
    keys=('conditional_nn','conditional_soft','unconditional_nn','unconditional_soft')
    period=float(np.median([r['frames'] for r in training.records]))
    scored,summary=score_rows(rows,root,scene,period,feature_keys=keys)
    write_csv(out/'raw_frames.csv',rows);write_csv(out/'scores.csv',scored)
    extraction=sum(r['gpu_seconds'] for r in testing.records)
    summary.update(scene=scene,diagnostic=diagnostic,median_train_period=period,
                   cache_matching_seconds=compute_seconds,matching_wall_seconds=time.perf_counter()-begin,
                   test_feature_extraction_seconds=extraction,
                   offline_extract_then_match_seconds=extraction+compute_seconds,
                   efficiency_note='Offline cache accounting; not measured streaming latency',
                   parameters=sum(p.numel() for p in phase_model.parameters()),
                   memory_prototypes_per_position=banks['prototypes_per_position'],
                   conditional_bytes=banks['conditional'].numel()*banks['conditional'].element_size(),
                   unconditional_bytes=banks['unconditional'].numel()*banks['unconditional'].element_size(),
                   peak_gb=torch.cuda.max_memory_allocated()/1e9,seconds=time.perf_counter()-started)
    write_json(out/'metrics.json',summary)
    if scene=='R02':
        sensitivity={}
        for shift in (-1,0,1):
            _,sensitivity[str(shift)]=score_rows(rows,root,scene,period,'common',shift,keys)
        write_json(out/'alignment_sensitivity.json',sensitivity)
    write_json(out/('diagnostic_completed.json' if diagnostic else 'completed.json'),summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--scene',choices=SCENES,required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--root',default='IPAD_dataset')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--bins',type=int,default=20)
    p.add_argument('--k',type=int,default=10)
    p.add_argument('--max-fit-frames',type=int,default=512)
    p.add_argument('--kmeans-iterations',type=int,default=20)
    p.add_argument('--temperature',type=float,default=.1)
    p.add_argument('--diagnostic',action='store_true')
    run(**vars(p.parse_args()))
