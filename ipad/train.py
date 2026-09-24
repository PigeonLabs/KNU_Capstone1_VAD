import argparse
import json
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, RandomSampler

from .common import SCENES, atomic_checkpoint, environment, prepare_clip, seed_everything, write_json
from .data import Clips
from .model import IPAD, objective


def parser():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='IPAD_dataset')
    p.add_argument('--scene', choices=SCENES, default='R01')
    p.add_argument('--output', default='runs/paper/R01/seed0')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--subset', choices=['train','all'], default='all')
    p.add_argument('--axis', choices=['tokens','memory'], default='tokens')
    p.add_argument('--shrink', type=float, default=0.)
    p.add_argument('--radius', type=int, default=0)
    p.add_argument('--no-memory', action='store_true')
    p.add_argument('--precision', choices=['fp32','bf16'], default='fp32')
    p.add_argument('--checkpoint-activations', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--steps-per-epoch', type=int, default=0, help='Diagnostic only; marks run as incomplete protocol')
    return p


def run(a):
    seed_everything(a.seed)
    torch.set_num_threads(8)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'config.json').exists() and not a.resume:
        raise FileExistsError(f'Run exists: {out}; use --resume or another output')
    ds = Clips(a.root, a.scene, subset=a.subset)
    period = ds.median_period
    if a.smoke_steps:
        chosen = np.linspace(0,len(ds)-1,a.batch_size,dtype=int).tolist()
        ds = Subset(ds, chosen)
    generator = torch.Generator().manual_seed(a.seed)
    # Preserve the original first-epoch order, but decouple worker startup from sampling.
    torch.empty((),dtype=torch.int64).random_(generator=generator)
    worker_generator=torch.Generator().manual_seed(a.seed)
    loader = DataLoader(ds, batch_size=a.batch_size, sampler=RandomSampler(ds,generator=generator), num_workers=a.workers,
                        drop_last=True, pin_memory=True, persistent_workers=a.workers>0, generator=worker_generator)
    device = torch.device('cuda')
    model = IPAD(memory=not a.no_memory, axis=a.axis, shrink=a.shrink, radius=a.radius,
                 checkpoint=a.checkpoint_activations).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    if a.resume:
        state = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
        previous = state['config']
        for key in ['scene','seed','subset','axis','shrink','radius','no_memory','precision','batch_size','steps_per_epoch','smoke_steps']:
            if previous[key] != vars(a)[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']
        torch.set_rng_state(state['torch_rng'])
        torch.cuda.set_rng_state_all(state['cuda_rng'])
        generator.set_state(state.get('sampler_rng',state['loader_rng']))
        if 'sampler_rng' in state:
            worker_generator.set_state(state['loader_rng'])
        del state
    history_path=out/'history.jsonl'
    previous_rows=[json.loads(line) for line in history_path.read_text().splitlines()] if a.resume and history_path.exists() else []
    committed_rows=[row for row in previous_rows if row['epoch']<=start_epoch]
    if a.resume and previous_rows!=committed_rows:
        history_path.write_text(''.join(json.dumps(row)+'\n' for row in committed_rows))
    previous_seconds=sum(row['seconds'] for row in committed_rows)
    source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('ipad').rglob('*.py')}
    config = {**vars(a), 'median_train_period': period, 'model_options': model.options,
              'environment': environment(), 'parameters': sum(p.numel() for p in model.parameters()),
              'complete_protocol': not bool(a.smoke_steps or a.steps_per_epoch) and a.subset=='all' and a.epochs==50,
              'data_clips': len(ds), 'source_sha256': source_hashes}
    write_json(out/'config.json', config)
    print(json.dumps(config), flush=True)
    model.train()
    torch.cuda.reset_peak_memory_stats()
    wall_start = time.perf_counter()
    losses = []
    smoke_batch = next(iter(loader)) if a.smoke_steps else None
    final_epoch = start_epoch
    for epoch in range(start_epoch, 1 if a.smoke_steps else a.epochs):
        totals = dict(loss=0., reconstruction=0., period=0., entropy=0.)
        n = 0
        iterator = [smoke_batch]*a.smoke_steps if a.smoke_steps else loader
        epoch_start = time.perf_counter()
        for step,batch in enumerate(iterator):
            if a.steps_per_epoch and step>=a.steps_per_epoch:
                break
            x = prepare_clip(batch['clip'], device)
            phase = batch['phase'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.precision=='bf16'):
                output = model(x)
                loss, parts = objective(output,x,phase)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            values = {'loss': loss.item(), **{k:v.item() for k,v in parts.items()}}
            for k,v in values.items(): totals[k]+=v
            n+=1
            if a.smoke_steps: losses.append(values)
            with (out/'batch_history.jsonl').open('a') as f:
                f.write(json.dumps({'time_unix':time.time(),'epoch':epoch+1,'step':step+1,
                     'videos':list(batch['video']),'frames':batch['frame'].tolist(),
                     'gradient_norm':grad_norm.item(),'peak_gb':torch.cuda.max_memory_allocated()/1e9,**values})+'\n')
            if step%25==0 or a.smoke_steps:
                progress = {'scene':a.scene,'epoch':epoch+1,'epochs':a.epochs,'step':step+1,
                            'steps_per_epoch':len(loader),'elapsed_seconds':previous_seconds+time.perf_counter()-wall_start,
                            'gradient_norm':grad_norm.item(), 'peak_gb':torch.cuda.max_memory_allocated()/1e9,**values}
                write_json(out/'progress.json',progress)
                print(json.dumps(progress),flush=True)
            del output, loss, parts, x
        final_epoch = epoch+1
        row = {'epoch':final_epoch,'steps':n,'seconds':time.perf_counter()-epoch_start,
               **{k:v/n for k,v in totals.items()}}
        with (out/'history.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if final_epoch%5==0 or final_epoch==a.epochs or a.smoke_steps:
            atomic_checkpoint(out/'last.pt', {'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                              'epoch':final_epoch,'config':config,'torch_rng':torch.get_rng_state(),
                              'cuda_rng':torch.cuda.get_rng_state_all(),'sampler_rng':generator.get_state(),
                              'loader_rng':worker_generator.get_state(),'rng_layout':'split_sampler_worker_v1'})
    atomic_checkpoint(out/'model.pt',{'model':model.state_dict(),'config':config,'epoch':final_epoch})
    if a.smoke_steps:
        model.eval()
        x=prepare_clip(smoke_batch['clip'],device)
        with torch.no_grad():
            before=model(x)['phase_logits'].cpu()
        state=torch.load(out/'model.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['model'])
        del state
        with torch.no_grad(): after=model(x)['phase_logits'].cpu()
        if not torch.equal(before,after): raise AssertionError('Checkpoint roundtrip changed inference')
        write_json(out/'smoke.json',{'losses':losses,'checkpoint_exact':True,
                   'loss_decreased':losses[-1]['reconstruction']<losses[0]['reconstruction'],
                   'peak_gb':torch.cuda.max_memory_allocated()/1e9,'environment':environment()})
    write_json(out/'completed.json',{'epochs':final_epoch,'seconds':previous_seconds+time.perf_counter()-wall_start,
               'peak_gb':torch.cuda.max_memory_allocated()/1e9,'complete_protocol':config['complete_protocol']})


if __name__=='__main__':
    run(parser().parse_args())
