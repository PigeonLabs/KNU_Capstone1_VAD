"""Sequential approved stage-eight suite; no automatic resume of disk pauses."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve,digest
from ipad.stage8 import SCENES,METHODS
from scripts.stage8_summary import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','8-1','8-2','8-3'],default='all');a=p.parse_args();os.chdir(ROOT);reserve();root=ROOT/'runs/stage8';root.mkdir(exist_ok=True)
    lock=(root/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'state':'running','pid':os.getpid(),'started_at':time.time(),'completed_steps':[],'completed_stages':[],'publication_failures':[]}
    def update(**kw): state.update(kw);write_json(root/'status.json',state)
    def run(name,command):
        reserve();update(step=name,command=command)
        with (root/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'name':name,'command':command})+'\n')
        dest=root/state['stage'];dest.mkdir(exist_ok=True)
        with (dest/(name.replace('/','_')+'.log')).open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['completed_steps'].append(name);update()
    def action(name,task,scene='R01',seed=0,method='anchored'):
        run(name,[sys.executable,'-m','ipad.stage8','--action',task,'--scene',scene,'--seed',str(seed),'--method',method])
    try:
        update()
        for stage in (['8-1','8-2','8-3'] if a.stage=='all' else [a.stage]):
            update(stage=stage,stage_started_at=time.time())
            if stage=='8-1':
                action('smoke','smoke')
                for method in METHODS[1:]:action(f'train/R01/seed0/{method}','train',method=method)
                freeze=root/'freeze.json'
                if not freeze.exists():write_json(freeze,{'time':time.time(),'protocol_sha256':digest('docs/stage8_protocol.md'),'settings_frozen_before_test':True,'methods':METHODS,'primary':'anchored','rank':4,'alpha':8,'epochs':10,'batch_size':32,'lr':1e-4,'anchor_weight':1.,'seeds':[0,1,2],'test_used_for_selection':False})
            else:
                for seed in ([0] if stage=='8-2' else [1,2]):
                    for scene in SCENES:
                        for method in METHODS:
                            action(f'train/{scene}/seed{seed}/{method}','train',scene,seed,method)
                            action(f'unit/{scene}/seed{seed}/{method}','unit',scene,seed,method)
                if stage=='8-3':
                    with (root/'gpu_before.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True)
                    for seed in range(3):
                        for scene in SCENES:
                            for method in METHODS:action(f'accuracy/{scene}/seed{seed}/{method}','accuracy',scene,seed,method)
                    for scene in SCENES:
                        for method in METHODS:run(f'benchmark/{scene}/{method}',[sys.executable,'-m','ipad.stage8_stream','--scene',scene,'--method',method])
            run(f'verify_{stage}',[sys.executable,'scripts/verify_stage8.py','--stage',stage]);summarize(stage)
            state['completed_stages'].append(stage);update(last_completed_experiment=stage)
            try:run(f'publish_{stage}',[sys.executable,'scripts/publish_stage.py','--stage',stage,'--approved-push','--message',f'{stage}: 정상 영상 LoRA 적응 실험 및 검증 결과'])
            except subprocess.CalledProcessError as e:
                state['publication_failures'].append({'stage':stage,'error':str(e)});update()
        update(state='completed' if not state['publication_failures'] else 'completed_with_publication_failure',finished_at=time.time(),free_gib=shutil.disk_usage(ROOT).free/2**30)
    except Exception as e:
        update(state='paused_low_disk' if (ROOT/'runs/disk_pause.json').exists() or shutil.disk_usage(ROOT).free<=10*2**30 else 'failed',error=str(e),finished_at=time.time());raise


if __name__=='__main__':main()
