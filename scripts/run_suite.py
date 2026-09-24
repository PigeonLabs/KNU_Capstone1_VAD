"""Sequential, restartable research queue. No scheduler/service is installed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ipad.common import SCENES, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['paper','extensions','ablations','all'],default='all')
    p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    os.chdir(ROOT)
    lock=(ROOT/'runs/suite.lock').open('w')
    try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError: raise SystemExit('Another suite is already running')
    status_path=ROOT/'runs/suite_status.json'
    started=time.time()
    completed=[]
    def execute(name,args,marker,force=False):
        if Path(marker).exists() and not force:
            completed.append(name+' (existing)'); return
        cmd=[sys.executable,*args]
        status={'state':'running','pid':os.getpid(),'step':name,'command':cmd,'started_at':started,
                'updated_at':time.time(),'completed':completed,'stage':a.stage,'seed':a.seed}
        write_json(status_path,status)
        logfile=ROOT/'runs'/f'{name}.log'
        print(name,flush=True)
        with logfile.open('a') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONUNBUFFERED':'1',
                    'OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'})
        if proc.returncode or not Path(marker).exists():
            status.update(state='failed',returncode=proc.returncode,log=str(logfile),updated_at=time.time())
            write_json(status_path,status)
            raise SystemExit(f'{name} failed; see {logfile}')
        completed.append(name)
        if name.startswith(('eval_', 'dino_reconstruction_', 'prototype_')):
            subprocess.run([sys.executable,'-m','ipad.report','--seed',str(a.seed)],check=True)
    if a.stage in ['paper','all']:
        for scene in SCENES:
            run=Path(f'runs/paper/{scene}/seed{a.seed}')
            args=['-m','ipad.train','--scene',scene,'--seed',str(a.seed),'--output',str(run)]
            if (run/'last.pt').exists():args+=['--resume']
            execute(f'paper_{scene}_s{a.seed}',args,run/'completed.json')
            execute(f'eval_{scene}_s{a.seed}',['-m','ipad.evaluate','--checkpoint',str(run/'model.pt')],run/'evaluation/metrics.json')
    if a.stage in ['extensions','all']:
        for scene in SCENES:
            run=Path(f'runs/paper/{scene}/seed{a.seed}')
            execute(f'dino_reconstruction_{scene}_s{a.seed}',
                    ['-m','ipad.evaluate','--checkpoint',str(run/'model.pt'),'--dino'],run/'evaluation_dino/metrics.json')
            execute(f'dino_cache_{scene}',['-m','ipad.features','--scene',scene],f'cache/dino/{scene}/complete.json')
            out=f'runs/prototype/{scene}/seed{a.seed}'
            execute(f'prototype_{scene}_s{a.seed}',['-m','ipad.prototype','--scene',scene,'--seed',str(a.seed),'--output',out],f'{out}/completed.json')
    if a.stage in ['ablations','all']:
        for scene in SCENES:
            run=Path(f'runs/no_memory/{scene}/seed{a.seed}')
            args=['-m','ipad.train','--scene',scene,'--seed',str(a.seed),'--output',str(run),'--no-memory']
            if (run/'last.pt').exists():args+=['--resume']
            execute(f'no_memory_{scene}_s{a.seed}',args,run/'completed.json')
            execute(f'eval_no_memory_{scene}_s{a.seed}',['-m','ipad.evaluate','--checkpoint',str(run/'model.pt')],run/'evaluation/metrics.json')
    execute(f'report_s{a.seed}',['-m','ipad.report','--seed',str(a.seed)],f'reports/results_seed{a.seed}.json',force=True)
    write_json(status_path,{'state':'completed','pid':os.getpid(),'started_at':started,'finished_at':time.time(),
               'completed':completed,'stage':a.stage,'seed':a.seed})


if __name__=='__main__':main()
