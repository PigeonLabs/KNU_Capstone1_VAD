"""Sequential, disk-guarded execution of approved stage-seven steps."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve
from ipad.stage7 import SCENES,load
from scripts.stage7_summary import summarize


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['7-1','7-2','7-3'],required=True);a=p.parse_args();os.chdir(ROOT);reserve();root=ROOT/'runs/stage7';root.mkdir(exist_ok=True)
    old=load(root/'status.json') if (root/'status.json').exists() else {}
    state={'state':'running','stage':a.stage,'pid':os.getpid(),'started_at':old.get('started_at',time.time()),'stage_started_at':time.time(),'completed':old.get('completed',[]),'automatic_publication_authorized':True}
    def update(**kw):state.update(kw);write_json(root/'status.json',state)
    def run(name,module,args):
        reserve();dest=root/a.stage;dest.mkdir(exist_ok=True);command=[sys.executable,'-m',module,*args];update(step=name,command=command)
        with (dest/(name.replace('/','_')+'.log')).open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        if name not in state['completed']:state['completed'].append(name)
        update()
    try:
        update()
        if a.stage in ['7-1','7-2']:
            for seed in range(3):
                for scene in SCENES:
                    for b,k in [('B',10),('S',5)]:
                        module='ipad.stage7' if a.stage=='7-1' else 'ipad.stage7_alerts'
                        args=['--scene',scene,'--seed',str(seed),'--backbone',b,'--k',str(k)]
                        if a.stage=='7-1':args=['--action','diagnose',*args]
                        run(f'{a.stage}/{scene}/seed{seed}/{b}_k{k}',module,args)
        else:
            with (root/'gpu_before_stream.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True)
            for scene in SCENES:
                for precision in ['fp32','bf16']:
                    run(f'7-3/accuracy/{scene}/{precision}','ipad.stage7_stream',['--action','accuracy','--scene',scene,'--precision',precision])
            for scene in SCENES:
                for variant in ['fp32','bf16','bf16_mixed']:
                    run(f'7-3/benchmark/{scene}/{variant}','ipad.stage7_stream',['--action','benchmark','--scene',scene,'--variant',variant])
        summarize(a.stage);update(last_completed_experiment=a.stage)
        subprocess.run([sys.executable,'scripts/publish_stage.py','--stage',a.stage,'--approved-push','--message',f'{a.stage} 완료: 진단·경보 신뢰성·전체 스트림 검증'],check=True)
        update(state='completed',finished_at=time.time())
    except Exception as e:update(state='failed',error=str(e),finished_at=time.time());raise


if __name__=='__main__':main()
