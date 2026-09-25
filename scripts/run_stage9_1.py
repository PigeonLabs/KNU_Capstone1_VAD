"""Normal-only quantization execution/compile experiment, 12 fixed conditions."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import fcntl
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve

VARIANTS=['bf16','w8a16','w8a8','w4a16']
MODES=['eager','default','reduce-overhead']


def main():
    os.chdir(ROOT);reserve();out=ROOT/'runs/stage9/9-1';out.mkdir(parents=True,exist_ok=True)
    lock=(out/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'state':'running','stage':'9-1','pid':os.getpid(),'started_at':time.time(),'completed':[],'failed':[]}
    def update(**kw):state.update(kw);write_json(out/'status.json',state)
    try:
        update()
        with (out/'gpu_before.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True)
        for mode in MODES:
            for variant in VARIANTS:
                reserve();name=f'{variant}_{mode}';command=[sys.executable,'scripts/stage9_quant_compile.py','--variant',variant,'--mode',mode]
                env=os.environ.copy();cache=ROOT/'cache/stage9_compile'/name
                env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_FREEZING='0',TORCH_LOGS='recompiles,graph_breaks')
                update(step=name,command=command)
                with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':command,'compile_cache':str(cache)})+'\n')
                with (out/f'{name}.log').open('a') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
                (state['completed'] if p.returncode==0 else state['failed']).append(name);update()
        subprocess.run([sys.executable,'scripts/summarize_stage9_1.py'],check=True)
        update(state='completed' if not state['failed'] else 'completed_with_failed_conditions',finished_at=time.time(),free_gib=shutil.disk_usage(ROOT).free/2**30)
        with (out/'publication.log').open('a') as f:subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','9-1','--approved-push','--message','9-1: 실제 INT8 연산과 BF16·INT8·INT4 컴파일 비교'],stdout=f,stderr=subprocess.STDOUT,check=True)
        update(publication='pushed')
    except Exception as e:
        update(state='paused_low_disk' if (ROOT/'runs/disk_pause.json').exists() or shutil.disk_usage(ROOT).free<=10*2**30 else 'failed',error=str(e));raise


if __name__=='__main__':main()
