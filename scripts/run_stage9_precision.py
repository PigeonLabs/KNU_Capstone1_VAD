"""Same BF16-rounding control for all four paths, after primary suite finishes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve
from scripts.run_stage9_1 import VARIANTS


def main():
    os.chdir(ROOT);out=ROOT/'runs/stage9/9-1';mode='reduce-overhead-precise';state={'state':'waiting_primary','pid':os.getpid(),'mode':mode,'completed':[],'failed':[],'started_at':time.time()}
    def update(**kw):state.update(kw);write_json(out/'precision_control_status.json',state)
    try:
        update()
        while True:
            reserve();old=json.loads((out/'status.json').read_text())
            if old['state'].startswith('completed') and old.get('publication')=='pushed':break
            if old['state'] in ['failed','paused_low_disk']:raise RuntimeError('Primary suite requires inspection')
            time.sleep(5)
        update(state='running')
        for variant in VARIANTS:
            reserve();name=f'{variant}_{mode}';command=[sys.executable,'scripts/stage9_quant_compile.py','--variant',variant,'--mode',mode]
            env=os.environ.copy();cache=ROOT/'cache/stage9_compile'/name
            env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_FREEZING='0',TORCH_LOGS='recompiles,graph_breaks')
            update(step=name,command=command)
            with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':command,'compile_cache':str(cache),'additional_normal_only_control':True})+'\n')
            with (out/f'{name}.log').open('a') as f:p=subprocess.run(command,env=env,stdout=f,stderr=subprocess.STDOUT)
            (state['completed'] if p.returncode==0 else state['failed']).append(name);update()
        subprocess.run([sys.executable,'scripts/summarize_stage9_1.py'],check=True)
        update(state='completed' if not state['failed'] else 'completed_with_failed_conditions',finished_at=time.time())
        with (out/'precision_publication.log').open('a') as f:subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','9-1','--approved-push','--message','9-1: BF16 중간 반올림 보존 대조군 및 최종 수치·속도 검증'],stdout=f,stderr=subprocess.STDOUT,check=True)
        update(publication='pushed')
    except Exception as e:update(state='failed',error=str(e));raise


if __name__=='__main__':main()
