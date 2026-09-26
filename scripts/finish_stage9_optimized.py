"""Sequential final diagnostics after the preregistered optimized suite."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve,digest
BASE=ROOT/'runs/stage9/9-1/optimized'


def main():
    state={'state':'waiting_primary','started_at':time.time(),'completed':[],'failed':[]}
    def update(**kw):state.update(kw);write_json(BASE/'diagnostic_status.json',state)
    assert not (BASE/'diagnostic_status.json').exists()
    update()
    while True:
        reserve();primary=json.loads((BASE/'status.json').read_text())
        if primary['state'].startswith('completed'):break
        if primary['state'] in ['failed','paused_low_disk']:raise RuntimeError('Primary requires inspection')
        time.sleep(5)
    guardfile=(BASE/'diagnostic_disk_guard.log').open('a');guard=subprocess.Popen([sys.executable,'scripts/disk_guard.py'],stdout=guardfile,stderr=subprocess.STDOUT,cwd=ROOT)
    try:
        chosen={}
        for v in ['bf16','w8a16','w8a8','w4_native','w4_packed']:
            results=[json.loads(p.read_text()) for p in (BASE/'pilot').glob(f'{v}_*/completed.json')]
            chosen[v]=min(results,key=lambda r:r['timings']['model_ms']['mean'])['mode']
        write_json(BASE/'frozen_fast_diagnostic.json',{'chosen':chosen,'rule':'fastest normal pilot, irrespective of numerical pass; diagnostic only, NOT deployment selection','time':time.time(),'protocol_sha256':digest(ROOT/'docs/stage9_optimization_protocol.md')})
        commands=[]
        for v,m in chosen.items():
            if (BASE/'full'/f'{v}_{m}'/'completed.json').exists():continue
            commands.append((f'full_{v}_{m}',[sys.executable,'scripts/run_stage9_optimized.py','--phase','full','--variant',v,'--mode',m],ROOT/'cache/stage9_optimized/full'/f'{v}_{m}'))
        commands.append(('layer_diagnostic',[sys.executable,'scripts/trace_stage9_precision.py'],ROOT/'cache/stage9_optimized/layer_diagnostic'))
        for name,cmd,cache in commands:
            reserve();update(state='running',step=name)
            env=os.environ.copy();env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_FREEZING='0')
            with (BASE/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':cmd,'cache':str(cache),'scope':'additional full normal diagnostic; numerical failures retained'})+'\n')
            with (BASE/(name+'_diagnostic.log')).open('a') as f:r=subprocess.run(cmd,env=env,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
            (state['completed'] if r.returncode==0 else state['failed']).append(name);update()
        update(state='completed' if not state['failed'] else 'completed_with_failures',finished_at=time.time())
    except Exception as e:update(state='failed',error=str(e));raise
    finally:
        if guard.poll() is None:guard.terminate();guard.wait()
        guardfile.close()


if __name__=='__main__':main()
