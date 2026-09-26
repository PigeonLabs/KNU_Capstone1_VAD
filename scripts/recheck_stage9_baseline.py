"""Preserve and repeat baseline potentially affected during compiler autotuning."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve
BASE=ROOT/'runs/stage9/9-1/optimized'


def main():
    status=BASE/'baseline_recheck_status.json'
    assert not status.exists()
    write_json(status,{'state':'waiting_diagnostics','time':time.time()})
    while True:
        reserve();s=json.loads((BASE/'diagnostic_status.json').read_text())
        if s['state'].startswith('completed'):break
        if s['state']=='failed':raise RuntimeError('Inspect diagnostic failure')
        time.sleep(3)
    source=BASE/'full/bf16_autotune_graph';saved=BASE/'restarts/bf16_autotune_graph_potential_autotune_overlap'
    saved.parent.mkdir(parents=True,exist_ok=True);assert not saved.exists();source.rename(saved)
    cache=ROOT/'cache/stage9_optimized/baseline_recheck'
    env=os.environ.copy();env.update(TORCHINDUCTOR_CACHE_DIR=str(cache/'inductor'),TRITON_CACHE_DIR=str(cache/'triton'),TORCHINDUCTOR_COMPILE_THREADS='2',TORCHINDUCTOR_FREEZING='0')
    cmd=[sys.executable,'scripts/run_stage9_optimized.py','--phase','full','--variant','bf16','--mode','autotune_graph']
    with (BASE/'commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':cmd,'cache':str(cache),'previous_result_preserved':str(saved),'reason':'potential GPU test overlap in compiler autotuning'})+'\n')
    log=(BASE/'baseline_recheck.log').open('a');guard=subprocess.Popen([sys.executable,'scripts/disk_guard.py'],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    write_json(status,{'state':'running','command':cmd,'cache':str(cache),'time':time.time()})
    try:
        result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        write_json(status,{'state':'completed' if result.returncode==0 else 'failed','returncode':result.returncode,'time':time.time()})
        if result.returncode:raise RuntimeError('Baseline recheck failed')
    finally:
        if guard.poll() is None:guard.terminate();guard.wait()
        log.close()


if __name__=='__main__':main()
