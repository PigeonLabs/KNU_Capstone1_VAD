"""Finish approved diagnostics after repeats; no recurring scheduled task."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve
os.chdir(ROOT);root=ROOT/'runs/stage3'
status={'state':'waiting_for_repeats','pid':os.getpid(),'completed':[],'started_at':time.time()}
def update(**kw):status.update(kw);write_json(root/'temporal_status.json',status)
try:
    update()
    while True:
        reserve();s=json.loads((root/'status.json').read_text())
        if s['state']=='failed':raise RuntimeError('Repeat runner failed; do not proceed')
        if s['state']=='completed':break
        time.sleep(3)
    for scene in ['R01','R02','R03','R04']:
        reserve();update(state='running',scene=scene)
        with (root/f'{scene}_temporal.log').open('a') as log:
            subprocess.run([sys.executable,'-m','ipad.phase_temporal','--scene',scene],check=True,stdout=log,stderr=subprocess.STDOUT)
        status['completed'].append(scene);update()
        subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','stage3','--message',f'실험 3C: {scene} 정상 holdout 시간 변형 진단'],check=True)
    subprocess.run([sys.executable,'scripts/analyze_stage3.py'],check=True,stdout=(root/'bootstrap_analysis.log').open('w'))
    subprocess.run([sys.executable,'scripts/stage3_findings.py'],check=True)
    update(state='completed',finished_at=time.time())
    subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','stage3','--message','실험 3단계: 3-seed 비교·영상 bootstrap·시간 진단 최종 분석'],check=True)
except Exception as e:
    update(state='failed',error=str(e));raise
