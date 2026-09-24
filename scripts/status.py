"""Read-only live status; does not mark any experiment complete."""
import json
import os
from pathlib import Path
import time

root=Path(__file__).resolve().parents[1]
p=root/'runs/suite_status.json'
if not p.exists():raise SystemExit('No suite status yet')
s=json.loads(p.read_text())
try:os.kill(s['pid'],0);alive=True
except ProcessLookupError:alive=False
print(json.dumps({**s,'process_alive':alive},indent=2,ensure_ascii=False))
for progress in sorted((root/'runs').glob('*/R0*/seed*/progress.json')):
    d=json.loads(progress.read_text())
    done=(d['epoch']-1)*d['steps_per_epoch']+d['step']
    total=d['epochs']*d['steps_per_epoch']
    seconds_per_step=d['elapsed_seconds']/done
    print(progress.parent.relative_to(root),f'{done}/{total}',
          f'rough remaining {max(0,total-done)*seconds_per_step/3600:.2f} hours',
          f'last update {time.time()-progress.stat().st_mtime:.0f}s ago')
