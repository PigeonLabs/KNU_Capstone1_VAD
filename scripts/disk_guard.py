"""Pause only this project's experiment processes at the user's disk reserve."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import psutil

ROOT=Path(__file__).resolve().parents[1]
RESERVE=10*1024**3  # conservative 10 GiB, slightly more than decimal 10 GB


def below_reserve(free,reserve=RESERVE):return free<=reserve


def experiment_roots():
    matches=[]
    for p in psutil.process_iter(['pid','create_time']):
        if p.pid==os.getpid():continue
        try:
            if Path(p.cwd())!=ROOT:continue
            args=p.cmdline()
            modules={'ipad.train','ipad.features','ipad.prototype','ipad.evaluate','ipad.phase_routing','ipad.phase_temporal'}
            scripts={'scripts/run_suite.py','scripts/publish_stage.py','scripts/run_stage3.py'}
            if any(a in modules for a in args) or any(a in scripts or a.endswith('/'+a2) for a in args for a2 in scripts):
                matches.append(p)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return matches


def pause(free):
    stopped={}
    # Stop parents first so they cannot start another stage while children are stopped.
    for p in experiment_roots():
        try:
            children=p.children(recursive=True)
            for process in [p,*children]:
                if process.pid in stopped:continue
                started=process.create_time()
                process.send_signal(signal.SIGSTOP)
                stopped[process.pid]={'pid':process.pid,'create_time':started}
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    event={'state':'paused_low_disk','time_unix':time.time(),'free_bytes':free,
           'reserve_bytes':RESERVE,'automatic_resume':False,'processes':list(stopped.values()),
           'reason':'User requested a 10 GB disk safety margin; experiments paused without deleting files.'}
    path=ROOT/'runs/disk_pause.json'
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(event,indent=2)+'\n');temp.replace(path)
    subprocess.run(['notify-send','-u','critical','IPAD 실험 일시중지',
                    f'디스크 여유 {free/1024**3:.2f} GiB: 10 GiB 안전마진에 도달했습니다. 자동 재개하지 않습니다.'],check=False)
    print(json.dumps(event),flush=True)
    return event


def main():
    p=argparse.ArgumentParser();p.add_argument('--once',action='store_true');a=p.parse_args()
    (ROOT/'runs').mkdir(exist_ok=True)
    lock=(ROOT/'runs/disk_guard.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Disk guard already running')
    last=0
    while True:
        free=shutil.disk_usage(ROOT).free
        if (ROOT/'runs/disk_pause.json').exists():
            # Latched until explicit user-directed resume; never SIGCONT automatically.
            if a.once:break
            time.sleep(1);continue
        if below_reserve(free):pause(free)
        if time.time()-last>30 or a.once:
            (ROOT/'runs/disk_guard_status.json').write_text(json.dumps(
                {'pid':os.getpid(),'free_bytes':free,'reserve_bytes':RESERVE,'checked_at':time.time()})+'\n')
            last=time.time()
        if a.once:break
        time.sleep(1)


if __name__=='__main__':main()
