"""Publish each completed approved ablation unit; no experiment is started here."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCENES=('R01','R02','R03','R04')


def read(path):return json.loads(path.read_text()) if path.exists() else None


def event(kind,**values):
    with (ROOT/'runs/events.jsonl').open('a') as f:
        f.write(json.dumps({'time_unix':time.time(),'event':kind,**values},ensure_ascii=False)+'\n')


def main():
    lock=(ROOT/'runs/publication_watch.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Publication watcher already running')
    ledger_path=ROOT/'runs/publication_ledger.json'
    ledger=read(ledger_path) or {}
    observed=None
    while True:
        if (ROOT/'runs/disk_pause.json').exists():time.sleep(2);continue
        status=read(ROOT/'runs/suite_status.json') or {}
        signature=(status.get('state'),status.get('step'))
        if signature!=observed:
            event('observed_suite_state',status=status);observed=signature
        for scene in SCENES:
            for unit,marker in [('학습','completed.json'),('평가','evaluation/metrics.json')]:
                key=f'{scene}:{unit}'
                if key in ledger or not (ROOT/f'runs/no_memory/{scene}/seed0'/marker).exists():continue
                event('publication_started',scene=scene,unit=unit)
                log=ROOT/f'runs/publish_ablation_{scene}_{"train" if unit=="학습" else "eval"}.log'
                with log.open('a') as f:
                    result=subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','ablation',
                         '--message',f'1단계 추가 검증: {scene} 메모리 제거 {unit} 로그·결과 기록'],
                         cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
                if result.returncode:
                    event('publication_failed',scene=scene,unit=unit,log=str(log.relative_to(ROOT)))
                    raise SystemExit(f'Publication failed; inspect {log}')
                publication=read(ROOT/'runs/publication_status.json')
                ledger[key]={'commit':publication['commit'],'published_at':time.time()}
                temp=ledger_path.with_suffix('.tmp');temp.write_text(json.dumps(ledger,indent=2,ensure_ascii=False)+'\n');temp.replace(ledger_path)
                event('publication_completed',scene=scene,unit=unit,commit=publication['commit'])
        if len(ledger)==8:
            event('all_approved_ablations_published');return
        time.sleep(2)


if __name__=='__main__':main()
