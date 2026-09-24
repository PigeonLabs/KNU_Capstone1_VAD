"""Detach the approved experiment queue while preserving the active virtualenv."""
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
(root/'runs').mkdir(exist_ok=True)
with (root/'runs/suite_console.log').open('a') as log:
    process=subprocess.Popen([sys.executable,str(root/'scripts/run_suite.py'),*sys.argv[1:]],
            cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
            start_new_session=True,close_fds=True,
            env={**os.environ,'PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8'})
time.sleep(2)
if process.poll() is not None:
    raise SystemExit(f'Launch exited with {process.returncode}; inspect runs/suite_console.log')
print(f'Experiment runner PID {process.pid}. Status: .venv/bin/python scripts/status.py')
