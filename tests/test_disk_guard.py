import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('disk_guard',Path(__file__).parents[1]/'scripts/disk_guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def test_disk_margin_inclusive():
    assert guard.below_reserve(guard.RESERVE)
    assert guard.below_reserve(guard.RESERVE-1)
    assert not guard.below_reserve(guard.RESERVE+1)


def test_pause_targets_only_supplied_project_processes_and_latches(tmp_path,monkeypatch):
    import json
    import signal
    calls=[]
    class Process:
        def __init__(self,pid,children=()):self.pid=pid;self.descendants=children
        def children(self,recursive=True):return self.descendants
        def create_time(self):return float(self.pid)
        def send_signal(self,sig):calls.append((self.pid,sig))
    child=Process(202);parent=Process(201,[child])
    (tmp_path/'runs').mkdir()
    monkeypatch.setattr(guard,'ROOT',tmp_path)
    monkeypatch.setattr(guard,'experiment_roots',lambda:[parent,child])
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**kw:None)
    guard.pause(guard.RESERVE)
    assert calls==[(201,signal.SIGSTOP),(202,signal.SIGSTOP)]
    event=json.loads((tmp_path/'runs/disk_pause.json').read_text())
    assert event['automatic_resume'] is False and event['state']=='paused_low_disk'
    assert len(event['processes'])==2
