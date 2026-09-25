import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('pub',Path(__file__).parents[1]/'scripts/publish_stage.py')
pub=importlib.util.module_from_spec(spec);spec.loader.exec_module(pub)


def test_push_without_explicit_batch_approval_never_calls_git(monkeypatch,tmp_path):
    monkeypatch.setattr(pub,'ROOT',tmp_path)
    monkeypatch.setattr(pub,'ensure_room',lambda:None)
    monkeypatch.chdir(tmp_path)
    calls=[];monkeypatch.setattr(pub,'git',lambda *a,**kw:calls.append(a))
    with pytest.raises(RuntimeError,match='explicit user approval'):
        pub.publish('stage3','must not push',push=True)
    assert calls==[]
