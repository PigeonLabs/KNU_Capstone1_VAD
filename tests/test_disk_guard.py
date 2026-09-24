import importlib.util
from pathlib import Path

spec=importlib.util.spec_from_file_location('disk_guard',Path(__file__).parents[1]/'scripts/disk_guard.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def test_disk_margin_inclusive():
    assert guard.below_reserve(guard.RESERVE)
    assert guard.below_reserve(guard.RESERVE-1)
    assert not guard.below_reserve(guard.RESERVE+1)
