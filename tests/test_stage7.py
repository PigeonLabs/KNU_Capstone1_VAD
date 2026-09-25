import csv
import json
import numpy as np
from ipad.stage7 import spans,write_csv


def test_empty_and_boundary_events(tmp_path):
    assert spans([])==[]
    assert spans([True,True,False,True])==[(0,1),(3,3)]
    assert json.loads(json.dumps(spans([True,True])))==[[0,1]]
    p=tmp_path/'empty.csv';write_csv(p,[],['video','start'])
    with p.open() as f:
        r=csv.DictReader(f);assert r.fieldnames==['video','start'];assert list(r)==[]
