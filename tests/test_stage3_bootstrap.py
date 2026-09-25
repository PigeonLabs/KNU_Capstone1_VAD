import importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

spec=importlib.util.spec_from_file_location('analysis3',Path(__file__).parents[1]/'scripts/analyze_stage3.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_block_auc_matches_weighted_auc_including_ties_and_single_class_videos():
    y=np.array([0,1,0,1,0,0]);scores=np.array([.2,.8,.5,.5,.1,.3]);v=np.array(['a','a','b','b','c','c'])
    counts=np.array([[1,1,1],[2,1,0],[0,0,3]])
    values=m.draw_auc(m.auc_blocks(y,scores,v),counts)
    for i,c in enumerate(counts[:2]):
        weights=np.array([c[ord(x)-ord('a')] for x in v]);assert abs(values[i]-roc_auc_score(y,scores,sample_weight=weights)*100)<1e-10
    assert np.isnan(values[2])
