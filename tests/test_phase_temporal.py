import numpy as np
from ipad.phase_temporal import transforms


def test_time_transforms_map_valid_source_indices_and_preserve_original():
    variants=transforms(200)
    for kind,(index,interval) in variants.items():
        assert index.min()>=0 and index.max()<200
        assert (interval is None)==(kind in ['normal','speed_0.9','speed_1.1'])
    np.testing.assert_equal(variants['normal'][0],np.arange(200))
    assert len(variants['pause'][0])==224
    assert len(variants['skip'][0])==176
    assert (np.diff(variants['reverse'][0])<0).any()
    assert (np.diff(variants['pause'][0])==0).sum()>=24
