import numpy as np
import torch
from ipad.stage5 import causal_time,calibrate,apply_calibration,alarm_flags,label_segments,Head,selected_frames


def test_causal_prefix_future_independence_and_period_wrap():
    phases=np.arange(100)*2%200
    a=causal_time(phases,100)
    assert np.isnan(a[:20]).all() and np.allclose(a[20:],0)
    b=phases.copy();b[70:]=13
    np.testing.assert_array_equal(a[:70],causal_time(b,100)[:70])


def test_fixed_calibration_no_test_dependence():
    rows=[{'video':'00','appearance':float(i),'temporal':float(2*i)} for i in range(100)]
    cal=calibrate(rows);before=dict(cal)
    scores=apply_calibration([{'appearance':1e6,'temporal':1e7}],cal)
    assert cal==before and scores[0]['combined_score']>1
    assert cal['calibration_videos']==['00']


def test_alarm_consecutive_reset_and_single_emission():
    active,emit=alarm_flags([0,2,2,2,2,0,2,2,2],1)
    assert active.tolist()==[False,False,False,True,True,False,False,False,True]
    assert np.flatnonzero(emit).tolist()==[3,8]
    assert not alarm_flags([2,2],1)[0].any()


def test_segments_include_edges_and_single_class():
    assert label_segments([1,1,0,1])==[(0,1),(3,3)]
    assert label_segments([0,0])==[]
    assert label_segments([1,1])==[(0,1)]


def test_backbone_pool_independence_and_frame_targets():
    a=[{'video':'0','frames':100,'path':'B'},{'video':'1','frames':120,'path':'B'}]
    b=[dict(r,path='S') for r in a]
    pool,n=selected_frames(a,0)
    assert (pool,n)==selected_frames(b,0)
    assert all(15<=t<a[ri]['frames']-7 for ri,t in pool)


def test_head_channel_variants_batch_independence():
    for c in [384,768]:
        model=Head(c).eval();x=torch.randn(2,16,c)
        with torch.no_grad():a=model(x);b=model(x[:1])
        assert a.shape==(2,200)
        torch.testing.assert_close(a[:1],b)
