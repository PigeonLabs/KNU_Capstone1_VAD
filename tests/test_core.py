import numpy as np
import torch
from PIL import Image

from ipad.data import Clips, validation_videos
from ipad.metrics import binary_metrics, normalize, period_errors
from ipad.model import PeriodMemory
from ipad.evaluate import score_rows


def test_memory_batch_independence_and_gradients():
    torch.manual_seed(1)
    for axis in ['tokens','memory']:
        m=PeriodMemory(slots=20,channels=4,classes=5,axis=axis)
        x=torch.randn(2,4,2,2,2,requires_grad=True)
        logits=torch.tensor([[1.,0,0,0,0],[0.,0,0,0,4]],requires_grad=True)
        full,entropy=m(x,logits)
        for i in range(2):
            single,_=m(x[i:i+1],logits[i:i+1])
            torch.testing.assert_close(full[i:i+1],single)
        (full.square().mean()+entropy).backward()
        assert torch.isfinite(x.grad).all() and torch.isfinite(logits.grad).all()
        assert logits.grad.abs().sum()>0


def test_memory_matches_paper_axis_and_slot_endpoint():
    m=PeriodMemory(slots=10,channels=2,classes=4)
    x=torch.tensor([[[[[1.,2.]]],[[[3.,4.]]]]])
    logits=torch.tensor([[0.,0.,0.,10.]])
    f=x.flatten(2).transpose(1,2)
    a=f@m.weight.T
    selected=3*10//4
    mask=torch.ones(10); mask[selected]+=logits.softmax(-1).max()
    expected=(a*mask).softmax(1)@m.weight
    output,_=m(x,logits)
    torch.testing.assert_close(output.flatten(2).transpose(1,2),expected)
    assert selected==7  # floor, not round; never out of range


def test_period_error_uses_normal_reference_and_wrap():
    e=period_errors([198,199,0,1,2,3,4],200)
    assert np.isnan(e[:2]).all() and np.isnan(e[-2:]).all()
    np.testing.assert_allclose(e[2:-2],0)
    stopped=period_errors([25]*7,200)
    assert (stopped[2:-2]>0).all()


def test_metrics_direction_and_constant_scores():
    assert binary_metrics([0,0,1,1],[0,.2,.8,1])['auroc']==100
    assert binary_metrics([0,0],[.1,.8])['auroc'] is None
    np.testing.assert_equal(normalize([3,3]),[0,0])
    assert binary_metrics([0,1],[0,0])['auroc']==50


def make_data(tmp_path):
    root=tmp_path/'dataset'
    for split in ['training','testing']:
        for vid in ['01','02']:
            folder=root/'R01'/split/'frames'/vid;folder.mkdir(parents=True)
            for i in range(30): Image.fromarray(np.full((8,8,3),i,dtype=np.uint8)).save(folder/f'{i:03d}.jpg')
    labels=root/'R01'/'test_label';labels.mkdir()
    np.save(labels/'001.npy',np.array([0]*15+[1]*15))
    np.save(labels/'002.npy',np.zeros(29))
    return root


def test_loader_support_phase_and_no_test_oracle(tmp_path):
    root=make_data(tmp_path)
    ds=Clips(root,'R01',cache=None)
    assert len(ds)==30
    assert ds[14]['video']=='01' and ds[14]['frame']==22
    assert ds[15]['video']=='02' and ds[15]['start']==0
    assert ds[14]['phase']==14*200//30
    assert len(validation_videos(root,'R01'))==1
    test=Clips(root,'R01','testing',cache=None)
    assert len(test)==15 and 'phase' not in test[0] and 'label' not in test[0]
    assert len(Clips(root,'R01','testing',cache=None,label_policy='common'))==30


def test_scene_normalization_alignment_and_sensitivity(tmp_path):
    root=make_data(tmp_path)
    rows=[{'video':v,'frame':s+8,'video_length':30,'phase':s,'mse':float(s+1)} for v in ['01','02'] for s in range(15)]
    result,summary=score_rows(rows,root,'R01',200)
    assert summary['excluded_videos']==['02']
    assert summary['frames']==11
    assert result[0]['frame']==10 and result[-1]['frame']==20
    assert not summary['paper_comparable_label_coverage']
    _,sensitivity=score_rows(rows,root,'R01',200,'common',1)
    assert sensitivity['frames']==22


def test_partial_predictions_are_not_full_coverage(tmp_path):
    root=make_data(tmp_path)
    rows=[{'video':'01','frame':s+8,'video_length':30,'phase':s,'mse':float(s+1)} for s in range(15)]
    _,summary=score_rows(rows,root,'R01',200)
    assert summary['missing_videos']==['02']
    assert not summary['paper_comparable_label_coverage']
    assert 'negative_psnr' not in rows[0]


def test_duplicate_frames_fail_instead_of_corrupting_window(tmp_path):
    import pytest
    root=make_data(tmp_path)
    rows=[{'video':'01','frame':8,'video_length':30,'phase':0,'mse':1.}]*5
    with pytest.raises(ValueError,match='Duplicate/noncontiguous'):
        score_rows(rows,root,'R01',200)
