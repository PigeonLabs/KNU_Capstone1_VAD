import numpy as np
import torch
from ipad.stage6 import weighted_quantile,fit_calibrator,apply_calibrator,choose_normal_quantile,pareto,match


def normal_rows():
    return [{'video':str(v),'frame':t,'phase':(t*3)%200,'appearance':.1+(t%17)*.001,'temporal':float(t%5)} for v in range(3) for t in range(140)]


def test_equal_video_weight_ignores_video_duplication():
    a=[0,1,2,10,11,12];v=['a']*3+['b']*3
    for q in [.1,.5,.8,.995]:
        assert weighted_quantile(a,v,q)==weighted_quantile(a[:3]*5+a[3:],['a']*15+['b']*3,q)


def test_phase_calibration_sparse_fallback_and_no_future_dependence():
    data=normal_rows();cal=fit_calibrator(data,'phase_mean');test=data[:25];prefix=apply_calibrator(test,cal)
    extra=[{**r,'appearance':1e5,'temporal':1e6} for r in data[25:]]
    assert prefix==apply_calibrator(test+extra,cal)[:25]
    sparse=fit_calibrator(data[:10],'phase_mean')
    assert all(v['weight']==0 for v in sparse['phase'].values())


def test_normal_video_cv_is_disjoint_and_uses_normal_constraints():
    result=choose_normal_quantile(normal_rows(),'phase_mean')
    assert all(r['heldout_video'] not in r['fit_videos'] for r in result['folds'])
    allowed=[r['q'] for r in result['grid'] if r['constraint_met']]
    assert result['q']==(min(allowed) if allowed else .999)
    assert result['constraint_met']==bool(allowed)


def test_pareto_dominance_ties_and_timing_tolerance():
    points=[{'variant':'a','auroc':80,'latency_ms':4.,'memory_gib':1.},
            {'variant':'b','auroc':79,'latency_ms':5.,'memory_gib':1.},
            {'variant':'c','auroc':81,'latency_ms':4.1,'memory_gib':1.}]
    assert pareto(points)==['a','c']
    assert pareto(points,.05)==['c']
    assert pareto([points[0],{**points[0],'variant':'tie'}])==['a','tie']


def test_float_matching_agrees_with_reference():
    from ipad.stage5 import match as reference
    torch.manual_seed(0);x=torch.randn(3,4,8);bank=torch.nn.functional.normalize(torch.randn(4,5,8),dim=-1)
    torch.testing.assert_close(match(x,bank),reference(x,bank))


def test_bf16_match_accumulates_to_fp32_on_gpu():
    if not torch.cuda.is_available():return
    torch.manual_seed(0);x=torch.randn(3,4,8,device='cuda');bank=torch.nn.functional.normalize(torch.randn(4,5,8,device='cuda'),dim=-1).bfloat16()
    result=match(x,bank);assert result.dtype==torch.float32 and torch.isfinite(result).all()
    a=torch.nn.functional.normalize(x.float(),dim=-1).bfloat16().float()
    expected=(1-torch.einsum('bpc,pkc->bpk',a,bank.float()).max(-1).values).clamp_min(0).mean(-1)
    torch.testing.assert_close(result,expected,atol=2e-6,rtol=2e-6)
