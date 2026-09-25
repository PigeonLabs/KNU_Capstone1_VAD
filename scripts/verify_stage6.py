"""Audit the frozen stage-six experiments and write a conservative synthesis."""
import os
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ipad import stage5 as s5
from ipad import stage6 as s6
from ipad.common import write_json


def identity(data):
    return [(r['video'], r['frame'], r['label']) for r in data]


def equal_scores(left, right):
    assert identity(left) == identity(right)
    np.testing.assert_allclose([r['combined_score'] for r in left],
                               [r['combined_score'] for r in right], rtol=1e-10, atol=1e-10)


def main():
    os.chdir(ROOT)
    root = Path('runs/stage6')
    assert s6.load(root/'status.json')['state'] == 'completed'
    units = []
    benches = []
    for scene in s6.SCENES:
        for seed in range(3):
            target = identity(s6.rows(s5.unit_path('B', scene, seed, 10)/'scores.csv'))
            for b, p, k in s6.VARIANTS:
                out = s6.folder(scene, seed, b, p, k)
                m = s6.load(out/'completed.json')
                source = s5.unit_path(b, scene, seed, 10)
                cfg = s6.load(source/'config.json')
                val = s6.rows(out/'validation_raw.csv')
                assert set(r['video'] for r in val) == set(cfg['calibration_videos'])
                assert not set(cfg['train_videos']) & set(cfg['calibration_videos'])
                cal = s6.load(out/'calibration.json')
                np.testing.assert_allclose(s5.calibrate(val)['threshold'], cal['threshold'], rtol=1e-10)
                scores = s6.rows(out/'scores.csv')
                assert identity(scores) == target
                assert all(np.isfinite(r[c]) for r in scores for c in ['appearance', 'temporal', 'combined_score'])
                raw = s6.rows(out/'causal_raw.csv')
                equal_scores(scores, s5.aligned(s5.apply_calibration(raw, cal), scene))
                for video in sorted({r['video'] for r in raw}):
                    vr = [r for r in raw if r['video'] == video]
                    assert vr[0]['frame'] == 35 and vr[-1]['frame'] == vr[0]['video_length']-1
                    rebuilt = s5.causal_time([r['phase'] for r in vr], m['period'])
                    np.testing.assert_allclose(rebuilt[20:], [r['temporal'] for r in vr[20:]], atol=1e-10)
                op = m['operation']
                assert op['detected_segments'] + op['missed_segments'] == op['eligible_segments']
                if p == 'fp32' and k in (5, 10):
                    equal_scores(scores, s6.rows(s5.unit_path(b, scene, seed, k)/'scores.csv'))
                if p == 'fp32' and k == 2:
                    assert (out/'memory_selection.csv').read_bytes() == (source/'memory_selection.csv').read_bytes()
                if p == 'bf16':
                    assert s6.digest(m['feature_hashes_path']) == m['feature_hashes_sha256']
                    assert s6.load(root/'6-1/streams'/scene/b/'completed.json')['backbone_dtype'] == 'torch.bfloat16'
                units.append({'scene': scene, 'seed': seed, 'variant': s6.key(b,p,k), 'frames': len(scores), 'passed': True})
                if seed == 0:
                    bm = s6.load(out/'benchmark/completed.json')
                    expected = 'torch.float32' if p == 'fp32' else 'torch.bfloat16'
                    assert all(bm[c] == expected for c in ['backbone_dtype','head_dtype','bank_dtype'])
                    lat = s5.read_csv(out/'benchmark/latency_frames.csv')
                    assert {r['mode'] for r in lat} == {'capacity','paced_30fps'}
                    assert all(float(r['processing_ms']) > 0 and float(r['queue_ms']) >= 0 for r in lat)
                    for mode, repeats in [('capacity',3),('paced_30fps',1)]:
                        group = [r for r in lat if r['mode']==mode]
                        assert len(set(r['repeat'] for r in group)) == repeats
                        assert min(int(r['frame']) for r in group if r['valid_score']=='True') == 35
                    benches.append(bm)
    summary = s6.load(root/'6-1/summary.json')
    assert s6.pareto(summary['points']) == summary['pareto']
    assert s6.pareto(summary['points'], .05) == summary['pareto_latency_5pct_ties']
    calibration_checks = []
    for scene in s6.SCENES:
        for seed in range(3):
            for b,k in [('B',10),('S',5)]:
                dest = root/'6-2'/scene/f'seed{seed}'/f'{b}_k{k}'
                source = s5.unit_path(b,scene,seed,k)
                completed = s6.load(dest/'completed.json')
                provenance = s6.load(dest/'normal_source.json')
                assert s6.digest(source/'validation_raw.csv') == provenance['validation_sha256']
                assert not set(provenance['train_videos']) & set(provenance['calibration_videos'])
                validation = s6.rows(source/'validation_raw.csv')
                raw = s6.rows(source/'causal_raw.csv')
                cv = s6.load(dest/'normal_cv.json')
                calibrators = s6.load(dest/'calibrators.json')
                equal_scores(s6.rows(dest/'baseline/scores.csv'), s6.rows(source/'scores.csv'))
                for kind in ['balanced','phase_mean','phase_max']:
                    assert calibrators[kind] == s6.fit_calibrator(validation,kind)
                    assert cv[kind] == s6.choose_normal_quantile(validation,kind)
                    for fold in cv[kind]['folds']:
                        assert fold['heldout_video'] not in fold['fit_videos']
                        assert set(fold['fit_videos']) | {fold['heldout_video']} == set(provenance['calibration_videos'])
                    fitted = s6.apply_calibrator(validation,calibrators[kind])
                    test = s6.apply_calibrator(raw,calibrators[kind])
                    for rule,q in [('fixed',.995),('cv',cv[kind]['q'])]:
                        method = f'{kind}_{rule}'
                        threshold = s6.weighted_quantile([r['combined_score'] for r in fitted],[r['video'] for r in fitted],q)
                        np.testing.assert_allclose(threshold,completed['methods'][method]['threshold'],rtol=1e-10)
                        equal_scores(s6.rows(dest/method/'scores.csv'),s5.aligned(test,scene))
                        op = completed['methods'][method]['operation']
                        assert op['detected_segments'] + op['missed_segments'] == op['eligible_segments']
                calibration_checks.append({'scene':scene,'seed':seed,'anchor':f'{b}_k{k}','all_seven_methods_verified':True})
    audit = {'passed':True,'accuracy_units':len(units),'benchmark_units':len(benches),
             'calibration_anchor_units':len(calibration_checks),'calibration_variants':7,
             'checks':units,'calibration_checks':calibration_checks,
             'minimum_stream_batch_phase_agreement':min(b['phase_agreement'] for b in benches),
             'maximum_stream_batch_appearance_difference':max(b['appearance_max_difference'] for b in benches)}
    write_json(root/'final_verification.json',audit)
    report(root,summary,s6.load(root/'6-2/summary.json'),benches,audit)
    print({k:v for k,v in audit.items() if k not in ['checks','calibration_checks']})


def report(root,eff,cal,benches,audit):
    lines = ['# 6-1·6-2 통합 실험 결과','',
      'R01–R04와 seed 0·1·2를 사용했다. 정상 학습 80%/보정 20%를 분리하고, 과거 프레임만 사용하는 탐지 규칙을 유지했다. 후보·임계값 선택 규칙은 실행 전에 고정했다.','',
      '## 6-1: 정확도·속도·GPU 메모리의 절충','',
      '| 구성 | AUROC ± seed SD (%) | 기준선 차이(pp) | capacity FPS | steady p95 평균(ms) | peak allocated 최대(GiB) | 관측 Pareto |',
      '|---|---:|---:|---:|---:|---:|---|']
    base = next(p for p in eff['points'] if p['variant']=='B_fp32_k10')
    for p in eff['points']:
        lines.append(f"| {p['variant']} | {p['auroc']:.2f} ± {p['std']:.2f} | {p['auroc']-base['auroc']:+.2f} | {p['fps']:.1f} | {p['latency_ms']:.2f} | {p['memory_gib']:.3f} | {'예' if p['variant'] in eff['pareto'] else '-'} |")
    lines += ['', '기준선 대비 평균 AUROC 손실이 1pp 이내인 후보: '+', '.join(p['variant'] for p in eff['points'] if p['auroc'] >= base['auroc']-1)+'. 이는 관측 평균에 적용한 기준이며 비열등성 검정이 아니다.', '',
              '| 구성 | 30 FPS paced E2E p95 최댓값(ms) | 33.3ms 기한 초과 비율 최댓값(%) |',
              '|---|---:|---:|']
    for p in eff['points']:
        lines.append(f"| {p['variant']} | {p['paced_p95_ms']:.2f} | {p['max_deadline_miss_fraction']*100:.2f} |")
    lines += ['', '관측 비지배 집합: '+', '.join(eff['pareto'])+'.',
              '지연 차이 5%를 동률로 본 보조 집합: '+', '.join(eff['pareto_latency_5pct_ties'])+'.',
              '이 집합은 12개 후보와 현재 테스트셋에서 관측한 결과이다. 전역 최적 또는 통계적 우월성을 뜻하지 않는다. 정확도는 3 seed, 속도는 seed 0의 4개 장면을 측정했다.',
              '모든 후보를 이번 실행에서 재측정했다. 현재 GPU, 배치 1, 원본 JPEG 최대 256프레임, capacity 3회와 30 FPS paced 1회이다. 카메라·네트워크 지연은 제외하고 OS 파일 캐시는 따뜻할 수 있다. 최초 35프레임은 점수 없는 초기 구간이다.', '',
              '측정 전 별도 ComfyUI 프로세스가 GPU 메모리 약 550 MiB를 점유한 것을 확인했다. 해당 작업을 종료하지 않았고 본 프로젝트 실험은 순차 실행했다. 전용 GPU를 독점한 측정이라고 주장하지 않는다. 전체 현황은 gpu_before_benchmark.txt에 보존한다.', '',
              '| 정밀도 | stream/batch 위상 일치율 최솟값 | 외형 점수 최대 절대 차이 | paced E2E p95 장면·구성 최댓값(ms) |',
              '|---|---:|---:|---:|']
    for precision in ['fp32','bf16']:
        group = [b for b in benches if b['precision']==precision]
        lines.append(f"| {precision} | {min(b['phase_agreement'] for b in group)*100:.2f}% | {max(b['appearance_max_difference'] for b in group):.8f} | {max(b['timings']['paced_30fps']['end_to_end_p95_ms'] for b in group):.2f} |")
    lines += ['', 'BF16은 백본·위상 예측기·메모리의 실제 BF16 연산이다. RGB/특징 정규화와 거리 누산은 FP32이다. 정상 보정과 테스트 특징을 실제 BF16 백본으로 다시 추출했다. FP32 정확도는 기존 FP16 저장 캐시를 사용한다. 배치 추출과 단일 프레임 추론 사이의 수치 차이를 위에 공개하며, 전체 스트림의 경보 정확도가 동일하다고 단정하지 않는다.', '',
              '## 6-2: 정상 영상 기반 보정의 효과','',
              '주 비교는 사전 지정한 phase_mean_cv 대 baseline이다. B/k10과 S/k5는 6-1 테스트 결과를 보고 고른 것이 아니다. AUROC·FPR·recall은 장면과 seed에 동일한 가중치를 준 평균이며 전체 프레임을 합친 micro 지표가 아니다.', '',
              '| anchor | 방법 | AUROC (%) | 활성 경보 FPR (%) | 구간 recall (%) | 오경보/정상 1000프레임 |',
              '|---|---|---:|---:|---:|---:|']
    for anchor in ['B_k10','S_k5']:
        for method in ['baseline','balanced_fixed','balanced_cv','phase_mean_fixed','phase_mean_cv','phase_max_fixed','phase_max_cv']:
            m = cal['macro'][anchor+'/'+method]
            lines.append(f"| {anchor} | {method} | {m['auroc']:.2f} | {m['active_fpr']*100:.2f} | {m['recall']*100:.2f} | {m['false_alarms']:.2f} |")
    lines += ['', '주 후보의 변화:','']
    for anchor in ['B_k10','S_k5']:
        a = cal['macro'][anchor+'/baseline']; b = cal['macro'][anchor+'/phase_mean_cv']
        simultaneous = b['active_fpr'] < a['active_fpr'] and b['recall'] > a['recall']
        lines.append(f"- {anchor}: AUROC {b['auroc']-a['auroc']:+.2f}pp, 활성 경보 FPR {(b['active_fpr']-a['active_fpr'])*100:+.2f}pp, 구간 recall {(b['recall']-a['recall'])*100:+.2f}pp. 오탐 감소·탐지율 증가의 동시 개선은 {'관측됐다' if simultaneous else '확인되지 않았다'}.")
    lines += ['', '장면별 주 비교(기준선 → phase_mean_cv, 각 3 seed 평균):', '',
              '| anchor | 장면 | AUROC (%) | 활성 FPR (%) | 구간 recall (%) |',
              '|---|---|---:|---:|---:|']
    for b,k in [('B',10),('S',5)]:
        for scene in s6.SCENES:
            us = [u for u in cal['units'] if u['backbone']==b and u['k']==k and u['scene']==scene]
            cells = []
            for field in ['auroc','active_alarm_fpr','segment_recall']:
                vals = []
                for method in ['baseline','phase_mean_cv']:
                    values = [u['methods'][method]['metrics']['combined_score']['auroc'] if field=='auroc' else u['methods'][method]['operation'][field]*100 for u in us]
                    vals.append(float(np.mean(values)))
                cells.append(f'{vals[0]:.2f} → {vals[1]:.2f}')
            lines.append(f"| {b}/k{k} | {scene} | "+' | '.join(cells)+' |')
    lines += ['', '| anchor / 보정 | 정상 CV 제약 충족 |', '|---|---:|']
    for key,c in cal['normal_cv_constraints'].items():
        lines.append(f"| {key} | {c['met']}/{c['total']} |")
    lines += ['', 'CV는 정상 보정 영상 하나를 제외해 검증하고 나머지 영상에서 통계·임계값을 적합한다. 평균 활성 FPR≤1%, 최악 영상≤5% 조건을 만족하는 가장 낮은 분위수를 고른다. 실패하면 q=.999로 대체하되 오탐 보장을 주장하지 않는다. 영상 간 가중치를 같게 하고, 예측 위상별 통계를 전역 통계로 수축했다. 테스트 중 통계·임계값은 고정했다.', '',
      '## 해석 범위와 재현','',
      '- 같은 테스트셋을 앞 단계에서 이미 관찰했다. 이번 결과는 정상 영상 간 보정과 기존 테스트셋에서의 추가 탐색이며, 새 공장·카메라·공정에 대한 일반화 입증이 아니다.',
      '- 정상 영상 CV는 경험적 검증이다. 프레임 상관성과 소수 영상 때문에 분포 무관 보장이나 conformal coverage로 표현하지 않는다.',
      '- 구간 recall은 정답 1의 연속 구간 기준이다. 시작 전에 활성화된 경보도 포함하므로 높은 오탐과 함께 판단한다. 지연은 탐지 구간만 계산하며 미탐·초기 구간·선행 경보는 상세 파일에 별도 기록했다.',
      '- 6-1의 경량화 후보와 6-2 보정을 결합한 모델은 이번에 공동 평가하지 않았다. 각각의 개선을 더해 새 모델 성능이라고 주장하지 않는다.',
      '- R02의 길이 불일치 영상 12·13·14는 주 결과에서 제외하고 공통 길이 및 ±1프레임 민감도를 보존했다.',
      f"- {audit['accuracy_units']}개 정확도 실행, {audit['benchmark_units']}개 속도 실행, {audit['calibration_anchor_units']}개 anchor × 7개 보정 변형의 분할·점수·CV·임계값·프레임 정렬 검산을 통과했다.",
      '- 실행: `.venv/bin/python scripts/run_stage6.py`. 최종 검산: `.venv/bin/python scripts/verify_stage6.py`. 부분 결과는 자동으로 덮어쓰지 않는다.',
      '- 원본·체크포인트·특징 바이너리는 로컬에 보존했다. 코드·규약·환경·명령·정상 CV 전 후보·프레임 점수·분석·해시만 GitHub에 게시한다.']
    (root/'research_findings.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
