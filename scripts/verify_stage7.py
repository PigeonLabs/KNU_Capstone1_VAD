"""Audit completed diagnostic/alert/stream experiments and produce Korean findings."""
import os
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad import stage5 as s5,stage6 as s6,stage7_alerts as alerts,stage7_stream as stream
from ipad.stage7 import SCENES,load,rows
from ipad.common import write_json


def identity(data):return [(r['video'],r['frame'],r['label']) for r in data]


def equal(left,right):
    assert identity(left)==identity(right)
    np.testing.assert_allclose([r['combined_score'] for r in left],[r['combined_score'] for r in right],rtol=1e-10,atol=1e-10)


def metrics_equal(a,b):
    for score in a:
        for metric in a[score]:
            if isinstance(a[score][metric],(int,float)):
                np.testing.assert_allclose(a[score][metric],b[score][metric],atol=1e-10)
            else:assert a[score][metric]==b[score][metric]


def command_manifest(root):
    records=[]
    for stage,module in [('7-1','ipad.stage7'),('7-2','ipad.stage7_alerts')]:
        for scene in SCENES:
            for seed in range(3):
                for b,k in [('B',10),('S',5)]:
                    config=root/stage/scene/f'seed{seed}'/f'{b}_k{k}'/'config.json'
                    args=(['--action','diagnose'] if stage=='7-1' else [])+['--scene',scene,'--seed',str(seed),'--backbone',b,'--k',str(k)]
                    records.append({'stage':stage,'command':[str(ROOT/'.venv/bin/python'),'-m',module,*args],'config':str(config),'started_at_from_config':load(config)['started_at']})
    for scene in SCENES:
        for precision in ['fp32','bf16']:
            records.append({'stage':'7-3','command':[str(ROOT/'.venv/bin/python'),'-m','ipad.stage7_stream','--action','accuracy','--scene',scene,'--precision',precision],'config':str(root/'7-3/streams'/scene/precision/'config.json')})
        for variant in stream.VARIANTS:
            records.append({'stage':'7-3','command':[str(ROOT/'.venv/bin/python'),'-m','ipad.stage7_stream','--action','benchmark','--scene',scene,'--variant',variant],'config':str(stream.folder(scene,0,variant)/'benchmark/config.json')})
    write_json(root/'command_manifest.json',{'origin':'Reconstructed from fixed runner dispatch and saved per-unit configurations, not an independent syscall trace. Failed attempts remain in7-1/failed_attempts and runner.log.','commands':records})


def main():
    os.chdir(ROOT);s6.reserve();root=Path('runs/stage7');assert load(root/'status.json')['state']=='completed';audit={'passed':True,'diagnostic_units':0,'alert_units':0,'stream_units':0,'benchmark_units':0,'normal_calibration_folds':0}
    for scene in SCENES:
        s6.reserve()
        for seed in range(3):
            for b,k in [('B',10),('S',5)]:
                source=s5.unit_path(b,scene,seed,k);diag=load(root/'7-1'/scene/f'seed{seed}'/f'{b}_k{k}'/'completed.json');old=s6.load(s6.folder(scene,seed,b,'fp32',k)/'completed.json')
                assert diag['components']['combined_score']['operation']==old['operation'];audit['diagnostic_units']+=1
                unit=root/'7-2'/scene/f'seed{seed}'/f'{b}_k{k}';m=load(unit/'completed.json');val=rows(source/'validation_raw.csv');raw=rows(source/'causal_raw.csv');cal=load(unit/'calibration.json');assert cal==s5.calibrate(val)
                provenance=load(unit/'source.json');assert not set(provenance['train_videos'])&set(provenance['calibration_videos']);assert set(cal['calibration_videos'])==set(provenance['calibration_videos']);assert s6.digest(source/'validation_raw.csv')==provenance['validation_sha256']
                cv=load(unit/'normal_cv.json')
                for rule in alerts.RULES:assert cv[rule]==alerts.choose(val,rule)
                for method,info in m['methods'].items():
                    rule=info['rule'];scored=s5.apply_calibration(raw,cal);common=s5.aligned(alerts.transform(scored,rule),scene);saved=rows(unit/method/'scores.csv');equal(saved,common);metrics_equal(info['metrics'],s5.metric_rows(saved))
                    if method=='baseline':h=cal['threshold'];equal(saved,rows(source/'scores.csv'))
                    else:
                        fitted=alerts.transform(s5.apply_calibration(val,cal),rule);h=float(np.quantile([r['combined_score'] for r in fitted],cv[rule]['q']))
                    np.testing.assert_allclose(h,info['threshold'],rtol=1e-10)
                    op,_,_=alerts.operation(s5.aligned(scored,scene,common=False),scene,h,rule);assert op==info['operation']
                audit['alert_units']+=1
            for variant in stream.VARIANTS:
                unit=stream.folder(scene,seed,variant);m=load(unit/'completed.json');val=rows(unit/'validation_raw.csv');raw=rows(unit/'causal_raw.csv');cal=load(unit/'calibration.json');assert cal==s5.calibrate(val)
                cfg=load(s5.unit_path('B',scene,seed,10)/'config.json');assert not set(cfg['train_videos'])&set(cal['calibration_videos']);assert set(cal['calibration_videos'])==set(cfg['calibration_videos'])
                scored=s5.apply_calibration(raw,cal);common=s5.aligned(scored,scene);saved=rows(unit/'scores.csv');equal(saved,common);assert identity(saved)==identity(rows(s5.unit_path('B',scene,seed,10)/'scores.csv'));metrics_equal(m['metrics'],s5.metric_rows(saved))
                assert s6.digest(m['feature_hashes_path'])==m['feature_hashes_sha256']
                expected=sum(r['frames']-35 for r in s5.records('B',scene,'testing'));assert len(raw)==expected
                assert all(0<=r['phase']<200 and np.isfinite(r['appearance']) and np.isfinite(r['temporal']) for r in raw)
                for video in sorted({r['video'] for r in raw}):
                    g=[r for r in raw if r['video']==video];assert [r['frame'] for r in g]==list(range(35,g[0]['video_length']))
                    rebuilt=s5.causal_time([r['phase'] for r in g],m['period']);np.testing.assert_allclose(rebuilt[20:],[r['temporal'] for r in g[20:]],atol=1e-10)
                op,_,_=s5.operation_metrics(s5.aligned(scored,scene,common=False),scene,cal['threshold']);assert op==m['operation']
                folds=load(unit/'normal_calibration_lovo.json');assert folds==stream.sensitivity(val,raw,scene);audit['normal_calibration_folds']+=len(folds)
                audit['stream_units']+=1
                if seed==0:
                    bm=load(unit/'benchmark/completed.json');live=rows(unit/'benchmark/live_scores.csv');assert identity(live)==identity(rows(unit/'online_scores.csv'));metrics_equal(bm['metrics'],s5.metric_rows(s5.aligned(live,scene)))
                    quality=s5.read_csv(unit/'benchmark/stream_accuracy_comparison.csv');np.testing.assert_allclose(bm['phase_agreement'],np.mean([r['phase_equal']=='True' for r in quality]));np.testing.assert_allclose(bm['alarm_agreement'],np.mean([r['alarm_equal']=='True' for r in quality]))
                    timing=s5.read_csv(unit/'benchmark/latency_frames.csv');assert all(float(r['processing_ms'])>0 and float(r['queue_ms'])>=0 for r in timing)
                    capacity=[r for r in timing if r['mode']=='capacity'];assert len(capacity)==len(live)+35*len({r['video'] for r in live})
                    for video in sorted({r['video'] for r in capacity}):
                        g=[r for r in capacity if r['video']==video];assert min(int(r['frame']) for r in g if r['valid_score']=='True')==35
                    assert len({r['video'] for r in timing if r['mode']=='paced_30fps'})==1
                    assert bm['head_dtype']==('torch.bfloat16' if variant=='bf16' else 'torch.float32');assert bm['bank_dtype']==('torch.float32' if variant=='fp32' else 'torch.bfloat16');audit['benchmark_units']+=1
    assert [audit[k] for k in ['diagnostic_units','alert_units','stream_units','benchmark_units']]==[24,24,36,12]
    write_json(root/'final_verification.json',audit);command_manifest(root);report(root);print(audit)


def report(root):
    d=load(root/'7-1/summary.json');a=load(root/'7-2/summary.json');s=load(root/'7-3/summary.json')
    lines=['# 7단계 통합 결과','','R01–R04, seed 0·1·2. 정상 학습80%/보정20% 유지. 같은 테스트셋의 사후 진단과 후속 탐색이며 독립된 새 환경 검증은 아니다.','',
      '## 7-1: 오탐은 주로 어디서 발생했는가','',
      'B/k10 R01에서 외형 단독 활성 FPR62.61%, 시간 단독0.29%, 결합59.83%였다. 각 단독 점수는 자체 정상q99.5 threshold로 평가했다. 정상 오경보 프레임의91.90%가30프레임 이상 지속되는 구간에 속했다. 이 관측은 위상 추정기 재학습보다 지속적인 정상 외형 점수/보정 분포 차이의 점검이 우선임을 시사한다. 분포 차이의 물리적 원인(조명·카메라·공정 변화)은 검토 라벨 없이 확정하지 않았다.',
      '이에 7-2는 모델을 고정한 경보 규칙 비교로 확정했다. 새로운 GRU를 학습하지 않았다. 선택 근거와 고정 수치는 decision.json 및 사전 규약에 기록했다. 경보 후처리만으로 지속적인 정상 분포 차이가 해결된다고 가정하지 않았다.','',
      '## 7-2: 경보가 덜 울리는 것과 좋은 탐지는 다르다','',
      '| 모델 | 방법 | AUROC | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame |','|---|---|---:|---:|---:|---:|']
    for anchor in ['B_k10','S_k5']:
        for method in ['baseline','raw_cv','ewma_cv','hysteresis_cv','ewma_hysteresis_cv']:
            m=a['macro'][anchor+'/'+method];lines.append(f"| {anchor} | {method} | {m['auroc']:.2f} | {m['active_fpr']*100:.2f} | {m['recall']*100:.2f} | {m['false_alarms']:.2f} |")
    lines+=['','주 후보 ewma_hysteresis_cv는 alpha=.2의 과거 점수 평활과 진입/해제 임계값 분리를 사용한다. 규칙 수치는 테스트로 탐색하지 않았고 임계값 분위수는 정상 영상 CV만으로 결정했다.','']
    for anchor in ['B_k10','S_k5']:
        x=a['macro'][anchor+'/baseline'];y=a['macro'][anchor+'/ewma_hysteresis_cv']
        improved=y['active_fpr']<x['active_fpr'] and y['false_alarms']<x['false_alarms'] and y['recall']>=x['recall']
        lines.append(f"- {anchor}: AUROC {y['auroc']-x['auroc']:+.2f}pp, 활성 FPR {(y['active_fpr']-x['active_fpr'])*100:+.2f}pp, recall {(y['recall']-x['recall'])*100:+.2f}pp, 오경보 발생 {y['false_alarms']-x['false_alarms']:+.2f}/정상1000frame. 세 경보 지표의 동시 개선은 {'관측됐다' if improved else '확인되지 않았다'}.")
    lines+=['','오경보 발생 횟수가 줄어도 활성 오탐 시간이 늘거나 탐지율이 낮아질 수 있다. EWMA의 AUROC 상승만으로 운영 경보가 좋아졌다고 주장하지 않는다. 탐지된 구간만의 지연이 짧아지는 현상도 미탐 증가와 함께 해석해야 한다. 모든 CV 제약 실패와 .999 fallback을 공개했다.','',
      '## 7-3: 전체 영상 배치 1 경로의 정확도와 속도','',
      '| 구성 | 전체 batch1 AUROC ± seed SD | 활성 FPR (%) | 구간 recall (%) | 단일모델 capacity FPS 평균 | paced p95 최댓값(ms) | peak allocated 최대 GiB |','|---|---:|---:|---:|---:|---:|---:|']
    for variant in stream.VARIANTS:
        m=s['macro'][variant];lines.append(f"| {variant} | {m['auroc']:.2f} ± {m['std']:.2f} | {m['active_fpr']*100:.2f} | {m['recall']*100:.2f} | {m['fps']:.1f} | {m['paced_p95_ms']:.2f} | {m['memory_gib']:.3f} |")
    lines+=['','BF16 mixed는 백본과 메모리를 BF16으로 두고 원래 FP32 head 가중치와 연산을 유지한 구성이다. 정상 보정도 각각의 batch1 경로에서 다시 적합했다. 정확도 패스는 특징 추출을 공유하므로 그 실행 시간을 FPS로 쓰지 않았다. 속도는 별도 단일 모델 seed0의 전체 유효 테스트 JPEG capacity1회와 장면별 가장 긴 유효 영상 전체30FPS 재생으로 측정했다. 카메라/네트워크는 제외하며 GPU/OS 캐시 및 다른 프로세스 영향이 가능한 현재 장비 결과이다.','',
      '| 구성 | 6단계 batch 기준 AUROC와 차이(pp) | batch와 phase 일치율 최솟값(%) | batch1 정확도·전용 실시간 phase 일치율 최솟값(%) | 경보 일치율 최솟값(%) | 최대 경보 점수 차이 | 최대 기한 초과율(%) |','|---|---:|---:|---:|---:|---:|---:|']
    for variant in stream.VARIANTS:
        m=s['macro'][variant];lines.append(f"| {variant} | {m['auroc']-m['batch_auroc']:+.3f} | {m['batch_phase_agreement_min']*100:.2f} | {m['live_phase_agreement_min']*100:.2f} | {m['live_alarm_agreement_min']*100:.2f} | {m['live_score_difference_max']:.8f} | {m['deadline_miss_fraction']*100:.3f} |")
    lines+=['','mixed의 6단계 기준은 전체 BF16이며 mixed의 배치64와 직접 비교한 값은 아니다. 배치64와 배치1의 일치와, 실제 배치1 평가와 실제 단일모델 스트림의 일치를 구분한다.','',
      '## 정상 보정 집합 구성 민감도','',
      '| 구성 | heldout 정상 FPR: fold 평균 (%) | heldout 정상 FPR: 최악 fold (%) | 동일 장면·seed 내 테스트 활성 FPR 변동폭 평균/최대(pp) |','|---|---:|---:|---|']
    for variant in stream.VARIANTS:
        groups=[load(stream.folder(scene,seed,variant)/'normal_calibration_lovo.json') for scene in SCENES for seed in range(3)]
        rates=[f['heldout_active_fpr'] for g in groups for f in g]
        widths=[(max(f['test_operation']['active_alarm_fpr'] for f in g)-min(f['test_operation']['active_alarm_fpr'] for f in g))*100 for g in groups]
        lines.append(f"| {variant} | {np.mean(rates)*100:.2f} | {max(rates)*100:.2f} | {np.mean(widths):.2f} / {max(widths):.2f} |")
    lines+=['','각 정상 보정 영상을 하나씩 제외해 통계와 임계값을 다시 계산했다. 모델 학습80%는 고정했다. 변동폭은 동일 장면·seed 내부에서 제외한 보정 영상에 따라 달라진 값의 max−min이다. 각 실행의 전체 값은 normal_calibration_lovo.json에 모두 공개했다. 테스트가 가장 좋은 보정 분할을 골라 사용하지 않았다.','',
      '## 연구 해석과 한계','',
      '- 구간 recall은 정답1의 연속 구간 기준이며 시작 전에 이미 활성화된 경보도 탐지로 포함한다. 따라서 R01의 높은 recall은 높은 정상 오탐과 함께 해석한다. 첫35프레임에 시작한 구간은 별도로 기록하고, 탐지 지연 통계는 미탐을 제외한다.',
      '- 현재 높은 오탐의 중요한 관측 원인은 R01의 지속적인 정상 외형 점수 상승이다. 다음 개선은 단순 임계값/평활 계수 추가 탐색보다 정상 데이터의 대표성과 공정·촬영 조건에 대한 원인 검토가 우선이다.',
      '- 이번 경보 규칙은 낮은 발생 횟수, 낮은 활성 오탐, 높은 탐지율을 동시에 달성했다는 근거를 제공하지 못했다. 실시간 처리 속도와 실제 경보 신뢰성은 구분해야 한다.',
      '- 단계7-2의 경보 정책과 단계7-3 정밀도 변환을 결합한 모델을 새로 선택하거나 공동 성능으로 주장하지 않았다.',
      '- R02 길이 불일치 영상12·13·14 제외, 공통frame35..N-18/운영frame35..N-1 및 ±1 민감도를 유지했다. 최초35frame은 warmup이며 미래 프레임을 기다리지 않는다.',
      '- 진단 저장 단계의 빈 결과표/NumPy 정수 JSON 오류 2건과 부분 파일은 failed_attempts에 보존했다. 수정은 저장 경계 조건에 한정되었고 모델·임계값을 변경하지 않았다.',
      '- 전체 테스트와 실제 정상영상 정밀도 검증을 수행했다. 최종 분할·CV·점수·영상경계·전체frame수·실시간 경보 교차검산은 final_verification.json에 기록했다.',
      '- 원본·체크포인트·특징 바이너리는 로컬에 보존하고 코드·규약·로그·프레임 점수·분석·해시만 게시했다. 새 전체 특징 캐시는 만들지 않았다.',
      '- 재실행: `.venv/bin/python scripts/run_stage7.py --stage 7-1`(7-2/7-3 동일). 검산: `.venv/bin/python scripts/verify_stage7.py`. 완료 결과는 재사용하고 부분 실행은 자동 덮어쓰기하지 않는다.']
    (root/'research_findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
