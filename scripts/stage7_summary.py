"""Evidence tables for stage seven; no test-driven parameter selection."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.stage7 import SCENES,load


def mean(v):
    x=[z for z in v if z is not None];return float(np.mean(x)) if x else None


def summarize(stage):
    root=ROOT/'runs/stage7'/stage;units=[load(f) for f in sorted(root.glob('R*/seed*/*/completed.json'))]
    result={'stage':stage,'units':units};lines=[f'# {stage} 결과','','R01–R04, seed 0·1·2. 기존 테스트셋 사후 진단/추가 탐색이며 독립 일반화 검증이 아니다.','']
    if stage=='7-1':
        assert len(units)==24;macro={}
        lines+=['| 모델 | 점수 | AUROC | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame |','|---|---|---:|---:|---:|---:|']
        for b,k in [('B',10),('S',5)]:
            us=[u for u in units if u['backbone']==b]
            for c in ['appearance_score','temporal_score','combined_score']:
                ms=[u['components'][c] for u in us];r={'auroc':mean([m['metrics']['auroc'] for m in ms]),'active_fpr':mean([m['operation']['active_alarm_fpr'] for m in ms]),'recall':mean([m['operation']['segment_recall'] for m in ms]),'false_alarms':mean([m['operation']['false_alarms_per_1000_normal_frames'] for m in ms])};macro[f'{b}_k{k}/{c}']=r
                lines.append(f"| {b}/k{k} | {c} | {r['auroc']:.2f} | {r['active_fpr']*100:.2f} | {r['recall']*100:.2f} | {r['false_alarms']:.2f} |")
        result['macro']=macro
        lines+=['','| 장면 B/k10 | 외형 FPR (%) | 시간 FPR (%) | 결합 FPR (%) | 30frame 이상 지속하는 오경보 프레임 비율 (%) |','|---|---:|---:|---:|---:|']
        for scene in SCENES:
            us=[u for u in units if u['scene']==scene and u['backbone']=='B'];vals=[mean([u['components'][c]['operation']['active_alarm_fpr'] for u in us])*100 for c in ['appearance_score','temporal_score','combined_score']];total=sum(u['false_alarm_stretches']['total_normal_alarm_frames'] for u in us);long=sum(u['false_alarm_stretches']['normal_alarm_frames_in_stretches_ge30'] for u in us)
            lines.append(f'| {scene} | {vals[0]:.2f} | {vals[1]:.2f} | {vals[2]:.2f} | {100*long/max(total,1):.2f} |')
        lines+=['','외형·시간 각각 정상 보정 q99.5 및 3연속 경보를 사용한다. 각 component의 임계값이 달라 단독 성능 차이를 인과적 기여도라고 단정하지 않는다. 30frame은 길이 구분 기준이며 원본 촬영FPS를 가정하지 않는다.',
          '영상별 정상 보정/테스트 정상/테스트 이상 score 분위수, 정상 상대위상 오차, 모든 오경보 정상 구간 길이, 미탐 구간의 threshold 초과 여부를 보존했다. 테스트 라벨을 이용한 통계는 사후 진단이며 설정 선택용 검증 성능이 아니다.']
    elif stage=='7-2':
        assert len(units)==24;macro={};constraints={}
        lines+=['| anchor | 方法'.replace('方法','방법')+' | AUROC | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame | 탐지 지연 중앙값 평균(frame) |','|---|---|---:|---:|---:|---:|---:|']
        for b,k in [('B',10),('S',5)]:
            us=[u for u in units if u['backbone']==b]
            for method in ['baseline','raw_cv','ewma_cv','hysteresis_cv','ewma_hysteresis_cv']:
                ms=[u['methods'][method] for u in us];seeds=[mean([u['methods'][method]['metrics']['combined_score']['auroc'] for u in us if u['seed']==seed]) for seed in range(3)]
                r={'auroc':mean(seeds),'std':float(np.std(seeds,ddof=1)),'active_fpr':mean([m['operation']['active_alarm_fpr'] for m in ms]),'recall':mean([m['operation']['segment_recall'] for m in ms]),'false_alarms':mean([m['operation']['false_alarms_per_1000_normal_frames'] for m in ms]),'delay':mean([m['operation']['delay_median_frames_detected_only'] for m in ms])};macro[f'{b}_k{k}/{method}']=r
                delay='-' if r['delay'] is None else f"{r['delay']:.2f}"
                lines.append(f"| {b}/k{k} | {method} | {r['auroc']:.2f} | {r['active_fpr']*100:.2f} | {r['recall']*100:.2f} | {r['false_alarms']:.2f} | {delay} |")
            for rule in ['raw','ewma','hysteresis','ewma_hysteresis']:
                cs=[load(root/u['scene']/f"seed{u['seed']}"/f'{b}_k{k}'/'normal_cv.json')[rule] for u in us]
                constraints[f'{b}_k{k}/{rule}']={'met':sum(c['constraint_met'] for c in cs),'total':len(cs),'selected_q':[c['q'] for c in cs]}
        result.update(macro=macro,normal_cv_constraints=constraints)
        lines+=['','주 비교는 ewma_hysteresis_cv 대 baseline. alpha=.2, 해제 임계값=진입의.7배, 진입3연속/해제3연속. raw_cv는 정상 CV로 threshold만 바꾸는 대조군이다. 임계값은 정상 영상 leave-one-out CV로 고르고 모든 후보 결과를 기록했다.','', '| anchor/규칙 | 정상 CV 제약 충족 |','|---|---:|']
        for key,c in constraints.items():lines.append(f"| {key} | {c['met']}/{c['total']} |")
        lines+=['','정상 CV 제약은 평균 활성FPR≤1%, 영상최대≤5%, 평균 오경보≤정상1000frame당1회이다. 미충족 시q=.999 fallback이며 보장으로 표현하지 않는다. EWMA 점수의 AUROC도 표시하지만 경보 횟수 감소가 정확도·미탐 개선을 의미하지 않는다. 탐지 지연은 탐지된 구간에 한정되며 미탐·coldstart·선행경보는 별도 파일에 보존했다.']
    elif stage=='7-3':
        assert len(units)==36;macro={};benchmarks=[]
        lines+=['| 정밀도 | batch1 AUROC ± seed SD | batch 추출 기준선 AUROC | 활성 FPR (%) | 구간 recall (%) |','|---|---:|---:|---:|---:|']
        for variant in ['fp32','bf16','bf16_mixed']:
            us=[u for u in units if u['variant']==variant];seeds=[mean([u['metrics']['combined_score']['auroc'] for u in us if u['seed']==seed]) for seed in range(3)]
            r={'auroc':mean(seeds),'std':float(np.std(seeds,ddof=1)),'batch_auroc':mean([u['batch_metrics']['combined_score']['auroc'] for u in us]),'active_fpr':mean([u['operation']['active_alarm_fpr'] for u in us]),'recall':mean([u['operation']['segment_recall'] for u in us]),'batch_phase_agreement_min':min(u['batch_phase_agreement'] for u in us)};macro[variant]=r
            lines.append(f"| {variant} | {r['auroc']:.2f} ± {r['std']:.2f} | {r['batch_auroc']:.2f} | {r['active_fpr']*100:.2f} | {r['recall']*100:.2f} |")
        lines+=['','| 정밀도 | 전체영상 capacity FPS 평균 | 가장 긴 영상 paced E2E p95 최댓값(ms) | 최대 기한 초과율(%) | peak allocated 최대 GiB | 실제 단일모델 seed0 AUROC |','|---|---:|---:|---:|---:|---:|']
        for variant in ['fp32','bf16','bf16_mixed']:
            bs=[load(root/scene/'seed0'/f'B_k10_{variant}'/'benchmark/completed.json') for scene in SCENES];benchmarks+=bs
            r={'fps':mean([b['timings']['capacity']['input_fps'] for b in bs]),'paced_p95_ms':max(b['timings']['paced_30fps']['end_to_end_p95_ms'] for b in bs),'deadline_miss_fraction':max(b['timings']['paced_30fps']['deadline_miss_fraction'] for b in bs),'memory_gib':max(b['peak_allocated_gib'] for b in bs),'seed0_live_auroc':mean([b['metrics']['combined_score']['auroc'] for b in bs]),'live_phase_agreement_min':min(b['phase_agreement'] for b in bs),'live_score_difference_max':max(b['score_max_difference'] for b in bs),'live_alarm_agreement_min':min(b['alarm_agreement'] for b in bs)};macro[variant].update(r)
            lines.append(f"| {variant} | {r['fps']:.1f} | {r['paced_p95_ms']:.2f} | {r['deadline_miss_fraction']*100:.2f} | {r['memory_gib']:.3f} | {r['seed0_live_auroc']:.2f} |")
        result.update(macro=macro,benchmarks=benchmarks)
        lines+=['','FP32/BF16 모두 정상 보정·전체 테스트를 원본 JPEG에서 batch1로 다시 추출했다. bf16_mixed는 백본·bank BF16에 head만 FP32로 바꾼 대조이다. mixed의 batch 기준선 열은6단계 전체 BF16이며 동일한 mixed batch 실험이 아니다.',
          '정확도 패스의 공유 추출 시간은 속도 측정에서 제외했다. 별도 단일 모델 seed0로 전체 유효 테스트 capacity1회와 장면별 가장 긴 영상 전체30FPS 재생을 측정했다. 실제 카메라/네트워크 지연과 새 환경 일반화는 포함하지 않는다.',
          'normal_calibration_lovo.json에는 보정 영상 하나씩 제외한 모든 통계·임계값·heldout 정상 경보·테스트 민감도를 기록했다. 모델 재학습 분할 검증이 아닌 보정 집합 구성 민감도이다. R02 12·13·14 제외/±1 민감도 유지.']
    else:
        raise ValueError(stage)
    write_json(root/'summary.json',result);(root/'results.md').write_text('\n'.join(lines)+'\n')
