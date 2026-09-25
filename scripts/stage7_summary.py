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
    else:
        raise ValueError(stage)
    write_json(root/'summary.json',result);(root/'results.md').write_text('\n'.join(lines)+'\n')
