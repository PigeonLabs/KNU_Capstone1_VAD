"""Write a Korean evidence summary from completed experiments only."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    root=ROOT/'runs/stage3';b=json.loads((root/'bootstrap.json').read_text())
    names={'unconditional':'무조건부 NN','legacy_hard':'기존 단일 위상 NN','posterior_hard':'합산 확률 단일 위상 NN',
           'top3_weighted':'상위 3개 위상 가중 거리(주 후보)','top3_nn':'상위 3개 위상 NN',
           'neighbor_nn':'인접 위상 NN','confidence_fallback':'신뢰도 fallback','conditional_all':'조건부 bank 전체 NN'}
    lines=['# 3단계 연구 결과 해석','',
           'R01–R04, seed 0·1·2에서 동일한 사전 고정 설정으로 비교했다. 2단계 이후 관측한 데이터에 대한 탐색 연구이며 독립 외부 검증이 아니다.',
           '원본 영상·라벨·특징 캐시와 1·2단계 결과는 변경하지 않았다.','',
           '## 실제 테스트 결과','',
           '| 방법 | seed 0 평균 | seed 1 평균 | seed 2 평균 | 3-seed 평균 ± 표준편차 |',
           '|---|---:|---:|---:|---:|']
    for k,name in names.items():
        vals=[b['seed_summary'].get(str(s),{}).get(k) for s in range(3)]
        cells=[f'{v:.2f}' if v is not None else '미완료' for v in vals]
        total=f'{np.mean(vals):.2f} ± {np.std(vals,ddof=1):.2f}' if all(v is not None for v in vals) else '미완료'
        lines.append('| '+name+' | '+' | '.join(cells)+' | '+total+' |')
    lines+=['','평균은 각 seed의 4개 장면 AUROC 단순 평균이다. 표준편차는 학습/클러스터링 seed 변동이며 데이터 모집단 불확실성과 다르다.',
            '모든 변형·AUPRC·장면별 결과는 results.md와 장면별 metrics.json에 보존한다. R02 12·13·14 영상은 주 평가에서 제외했다.','',
            '## 영상 단위 불확실성','',
            'seed 0에서 2,000회 paired video-cluster bootstrap을 수행했다. 각 영상의 프레임을 함께 재표집한다. 양성/음성이 모두 존재하는 표본만 사용하며 4개 장면 모두 유효한 표본으로 macro 차이 구간을 계산했다.',
            '| 비교 | AUROC 차이 95% 구간(pp) | 유효 표본 |','|---|---:|---:|']
    for k,ci in b['macro_delta_ci95'].items():lines.append(f"| {k} | [{ci[0]:.2f}, {ci[1]:.2f}] | {b['macro_valid_draws'][k]} |")
    lines+=['','이 구간은 고정된 모델/테스트셋의 영상 표본 변동을 나타낸다. 여러 비교의 유의성 보정이나 독립 외부 검증을 대체하지 않는다.','',
            '## 위상 조건 해석','',
            '- 기존 hard와 무조건부 seed 0 점수는 2단계 원 점수와 최대 오차 2e-6 이내로 일치했다. 모든 seed/변형의 평가 video/frame/label ID는 2단계와 동일하다.',
            '- 정상 holdout은 80% 학습 전용 모델/메모리로 평가했다. fine 200-class 정확도를 실제 routing의 20-bin 정확도와 구분했다.',
            '- 무조건부 메모리는 정상 후보를 더 많이 비교하므로 정상 거리 자체가 작다는 사실만으로 더 정확한 위상 또는 더 좋은 이상탐지라고 결론내리지 않는다.',
            '- conditional 전체 bank와 무조건부 bank의 차이는 routing뿐 아니라 위상별/전체 클러스터링 구성의 차이도 포함한다.',
            '- R01 development bank의 19번 bin은 비어 있다. clip 시작 위치 정의상 끝쪽 위상 표본이 적고, 기존 hard는 원형 최근접 점유 bin을 사용한다.',
            '- 인접/무작위 대조는 프레임마다 후보 bin 수가 같다. 현재 각 점유 bin은 위치당 10개 prototype이어서 후보 prototype 수도 같다. random3와 top3도 후보 수가 같다.',
            '- 신뢰도 cutoff는 정상 개발 모델에서 정한 후 전체 학습 모델로 옮겼다. 재학습으로 confidence 분포가 변할 수 있어 fallback은 탐색적 결과이다.',
            '- validation_raw.csv의 confidence_fallback은 cutoff를 정하기 전의 비활성 기준(무조건부)이다. 실제 테스트 fallback은 calibration.json의 고정 cutoff를 사용한다.','']
    if all(str(s) in b['seed_summary'] and 'top3_weighted' in b['seed_summary'][str(s)] for s in range(3)):
        delta=np.mean([b['seed_summary'][str(s)]['top3_weighted']-b['seed_summary'][str(s)]['unconditional'] for s in range(3)])
        lines+= [f'사전 지정 주 후보의 무조건부 대비 3-seed 평균 차이는 **{delta:+.2f}pp**다. '+
                 ('현재 결과는 위상 routing 완화만으로 무조건부 기준선을 개선했다는 주장을 지지하지 않는다.' if delta<=0 else '평균 개선만으로 통계적/실용적 우월성이 확정되지는 않는다.'),'']
    temporal={}
    for scene in ['R01','R02','R03','R04']:
        p=root/scene/'seed0/temporal/completed.json'
        if p.exists():temporal[scene]=json.loads(p.read_text())
    lines+=['## 정상 holdout 시간 변형 진단','',
            '정지·역순·건너뛰기는 정상 영상 특징의 시간 index를 변형한 파생 진단이다. 실제 이상 유형별 정답이 아니며 기업 데이터에서의 유형 분류 성능을 의미하지 않는다.',
            '| 장면/변형 | 외형 무조건부 AUC | top3 외형 AUC | 시간 window5 AUC | 시간 window21 AUC | 외형+시간 AUC |',
            '|---|---:|---:|---:|---:|---:|']
    tkeys=['unconditional','top3_weighted','time5','time21','appearance_time_max']
    for scene,d in temporal.items():
        for kind in ['pause','reverse','skip']:
            vals=[d['variants'][kind][k]['auroc'] for k in tkeys]
            lines.append('| '+scene+'/'+kind+' | '+' | '.join(f'{v:.2f}' if v is not None else 'N/A' for v in vals)+' |')
    lines+=['','- 동일 원본 프레임 대조에서 무조건부 외형 점수는 시간 순서를 바꿔도 수치 오차 내 동일하다. 변형 구간에서 외형 AUROC가 높더라도 중앙 구간 선택·반복에 따른 분포 효과일 수 있으므로 시간 이상 탐지 증거로 해석하지 않는다. paired_source_frame_audit.json에 점수 변화 검증을 보존한다.',
            '- ±10% 속도 변형은 허용 가능한 변동이라는 가정 아래 별도 알림 비율을 기록했다. 실제 공정 정상 범위는 기업 멘토 검토가 필요하다.',
            '- 임계치는 변형 전 정상 holdout의 p95이다. 같은 정상 holdout을 이용한 임계치 보정/오탐 비율이므로 독립 일반화 오탐률로 주장하지 않는다.',
            '- 미래 프레임을 쓰는 clip/window 및 테스트 전체 정규화를 사용했다. 실제 온라인 지연, 원인 분류 정확도, 공정 현장 실시간성을 입증한 결과가 아니다.',
            '- 원시 index manifest와 진단 프레임 점수, AUROC/AUPRC, p95 양성 재현율 및 알림 비율은 각 temporal/ 폴더에 보존한다.','',
            '## 연구 방향','',
            '무조건부 DINOv2 메모리를 강한 외형 기준선으로 유지한다. 위상 조건부가 항상 유리하다는 전제를 피하고, 실제 시간 이상 유형 라벨을 확보해 시간적 제약의 독립적 기여를 검증하는 방향이 적절하다. 새로운 학습 방법이나 추가 연구 단계는 이 보고서에서 자동 실행하지 않는다.']
    (root/'research_findings.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
