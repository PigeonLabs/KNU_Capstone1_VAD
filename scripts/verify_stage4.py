"""Audit all stage 4 outputs and create an integrated evidence report."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json


def read(path):
    with path.open() as f:return list(csv.DictReader(f))


def identity(rows):return [(r['video'],int(r['frame']),int(r['label'])) for r in rows]


def main():
    root=ROOT/'runs/stage4';checks=[];source_hashes=set()
    for scene in ['R01','R02','R03','R04']:
        for seed in range(3):
            a=root/'4-1'/scene/f'seed{seed}';b=root/'4-3'/scene/f'seed{seed}'
            assert (a/'completed.json').exists() and (b/'completed.json').exists()
            rows_a=read(a/'scores.csv');rows_b=read(b/'scores.csv');assert identity(rows_a)==identity(rows_b)
            for ra,rb in zip(rows_a,rows_b):assert float(ra['unconditional_score'])==float(rb['unconditional_score'])
            prior=json.loads((b/'normal_prior.json').read_text());transition=np.array(prior['transition'])
            assert transition.shape==(20,20) and (transition>0).all()
            np.testing.assert_allclose(transition.sum(1),1,atol=1e-12)
            original=read(ROOT/f'runs/stage3/{scene}/seed{seed}/scores.csv')
            bank_sizes=[]
            for k in [1,2,5,10]:
                out=root/'4-2'/scene/f'seed{seed}'/f'k{k}';assert (out/'completed.json').exists()
                assert identity(read(out/'scores.csv'))==identity(original)
                m=json.loads((out/'metrics.json').read_text());bank_sizes.append(m['single_bank_bytes']/k)
                if k==10:assert max(json.loads((out/'baseline_equivalence.json').read_text()).values())<2e-6
                if seed==0:
                    benchmark=json.loads((out/'benchmark.json').read_text())
                    for v in benchmark['timings'].values():
                        assert len(v['cache_seconds'])==10 and len(v['dino_pipeline_seconds'])==3
                        assert min(v['cache_seconds']+v['dino_pipeline_seconds'])>0
                source_hashes.add(json.loads((out/'config.json').read_text())['source_sha256']['ipad/stage4.py'])
            assert len(set(bank_sizes))==1
            checks.append({'scene':scene,'seed':seed,'common_4_1_4_3_frames':len(rows_a),'budget_frames':len(original),
                           'same_baseline_scores':True,'budget_bytes_linear':True,'transition_valid':True})
    assert len(source_hashes)==1
    write_json(root/'final_verification.json',{'passed':True,'checks':checks,'stage4_source_hash':list(source_hashes)[0]})
    s={stage:json.loads((root/stage/'summary.json').read_text()) for stage in ['4-1','4-2','4-3']}
    a=s['4-1']['macro'];c=s['4-3']['macro'];budget=s['4-2']['macro']
    lines=['# 4-1·4-2·4-3 통합 결과','',
           '세 실험은 사전 고정 설정, R01–R04, seed0·1·2에서 완료했다. 모든 결과는 오프라인·기존 테스트셋 탐색 결과이다. 실제 이상 유형별 라벨은 확보하지 않았다.','',
           '## 4-1과 4-3: 같은 평가 프레임의 시간 정보 비교','',
           '| 방법 | 3-seed 평균 AUROC ± 표준편차 |','|---|---:|']
    for name,store,key in [('외형 무조건부',a,'None/unconditional'),('외형 + window5',a,'None/appearance_time5'),
       ('외형 + window21 (4-1 주 후보)',a,'None/appearance_time21'),('외형 + 전이 KL',c,'None/appearance_transition'),
       ('외형 + 체류시간',c,'None/appearance_duration'),('외형 + 전이·체류시간 (4-3 주 후보)',c,'None/appearance_process')]:
        v=store[key];lines.append(f"| {name} | {v['mean']:.2f} ± {v['std']:.2f} |")
    lines+=['','### 장면별 3-seed 평균','', '| 장면 | 외형 기준 | 4-1 외형+시간21 | 4-3 외형+전이·체류 |', '|---|---:|---:|---:|']
    for scene in ['R01','R02','R03','R04']:
        means=[]
        for stage,method in [('4-1','unconditional'),('4-1','appearance_time21'),('4-3','appearance_process')]:
            vals=[r['metrics'][method]['auroc'] for r in s[stage]['scene_results'] if r['scene']==scene]
            means.append(float(np.mean(vals)))
        lines.append('| '+scene+' | '+' | '.join(f'{v:.2f}' for v in means)+' |')
    lines+=['','같은 프레임에서 외형 기준 점수가 완전히 일치함을 확인했다. window21 경계 때문에 3단계 전체 프레임 결과와 직접 차감하지 않는다.',
            f"4-1 주 후보 - 외형 기준선 차이: {a['None/appearance_time21']['mean']-a['None/unconditional']['mean']:+.2f}pp.",
            f"4-3 주 후보 - 외형 기준선 차이: {c['None/appearance_process']['mean']-c['None/unconditional']['mean']:+.2f}pp.",'',
            '각 실험의 seed0 paired video bootstrap 구간과 모든 변형은 개별 results.md에 기록했다. 반복 seed 표준편차는 데이터 모집단 불확실성과 다르다.','',
            '## 4-2: 메모리 용량','',
            '| 위상별 k / full 대비 bank 크기 | 무조건부 NN | hard | top3 가중 |','|---|---:|---:|---:|']
    for k in [1,2,5,10]:
        lines.append(f'| {k} / {k*10}% | '+' | '.join(f"{budget[f'{k}/{m}']['mean']:.2f}" for m in ['unconditional','legacy_hard','top3_weighted'])+' |')
    full=budget['10/unconditional']
    lines+=['','무조건부 메모리 축소에 따른 full 대비 평균 손실: '+', '.join(f"k={k}: {full['mean']-budget[f'{k}/unconditional']['mean']:.2f}pp" for k in [1,2,5])+'.',
            '1pp 기준은 관측된 macro 평균의 실용성 판단이며 통계적 비열등성 검정은 아니다. 장면별·seed별 손실은 개별 metrics.json에서 확인할 수 있다.']
    lines+=['','같은 k에서 조건부·무조건부 bank의 표본과 총개수를 맞췄다. full 대비1pp 손실 기준은 사전에 정한 실용성 기준이며 별도 테스트 최적화 결과를 독립 검증으로 주장하지 않는다.',
            '캐시 매칭과 DINO 포함 속도는 개별 표에서 분리한다. RAM에 디코딩된 입력의 warm 측정으로 디스크/카메라 I/O나 엣지 장치 성능을 포함하지 않는다. 비교 구현 전체 peak GPU 메모리를 단일 방법 배포 메모리처럼 비교하지 않는다.','',
            '## 연구 해석과 한계','',
            '- 위상별 외형 메모리 선택과 시간 진행 검사의 역할을 구분한다. 방법별 최고 장면을 골라 합산하지 않고 동일 변형의 네 장면 평균을 보고한다.',
            '- 장면 테스트 전체 정규화와 미래 프레임을 포함한 clip/window를 사용한다. 온라인 탐지·실제 공장 지연·독립 정상 오탐률을 검증한 결과는 아니다.',
            '- 4-3은 정상 학습 영상에서 모델이 예측한 위상으로 전이/체류 분포를 학습했다. 위상 jitter, 관측 기간, 학습/테스트 confidence 차이가 영향을 줄 수 있다.',
            '- R02 영상12·13·14는 주 결과에서 제외하고 공통 길이 ±1 정렬 민감도를 별도로 보존한다.',
            '- 실험별 고정 규약·코드·프레임 점수·원 정답·환경/모델 해시·실패 로그를 보존한다. 새 연구는 추가 승인 없이 시작하지 않는다.']
    lines+=['','## 캡스톤 논문 구성 제안','',
        '현재 결과에서는 4-1의 외형·주기 진행 점수 결합을 주 방법으로, 4-2의 메모리 절충을 효율 분석으로, 4-3을 추가 복잡성이 개선으로 이어지지 않은 비교 실험으로 배치하는 구성이 적절하다.',
        '주장은 “예측 위상을 외형 메모리 선택에 쓰는 것과 시간 진행 이상 검사에 쓰는 것은 효과가 다르다”로 한정한다. 4-3의 현재 KL·체류시간 모델이 실패한 것이며 모든 전이 모델의 가능성을 부정하는 결과는 아니다.',
        '4-2의 절반 메모리와 4-1의 시간 점수를 함께 적용한 조합은 이번 세 실험에서 직접 검증하지 않았다. 두 결과를 결합해 작은 메모리에서도 80.97%가 달성됐다고 주장하지 않는다.',
        '알고리즘의 최초성·기존 방법 대비 일반화·독립 데이터 성능은 이 실험만으로 입증되지 않는다. 캡스톤 기여는 재현 분석, 위상 활용 방식의 분리 검증, 정확도와 메모리의 절충을 정량화한 것으로 서술한다.']
    lines+=['','## 실행 및 검증 명령','', '```bash', '.venv/bin/python scripts/run_stage4.py', '.venv/bin/python scripts/verify_stage4.py', '```', '', '기존 완료 결과가 있으면 해당 단위는 건너뛴다. 부분 결과가 있으면 검사 없이 덮어쓰거나 자동 재개하지 않는다. 실행기는 사용자 승인된 실험별 main 자동 게시를 포함한다.']
    (root/'research_findings.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'passed':True,'complete_units':72,'same_support_frames':sum(r['common_4_1_4_3_frames'] for r in checks)}))


if __name__=='__main__':main()
