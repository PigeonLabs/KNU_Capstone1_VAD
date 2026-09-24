"""Generate a status report without presenting unfinished experiments as results."""
import argparse
import json
from pathlib import Path

from .common import PAPER_AUC, SCENES, write_json


def read(path):return json.loads(path.read_text()) if path.exists() else None


def build(seed=0):
    records={};table=[]
    for scene in SCENES:
        base=Path(f'runs/paper/{scene}/seed{seed}')
        record={'paper_target':PAPER_AUC[scene],'progress':read(base/'progress.json'),
                'completed':read(base/'completed.json'),'baseline':read(base/'evaluation/metrics.json'),
                'dino_reconstruction':read(base/'evaluation_dino/metrics.json'),
                'prototype':read(Path(f'runs/prototype/{scene}/seed{seed}/metrics.json')),
                'no_memory':read(Path(f'runs/no_memory/{scene}/seed{seed}/evaluation/metrics.json'))}
        records[scene]=record
        for family in ['baseline','dino_reconstruction','prototype','no_memory']:
            result=record[family]
            if result:
                for score,metric in result['metrics'].items():
                    table.append({'scene':scene,'method':family,'score':score,**metric,
                                  'full_label_coverage':result['paper_comparable_label_coverage'],
                                  'gap_to_paper_pp':None if metric['auroc'] is None else metric['auroc']-PAPER_AUC[scene]})
    cuda=read(Path('reports/cuda_check.json'))
    smoke=read(Path('runs/smoke/fp32/smoke.json'))
    result={'seed':seed,'scenes':records,'results':table,'cuda':cuda,
            'smoke_verified':bool(smoke and smoke['loss_decreased'] and smoke['checkpoint_exact']),
            'all_primary_evaluations_complete':all(r['baseline'] is not None for r in records.values()),
            'all_extensions_complete':all(bool(r['dino_reconstruction'] and r['prototype']) for r in records.values()),
            'all_memory_ablations_complete':all(r['no_memory'] is not None for r in records.values())}
    write_json(f'reports/results_seed{seed}.json',result)
    lines=['# IPAD 실제 데이터 재현 — 실행 상태 및 결과','',
           '완료되지 않은 학습·평가를 성능 재현 성공으로 표시하지 않습니다. seed 0 결과를 먼저 진단하고 seed 1·2 반복 여부를 결정합니다.','',
           '## 확인된 실행 가능성','',
           f'- CUDA 검증: {bool(cuda)}',f'- 실제 클립 반복 학습 및 체크포인트 검증: {result["smoke_verified"]}',
           '- 원 논문 기준 설정: 장면별 독립 모델, 16프레임, 256×256, batch 8, Adam 1e-4, 50 epochs, FP32.',
           '- 공식 소스 구조의 파라미터는 263,478,713개입니다. 논문 표의 35.9M과 일치하지 않습니다.','',
           '## 원 방법론 진행','', '| 장면 | 논문 AUROC | 진행 | 재현 AUROC |','|---|---:|---|---:|']
    aucs=[]
    for scene,r in records.items():
        progress=r['progress'];metrics=r['baseline']
        status='대기' if not progress else f'epoch {progress["epoch"]}/50, step {progress["step"]}/{progress["steps_per_epoch"]}'
        value=None
        if metrics:
            value=metrics['metrics']['negative_psnr_with_phase']['auroc'];status='평가 완료'
            if value is not None:aucs.append(value)
            if not metrics['paper_comparable_label_coverage']:status+=' · 라벨 제외 있음'
        display='—' if value is None else f'{value:.2f}'
        lines.append(f'| {scene} | {PAPER_AUC[scene]:.1f} | {status} | {display} |')
    if len(aucs)==4:lines.extend(['',f'4개 장면 평균: **{sum(aucs)/4:.2f}%**. R02 라벨 제외가 있으면 잠정 비교입니다.'])
    lines.extend(['','## 방법별 결과','','| 장면 | 방법 | 점수 | AUROC | AUPRC | 프레임 |',
                  '|---|---|---|---:|---:|---:|'])
    for row in table:
        auc='—' if row['auroc'] is None else f'{row["auroc"]:.2f}'
        ap='—' if row['auprc'] is None else f'{row["auprc"]:.2f}'
        lines.append(f'| {row["scene"]} | {row["method"]} | {row["score"]} | {auc} | {ap} | {row["frames"]} |')
    if not table:lines.append('| — | 전체 학습 후 생성 예정 | — | — | — | — |')
    lines.extend(['','## 데이터 및 해석의 제한','',
                  '- R02 영상 12·13·14: 영상/라벨 길이 불일치. 주 결과에서 제외하며 공통 길이 및 ±1 정렬 민감도 결과를 별도 저장합니다.',
                  '- 점수는 장면의 테스트 전체에서 정규화합니다. 이는 오프라인 논문 비교이며 온라인 임계값 보정 결과가 아닙니다.',
                  '- 길이 16의 중앙 프레임 재구성과 중심 window 5는 미래 프레임을 사용합니다. 실시간 인과 모델이라고 주장하지 않습니다.',
                  '- 유형별 이상 라벨이 없어 외형/시간 이상 원인 분류 성능은 주장하지 않습니다.',
                  '- DINOv2의 사전학습 데이터·표현력과 IPAD의 random initialization은 다릅니다. 특징 비교의 개선을 구조 하나의 인과 효과로 해석하지 않습니다.',
                  '- 모델 전체 재현이 완료되어도 논문 미기재 설정과 공식 소스 불일치는 남으며, 정확히 동일한 구현이라고 단정할 수 없습니다.',''])
    Path(f'reports/results_seed{seed}.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in result.items() if k.startswith('all_') or k=='smoke_verified'}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,default=0)
    build(**vars(p.parse_args()))
