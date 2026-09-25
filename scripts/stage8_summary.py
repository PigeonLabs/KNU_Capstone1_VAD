"""Korean reports with fixed-method, equal-scene and equal-seed summaries."""
from pathlib import Path
import json
import numpy as np
from ipad.stage8 import ROOT,SCENES,METHODS,unit,stream_dir,adapter_dir
from ipad.common import write_json
from ipad.stage7 import write_csv


def load(p): return json.loads(p.read_text())


def summarize(stage):
    out=ROOT/stage;lines=[]
    if stage=='8-1':
        m=load(out/'smoke/completed.json')
        lines=['R01 정상 영상의 구현·학습 진단 완료. 테스트 성능 결과가 아닙니다.','',
            f"추가 학습 파라미터 **{m['trainable_parameters']:,}개**. 초기/복원 최대오차 {m['initial_max_error']:.2g}/{m['restored_max_error']:.2g}, FP32병합 최대오차 {m['merged_max_error']:.3g}. 기존 가중치 불변·gradient 차단·유한 손실 확인.",'',
            f"고정 소수 프레임 진단 loss: 첫5회 {m['first5_loss']:.6f} → 마지막5회 {m['last5_loss']:.6f}. 진단 peak allocated {m['peak_allocated_gib']:.3f} GiB.",'',
            '| 방법 | 표본 | epoch | 첫/마지막 epoch loss | 학습·검증 시간(초) | GPU peak GiB |','|---|---:|---:|---:|---:|---:|']
        times=[]
        for method in METHODS[1:]:
            r=load(adapter_dir('R01',0,method)/'completed.json');times.append(r['seconds'])
            lines.append(f"| {method} | {r['samples']} | 10 | {r['first_loss']:.6f} / {r['last_loss']:.6f} | {r['seconds']:.1f} | {r['peak_allocated_gib']:.3f} |")
        lines+=['','손실 정의가 다르므로 두 방법의 총 loss 크기로 우열을 비교하지 않습니다. 고정10epoch를 사용하며 테스트를 확인한 checkpoint 선택이 없습니다.',
            f"R01 두 실행 평균 {np.mean(times):.1f}초. 다른 장면의 표본 수·특징 재추출·head/메모리 학습·전체 스트림 시간을 포함하지 않는 측정입니다."]
    else:
        seeds=[0] if stage=='8-2' else [0,1,2];precision='fp32_batch32' if stage=='8-2' else 'bf16_batch1';table=[]
        for seed in seeds:
            for scene in SCENES:
                for method in METHODS:
                    p=unit(scene,seed,method) if stage=='8-2' else stream_dir(scene,seed,method);m=load(p/'completed.json');op=m['operation']
                    table.append({'scene':scene,'seed':seed,'method':method,'precision':precision,'auroc':m['metrics']['combined_score']['auroc'],
                        'auprc':m['metrics']['combined_score']['auprc'],'appearance_auroc':m['metrics']['appearance_score']['auroc'],
                        'active_fpr_pct':op['active_alarm_fpr']*100,'recall_pct':op['segment_recall']*100,
                        'false_alarms_per1000':op['false_alarms_per_1000_normal_frames'],'frames':m['common_frames']})
        write_csv(out/'summary.csv',table);aggregate=[]
        for method in METHODS:
            selected=[r for r in table if r['method']==method];metric={}
            for key in ['auroc','auprc','appearance_auroc','active_fpr_pct','recall_pct','false_alarms_per1000']:
                values=[float(np.mean([r[key] for r in selected if r['seed']==seed])) for seed in seeds]
                metric[key]={'mean':float(np.mean(values)),'seed_sd':float(np.std(values,ddof=1)) if len(seeds)>1 else None,'per_seed_macro':values}
            aggregate.append({'method':method,'metrics':metric})
        write_json(out/'aggregate.json',aggregate)
        lines=[f"R01–R04, seed {','.join(map(str,seeds))}, {precision}. 장면별 단순 평균 후 seed 평균±표준편차. 주 후보는 사전 고정 anchored이며 테스트 최상 모델을 고르지 않습니다.",'',
            '| 방법 | AUROC (%) | AUPRC (%) | 외형 AUROC (%) | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame |','|---|---:|---:|---:|---:|---:|---:|']
        for item in aggregate:
            values=[]
            for key in ['auroc','auprc','appearance_auroc','active_fpr_pct','recall_pct','false_alarms_per1000']:
                v=item['metrics'][key];values.append(f"{v['mean']:.2f}"+(f" ± {v['seed_sd']:.2f}" if v['seed_sd'] is not None else ''))
            lines.append('| '+item['method']+' | '+' | '.join(values)+' |')
        base=aggregate[0]['metrics'];anchor=aggregate[2]['metrics'];diff={k:anchor[k]['mean']-base[k]['mean'] for k in base}
        write_json(out/'primary_difference.json',diff)
        lines+=['',f"anchored − frozen: AUROC {diff['auroc']:+.2f}pp, 활성 FPR {diff['active_fpr_pct']:+.2f}pp, 구간 recall {diff['recall_pct']:+.2f}pp.",
            '','정상 오탐시간·발생횟수·미탐을 함께 해석합니다. 구간 recall은 시작 전부터 켜진 경보를 포함하고 지연은 탐지된 구간만 계산하므로 이 수치만으로 운영 신뢰성 향상을 주장하지 않습니다. R02 12/13/14 제외 및 동일 유효 프레임 유지. 이미 관찰한 테스트의 후속 탐색이며 새 환경 검증이 아닙니다.']
        if stage=='8-3':
            lines+=['','| 방법 | capacity FPS (장면 평균) | paced p95 최대(ms) | peak allocated 최대 GiB | 최대 deadline miss (%) |','|---|---:|---:|---:|---:|']
            for method in METHODS:
                bs=[load(stream_dir(scene,0,method)/'benchmark/completed.json') for scene in SCENES]
                lines.append(f"| {method} | {np.mean([r['timings']['capacity']['input_fps'] for r in bs]):.1f} | {max(r['timings']['paced_30fps']['end_to_end_p95_ms'] for r in bs):.2f} | {max(r['peak_allocated_gib'] for r in bs):.3f} | {100*max(r['timings']['paced_30fps']['deadline_miss_fraction'] for r in bs):.3f} |")
            lines+=['','실측: seed0, 단일모델·전체 유효 JPEG capacity1회·각 장면 최장영상 전체30FPS paced1회. 카메라/네트워크 제외, OS 캐시와 다른 프로세스 영향 가능. LoRA 병합은 추가분기를 제거하며 백본 경량화를 뜻하지 않습니다.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    completed=[s for s in ['8-1','8-2','8-3'] if (ROOT/s/'results.md').exists()]
    report=['# 8단계 정상 데이터 기반 DINOv2 LoRA 결과','',
        '고정 DINOv2 대비 정상 적응이 이상 구분 능력과 실시간성을 유지하면서 외형 오탐을 줄이는지 검증합니다. 단계별 완료 여부는 아래 실제 산출물로 구분합니다.','']
    for s in completed: report += [f'## {s}','',(ROOT/s/'results.md').read_text()]
    report+=['## 재현 및 제한','','규약: docs/stage8_protocol.md. `.venv/bin/python scripts/run_stage8.py --stage all`로 실행하며 완료된 단위는 재사용하고 부분 결과는 자동 덮어쓰지 않습니다. 테스트 라벨로 설정을 바꾸지 않았고 정상80%만 adapter/head/bank를 학습합니다. 보정20%로 q99.5 및 3연속 경보를 고정합니다. 증강이 실제 이상 의미를 보존하거나 정상 손실 감소가 이상 구분을 보장한다고 가정하지 않습니다. 원본/모델/특징은 로컬, 공개는 분석·로그·점수·SHA256만입니다.']
    (ROOT/'research_findings.md').write_text('\n'.join(report)+'\n')
