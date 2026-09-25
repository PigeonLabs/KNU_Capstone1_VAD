"""Cross-check complete stage-five results and produce a conservative Korean synthesis."""
import csv
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.stage5 import SCENES,unit_path,calibrate,apply_calibration,causal_time


def load(p):return json.loads(p.read_text())


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def identity(rows):return [(r['video'],int(r['frame']),int(r['label'])) for r in rows]


def main():
    root=ROOT/'runs/stage5';checks=[];sources=set()
    for scene in SCENES:
        for seed in range(3):
            baseline=read(unit_path('B',scene,seed,10)/'scores.csv');target=identity(baseline)
            pools=[]
            for b,k in [('B',10),('B',5),('S',10),('S',5)]:
                out=unit_path(b,scene,seed,k);assert (out/'completed.json').exists();cfg=load(out/'config.json');m=load(out/'metrics.json')
                assert not set(cfg['train_videos'])&set(cfg['calibration_videos']);sources.add(cfg['source_sha256']['ipad/stage5.py'])
                assert identity(read(out/'scores.csv'))==target
                cal=load(out/'calibration.json');v=read(out/'validation_raw.csv')
                for r in v:
                    r['appearance']=float(r['appearance']);r['temporal']=float(r['temporal'])
                rebuilt=calibrate(v)
                np.testing.assert_allclose(rebuilt['threshold'],cal['threshold'],rtol=1e-10)
                assert set(cal['calibration_videos'])==set(cfg['calibration_videos'])
                raw=read(out/'causal_raw.csv')
                for video in sorted({r['video'] for r in raw}):
                    vs=[r for r in raw if r['video']==video];assert int(vs[0]['frame'])==35 and int(vs[-1]['frame'])==int(vs[0]['video_length'])-1
                    recomputed=causal_time([int(r['phase']) for r in vs],m['period'])
                    np.testing.assert_allclose(recomputed[20:],[float(r['temporal']) for r in vs[20:]],atol=1e-10)
                pools.append([(r['video'],int(r['frame'])) for r in read(out/'memory_selection.csv')])
                benchmark=root/'5-3'/scene/f'seed{seed}'/f'{b}_k{k}';assert (benchmark/'completed.json').exists()
                op=load(benchmark/'operation.json');assert op['detected_segments']+op['missed_segments']==op['eligible_segments']
                if seed==0:
                    bm=load(benchmark/'benchmark.json');lat=read(benchmark/'latency_frames.csv')
                    assert {r['mode'] for r in lat}=={'capacity','paced_30fps'}
                    assert all(float(r['processing_ms'])>0 and float(r['queue_ms'])>=0 for r in lat)
                    assert bm['batch_size']==1 and bm['fps_assumption']==30
                checks.append({'scene':scene,'seed':seed,'backbone':b,'k':k,'common_frames':len(target),'normal_split_disjoint':True,'calibration_rebuilt':True,'temporal_recomputed':True})
            assert all(p==pools[0] for p in pools)
            for b in ['B','S']:
                full=load(unit_path(b,scene,seed,10)/'memory.json');half=load(unit_path(b,scene,seed,5)/'memory.json');assert full['bytes']==2*half['bytes']
    report={'passed':True,'model_units':len(checks),'benchmark_seed0_configurations':16,'checks':checks,'source_hashes':sorted(sources),
            'source_note':'S cache metadata timing fields corrected before S extraction; implementation_events.jsonl records the change; model definitions unchanged'}
    write_json(root/'final_verification.json',report)
    sums={s:load(root/s/'summary.json') for s in ['5-1','5-2','5-3']};a=sums['5-1']['macro'];b=sums['5-2']['macro']
    lines=['# 5-1·5-2·5-3 통합 결과','',
      '현재 GPU/단일30FPS/batch1 기준으로 온라인 전환과경량화를 검증했다. 정상80%에서 모델·memory를 학습하고 정상20%에서 보정·임계값을 고정했다. 테스트로 설정을 선택하지 않았다.','',
      '## 5-1: 같은 모델 용량에서 온라인 전환','', '| 방법 | 평균 AUROC ± seed 표준편차 (%) |','|---|---:|']
    for key,label in [('centered_testnorm','중앙 입력·테스트 전체 정규화'),('centered_fixed','중앙 입력·정상 고정 보정'),('causal_appearance','과거 입력·외형 단독'),('causal_combined','과거 입력·외형+시간')]:
        v=a['B_k10/'+key];lines.append(f"| {label} | {v['mean']:.2f} ± {v['std']:.2f} |")
    lines+=['','기존4단계와 달리 정상 검증영상을 학습에서 제외했고 위상 타깃을 예측 프레임에 맞춰 재학습했다. 4단계80.97%와 단순차감하지 않는다.','',
      '## 5-2: 동일 온라인 규칙의2×2 비교','', '| 구성 | 평균 AUROC ± 표준편차 | B/k10 대비 차이(pp) |','|---|---:|---:|']
    base=b['B_k10/causal_combined']['mean']
    for key in ['B_k10','B_k5','S_k10','S_k5']:
        v=b[key+'/causal_combined'];lines.append(f"| {key} | {v['mean']:.2f} ± {v['std']:.2f} | {v['mean']-base:+.2f} |")
    lines+=['','k5는 동일 backbone의k10 대비 prototype 저장량50%. backbone 전체·VRAM50%를 의미하지 않는다.1pp손실 기준은 관측 평균의 실용성 기준이며 비열등성 검정이 아니다.','',
      '## 5-3: 현재GPU의 실제 batch1 평가','', '| 구성 | capacity FPS 장면평균 | paced E2E p95의 장면최댓값(ms) | paced 최대 대기(ms) | peak allocated 최대(GiB) |','|---|---:|---:|---:|---:|']
    for key in ['B_k10','B_k5','S_k10','S_k5']:
        ms=[load(root/'5-3'/scene/'seed0'/key/'benchmark.json') for scene in SCENES]
        fps=np.mean([m['timings']['capacity']['input_fps'] for m in ms]);p95=max(m['timings']['paced_30fps']['end_to_end_p95_ms'] for m in ms);queue=max(m['timings']['paced_30fps']['max_queue_ms'] for m in ms);peak=max(m['peak_allocated_gib'] for m in ms)
        lines.append(f'| {key} | {fps:.1f} | {p95:.2f} | {queue:.2f} | {peak:.3f} |')
    lines+=['','## 고정 임계값의 경보 품질','',
      '| 구성 | 장면·seed 평균 정상 frame FPR (%) | 평균 구간 탐지율 (%) | 평가 구간 합계 | 미탐 구간 | 시작 전부터 경보 중인 구간 |',
      '|---|---:|---:|---:|---:|---:|']
    for key in ['B_k10','B_k5','S_k10','S_k5']:
        ms=[load(root/'5-3'/scene/f'seed{seed}'/key/'operation.json') for seed in range(3) for scene in SCENES]
        v=sums['5-3']['macro'][key]
        lines.append(f"| {key} | {v['mean_scene_seed_frame_fpr']*100:.2f} | {v['mean_segment_recall']*100:.2f} | {sum(m['eligible_segments'] for m in ms)} | {sum(m['missed_segments'] for m in ms)} | {sum(m['preexisting_alarm_segments'] for m in ms)} |")
    lines+=['','구간 합계는 seed 반복을 포함하므로 독립 사건 수가 아니다. 이미 켜진 경보도 구간 탐지로 계산한다. 따라서 오탐이 많은 모델의 높은 탐지율·0프레임 지연을 좋은 경보 성능으로 해석하면 안 된다.',
      '정상 검증 q99.5로 고정한 임계값이 테스트 정상 구간에서도0.5% FPR를 보장하지 않는다. 처리 속도 충족과 실제 경보의 신뢰성은 별개의 결과이다. 테스트 결과를 보고 임계값을 바꾸지 않았다.','',
      '| 장면 (B/k10) | 3-seed 평균 정상 frame FPR (%) | 3-seed 평균 구간 탐지율 (%) |','|---|---:|---:|']
    for scene in SCENES:
        ms=[load(root/'5-3'/scene/f'seed{seed}'/'B_k10'/'operation.json') for seed in range(3)]
        lines.append(f"| {scene} | {np.mean([m['frame_fpr'] for m in ms])*100:.2f} | {np.mean([m['segment_recall'] for m in ms])*100:.2f} |")
    lines+=['','30FPS paced replay는 arrival부터 대기·JPEG read/decode·DINO·head·memory·경보 처리까지 측정했다. OS cache는 warm일 수 있고 카메라·네트워크·다른장비 성능은미포함이다. 최초35프레임은 coldstart이며 이후에는 미래 프레임을 기다리지 않는다.','',
      '전체 테스트에서의 오탐·미탐·지연 결과는 [5-3 상세표](../stage5_3_streaming/results.md)에 있다. 평균 AUROC나 처리 FPS만으로 실시간 공정 경보가 실용적이라고 결론내리지 않는다. 특히 고정 정상 임계값에서의 test 정상 오탐과 segment recall을 함께 판단해야 한다.',
      '정답1 연속구간은 실제 고장 유형/독립사건 라벨이 아니다. 지연은 탐지된 구간에서만 계산하고 미탐과 coldstart 구간을 따로 기록했다. 영상의 실제촬영FPS는 확인되지 않았으므로30FPS 환산은 시나리오이다.','',
      '## 재현 및 한계','', '- R02영상12·13·14 제외. 같은프레임/label 비교 및 ±1 민감도 보존.',
      '- 모든48모델의 분할·보정재계산·시간점수·memory비율·공통frame 검산 통과.3seed 표준편차는 새로운 데이터 일반화를 입증하지 않는다.',
      '- 모델·원본·특징캐시는 로컬보존,분석자료/로그/점수/SHA256만GitHub게시.',
      '- `.venv/bin/python scripts/run_stage5.py`로 순차실행, `.venv/bin/python scripts/verify_stage5.py`로 교차검산.부분결과는 자동덮어쓰기하지 않는다.']
    lines+=['','## 해석 및 다음 우선순위','',
      '5-1에서는 미래 프레임과 테스트 전체 정규화를 제거해도 외형·시간 결합의 AUROC 이점이 유지됐다. 5-2에서는 B 백본의 메모리만 절반으로 줄인 구성이 0.47pp 손실로 사전 1pp 기준 안에 들었다. S 백본은 처리 속도와 GPU 메모리를 개선했으나 평균 손실이 1.17pp 이상으로 그 기준을 넘었다.',
      '현재 장비에서 정확도를 우선하면 B/k10, 프로토타입 저장량 절충은 B/k5가 후보이다. S/k10은 더 큰 정확도 손실을 허용할 때의 후보이며 다른 장비에서의 속도는 추가 측정이 필요하다.',
      '30 FPS 처리 속도와 별개로 고정 경보 임계값의 일반화가 실패했다. R01의 높은 오탐과 R02의 낮은 구간 탐지율을 고려하면 현재 모델을 신뢰할 수 있는 현장 경보 시스템으로 주장할 수 없다. 다음 연구의 우선순위는 추가 경량화보다 정상 보정 분포의 대표성·오탐 제어·미탐 개선이다. 테스트 정답에 맞춘 임계값 조정은 하지 않았다.','',
      '짧은 검산 구간에서 FP16 캐시와 같은 정밀도로 맞춘 단일 프레임 추출의 외형 점수 최대 차이는 전체 구성에서 4.55e-7 미만이었고 위상 argmax가 모두 일치했다. 전체 경보 정확도는 캐시 평가이며 실시간 FP32 전체 테스트의 재평가라고 주장하지 않는다.']
    (root/'research_findings.md').write_text('\n'.join(lines).replace('현재 GPU/단일30FPS/batch1','현재 GPU·단일 30 FPS·배치 1').replace('과경량화','과 경량화').replace('정상80%','정상 80%').replace('정상20%','정상 20%').replace('기존4단계','기존 4단계').replace('4단계80.97%','4단계 80.97%').replace('규칙의2×2','규칙의 2×2').replace('정상 구간에서도0.5%','정상 구간에서도 0.5%').replace('최초35프레임','최초 35프레임').replace('성능은미포함','성능은 미포함').replace('모든48모델','모든 48개 모델')+'\n');print(json.dumps({'passed':True,'model_units':len(checks)}))


if __name__=='__main__':main()
