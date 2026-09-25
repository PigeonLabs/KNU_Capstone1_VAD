"""Run preregistered efficiency and calibration experiments sequentially."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve
from ipad.stage6 import SCENES,VARIANTS,key,folder,load,pareto


def mean(values):
    vals=[v for v in values if v is not None];return float(np.mean(vals)) if vals else None


def fmt(x,digits=2):return '-' if x is None else f'{x:.{digits}f}'


def summarize(stage):
    root=ROOT/'runs/stage6'/stage;result={'stage':stage,'complete':True};lines=[f'# {stage} 실험 결과','',
      'R01–R04, seed 0·1·2. 정상 80% 학습 / 20% 보정 분리와 causal frame 규약 유지. 기존 테스트셋에서 추가 탐색한 결과이며 독립 데이터 일반화 증거가 아니다.','']
    if stage=='6-1':
        points=[]
        for b,p,k in VARIANTS:
            units=[load(folder(scene,seed,b,p,k)/'metrics.json') for seed in range(3) for scene in SCENES]
            seeds=[mean([u['metrics']['combined_score']['auroc'] for u in units if u['seed']==seed]) for seed in range(3)]
            benches=[load(folder(scene,0,b,p,k)/'benchmark/completed.json') for scene in SCENES]
            points.append({'variant':key(b,p,k),'auroc':mean(seeds),'std':float(np.std(seeds,ddof=1)),'seed_aurocs':seeds,
              'latency_ms':mean([v['timings']['capacity']['steady_p95_ms'] for v in benches]),
              'memory_gib':max(v['peak_allocated_gib'] for v in benches),'fps':mean([v['timings']['capacity']['input_fps'] for v in benches]),
              'paced_p95_ms':max(v['timings']['paced_30fps']['end_to_end_p95_ms'] for v in benches),
              'max_deadline_miss_fraction':max(v['timings']['paced_30fps']['deadline_miss_fraction'] for v in benches),
              'bank_runtime_bytes_mean':mean([u['bank_runtime_bytes'] for u in units]),'metrics':units,'benchmarks':benches})
        frontier=pareto(points);tolerant=pareto(points,.05);baseline=next(p['auroc'] for p in points if p['variant']=='B_fp32_k10')
        result.update(points=points,pareto=frontier,pareto_latency_5pct_ties=tolerant)
        lines+=['| 구성 | 평균 AUROC ± seed SD | 기준선 차이 pp | capacity FPS | steady p95 평균 ms | peak allocated 최대 GiB | 관측 Pareto |', '|---|---:|---:|---:|---:|---:|---|']
        for p in points:lines.append(f"| {p['variant']} | {p['auroc']:.2f} ± {p['std']:.2f} | {p['auroc']-baseline:+.2f} | {p['fps']:.1f} | {p['latency_ms']:.2f} | {p['memory_gib']:.3f} | {'예' if p['variant'] in frontier else '-'} |")
        lines+=['','정확도 최대 / steady-state 처리 p95 최소 / peak allocated VRAM 최소의 비지배 집합이다.12개 후보 내 관측 결과이며 전역 최적·통계적으로 확정된 우월성이 아니다.',
          '지연 차이5%를 동률로 보는 보조 집합: '+', '.join(tolerant)+'.',
          'FPS는 이번 실행에서 모든 후보를 재측정했다. 현재 GPU,원본 JPEG read/decode,batch1,capacity3회/30FPS paced1회,최대256frames. 카메라/네트워크 제외,OS cache가 warm일 수 있다. 지연과 memory는 seed0,정확도는3seed이다.',
          'BF16은 실제 backbone/head/bank dtype 변환과 재추출 평가이며 RGB/feature normalization/거리 누산은FP32이다. 새 전체 feature cache를 저장하지 않았고 영상별 추출 해시와 raw score를 보존했다.',
          'FP32 k10/k5 기준선 점수 등가성, k2 정상 sample pool 동등성,각 precision의 정상 보정,stream/batched 추출 오차는 실행 폴더에 있다.']
    else:
        macro={};methods=['baseline','balanced_fixed','balanced_cv','phase_mean_fixed','phase_mean_cv','phase_max_fixed','phase_max_cv'];all_units=[]
        lines+=['| anchor | 보정·임계값 | AUROC | active 경보 FPR (%) | 구간 recall (%) | 오경보 / 정상1000frame | 탐지 구간 지연 중앙값 평균(frame) |', '|---|---|---:|---:|---:|---:|---:|']
        for b,k in [('B',10),('S',5)]:
            units=[load(root/scene/f'seed{seed}'/f'{b}_k{k}'/'completed.json') for seed in range(3) for scene in SCENES];all_units+=units
            for method in methods:
                ms=[u['methods'][method] for u in units];seeds=[mean([u['methods'][method]['metrics']['combined_score']['auroc'] for u in units if u['seed']==seed]) for seed in range(3)]
                row={'auroc':mean(seeds),'std':float(np.std(seeds,ddof=1)),'seed_aurocs':seeds,'active_fpr':mean([m['operation']['active_alarm_fpr'] for m in ms]),
                     'recall':mean([m['operation']['segment_recall'] for m in ms]),'false_alarms':mean([m['operation']['false_alarms_per_1000_normal_frames'] for m in ms]),
                     'delay_detected_only':mean([m['operation']['delay_median_frames_detected_only'] for m in ms]),
                     'missed_segments_sum_seeds':sum(m['operation']['missed_segments'] for m in ms)}
                macro[f'{b}_k{k}/{method}']=row
                lines.append(f"| {b}/k{k} | {method} | {row['auroc']:.2f} | {row['active_fpr']*100:.2f} | {row['recall']*100:.2f} | {row['false_alarms']:.2f} | {fmt(row['delay_detected_only'])} |")
        result.update(macro=macro,units=all_units);cv_counts={}
        for b,k in [('B',10),('S',5)]:
            for kind in ['balanced','phase_mean','phase_max']:
                cs=[load(root/scene/f'seed{seed}'/f'{b}_k{k}'/'normal_cv.json')[kind] for seed in range(3) for scene in SCENES]
                cv_counts[f'{b}_k{k}/{kind}']={'met':sum(c['constraint_met'] for c in cs),'total':len(cs),'selected_q':[c['q'] for c in cs]}
        result['normal_cv_constraints']=cv_counts
        lines+=['','## 正常CV'.replace('正常','정상 '),'','| anchor / 보정 | CV 오탐 제약 충족 실행 |','|---|---:|']
        for name,c in cv_counts.items():lines.append(f"| {name} | {c['met']}/{c['total']} |")
        lines+=['','CV는 정상 validation 영상을 하나씩 제외하고 나머지로 보정한다. 평균active FPR≤1%,최대영상FPR≤5%를 만족하는 가장 낮은q를 선택했다. 실패한 실행은q=.999 fallback이며 오탐 보장을 주장하지 않는다.',
          '주 비교는 phase_mean_cv 대 baseline. fixed는 영상 균형 q99.5,cv는 정상 영상만으로 고른q. 영상/seed별 결과·CV 분할과 전q 후보·보정 통계·정답/경보 frame·미탐/경보선행/coldstart 구간은 보존했다.',
          '탐지율은 정답1 연속구간 기준이며 이미 활성화된 경보도 포함한다. 지연은 탐지된 구간만의 값이고 미탐은 별도로 기록한다. 같은 테스트셋을 반복 관찰했으므로 새로운 환경 일반화가 입증된 것은 아니다.']
    lines+=['','R02영상12·13·14는 주 결과에서 제외하고±1정렬 민감도를 보존했다. 모든 원본·기존 결과를 유지하고 바이너리는 로컬에 보존한다.']
    write_json(root/'summary.json',result);(root/'results.md').write_text('\n'.join(lines)+'\n')


def main():
    os.chdir(ROOT);reserve();root=ROOT/'runs/stage6';root.mkdir(exist_ok=True);state={'state':'running','pid':os.getpid(),'started_at':time.time(),'completed':[],'automatic_publication_authorized':True}
    def update(**kw):state.update(kw);write_json(root/'status.json',state)
    def run(stage,name,args):
        reserve();dest=root/stage;dest.mkdir(exist_ok=True);command=[sys.executable,'-m','ipad.stage6',*args];update(step=name,command=command)
        with (dest/(name.replace('/','_')+'.log')).open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(name);update()
    try:
        update()
        for seed in range(3):
            for scene in SCENES:
                for b in ['B','S']:
                    for k in [10,5,2]:run('6-1',f'fp32/{scene}/seed{seed}/{b}_k{k}',['--action','fp32','--scene',scene,'--seed',str(seed),'--backbone',b,'--k',str(k)])
        for scene in SCENES:
            for b in ['B','S']:run('6-1',f'bf16/{scene}/{b}',['--action','bf16','--scene',scene,'--backbone',b])
        with (root/'gpu_before_benchmark.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True)
        for scene in SCENES:
            for b,p,k in VARIANTS:run('6-1',f'benchmark/{scene}/{key(b,p,k)}',['--action','benchmark','--scene',scene,'--backbone',b,'--precision',p,'--k',str(k)])
        summarize('6-1');update(last_completed_experiment='6-1')
        subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','6-1','--approved-push','--message','6-1 완료: 실제 BF16·메모리 예산·관측 Pareto'],check=True)
        for seed in range(3):
            for scene in SCENES:
                for b,k in [('B',10),('S',5)]:run('6-2',f'calibration/{scene}/seed{seed}/{b}_k{k}',['--action','calibration','--scene',scene,'--seed',str(seed),'--backbone',b,'--k',str(k)])
        summarize('6-2');update(last_completed_experiment='6-2')
        subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','6-2','--approved-push','--message','6-2 완료: 정상 영상 CV·위상별 보정·오탐/미탐 비교'],check=True)
        update(state='completed',finished_at=time.time())
    except Exception as e:update(state='failed',error=str(e),finished_at=time.time());raise


if __name__=='__main__':main()
