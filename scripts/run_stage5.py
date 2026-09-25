"""Sequential approved stage-five experiments with automatic completion publication."""
import csv
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
from ipad.stage5 import SCENES,unit_path


def load(p):return json.loads(p.read_text())


def summarize(stage):
    root=ROOT/'runs/stage5'/stage;lines=[f'# {stage} 실험 결과','',
        'R01–R04,seed0·1·2. 정상80% 학습/20% 고정 보정. 테스트 정답은 학습·보정·임계값 선택에 사용하지 않았다.','']
    result={'stage':stage,'macro':{},'units':[]}
    if stage in ['5-1','5-2']:
        configs=[('B',10)] if stage=='5-1' else [('B',10),('B',5),('S',10),('S',5)]
        methods=['centered_testnorm','centered_fixed','causal_appearance','causal_combined'] if stage=='5-1' else ['causal_combined']
        lines+=['| 모델 | 점수 | seed0 | seed1 | seed2 | 평균 ± 표준편차 AUROC (%) |','|---|---|---:|---:|---:|---:|']
        for backbone,k in configs:
            ms=[load(unit_path(backbone,scene,seed,k)/'metrics.json') for seed in range(3) for scene in SCENES];result['units']+=ms
            for method in methods:
                vals=[]
                for seed in range(3):
                    items=[m for m in ms if m['seed']==seed]
                    vals.append(float(np.mean([m[method]['combined_score']['auroc'] if method.startswith('centered') else m['metrics']['appearance_score' if method=='causal_appearance' else 'combined_score']['auroc'] for m in items])))
                item={'seeds':vals,'mean':float(np.mean(vals)),'std':float(np.std(vals,ddof=1))};result['macro'][f'{backbone}_k{k}/{method}']=item
                lines.append(f'| {backbone}/k{k} | {method} | '+' | '.join(f'{v:.2f}' for v in vals)+f" | {item['mean']:.2f} ± {item['std']:.2f} |")
        lines+=['','공통 frame35..N-18에서 비교한다. stage4와 정상 학습 범위·타깃·보정 규칙이 달라 직접 차감하지 않는다.','',
          '장면별 AUROC/AUPRC,full causal support,프레임 점수,보정 median/q99.5/threshold,매배치 loss/gradient와 checkpoint 검증은 각 실행 폴더에 있다. R02영상12/13/14는 주 결과에서 제외하고 ±1 민감도를 보존한다.']
        if stage=='5-1':lines+=['','causal 입력은 현재까지16프레임,시간 검사는 과거21개 예측이며 첫 점수는 frame35이다. centered 비교군만 미래17프레임을 사용한다. 온라인 점수 보정은 정상 holdout에서 고정했고 테스트 중 업데이트하지 않는다.']
        else:lines+=['','메모리 k5는 k10의 정확히 절반 prototype이며 실제 점유 bin 수에 따라 총개수가 달라진다. backbone 변경 시 head·memory·normal calibration을 재구축했다. B/k10은5-1 결과를 재사용했다.']
    else:
        lines+=['| 모델 | 정상 frame FPR (%) | 오경보/정상1000frame | 구간 탐지율 (%) | 탐지 구간 지연 중앙값 (frame) |','|---|---:|---:|---:|---:|']
        for backbone,k in [('B',10),('B',5),('S',10),('S',5)]:
            ms=[load(root/scene/f'seed{seed}'/f'{backbone}_k{k}'/'completed.json') for seed in range(3) for scene in SCENES];result['units']+=ms
            ops=[m['operation'] for m in ms];mean=lambda key:float(np.mean([o[key] for o in ops if o[key] is not None]))
            row={'mean_scene_seed_frame_fpr':mean('frame_fpr'),'mean_false_alarms_per_1000_normal_frames':mean('false_alarms_per_1000_normal_frames'),
                 'mean_segment_recall':mean('segment_recall'),'mean_scene_seed_detected_delay_median':mean('delay_median_frames_detected_only')}
            result['macro'][f'{backbone}_k{k}']=row
            lines.append(f"| {backbone}/k{k} | {row['mean_scene_seed_frame_fpr']*100:.2f} | {row['mean_false_alarms_per_1000_normal_frames']:.2f} | {row['mean_segment_recall']*100:.2f} | {row['mean_scene_seed_detected_delay_median']:.2f} |")
        lines+=['','上表は장면·seed별 지표의 단순평균이다. 지연은 탐지된 구간만의 중앙값을 평균한 값이며 미탐·coldstart 구간 수는개별 operation.json에 함께 기록한다. 고장 유형별 정확도나 독립 사건 정답을 의미하지 않는다.'.replace('上表は','위 표는 '),'',
          '## 실제 batch1 측정 (seed0)','',
          '| 장면 | 모델 | capacity FPS | processing p95 ms | paced E2E p95 ms | paced deadline miss (%) | max queue ms | peak allocated GiB |',
          '|---|---|---:|---:|---:|---:|---:|---:|']
        for scene in SCENES:
            for b,k in [('B',10),('B',5),('S',10),('S',5)]:
                bench=load(root/scene/'seed0'/f'{b}_k{k}'/'benchmark.json');a=bench['timings']['capacity'];p=bench['timings']['paced_30fps']
                lines.append(f"| {scene} | {b}/k{k} | {a['input_fps']:.1f} | {a['processing_p95_ms']:.2f} | {p['end_to_end_p95_ms']:.2f} | {p['deadline_miss_fraction']*100:.2f} | {p['max_queue_ms']:.2f} | {bench['peak_allocated_gib']:.3f} |")
        lines+=['','현재 RTX PRO6000,FP32,batch1,원본 JPEG read/decode 포함. capacity3회/30FPS paced replay1회,각 영상최대256frame. OS page cache가 warm일 수 있으며 카메라·네트워크 지연은 미포함이다. 실제 촬영 FPS가 확인된 데이터는 아니므로30FPS는 도착률 시나리오이다. 경보 정확도는 전체 causal test cache 평가이며 위 짧은 replay와 구분한다.','',
          'raw latency·queue·직접 경보·stream/cache 비교·frame별 경보 상태·미탐을 포함한 구간별 지연은 각 실행 폴더에 있다.']
    result['complete']=True;write_json(root/'summary.json',result);(root/'results.md').write_text('\n'.join(lines)+'\n')


def main():
    os.chdir(ROOT);reserve();root=ROOT/'runs/stage5';root.mkdir(exist_ok=True)
    state={'state':'running','pid':os.getpid(),'started_at':time.time(),'completed':[],'automatic_publication_authorized':True}
    def update(**kw):state.update(kw);write_json(root/'status.json',state)
    def run(stage,key,args):
        reserve();out=root/stage;out.mkdir(exist_ok=True);command=[sys.executable,'-m','ipad.stage5',*args];update(step=key,command=command)
        with (out/(key.replace('/','_')+'.log')).open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
        state['completed'].append(key);update()
    try:
        update()
        for stage in ['5-1','5-2','5-3']:
            if stage=='5-2':
                for scene in SCENES:run(stage,'cache_S/'+scene,['--action','cache','--scene',scene])
            if stage=='5-3':
                with (root/'gpu_before_benchmark.txt').open('w') as f:subprocess.run(['nvidia-smi'],stdout=f,check=True)
            for seed in range(3):
                for scene in SCENES:
                    configs=[('B',10)] if stage=='5-1' else [('B',5),('S',10),('S',5)] if stage=='5-2' else [('B',10),('B',5),('S',10),('S',5)]
                    for b,k in configs:
                        run(stage,f'{stage}/{scene}/seed{seed}/{b}_k{k}', ['--action','benchmark' if stage=='5-3' else 'unit','--scene',scene,'--seed',str(seed),'--backbone',b,'--k',str(k)])
            summarize(stage);update(last_completed_experiment=stage)
            subprocess.run([sys.executable,'scripts/publish_stage.py','--stage',stage,'--approved-push','--message',f'실험 {stage} 완료: 온라인·경량화·경보 평가 기록'],check=True)
        update(state='completed',finished_at=time.time())
    except Exception as e:update(state='failed',error=str(e),finished_at=time.time());raise


if __name__=='__main__':main()
