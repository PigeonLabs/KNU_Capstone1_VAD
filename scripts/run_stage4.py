"""Run each approved experiment fully, report, then automatically publish its results."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import csv

import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve
from scripts.analyze_stage3 import auc_blocks,draw_auc


def summarize(stage):
    root=ROOT/'runs/stage4'/stage
    found=sorted(root.glob('R*/seed*/k*/metrics.json' if stage=='4-2' else 'R*/seed*/metrics.json'))
    values=[json.loads(p.read_text()) for p in found]
    names=(['unconditional','legacy_hard','top3_weighted'] if stage=='4-2' else
           ['unconditional','time5','time21','appearance_time5','appearance_time21'] if stage=='4-1' else
           ['unconditional','transition_kl','duration','process','phase_entropy','appearance_transition','appearance_duration','appearance_process','appearance_time21'])
    budgets=[1,2,5,10] if stage=='4-2' else [None]
    summary={'stage':stage,'units':len(values),'complete':len(values)==(48 if stage=='4-2' else 12),'macro':{},'scene_results':values}
    lines=[f'# {stage} 실험 결과','',f'완료 실행 단위: {len(values)}. R01–R04 / seed 0·1·2.',
      '테스트 결과로 설정을 선택하지 않았으며 사전 규약의 모든 변형을 보고한다.','',
      '| k | 방법 | seed0 macro | seed1 macro | seed2 macro | 평균 ± seed 표준편차 |',
      '|---|---|---:|---:|---:|---:|']
    for k in budgets:
        for method in names:
            perseed=[]
            for seed in range(3):
                group=[v for v in values if v['seed']==seed and (k is None or v['k']==k)]
                perseed.append(float(np.mean([v['metrics'][method]['auroc'] for v in group])) if len(group)==4 else None)
            if all(v is not None for v in perseed):
                summary['macro'][f'{k}/{method}']={'seeds':perseed,'mean':float(np.mean(perseed)),'std':float(np.std(perseed,ddof=1))}
                lines.append('| '+str(k or '-')+' | '+method+' | '+' | '.join(f'{v:.2f}' for v in perseed)+f' | {np.mean(perseed):.2f} ± {np.std(perseed,ddof=1):.2f} |')
    if stage!='4-2' and summary['complete']:
        primary='appearance_time21' if stage=='4-1' else 'appearance_process'
        diffs=[];rng=np.random.default_rng(1729);bootstrap={}
        for scene in ['R01','R02','R03','R04']:
            with (root/scene/'seed0/scores.csv').open() as f:rows=list(csv.DictReader(f))
            y=[int(r['label']) for r in rows];videos=[r['video'] for r in rows];n=len(set(videos))
            counts=rng.multinomial(n,np.full(n,1/n),size=2000)
            a=draw_auc(auc_blocks(y,[float(r[primary+'_score']) for r in rows],videos),counts)
            b=draw_auc(auc_blocks(y,[float(r['unconditional_score']) for r in rows],videos),counts)
            d=a-b;diffs.append(d);finite=d[np.isfinite(d)]
            bootstrap[scene]={'ci95_pp':np.quantile(finite,[.025,.975]).tolist(),'valid_draws':len(finite)}
        matrix=np.stack(diffs);valid=np.isfinite(matrix).all(0);ci=np.quantile(matrix[:,valid].mean(0),[.025,.975]).tolist()
        summary['bootstrap']={'primary':primary,'draws':2000,'unit':'video','seed0_only':True,'scenes':bootstrap,'macro_ci95_pp':ci,'valid_draws':int(valid.sum())}
        lines+=['',f'주 후보 {primary} - 무조건부 기준선의 seed0 paired video bootstrap 차이95% 구간: [{ci[0]:.2f}, {ci[1]:.2f}] pp.',
                'window21로 인한 경계 제외 이후 모든 방법이 동일한 평가 frame/video/label ID를 사용한다. 이전 3단계 전체 support 결과와 직접 차감하지 않는다.']
    if stage=='4-2':
        lines+=['','## 메모리와 효율 (seed0 측정)','',
          '| 장면 | k | 위치당 총 prototype | bank MiB | 무조건부 cache FPS | 무조건부 DINO 포함 FPS | hard DINO 포함 FPS | top3 DINO 포함 FPS |',
          '|---|---:|---:|---:|---:|---:|---:|---:|']
        for v in values:
            if v['seed']!=0:continue
            bench=json.loads((root/v['scene']/'seed0'/f"k{v['k']}"/'benchmark.json').read_text())['timings']
            lines.append(f"| {v['scene']} | {v['k']} | {v['prototypes_per_position']} | {v['single_bank_bytes']/1024**2:.2f} | {bench['unconditional']['cache_fps']:.1f} | {bench['unconditional']['pipeline_output_fps']:.1f} | {bench['legacy_hard']['pipeline_output_fps']:.1f} | {bench['top3_weighted']['pipeline_output_fps']:.1f} |")
        lines+=['','FPS는 warm-up 후 decoded RAM 입력 측정이다. 카메라·디스크 I/O는 포함하지 않으며 실제 공장/엣지 장치 streaming 성능이 아니다. cached 후보 제한 matching과 DINO 포함 측정은 분리한다. GPU peak는 비교 구현 전체 peak이며 단일 방법의 배포 peak가 아니다.',
          '같은 k에서 조건부·무조건부 bank는 같은 표본과 총 prototype 수를 사용한다. k10의 기존 점수 일치 여부는 baseline_equivalence.json에 있다.']
    if stage=='4-3':lines+=['','전이 분포는 정상 학습 영상의 모델 예측으로 학습했다. 체류시간에서는 영상 시작/끝의 잘린 run을 제외한다. 위상 jitter와 in-sample prediction 분포 차이가 한계이며 실제 이상 유형 분류나 독립 정상 오탐률을 주장하지 않는다.']
    lines+=['','세부 AUPRC·장면별 수치·원 점수/정답·설정·해시·R02 민감도·실행 로그는 각 실행 폴더에 보존한다. centered clip과 테스트 전체 정규화를 사용하는 오프라인 비교이다. R02 불일치 영상12·13·14는 주 결과에서 제외한다.']
    write_json(root/'summary.json',summary);(root/'results.md').write_text('\n'.join(lines)+'\n')


def main(stages):
    os.chdir(ROOT);reserve();root=ROOT/'runs/stage4';root.mkdir(exist_ok=True)
    state={'state':'running','pid':os.getpid(),'stages':stages,'started_at':time.time(),'completed':[],'automatic_publication_authorized':True}
    def update(**kw):state.update(kw);write_json(root/'status.json',state)
    try:
        update()
        for stage in stages:
            (root/stage).mkdir(exist_ok=True)
            for seed in range(3):
                for scene in ['R01','R02','R03','R04']:
                    for k in ([10,1,2,5] if stage=='4-2' else [10]):
                        reserve();key=f'{stage}/{scene}/seed{seed}'+(f'/k{k}' if stage=='4-2' else '')
                        command=[sys.executable,'-m','ipad.stage4','--stage',stage,'--scene',scene,'--seed',str(seed),'--k',str(k)]
                        update(step=key,command=command)
                        with (root/stage/f'{scene}_s{seed}_k{k}.log').open('a') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
                        state['completed'].append(key);update()
            summarize(stage);update(last_completed_experiment=stage)
            subprocess.run([sys.executable,'scripts/publish_stage.py','--stage',stage,'--approved-push','--message',f'실험 {stage} 완료: 전체 비교 결과·로그·해시 자동 게시'],check=True)
        update(state='completed',finished_at=time.time())
    except Exception as e:update(state='failed',error=str(e),finished_at=time.time());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stages',nargs='+',default=['4-1','4-2','4-3'],choices=['4-1','4-2','4-3']);main(p.parse_args().stages)
