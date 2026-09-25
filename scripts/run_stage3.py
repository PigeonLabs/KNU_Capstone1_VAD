"""Run the approved stage-3 units and publish each completed scene without scheduling."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import reserve


def report():
    root=ROOT/'runs/stage3';results={}
    for path in sorted(root.glob('R*/seed*/metrics.json')):
        value=json.loads(path.read_text());results[f"{value['scene']}/seed{value['seed']}"]=value
    write_json(root/'summary.json',results)
    lines=['단계별 완료 결과입니다. seed 0의 기존 hard/unconditional 점수와 2단계 점수 일치를 검증했습니다.',
           '테스트 결과로 변형을 선택하지 않으며, 주 후보는 사전 지정한 top3 확률 가중 거리입니다.','',
           '| 장면/seed | 무조건부 | 기존 hard | 인접 NN | top3 NN | top3 가중 | fallback | random3 | 전체 조건부 bank |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys=['unconditional','legacy_hard','neighbor_nn','top3_nn','top3_weighted','confidence_fallback','random3_nn','conditional_all']
    for name,r in results.items():
        lines.append('| '+name+' | '+' | '.join(f"{r['metrics'][k]['auroc']:.2f}" for k in keys)+' |')
    lines+=['','| 정상 holdout | 20-bin 정확도 | ±1 정확도 | 원형 MAE(bin) | cutoff 활성 |','|---|---:|---:|---:|---|']
    for name,r in results.items():
        d=r['normal_validation'];lines.append(f"| {name} | {d['bin20_accuracy']*100:.2f}% | {d['within_one_accuracy']*100:.2f}% | {d['circular_mae_bins']:.2f} | {r['calibration']['enabled']} |")
    lines+=['','원시 CSV에는 frame ID, 예측 확률, 모든 비교 점수와 정답이 포함됩니다. 세부 지표·AUPRC·R02 민감도는 장면/seed별 JSON에 있습니다.',
            '정상 holdout의 상대 위치는 진단용 참조입니다. 테스트 정답 위상을 사용하지 않습니다. centered window와 테스트 전체 정규화를 사용하므로 온라인 실시간 결과가 아닙니다.']
    (root/'results.md').write_text('\n'.join(lines)+'\n')


def main(seeds):
    os.chdir(ROOT);reserve();root=ROOT/'runs/stage3';root.mkdir(exist_ok=True)
    status={'state':'running','pid':os.getpid(),'seeds':seeds,'started_at':time.time(),'completed':[]}
    def update(**kw):status.update(kw);write_json(root/'status.json',status)
    try:
        for seed in seeds:
            for scene in ['R01','R02','R03','R04']:
                reserve();cmd=[sys.executable,'-m','ipad.phase_routing','--scene',scene,'--seed',str(seed)]
                update(step=f'{scene}/seed{seed}',command=cmd)
                with (root/f'{scene}_seed{seed}.log').open('a') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                status['completed'].append(f'{scene}/seed{seed}');report();update()
                subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','stage3','--message',
                    f'실험 3단계: {scene} seed {seed} 정상 위상 진단과 routing 비교'],check=True)
        update(state='completed',finished_at=time.time());report()
        subprocess.run([sys.executable,'scripts/publish_stage.py','--stage','stage3','--message',
                       '실험 3단계: 승인된 routing 비교 완료 현황 정리'],check=True)
    except Exception as e:
        update(state='failed',error=str(e),finished_at=time.time());raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seeds',type=int,nargs='+',default=[0]);main(p.parse_args().seeds)
