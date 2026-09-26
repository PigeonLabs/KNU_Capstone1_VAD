"""Archive analysis artifacts and publish only the user's approved project paths."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SCENES=('R01','R02','R03','R04')
REMOTE='https://github.com/PigeonLabs/KNU_Capstone1_VAD.git'
RESERVE=10*1024**3
STAGES={'stage1':'stage1_reproduction','stage2':'stage2_dinov2','ablation':'stage1_memory_ablation','stage3':'stage3_phase_routing','4-1':'stage4_1_appearance_temporal','4-2':'stage4_2_memory_budget','4-3':'stage4_3_process_prior','5-1':'stage5_1_causal','5-2':'stage5_2_lightweight','5-3':'stage5_3_streaming','6-1':'stage6_1_pareto','6-2':'stage6_2_calibration','7-1':'stage7_1_diagnostics','7-2':'stage7_2_alerts','7-3':'stage7_3_full_stream','8-1':'stage8_1_lora_validation','8-2':'stage8_2_lora_comparison','8-3':'stage8_3_lora_stream','quant-probe':'quantization_probe','9-1':'stage9_1_quant_compile'}


def load(path):return json.loads(path.read_text()) if path.exists() else None


def git(*args,check=True):
    return subprocess.run(['git',*args],cwd=ROOT,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=check)


def ensure_room(extra=0):
    if (ROOT/'runs/disk_pause.json').exists() or shutil.disk_usage(ROOT).free-extra<=RESERVE:
        raise RuntimeError('Disk reserve reached or pause is latched; publication stopped without resuming experiments')


def copy_analysis(source,dest):
    if not source.exists():return
    files=[source] if source.is_file() else source.rglob('*')
    for file in files:
        if not file.is_file() or file.suffix not in {'.json','.jsonl','.csv','.log','.txt','.md'}:continue
        target=dest if source.is_file() else dest/file.relative_to(source)
        ensure_room(file.stat().st_size)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(file,target)


def binary_inventory(stage):
    if stage in {'quant-probe','9-1'}:
        yield from ()
    elif stage=='stage1':
        for scene in SCENES:
            yield from (ROOT/'IPAD_dataset'/scene).rglob('*.jpg')
            yield from (ROOT/'IPAD_dataset'/scene).rglob('*.npy')
            yield from (ROOT/'runs/paper'/scene/'seed0').glob('*.pt')
    elif stage=='stage2':
        yield from (ROOT/'cache/dino').rglob('*.npy')
        for scene in SCENES:yield from (ROOT/'runs/prototype'/scene/'seed0').glob('*.pt')
        yield from (ROOT/'cache/torch_hub/checkpoints').glob('*.pth')
    elif stage in {'4-1','4-2','4-3'}:
        yield from (ROOT/'runs/stage4'/stage).rglob('*.pt')
    elif stage in {'5-1','5-2','5-3'}:
        yield from (ROOT/'runs/stage5'/stage).rglob('*.pt')
        if stage=='5-2':
            yield from (ROOT/'cache/dino_small').rglob('*.npy')
            yield from (ROOT/'cache/torch_hub/checkpoints').glob('*vits14*.pth')
    elif stage in {'6-1','6-2'}:
        yield from (ROOT/'runs/stage6'/stage).rglob('*.pt')
    elif stage in {'7-1','7-2','7-3'}:
        yield from (ROOT/'runs/stage7'/stage).rglob('*.pt')
    elif stage in {'8-1','8-2','8-3'}:
        yield from (ROOT/'runs/stage8'/stage).rglob('*.pt')
        yield from (ROOT/'runs/stage8'/stage).rglob('*.npy')
    elif stage=='stage3':
        yield from (ROOT/'runs/stage3').rglob('*.pt')
    else:
        for scene in SCENES:
            folder=ROOT/'runs/no_memory'/scene/'seed0'
            if (folder/'completed.json').exists():yield from folder.glob('*.pt')


def inventory(stage,dest):
    previous={}
    manifest=dest/'artifacts.jsonl'
    if manifest.exists():
        previous={r['path']:r for r in (json.loads(x) for x in manifest.read_text().splitlines())}
    with manifest.with_suffix('.tmp').open('w') as out:
        for file in sorted(binary_inventory(stage)):
            stat=file.stat();relative=str(file.relative_to(ROOT));old=previous.get(relative)
            if old and old['bytes']==stat.st_size and old['mtime_ns']==stat.st_mtime_ns:
                row=old
            else:
                with file.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
                after=file.stat()
                if (stat.st_size,stat.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
                    raise RuntimeError(f'Artifact changed during hashing: {relative}')
                row={'path':relative,'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns,'sha256':digest,'binary_uploaded':False}
            out.write(json.dumps(row)+'\n')
    manifest.with_suffix('.tmp').replace(manifest)


def snapshot(stage):
    dest=ROOT/'experiments'/STAGES[stage];dest.mkdir(parents=True,exist_ok=True)
    if stage=='9-1':
        copy_analysis(ROOT/'runs/stage9/9-1',dest)
        # Explicit analytical plot allowlist; dataset images remain excluded.
        plot=ROOT/'runs/stage9/9-1/kernel_study/kernel_latency.png'
        if plot.exists():
            ensure_room(plot.stat().st_size)
            shutil.copy2(plot,dest/'kernel_study/kernel_latency.png')
    elif stage=='quant-probe':
        copy_analysis(ROOT/'runs/quantization_probe',dest)
    elif stage=='stage1':
        for scene in SCENES:
            source=ROOT/f'runs/paper/{scene}/seed0'
            if not (source/'evaluation/metrics.json').exists():raise RuntimeError(f'Baseline incomplete: {scene}')
            for name in ['config.json','history.jsonl','batch_history.jsonl','completed.json','evaluation']:
                copy_analysis(source/name,dest/scene/name)
            copy_analysis(ROOT/f'runs/paper_{scene}_s0.log',dest/scene/'train.log')
            copy_analysis(ROOT/f'runs/eval_{scene}_s0.log',dest/scene/'evaluate.log')
        for name in ['data_audit.json','cuda_check.json','resume_verification.json','pytest.txt']:
            copy_analysis(ROOT/'reports'/name,dest/'verification'/name)
        copy_analysis(ROOT/'runs/smoke/fp32/smoke.json',dest/'verification/smoke.json')
    elif stage=='stage2':
        for scene in SCENES:
            source=ROOT/f'runs/prototype/{scene}/seed0'
            if not (source/'completed.json').exists():raise RuntimeError(f'DINOv2 prototype incomplete: {scene}')
            copy_analysis(ROOT/f'runs/paper/{scene}/seed0/evaluation_dino',dest/scene/'reconstruction')
            copy_analysis(source,dest/scene/'prototype')
            copy_analysis(ROOT/f'cache/dino/{scene}',dest/scene/'feature_metadata')
            for name in ['dino_reconstruction','dino_cache','prototype']:
                suffix='' if name=='dino_cache' else '_s0'
                copy_analysis(ROOT/f'runs/{name}_{scene}{suffix}.log',dest/scene/f'{name}.log')
        copy_analysis(ROOT/'reports/dino_verification.json',dest/'verification/dino_verification.json')
    elif stage in {'4-1','4-2','4-3'}:
        copy_analysis(ROOT/'runs/stage4'/stage,dest)
        copy_analysis(ROOT/'runs/stage4/status.json',dest/'runner_status.json')
        copy_analysis(ROOT/'runs/stage4/verification.txt',dest/'verification.txt')
        copy_analysis(ROOT/'runs/stage4/research_findings.md',ROOT/'experiments/stage4_summary/research_findings.md')
        copy_analysis(ROOT/'runs/stage4/final_verification.json',ROOT/'experiments/stage4_summary/final_verification.json')
    elif stage in {'5-1','5-2','5-3'}:
        copy_analysis(ROOT/'runs/stage5'/stage,dest)
        for name in ['status.json','verification.txt','operational_verification.txt','gpu_before_benchmark.txt','implementation_events.jsonl']:
            copy_analysis(ROOT/'runs/stage5'/name,dest/name)
        if stage=='5-2':copy_analysis(ROOT/'cache/dino_small',dest/'feature_metadata')
        for name in ['research_findings.md','final_verification.json']:
            copy_analysis(ROOT/'runs/stage5'/name,ROOT/'experiments/stage5_summary'/name)
    elif stage in {'6-1','6-2'}:
        copy_analysis(ROOT/'runs/stage6'/stage,dest)
        for name in ['status.json','runner.log','verification.txt','targeted_verification.txt','final_verification.log','precision_diagnostic.json','gpu_before_benchmark.txt','implementation_events.jsonl']:
            copy_analysis(ROOT/'runs/stage6'/name,dest/name)
        for name in ['research_findings.md','final_verification.json']:
            copy_analysis(ROOT/'runs/stage6'/name,ROOT/'experiments/stage6_summary'/name)
    elif stage in {'7-1','7-2','7-3'}:
        copy_analysis(ROOT/'runs/stage7'/stage,dest)
        for name in ['status.json','runner.log','verification.txt','full_verification.txt','precision_diagnostic.json','precision_diagnostic.log','decision.json','command_manifest.json','gpu_before_stream.txt','implementation_events.jsonl','final_verification.log']:
            copy_analysis(ROOT/'runs/stage7'/name,dest/name)
        for name in ['research_findings.md','final_verification.json']:
            copy_analysis(ROOT/'runs/stage7'/name,ROOT/'experiments/stage7_summary'/name)
    elif stage in {'8-1','8-2','8-3'}:
        copy_analysis(ROOT/'runs/stage8'/stage,dest)
        for name in ['status.json','runner.log','verification.txt','freeze.json','provenance.json','input_cache_audit.json','commands.jsonl','gpu_before.txt','implementation_events.jsonl']:
            copy_analysis(ROOT/'runs/stage8'/name,dest/name)
        for name in ['research_findings.md','final_verification.json']:
            copy_analysis(ROOT/'runs/stage8'/name,ROOT/'experiments/stage8_summary'/name)
    elif stage=='stage3':
        for scene in SCENES:
            for source in (ROOT/'runs/stage3'/scene).glob('seed*'):
                if (source/'completed.json').exists():copy_analysis(source,dest/scene/source.name)
        copy_analysis(ROOT/'runs/stage3/summary.json',dest/'summary.json')
        copy_analysis(ROOT/'runs/stage3/bootstrap.json',dest/'bootstrap.json')
        copy_analysis(ROOT/'runs/stage3/research_findings.md',dest/'research_findings.md')
        copy_analysis(ROOT/'runs/stage3/repeat_decision.json',dest/'repeat_decision.json')
        copy_analysis(ROOT/'runs/stage3/results.md',dest/'results.md')
        copy_analysis(ROOT/'runs/stage3/verification.txt',dest/'verification.txt')
        copy_analysis(ROOT/'runs/stage3/final_verification.json',dest/'final_verification.json')
        copy_analysis(ROOT/'runs/stage3/status.json',dest/'status.json')
        copy_analysis(ROOT/'runs/stage3/temporal_status.json',dest/'temporal_status.json')
        for log in (ROOT/'runs/stage3').glob('*.log'):copy_analysis(log,dest/log.name)
    else:
        for scene in SCENES:
            source=ROOT/f'runs/no_memory/{scene}/seed0'
            if (source/'completed.json').exists():
                copy_analysis(source,dest/scene)
                copy_analysis(ROOT/f'runs/no_memory_{scene}_s0.log',dest/scene/'train.log')
                copy_analysis(ROOT/f'runs/eval_no_memory_{scene}_s0.log',dest/scene/'evaluate.log')
    inventory(stage,dest)
    copy_analysis(ROOT/'runs/events.jsonl',ROOT/'experiments/process_events.jsonl')
    # Retain evidence of failures/restarts in the raw orchestration log.
    copy_analysis(ROOT/'runs/suite_console.log',ROOT/'experiments/suite_console.log')
    return dest


def make_readme():
    base=ROOT/'experiments/stage1_reproduction';dino=ROOT/'experiments/stage2_dinov2'
    rows=[];targets=[84.4,75.4,43.5,76.7];values=[]
    for scene,target in zip(SCENES,targets):
        d=load(base/scene/'evaluation/metrics.json')
        if d:
            value=d['metrics']['negative_psnr_with_phase']['auroc'];values.append(value)
            rows.append(f'| {scene} | {target:.2f} | {value:.2f} | {value-target:+.2f} | {d["frames"]:,} |')
    import re
    def embedded(path):
        if not path.exists(): return ''
        lines=[]; in_fence=False
        for line in path.read_text().splitlines():
            if line.lstrip().startswith('```'):
                in_fence=not in_fence
                lines.append(line)
                continue
            if not in_fence:
                heading=re.match(r'^(\s*)(#{1,6})([ \t]+.*)$',line)
                if heading:
                    line=heading.group(1)+'#'*min(6,len(heading.group(2))+2)+heading.group(3)
                def rebase(match):
                    raw=match.group(1); target=raw.strip()
                    if target.startswith(('http://','https://','mailto:','tel:','#','/','<')): return match.group(0)
                    first,sep,rest=target.partition(' ')
                    link,hashmark,fragment=first.partition('#')
                    if not link: return match.group(0)
                    relative=(path.parent.relative_to(ROOT)/link)
                    import os
                    link=os.path.normpath(str(relative)).replace(os.sep,'/')
                    rebuilt=link+('#'+fragment if hashmark else '')+((' '+rest) if sep else '')
                    return ']('+rebuilt+')'
                line=re.sub(r'(?<!!)\]\(([^)]*)\)',rebase,line)
            lines.append(line)
        return '\n'.join(lines)
    text=['# 산업 공정 영상 이상탐지: IPAD 재현과 DINOv2 비교','',
          'R01–R04 실제 공정 영상으로 IPAD 논문 방법을 재현하고, DINOv2 특징 비교부터 위상 메모리·온라인 탐지·경량화·양자화까지 단계별로 평가합니다. 각 절에는 핵심 결과와 전체 실험 보고서를 함께 두었습니다.','',
          '[단계별 진행](#단계별-진행) · [핵심 결과](#핵심-결과-요약) · [실행 방법](#실행-방법) · [평가 범위와 한계](#평가-범위와-한계) · [원 자료](#원-자료)','',
          '## 단계별 진행','',
          '| 단계 | 목적 | 상태 | 기록 |','|---|---|---|---|',
          '| 1단계 | Swin-T + 주기 메모리 + 재구성 + 주기 검사 | 4개 장면 50 epochs 완료 · seed 0 | [전체 자료](experiments/stage1_reproduction/) |',
          f'| 2단계 | DINOv2 입력–복원 특징 비교 / 비재구성 prototype | {"4개 장면 완료 · seed 0" if (dino/"R04/prototype/completed.json").exists() else "게시 준비 중"} | [전체 자료](experiments/stage2_dinov2/) |',
          '| 1단계 추가 검증 | 메모리 제거 ablation | 4개 장면 완료 · seed 0 | [자료](experiments/stage1_memory_ablation/) |',
          '| 3단계 | 위상 진단 및 불확실성을 고려한 메모리 선택 | R01–R04 × seed 0·1·2 및 시간 진단 완료 | [규약](docs/stage3_protocol.md) · [결과](experiments/stage3_phase_routing/) |','',
          '## 1단계 — 논문 방법론 재현','',
          '장면마다 독립 학습: 16프레임, 256×256, Video Swin-T, 200개 위상, 메모리 2,000개, window 5, Adam 1e-4, batch 8, 50 epochs, FP32, seed 0. 재구성·주기 점수를 장면별 정규화 후 같은 가중치로 결합합니다.','',
          '| 장면 | 논문 AUROC (%) | 구현 AUROC (%) | 차이 (pp) | 평가 프레임 |','|---|---:|---:|---:|---:|',*rows]
    position=text.index('## 1단계 — 논문 방법론 재현')-1
    additional=[]
    for st,title in [('4-1','외형·시간 검사'),('4-2','메모리 용량'),('4-3','전이·체류시간'),('5-1','온라인 기준선'),('5-2','백본·메모리 경량화'),('5-3','30 FPS 재생·오탐·미탐 평가'),('6-1','정밀도·메모리 Pareto'),('6-2','정상 영상 보정 일반화'),('7-1','오탐·미탐 원인 분해'),('7-2','causal 경보 규칙 비교'),('7-3','전체 영상 batch1 검증'),('8-1','LoRA 구현·학습 검증'),('8-2','정상 영상 LoRA 비교'),('8-3','반복·병합 BF16 실시간'),('9-1','양자화 실행 경로·컴파일·커널 진단')]:
        ready=(ROOT/'experiments'/STAGES[st]/'results.md').exists()
        label = ('완료 · seed 0' if st in ['8-1','8-2','9-1'] else '완료 · seed 0·1·2') if ready else '진행 전 또는 실행 중'
        display='9단계' if st=='9-1' else st
        additional.append(f'| {display} | {title} | {label} | [결과·기록](experiments/{STAGES[st]}/results.md) |')
    text[position:position]=additional
    summary=['','## 핵심 결과 요약','',
        '| 단계 | 주요 관찰 | 상세 결과 |','|---|---|---|',
        f'| 1단계 논문 재현 | 네 장면 AUROC 평균 **{sum(values)/4:.2f}%** (논문 표의 평균 70.00%). R02 영상 12·13·14는 정렬 불일치로 주 결과에서 제외했습니다. | [재현 결과·차이](REPRODUCTION.md) |' if len(values)==4 else '| 1단계 논문 재현 | 장면별 결과와 논문 구현 차이를 기록했습니다. | [재현 기록](experiments/stage1_reproduction/) |',
        '| 2단계 DINOv2 | 비재구성 위상 무조건부 최근접 prototype의 seed 0 평균 AUROC는 78.37%였습니다. 단일 seed 결과입니다. | [전체 비교](experiments/stage2_dinov2/) |',
        '| 3단계 위상 진단 | 정상 holdout에서 인접 위상 정확도는 장면별 약 83–96%였습니다. centered-window 테스트는 오프라인 진단입니다. | [3-seed 종합](experiments/stage3_phase_routing/research_findings.md) |',
        '| 4단계 시간 정보 | 외형+window 21의 AUROC는 80.97 ± 0.67% (3 seeds)였습니다. | [통합 분석](experiments/stage4_summary/research_findings.md) |',
        '| 5단계 온라인 전환 | 과거 프레임 기반 외형+시간 기준선은 AUROC 81.24 ± 0.12%였습니다. | [통합 분석](experiments/stage5_summary/research_findings.md) |',
        '| 6단계 효율 절충 | S_bf16_k5는 AUROC 79.85 ± 0.45%, capacity 260.6 FPS, peak 0.085 GiB의 관측 Pareto 후보였습니다. | [통합 분석](experiments/stage6_summary/research_findings.md) |',
        '| 7단계 경보 분석 | R01 B/k10에서 hysteresis는 오경보를 3.87→1.85회/정상 1,000프레임으로 낮췄고, 구간 recall도 59.70→48.31%로 낮췄습니다. | [통합 분석](experiments/stage7_summary/research_findings.md) |',
        '| 8단계 LoRA | 3-seed stream의 frozen AUROC는 81.08 ± 0.08%, anchored는 80.97 ± 0.20%였습니다. 정상 적응의 이득은 제한적이었습니다. | [통합 분석](experiments/stage8_summary/research_findings.md) |',
        '| 9단계 양자화·커널 | W4 packed는 INT4 저장 형식에 BF16 GEMM을 사용하며 payload는 46.17 MiB (BF16 165.14 MiB)였습니다. 수치 검사를 통과한 graph 경로의 backbone은 4.944 ms (BF16 3.306 ms), BF16 대비 patch cosine 거리는 약 0.1304였습니다. 전체 VAD AUROC는 측정하지 않았습니다. | [최적화](experiments/stage9_1_quant_compile/optimized/results.md) · [커널 분석](experiments/stage9_1_quant_compile/kernel_study/results.md) |']
    position=text.index('## 1단계 — 논문 방법론 재현')-1
    text[position:position]=summary
    if len(values)==4:text.extend(['',f'장면별 AUROC 단순 평균: **{sum(values)/4:.2f}%** (논문 70.00%).'])
    text.extend(['','**해석 제한:** R02는 영상/라벨 길이가 다른 영상 12·13·14를 제외합니다. 공개 코드의 전체 파라미터는 263.48M으로 논문 표 35.9M과 다릅니다. 점수 결합 등 미기재 사항을 명시적 가정으로 보완했으므로 원 논문과 완전히 같은 조건의 우월성 증거로 해석하지 않습니다. [차이와 가정](REPRODUCTION.md)','',
                 '## 2단계 — DINOv2 도입','',
                 '- **A: 입력–복원 특징 비교** — 기존 IPAD checkpoint를 유지하고 frozen ViT-B/14의 CLS 및 6·12층 patch 특징 차이를 비교합니다.',
                 '- **B: 비재구성 prototype** — 16프레임 CLS 위상 MLP와 20개 위상 구간의 공간별 메모리를 사용합니다. 위상별 최대 10개 prototype, cosine NN 및 soft projection(온도 0.1)을 비교합니다.',
                 '- 조건부/무조건부 비교는 같은 정상 특징 표본과 같은 총 prototype 수를 사용합니다. 모든 방법은 동일한 유효 평가 프레임을 사용합니다.',''])
    if (dino/'R04/prototype/completed.json').exists():
        methods=[('원 재현: 픽셀+주기','baseline','negative_psnr_with_phase'),
                 ('A: CLS','reconstruction','dino_cls'),('A: 최종층 patch','reconstruction','dino_patch12'),
                 ('A: 다층 patch+주기','reconstruction','dino_multilevel_with_phase'),
                 ('B: 위상 조건부 NN','prototype','conditional_nn'),('B: 위상 무조건부 NN','prototype','unconditional_nn'),
                 ('B: 위상 조건부 soft','prototype','conditional_soft'),('B: 위상 무조건부 soft','prototype','unconditional_soft')]
        text+=['| 방법 (AUROC %) | R01 | R02* | R03 | R04 | 평균 |','|---|---:|---:|---:|---:|---:|']
        for label,family,score in methods:
            vals=[]
            for scene in SCENES:
                path=base/scene/'evaluation/metrics.json' if family=='baseline' else dino/scene/family/'metrics.json'
                vals.append(load(path)['metrics'][score]['auroc'])
            text.append('| '+label+' | '+' | '.join(f'{v:.2f}' for v in vals)+f' | {sum(vals)/4:.2f} |')
        text+=['','표의 방법은 모든 장면에서 같은 점수 정의를 사용합니다. 장면마다 가장 높은 변형을 골라 평균내지 않습니다. AUPRC, 모든 점수 변형, 원 점수·정답 CSV, R02 정렬 민감도는 단계별 폴더에 보존합니다. seed 0 단일 실행이며 통계적 유의성 주장이 아닙니다.']
    else:text+=['DINOv2 결과는 전체 완료 후 다음 단계 commit으로 게시합니다.']
    ablation=ROOT/'experiments/stage1_memory_ablation'
    existing=[s for s in SCENES if (ablation/s/'evaluation/metrics.json').exists()]
    if existing:
        text+=['','### 1단계 추가 검증: 메모리 제거','', '| 장면 | 원 모델 | 메모리 제거+주기 |','|---|---:|---:|']
        for s in existing:
            a=load(base/s/'evaluation/metrics.json')['metrics']['negative_psnr_with_phase']['auroc']
            b=load(ablation/s/'evaluation/metrics.json')['metrics']['negative_psnr_with_phase']['auroc']
            text.append(f'| {s} | {a:.2f} | {b:.2f} |')
    text+=['','## 실행 방법','', '```bash','uv venv .venv --python 3.13',
           'uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128',
           '.venv/bin/python scripts/bootstrap_vendor.py',
           '# 원 IPAD 데이터의 R01~R04를 IPAD_dataset/ 아래에 배치',
           '.venv/bin/python -m ipad.data --cache',
           '.venv/bin/python -m pytest -q',
           '.venv/bin/python scripts/launch_suite.py --stage all',
           '.venv/bin/python scripts/status.py','```','',
           '현재 실행 환경은 RTX PRO 6000 Blackwell 96GB, PyTorch 2.11.0+cu128입니다. 다른 GPU에서는 호환 환경과 소규모 검증을 먼저 확인합니다. [세부 명령](docs/RUNNING.md)','',
           '## 평가 범위와 한계','',
           '- 각 실험 폴더에는 설정, seed, 환경·소스 해시, 명령, 학습 이력, 실패·재시작 기록, 프레임 정렬표와 프레임별 점수가 보관됩니다. 바이너리는 공개하지 않으며 경로·크기·SHA256 목록만 제공합니다.',
           '- R02 영상 12·13·14는 영상과 라벨의 프레임 수가 달라 주 결과에서 제외했습니다. 따라서 R02 전체와 4개 장면 평균은 완전한 원 논문 재현 수치로 표시하지 않습니다.',
           '- 1–4단계는 중앙 프레임 및 테스트 구간 정규화를 포함하는 오프라인 비교입니다. 5단계 이후는 분리된 정상 보정과 과거 프레임 기반 추론을 사용하지만 실제 카메라·다른 GPU의 성능을 보장하지 않습니다.',
           '- 9단계는 정상 프레임에서 실행 경로, 특징 수치 차이, 시간과 메모리를 진단했습니다. 전체 이상탐지 AUROC·경보 성능을 평가하지 않았으며, 양자화를 배포 후보로 확정한 결과가 아닙니다.',
           '[실행 안내](docs/RUNNING.md) · [실행 기록 정책](docs/EXPERIMENT_LOG_POLICY.md) · [논문–구현 차이](REPRODUCTION.md) · [3단계 사전 규약](docs/stage3_protocol.md)','',
           '## 원 자료','',
           '- [IPAD 논문 v1](https://arxiv.org/abs/2404.15033v1) · [공식 코드](https://github.com/LJF1113/IPAD), commit `22764cbeeda3946303d236babdd2664fd6241b91`.',
           '- [DINOv2 공식 구현](https://github.com/facebookresearch/dinov2), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.',
           '- upstream 코드의 재배포 대신 출처·SHA256을 보존하고 bootstrap에서 원본을 내려받습니다.','']
    if (ROOT/'experiments/stage3_phase_routing/results.md').exists():
        findings=ROOT/'experiments/stage3_phase_routing/research_findings.md'
        text += ['## 3단계 — 위상 진단과 선택 방식 비교', '', '[연구 해석·3-seed 요약·시간 진단](experiments/stage3_phase_routing/research_findings.md) · [사전 고정 규약](docs/stage3_protocol.md)', '', embedded(findings), '', embedded(ROOT/'experiments/stage3_phase_routing/results.md')]
    for st in ['4-1','4-2','4-3']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['', f'## {st} 추가 실험', '', f'[전체 기록](experiments/{STAGES[st]}/) · [고정 규약](docs/stage4_protocol.md)', '', path.read_text()]
    summary_path=ROOT/'experiments/stage4_summary/research_findings.md'
    if summary_path.exists():
        text += ['', '## 4단계 통합 해석', '', '[세 실험의 통합 보고서와 최종 검산](experiments/stage4_summary/research_findings.md)', '', embedded(summary_path)]
    for st in ['5-1','5-2','5-3']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['', f'## {st} 온라인·경량화 실험', '', f'[전체 기록](experiments/{STAGES[st]}/) · [고정 규약](docs/stage5_protocol.md)', '', path.read_text()]
    summary_path=ROOT/'experiments/stage5_summary/research_findings.md'
    if summary_path.exists():
        text += ['', '## 5단계 통합 해석', '', '[온라인·경량화·실시간 평가 통합 결과](experiments/stage5_summary/research_findings.md)', '', embedded(summary_path)]
    for st in ['6-1','6-2']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['', f'## {st} 추가 실험', '', f'[전체 기록](experiments/{STAGES[st]}/) · [고정 규약](docs/stage6_protocol.md)', '', path.read_text()]
    summary_path=ROOT/'experiments/stage6_summary/research_findings.md'
    if summary_path.exists():
        text += ['', '## 6단계 통합 해석', '', '[Pareto·일반화 통합 결과](experiments/stage6_summary/research_findings.md)', '', embedded(summary_path)]
    for st in ['7-1','7-2','7-3']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['',f'## {st} 실험','',f'[전체 기록](experiments/{STAGES[st]}/) · [규약](docs/stage7_protocol.md)','',path.read_text()]
    summary_path=ROOT/'experiments/stage7_summary/research_findings.md'
    if summary_path.exists():
        text += ['', '## 7단계 통합 결과', '', '[원인 분석·경보·전체 스트림 검증](experiments/stage7_summary/research_findings.md)', '', embedded(summary_path)]
    for st in ['8-1','8-2','8-3']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['',f'## {st} LoRA 실험','',f'[전체 기록](experiments/{STAGES[st]}/) · [규약](docs/stage8_protocol.md)','',path.read_text()]
    summary_path=ROOT/'experiments/stage8_summary/research_findings.md'
    if summary_path.exists():
        text += ['', '## 8단계 통합 결과', '', '[정상 적응·오탐·미탐·실시간 검증](experiments/stage8_summary/research_findings.md)', '', embedded(summary_path)]
    qp=ROOT/'experiments/quantization_probe/results.md'
    if qp.exists():text += ['','## 9단계 사전 진단 — INT8·INT4 양자화','','정상 96프레임을 이용한 구현 진단으로, 전체 AUROC·실시간 VAD 평가는 포함하지 않았습니다. [규약](docs/quantization_probe.md) · [전체 측정·로그](experiments/quantization_probe/)', '',embedded(qp)]
    q9=ROOT/'experiments/stage9_1_quant_compile/results.md'
    if q9.exists():text += ['','## 9단계 — 양자화 실행 경로와 컴파일 최적화','','[전체 기록](experiments/stage9_1_quant_compile/) · [규약](docs/stage9_protocol.md)','',embedded(q9)]
    optimized=ROOT/'experiments/stage9_1_quant_compile/optimized/results.md'
    if optimized.exists():
        text += ['', '## 9단계 후속 — INT8·INT4 최적화', '', '[전체 최적화 결과·수치 검증·원측정](experiments/stage9_1_quant_compile/optimized/results.md)', '', embedded(optimized)]
    audit=ROOT/'experiments/stage9_1_quant_compile/audit/results.md'
    if audit.exists():
        text += ['', '## 9단계 감사 — 양자화 경로 이상값 원인 분리', '', '[감사 기록·수치 검증](experiments/stage9_1_quant_compile/audit/results.md)', '', embedded(audit)]
    kernel=ROOT/'experiments/stage9_1_quant_compile/kernel_study/results.md'
    if kernel.exists():
        text += ['', '## 9단계 커널 진단 — INT4 병목의 하드웨어 분석', '', '[선형층·캐시·Nsight 측정 전체 결과](experiments/stage9_1_quant_compile/kernel_study/results.md) · [실험 규약](docs/stage9_kernel_protocol.md)', '', embedded(kernel)]
    (ROOT/'README.md').write_text('\n'.join(text)+'\n')


def publish(stage,message,push=False,approved_push=False):
    os.chdir(ROOT);ensure_room()
    if push and not approved_push:
        raise RuntimeError("Publication requires explicit authorization; approved stage 4/5/6 experiments have standing user authorization")
    lock=(ROOT/'runs/publication.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX)
    status={'state':'preparing','stage':stage,'started_at':time.time(),'repository':REMOTE}
    state_file=ROOT/'runs/publication_status.json'
    def state(**kw):status.update(kw);state_file.write_text(json.dumps(status,indent=2)+'\n')
    state()
    try:
        snapshot(stage);make_readme()
        if not (ROOT/'.git').exists():git('init','-b','main');git('remote','add','origin',REMOTE)
        if git('branch','--show-current').stdout.strip()!='main':raise RuntimeError('Publication only allowed from main')
        if git('remote','get-url','origin').stdout.strip()!=REMOTE:raise RuntimeError('Unexpected origin')
        git('fetch','origin')
        remote=git('rev-parse','--verify','origin/main',check=False)
        if remote.returncode==0:
            head=git('rev-parse','HEAD',check=False)
            if head.returncode or git('merge-base','--is-ancestor',remote.stdout.strip(),'HEAD',check=False).returncode:
                raise RuntimeError('Remote main changed; reconcile it before publishing. No force push attempted.')
        paths=['AGENTS.md','.gitignore','README.md','REPRODUCTION.md','requirements.txt','requirements.lock.txt','ipad','scripts','tests','docs','experiments']
        # Git stores new/changed blobs; unchanged archived experiments need no new copy.
        candidates=set(git('ls-files','-z','--modified','--others','--exclude-standard','--',*paths).stdout.split('\0'))
        candidates.update(git('diff','--cached','--name-only','-z','--',*paths).stdout.split('\0'))
        estimated=sum((ROOT/name).stat().st_size for name in candidates if name and (ROOT/name).is_file())
        reserve_estimate=3*estimated+64*1024**2
        ensure_room(reserve_estimate)
        state(changed_file_bytes=estimated,publication_reserve_bytes=reserve_estimate)
        git('add','--',*paths)
        for path in git('diff','--cached','--name-only').stdout.splitlines():
            if Path(path).suffix in {'.pt','.pth','.npy','.jpg','.pdf','.docx','.hwp'}:
                raise RuntimeError(f'Unapproved binary in index: {path}')
            if (ROOT/path).exists() and (ROOT/path).stat().st_size>50*1024**2:raise RuntimeError(f'Unexpected large Git file: {path}')
        if git('diff','--cached','--quiet',check=False).returncode:
            git('commit','-m',message)
        sha=git('rev-parse','HEAD').stdout.strip();state(state='committed',commit=sha)
        if push:
            ensure_room();git('push','-u','origin','main')
            tip=git('ls-remote','origin','refs/heads/main').stdout.split()[0]
            if tip!=sha:raise RuntimeError('Remote tip does not match committed revision')
            state(state='pushed',finished_at=time.time())
        print(json.dumps(status),flush=True)
    except Exception as e:
        state(state='failed',error=str(e),finished_at=time.time())
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=STAGES,required=True)
    p.add_argument('--message',required=True);p.add_argument('--no-push',action='store_true')
    p.add_argument('--approved-push',action='store_true',help='Only after explicit user approval of this completed experiment batch')
    a=p.parse_args();publish(a.stage,a.message,a.approved_push and not a.no_push,a.approved_push)
