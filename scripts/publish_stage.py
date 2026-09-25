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
STAGES={'stage1':'stage1_reproduction','stage2':'stage2_dinov2','ablation':'stage1_memory_ablation','stage3':'stage3_phase_routing','4-1':'stage4_1_appearance_temporal','4-2':'stage4_2_memory_budget','4-3':'stage4_3_process_prior','5-1':'stage5_1_causal','5-2':'stage5_2_lightweight','5-3':'stage5_3_streaming'}


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
    if stage=='stage1':
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
    if stage=='stage1':
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
    text=['# 산업 공정 영상 이상탐지: IPAD 재현과 DINOv2 비교','',
          'R01–R04 실제 공정 영상만 사용합니다. **1단계는 논문 방법론 재현, 2단계는 DINOv2 도입**입니다. 합성 데이터와 LoRA 전이는 이번 실험에서 제외합니다.','',
          '## 단계별 진행','',
          '| 단계 | 목적 | 상태 | 기록 |','|---|---|---|---|',
          '| 1단계 | Swin-T + 주기 메모리 + 재구성 + 주기 검사 | 4개 장면 50 epochs 완료 · seed 0 | [전체 자료](experiments/stage1_reproduction/) |',
          f'| 2단계 | DINOv2 입력–복원 특징 비교 / 비재구성 prototype | {"4개 장면 완료 · seed 0" if (dino/"R04/prototype/completed.json").exists() else "게시 준비 중"} | [전체 자료](experiments/stage2_dinov2/) |',
          '| 1단계 추가 검증 | 메모리 제거 ablation | 4개 장면 완료 · seed 0 | [자료](experiments/stage1_memory_ablation/) |',
          '| 3단계 | 위상 진단 및 불확실성을 고려한 메모리 선택 | R01–R04 × seed 0·1·2 및 시간 진단 완료 | [규약](docs/stage3_protocol.md) · [결과](experiments/stage3_phase_routing/) |','',
          '## 1단계 — 논문 방법론 재현','',
          '장면마다 독립 학습: 16프레임, 256×256, Video Swin-T, 200개 위상, 메모리 2,000개, window 5, Adam 1e-4, batch 8, 50 epochs, FP32, seed 0. 재구성·주기 점수를 장면별 정규화 후 같은 가중치로 결합합니다.','',
          '| 장면 | 논문 AUROC (%) | 구현 AUROC (%) | 차이 (pp) | 평가 프레임 |','|---|---:|---:|---:|---:|',*rows]
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
           '## 자료와 기록 정책','',
           '- `experiments/`: 단계별 설정·epoch 이력·실행 로그·프레임별 정답/점수·평가지표·정렬 민감도.',
           '- `artifacts.jsonl`: 로컬 원본/모델/특징 파일의 경로·크기·SHA256. **바이너리는 GitHub에 업로드하지 않았습니다.**',
           '- 과거 원 모델 stdout은 25배치 간격입니다. 이번 기록 정책 이후의 학습은 매 배치 JSONL을 추가합니다. 기록하지 않은 과거 값을 복원하지 않습니다.',
           '- 최신 사용자 지시에 따라 승인된 실험 전체 완료 시 결과·로그·해시를 main에 자동 게시합니다. 바이너리를 제외하고 원격 변경을 강제로 덮어쓰지 않습니다.',
           '- 디스크 여유가 **10 GiB 이하**가 되면 이 프로젝트의 실험을 일시중지하고 보고합니다. 자동 재개하지 않습니다.',
           '- 메모리 제거 실험은 1단계 추가 검증입니다. 3단계는 2026-09-25 승인받아 정상 위상 진단과 routing 비교부터 진행합니다.',
           '- 테스트 전체 정규화와 미래 프레임을 포함하는 centered window를 사용하므로 온라인/인과적 실시간 성능 주장이 아닙니다.',
           '[기록 규칙](docs/EXPERIMENT_LOG_POLICY.md) · [코드–논문 차이](REPRODUCTION.md) · [3단계 제안](docs/stage3_proposal.md)','',
           '## 원 자료','',
           '- [IPAD 논문 v1](https://arxiv.org/abs/2404.15033v1) · [공식 코드](https://github.com/LJF1113/IPAD), commit `22764cbeeda3946303d236babdd2664fd6241b91`.',
           '- [DINOv2 공식 구현](https://github.com/facebookresearch/dinov2), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.',
           '- upstream 코드의 재배포 대신 출처·SHA256을 보존하고 bootstrap에서 원본을 내려받습니다.','']
    if (ROOT/'experiments/stage3_phase_routing/results.md').exists():
        text += ['## 3단계 — 위상 진단과 선택 방식 비교', '', '[연구 해석·3-seed 요약·시간 진단](experiments/stage3_phase_routing/research_findings.md) · [사전 고정 규약](docs/stage3_protocol.md)', '', (ROOT/'experiments/stage3_phase_routing/results.md').read_text()]
    for st in ['4-1','4-2','4-3']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['', f'## {st} 추가 실험', '', f'[전체 기록](experiments/{STAGES[st]}/) · [고정 규약](docs/stage4_protocol.md)', '', path.read_text()]
    if (ROOT/'experiments/stage4_summary/research_findings.md').exists():
        text += ['', '## 4단계 통합 해석', '', '[세 실험의 통합 보고서와 최종 검산](experiments/stage4_summary/research_findings.md)']
    for st in ['5-1','5-2','5-3']:
        path=ROOT/'experiments'/STAGES[st]/'results.md'
        if path.exists():text += ['', f'## {st} 온라인·경량화 실험', '', f'[전체 기록](experiments/{STAGES[st]}/) · [고정 규약](docs/stage5_protocol.md)', '', path.read_text()]
    if (ROOT/'experiments/stage5_summary/research_findings.md').exists():
        text += ['', '## 5단계 통합 해석', '', '[온라인·경량화·실시간 평가 통합 결과](experiments/stage5_summary/research_findings.md)']
    (ROOT/'README.md').write_text('\n'.join(text))


def publish(stage,message,push=False,approved_push=False):
    os.chdir(ROOT);ensure_room()
    if push and not approved_push:
        raise RuntimeError("Publication requires explicit authorization; approved stage 4/5 experiments have standing user authorization")
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
        estimated=sum(f.stat().st_size for name in paths for f in ([ROOT/name] if (ROOT/name).is_file() else (ROOT/name).rglob('*')) if f.is_file())
        ensure_room(3*estimated)
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
