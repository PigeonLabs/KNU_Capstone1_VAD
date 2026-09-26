"""Publish evidence for fixed-input optimized low-bit paths; no accuracy claims."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import digest
from scripts.summarize_stage9_1 import kernel_audit
BASE=ROOT/'runs/stage9/9-1/optimized'


def load(p):return json.loads(p.read_text())


def main():
    status=load(BASE/'status.json');assert status['state'].startswith('completed'),status
    selection=load(BASE/'frozen_selection.json');data=[];failures=[];inputs={}
    for phase in ['pilot','full']:
        for p in sorted((BASE/phase).glob('*')):
            if not p.is_dir():continue
            config=load(p/'config.json');assert not config['test_labels_used']
            assert config['script_sha256']==digest(ROOT/'scripts/run_stage9_optimized.py')
            assert config['module_sha256']==digest(ROOT/'ipad/stage9_optimized.py')
            if (p/'failure.json').exists():failures.append({'condition':str(p.relative_to(BASE)),**load(p/'failure.json')});continue
            m=load(p/'completed.json');features=list(csv.DictReader((p/'feature_differences.csv').open()));timings=list(csv.DictReader((p/'timings.csv').open()))
            expected=16 if phase=='pilot' else 384
            assert len(features)==expected and len(timings)==expected*3
            keys=[(r['scene'],r['video'],r['frame'],r['input_bf16_sha256']) for r in features]
            if phase not in inputs:inputs[phase]=keys
            else:assert inputs[phase]==keys,'Input mismatch '+str(p)
            operators=[r['name'] for r in csv.DictReader((p/'profiler.csv').open())]
            audit=kernel_audit(operators);write_json(p/'kernel_audit.json',audit)
            m['kernel_audit']=audit;m['path']=str(p.relative_to(BASE));data.append(m)
    full=[m for m in data if m['phase']=='full'];good=[m for m in full if m['numeric_pass']]
    best={}
    for m in good:
        if m['variant'] not in best or m['timings']['model_ms']['mean']<best[m['variant']]['timings']['model_ms']['mean']:best[m['variant']]=m
    rows=[]
    for m in data:
        rows.append({'phase':m['phase'],'variant':m['variant'],'mode':m['mode'],'samples':m['samples'],'model_ms':m['timings']['model_ms']['mean'],'total_ms':m['timings']['total_ms']['mean'],'total_p95_ms':m['timings']['total_ms']['p95'],'numeric_pass':m['numeric_pass'],'same_eager_patch_max_abs':m['numeric']['same_eager_patch12_max_abs'],'fp32_patch_cosine':m['numeric']['fp32_patch12_cosine_mean'],'peak_allocated_mib':m['peak_allocated_mib'],'payload_mib':m['payload_bytes']/2**20,'startup_seconds':m['startup_seconds']})
    from ipad.stage7 import write_csv
    write_csv(BASE/'summary.csv',rows)
    lines=['# 9-1 INT8·INT4 최적화: 동일 전처리와 CUDA Graph 검증','',
      '기존 잘못된 커널 선택과 불필요한 CPU 동기화를 수정하고, BF16에도 똑같은 최적화 탐색을 적용했습니다. 이 결과는 정상 데이터에서의 실행 최적화입니다. 전체 이상탐지 AUROC·경보 성능은 평가하지 않았습니다.',
      '', '## 검증 방법','',
      '- R01–R04 고정 정상384frame, seed0, batch1. eager FP32 RGB/resize/정규화를 모든 경로에 고정했습니다. 프레임별 BF16 백본 입력 SHA256이 모두 일치합니다.',
      '- 5경로 × eager / eager CUDA Graph / compile CUDA Graph / max-autotune CUDA Graph =20개 pilot. 균등16frame에서 사전 수치선(동일 eager 대비 CLS/patch cosine평균≤1e-3, 최대절대차이≤.05)을 통과한 가장 빠른 모델 실행 방식을 고정한 뒤384frame×3회 검증했습니다. full 실패시 사전 지정 graph를 추가 확인합니다.',
      '- FP32 원가중치+FP32입력(TF32off), BF16 eager, 동일 양자화 eager의 세 기준을 기록했습니다. 같은 eager와 일치해도 양자화 자체가 정확도를 보존한다는 의미는 아닙니다.',
      '- 아래 total은 JPEG읽기·전처리·BF16변환·백본을 포함합니다. 전처리 후 동기화를 동일하게 적용해 model 구간과 분리했습니다. 기존 전처리까지 컴파일한 표와 직접 비교하지 않습니다.',
      '', '## 전체384frame 수치검사를 통과한 경로','',
      '| 경로 | 실행 방식 | 백본 평균 ms | 전처리 포함 평균 ms | 전체 p95 ms | 가중치payload MiB | GPU peak MiB |','|---|---|---:|---:|---:|---:|---:|']
    for v,m in best.items():
        lines.append(f"| {v} | {m['mode']} | {m['timings']['model_ms']['mean']:.3f} | {m['timings']['total_ms']['mean']:.3f} | {m['timings']['total_ms']['p95']:.3f} | {m['payload_bytes']/2**20:.2f} | {m['peak_allocated_mib']:.2f} |")
    if not best:lines.append('| 수치검사 통과 없음 | — | — | — | — | — | — |')
    lines+=['','| 경로 | 동일 eager patch 최대차이 | FP32 대비 patch cosine 평균 | BF16 eager 대비 patch cosine 평균 |','|---|---:|---:|---:|']
    for v,m in best.items():
        n=m['numeric'];lines.append(f"| {v} | {n['same_eager_patch12_max_abs']:.6f} | {n['fp32_patch12_cosine_mean']:.6f} | {n['bf16_eager_patch12_cosine_mean']:.6f} |")
    fastest={}
    for m in full:
        if m['variant'] not in fastest or m['timings']['model_ms']['mean']<fastest[m['variant']]['timings']['model_ms']['mean']:fastest[m['variant']]=m
    lines+=['','## 가장 빠른 전체384frame 경로: 수치 실패 포함 진단','',
        '다음 표는 수치 기준과 무관하게 가장 빠른 normal pilot을 추가384frame에서 검증한 결과입니다. **수치 초과 경로는 동등한 대체 모델로 채택하지 않습니다.**',
        '', '| 경로 | 방식 | 백본 ms | 총 ms | payload MiB | peak MiB | 수치 기준 |', '|---|---|---:|---:|---:|---:|---|']
    for v,m in fastest.items():
        lines.append(f"| {v} | {m['mode']} | {m['timings']['model_ms']['mean']:.3f} | {m['timings']['total_ms']['mean']:.3f} | {m['payload_bytes']/2**20:.2f} | {m['peak_allocated_mib']:.2f} | {'통과' if m['numeric_pass'] else '초과'} |")
    lines+=['','## 무엇을 최적화했나','',
      '- W8A16: Int8WeightOnlyConfig 가중치를 유지하고, 현재 GPU에서 느린 weight_int8pack_mm으로 치환하는 compiler 패턴을 프로세스 내부에서 비활성화했습니다. 단순 BF16 원가중치 복원 모델로 바꾼 것이 아닙니다.',
      '- W8A8: 실제 INT8 정수 GEMM을 유지하고 FakeTensor 감지의 GPU 텐서 문자열 처리를 타입 검사로 교체했습니다. 설치된 라이브러리 파일은 변경하지 않았습니다.',
      '- W4 native: 기존 tinygemm group128 커널에도 동일한 CUDA Graph와 autotune 기회를 적용했습니다.',
      '- W4 packed: 원래 group128 INT4 q/scale/zero 값을 유지한 nibble packing. 48층 모두 padded tinygemm과 같은 양자화 값을 확인했습니다. 768→1024 패딩을 피하고 필요할 때 BF16으로 복원해 GEMM을 수행합니다. **INT4 저장/BF16 계산**이며 순수 INT4 Tensor Core 연산 가속으로 주장하지 않습니다.',
      '- eager CUDA Graph는 기존 eager 연산을 캡처하여 CPU 호출 비용을 줄이고 compiler의 추가 수치 변경을 피하는 후보입니다. compile/max-autotune 실패나 수치 초과는 숨기지 않고 아래에 기록했습니다. 원본 경로와 패키지 파일은 보존했습니다.',
      '- payload 절감과 실제GPU peak는 다릅니다. CUDA Graph가 복원용 임시 버퍼를 유지할 수 있으므로 압축모델 크기만으로 실행 메모리 절감을 주장하지 않습니다.',
      '', '## 전체 탐색 기록','', '| 단계 | 경로 | 모드 | 백본 ms | 총 ms | 수치 기준 | patch 최대차이 | 첫 실행초 |','|---|---|---|---:|---:|---|---:|---:|']
    for r in rows:lines.append(f"| {r['phase']} | {r['variant']} | {r['mode']} | {r['model_ms']:.3f} | {r['total_ms']:.3f} | {'통과' if r['numeric_pass'] else '초과'} | {r['same_eager_patch_max_abs']:.6f} | {r['startup_seconds']:.2f} |")
    if failures:
        lines+=['','## 실행 실패','']
        for r in failures:lines.append(f"- {r['condition']}: 원인과 traceback은 해당 failure.json 및 로그에 보존했습니다. 완료로 집계하지 않습니다.")
    layer_file=BASE/'layer_diagnostic/completed.json'
    if layer_file.exists():
        layer=load(layer_file)
        lines+=['','## 중간층·FP32 기준 진단','',
            '각 장면 첫 정상샘플4개에서 FP32/BF16/compiled 중간 출력을 비교했습니다. token 준비 이후 Transformer 블록에서도 수치 차이가 관찰됐습니다. 위치 임베딩을 eager에서 미리 계산한 대조군은 eager 출력을 바꾸지 않았지만 compiler 차이를 없애지는 못했습니다. 중간값 반환 자체가 fusion을 바꿀 수 있어 마지막 token 출력과 실사용 백본 경로의 결과를 동일하다고 간주하지 않습니다.',
            '', '[중간층 차이 원자료](layer_diagnostic/layer_differences.csv) · [고정 위치 임베딩 대조군](layer_diagnostic/completed.json)']
        assert layer['state']=='completed'
        assert all(r['fixed_position_eager_change']['max_abs']==0 for r in layer['fixed_position_controls'])
    if (BASE/'baseline_recheck_status.json').exists():
        assert load(BASE/'baseline_recheck_status.json')['state']=='completed'
        lines+=['','BF16 autotune 최초 시도의 compiler 초기화 구간에 별도 GPU 회귀검사가 겹쳤을 가능성을 발견해, 해당 기록을 restarts 아래 보존하고 모든 GPU 작업 종료 후 새 compiler cache로 단독 재측정했습니다. 표에는 재측정값만 사용합니다. [측정 감사](measurement_review.json)']
    lines+=['','## 제한 및 재현','',
      '- 이 결과만으로 이상탐지 성능 유지나 전체 VAD30FPS를 주장하지 않습니다. 중간층 진단은 추가 원인 자료이며 수치선이나 양자화 설정을 사후 변경하는 데 쓰지 않았습니다.',
      '- 현재 RTX PRO 6000과 desktop GPU 공유 환경에서의 측정입니다. pilot 선택 편향을 줄이기 위해384frame에서 재측정했지만, 독립 날짜/장비 성능 보증은 아닙니다.',
      '- 실제커널 profiler.csv, 입력/특징차이 feature_differences.csv, 3반복 timings.csv, 환경/config, compiler SHA 목록은 경로별 보존합니다. 원본영상·가중치·컴파일binary는 게시하지 않습니다.',
      '', '```bash', '.venv/bin/python scripts/run_stage9_optimized.py --suite', '.venv/bin/python scripts/summarize_stage9_optimized.py', '```', '', '기존 결과가 있으면 덮어쓰기를 거부합니다. [사전 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/stage9_optimization_protocol.md)']
    (BASE/'results.md').write_text('\n'.join(lines)+'\n')
    write_json(BASE/'verification.json',{'state':'passed','input_bytes_identical':True,'test_labels_used':False,'full_validated':{v:m['mode'] for v,m in best.items()},'full_fast_diagnostic':{v:m['mode'] for v,m in fastest.items()},'successful_conditions':len(data),'failed_conditions':[f['condition'] for f in failures],'numeric_failed_conditions':[m['path'] for m in data if not m['numeric_pass']],'accuracy_evaluated':False})
    print(json.dumps(load(BASE/'verification.json')))


if __name__=='__main__':main()
