"""Verify fixed conditions and report negative/failed compile outcomes honestly."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ipad.common import write_json
from ipad.phase_routing import digest
from ipad.stage5 import read_csv
from scripts.run_stage9_1 import VARIANTS,MODES


def main():
    out=ROOT/'runs/stage9/9-1';data={};failures=[];samples=None;numeric_failures=[];rows=[]
    modes=list(MODES)
    extra='reduce-overhead-precise'
    if all(any((out/f'{v}_{extra}'/f).exists() for f in ['completed.json','failure.json']) for v in VARIANTS):modes.append(extra)
    for mode in modes:
        for variant in VARIANTS:
            key=f'{variant}_{mode}';p=out/key;config=json.loads((p/'config.json').read_text())
            if samples is None:samples=config['samples']
            assert config['samples']==samples and config['test_labels_used'] is False
            if (p/'failure.json').exists():failures.append({'condition':key,**json.loads((p/'failure.json').read_text())});continue
            m=json.loads((p/'completed.json').read_text());data[variant,mode]=m
            operators=json.loads((p/'operators.json').read_text())
            audit={'integer_gemm':any(name=='aten::_int_mm' or 'gemm_s8' in name.lower() for name in operators),
                   'weight_only_int8_kernel':any('_weight_int8pack_mm' in name for name in operators),
                   'packed_int4':any('_weight_int4pack_mm' in name or 'tinygemm' in name for name in operators),
                   'cuda_graph_api':any('cudagraph' in name.lower() or 'cugraph' in name.lower() for name in operators),
                   'note':'Reclassified from raw profiler names: CompiledFxGraph is not CUDA Graph; weight_int8pack_mm is not proof of integer GEMM.'}
            write_json(p/'kernel_audit.json',audit);m['audited_kernel_evidence']=audit
            assert m['samples']==384 and m['timed_calls']==1152 and m['repeats']==3
            t=read_csv(p/'timings.csv');f=read_csv(p/'feature_differences.csv');assert len(t)==1152 and len(f)==384
            ids=[(r['scene'],r['video'],r['frame']) for r in samples]
            assert [(r['scene'],r['video'],int(r['frame'])) for r in f]==ids
            for rep in range(3):assert [(r['scene'],r['video'],int(r['frame'])) for r in t if int(r['repeat'])==rep]==ids
            assert np.isclose(np.mean([float(r['total_ms']) for r in t]),m['timings']['total_ms']['mean'])
            assert all(np.isfinite(float(r['total_ms'])) and float(r['total_ms'])>0 for r in t)
            if not m['numeric_pass']:numeric_failures.append(key)
            assert not m['compiler_stats'].get('graph_break',{}),m['compiler_stats']
            if mode!='eager':assert m['compiler_stats'].get('stats',{}).get('unique_graphs',0)>=1
            rows.append({'variant':variant,'mode':mode,'total_ms':m['timings']['total_ms']['mean'],'p95_ms':m['timings']['total_ms']['p95'],'model_ms':m['timings']['model_wall_ms']['mean'],
                'gpu_interval_ms':m['timings']['gpu_interval_ms']['mean'],'peak_allocated_mib':m['peak_allocated_mib'],'first_call_seconds':m['first_call_seconds'],'numeric_pass':m['numeric_pass']})
    from ipad.stage7 import write_csv
    write_csv(out/'summary.csv',rows)
    lines=['# 9-1 실제 양자화 연산 경로와 컴파일 최적화','',
        '9단계는 양자화이며 기존 8단계 LoRA와 별개입니다. 고정 ViT-B/14, R01–R04 정상 학습 영상의 동일 384프레임, batch 1, 3회 반복. 아래 시간은 JPEG 읽기·전처리·전송·백본·CUDA 동기화를 포함하며 위상 예측기·메모리·경보는 제외합니다. 전체 VAD 성능과 AUROC는 이번 단계에서 측정하지 않았습니다.','',
        '| 경로 | 실행 | 평균 ms | p95 ms | 해당 eager 대비 속도 | 같은 실행 BF16 대비 속도 | GPU peak MiB | 초기 호출 초 | 수치 검사 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for mode in modes:
        for variant in VARIANTS:
            m=data.get((variant,mode))
            if m is None:lines.append(f'| {variant} | {mode} | 실행 실패 | — | — | — | — | — | 미검증 |');continue
            avg=m['timings']['total_ms']['mean'];eager=data.get((variant,'eager'));bf=data.get(('bf16',mode))
            speed=f"{eager['timings']['total_ms']['mean']/avg:.2f}×" if eager else '—';rel=f"{bf['timings']['total_ms']['mean']/avg:.2f}×" if bf else '—'
            lines.append(f"| {variant} | {mode} | {avg:.3f} | {m['timings']['total_ms']['p95']:.3f} | {speed} | {rel} | {m['peak_allocated_mib']:.1f} | {m['first_call_seconds']:.2f} | {'통과' if m['numeric_pass'] else '기준 초과'} |")
    lines+=['','속도 배율은 클수록 빠릅니다. 양자화·컴파일 효과를 분리하기 위해 양자화 compiled를 BF16 eager와만 비교하지 않습니다. 초기 호출은 새 조건 캐시에서의 모델 compile 호출 비용이며 eager 참조 계산과 GPU 초기화 이후입니다. warmup 12회와 초기/정상 peak 메모리는 원 JSON에 별도 기록했습니다.','',
        '## 실제 연산 경로','',
        '| 조건 | 정수 GEMM 연산/커널 흔적 | packed INT4 흔적 | CUDA Graph 흔적 |', '|---|---|---|---|']
    for (v,mode),m in data.items():
        e=m['audited_kernel_evidence'];lines.append(f"| {v}/{mode} | {'확인' if e['integer_gemm'] else '이름에서 미확인'} | {'확인' if e['packed_int4'] else '없음'} | {'확인' if e['cuda_graph_api'] else '이름에서 미확인'} |")
    lines+=['','원시 kernel_evidence 필드의 단순 이름 검색은 CompiledFxGraph를 CUDA Graph로, weight_int8pack_mm을 정수 GEMM으로 오인할 수 있어 kernel_audit.json에서 다시 구분했습니다. 원 기록은 보존했습니다. 연산·커널 이름은 profiler.csv, 생성된 코드의 호출/자료형 발췌는 generated_kernel_evidence.json, 전체 로컬 생성물 SHA256은 generated_cache_hashes.json에 있습니다. profiler는 별도 5프레임 실행이며 시간 측정에 포함하지 않았습니다. 컴파일로 연산이 합쳐져 이름이 바뀐 경우 이름 미검출만으로 해당 정밀도 연산이 없다고 단정하지 않습니다.','',
        '## 특징 수치 변화','',
        '| 조건 | 동일 경로 eager 대비 patch cosine 차이 | 동일 eager 대비 patch 최대 오차 | BF16 eager 대비 patch cosine 차이 |', '|---|---:|---:|---:|']
    for (v,mode),m in data.items():
        n=m['numeric'];lines.append(f"| {v}/{mode} | {n['same_eager_patch12_cosine_mean']:.7f} | {n['same_eager_patch12_max_abs']:.6f} | {n['bf16_patch12_cosine_mean']:.6f} |")
    lines+=['','수치 검사선은 동일 양자화 eager 대비 CLS와 patch 모두 평균 cosine 거리≤1e-3, 최대절대차이≤0.05입니다. BF16 대비 양자화 손실과 동일 양자화에서 compile에 의한 차이를 구분합니다. 정상 특징 수치 검사가 통과해도 이상탐지 AUROC·경보 동등성을 보장하지 않습니다.']
    if numeric_failures:lines+=['',f"수치 검사 기준 초과 조건: {', '.join(numeric_failures)}. 속도가 빨라도 동등한 대체 모델로 채택하지 않습니다."]
    if failures:
        lines+=['','## 실행 실패','']
        for f in failures:lines.append(f"- {f['condition']}: `{f['error']}`. 원 traceback과 부분 산출물을 보존했습니다.")
    lines+=['','## 해석과 재현','',
        '- 초기 12개 조건을 고정했고, 정상 특징 수치 차이를 관찰한 뒤 BF16 중간 반올림 보존 대조군 4개를 추가했습니다. 추가 조건은 reduce-overhead-precise이며 네 연산 경로에 동일하게 적용했습니다. calibration/test 영상을 사용하지 않았습니다. 모델 재학습과 전체 VAD 임계값 선택은 수행하지 않았습니다.',
        '- BF16/weight-only 경로보다 INT8 정수 행렬곱이 빠르더라도 활성값 양자화·스케일 보정·커널 호출·입출력 비용을 포함한 전체 시간을 함께 비교합니다.',
        '- default와 reduce-overhead를 같은 백본/입력에 적용했습니다. 후자는 CUDA Graph를 요청하며 실제 적용은 로그/커널 자료로 확인합니다. 실행 중 graph break/eager fallback은 허용하지 않았습니다.',
        '- 모델 가중치 저장 크기와 compiler buffer/워크스페이스를 포함하는 runtime GPU peak는 다릅니다. compiled가 더 많은 메모리를 쓸 수 있습니다. GPU 구간시간도 커널 호출 사이의 빈 시간을 포함하므로 순수 연산시간 합계가 아닙니다.',
        '- 현재 GPU, 정상 프레임, warm OS cache 가능 조건입니다. 실제 카메라·네트워크·전체 VAD·다른 장비의 실시간성을 주장하지 않습니다.',
        '- compiler worker 수는2, torch CPU thread는8로 제한했습니다. 10 GiB 이하 중지, 자동 재개 금지, 원본/기존 결과 보존. 가중치/영상/컴파일 binary는 로컬에만 보존합니다.','',
        '```bash', '.venv/bin/python scripts/run_stage9_1.py', '.venv/bin/python scripts/summarize_stage9_1.py','```','',
        '[고정 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/stage9_protocol.md) · [TorchAO 연산 경로 문서](https://docs.pytorch.org/ao/stable/workflows/inference.html)']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    verification={'artifact_audit_passed':True,'planned_primary_conditions':12,'included_precision_controls':4 if len(modes)==4 else 0,'successful_conditions':len(data),'failed_conditions':[r['condition'] for r in failures],
        'numeric_failed_conditions':numeric_failures,'input_frames':len(samples),'same_input_hashes':True,'timed_calls_per_condition':1152,'test_labels_used':False,
        'protocol_sha256':digest(ROOT/'docs/stage9_protocol.md')}
    write_json(out/'verification.json',verification);print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
