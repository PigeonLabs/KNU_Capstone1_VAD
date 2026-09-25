"""Preserve the original stage 9-1 data and add a causal-audit correction."""
import csv
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.phase_routing import digest


def load(p):return json.loads(p.read_text())


def main():
    out=ROOT/'runs/stage9/9-1/audit'
    required=['repr','numeric','woq_original','woq_bypass']
    data={name:load(out/name/'completed.json') for name in required}
    for name in required:
        config=load(out/name/'config.json')
        assert data[name]['state']=='completed' and config['test_labels_used'] is False
        assert config['script_sha256']==digest(ROOT/'scripts/diagnose_stage9.py')
    a=data['repr'];rows=data['numeric']['rows']
    exact=max(v['max_abs'] for name in ['original_before','type_check_only','original_restored'] for r in a[name]['differences'] for v in r.values())
    assert exact==0
    assert max(r['eager_repeat_patch12_max_abs'] for r in rows)==0
    assert max(r['eager_split_patch12_max_abs'] for r in rows)==0
    def ops(kind):return list(csv.DictReader((out/kind/'profiler.csv').open()))
    original_ops=ops('woq_original');bypass_ops=ops('woq_bypass')
    original_count=sum(int(r['count']) for r in original_ops if r['name']=='aten::_weight_int8pack_mm')
    bypass_count=sum(int(r['count']) for r in bypass_ops if r['name']=='aten::_weight_int8pack_mm')
    assert original_count>0 and bypass_count==0
    lines=['# 9-1 이상값 재검증 및 해석 정정','',
      '사용자의 이상값 지적 후 실시한 정상 데이터 원인 분리 감사입니다. 기존16조건의 원시 결과와 수치 검사 실패는 보존합니다. 최초 표는 해당 라이브러리·컴파일 경로에서의 관측치이며, INT8의 본질적 속도나 유효한 최적 구현의 비교로 해석하면 안 됩니다.',
      '', '## 1. W8A8 eager의 불필요한 텐서 문자열 처리','',
      'TorchAO safe_int_mm의 `"FakeTensor" in input.__repr__()`만 `isinstance(input, FakeTensor)`로 교체했습니다. 파일은 바꾸지 않고 진단 프로세스 내부에서만 교체한 뒤 원복했습니다. 실제 INT8 GEMM/양자화/가중치/입력은 유지했습니다.',
      '', '| 순서 | 모델 시간 평균 (ms) | 3회 profiler의 scalar 추출 횟수 | 원본 대비 특징 최대 절대차이 |','|---|---:|---:|---:|']
    for name,title in [('original_before','원본 A'),('type_check_only','타입 검사 B'),('original_restored','원복 A')]:
        pr=list(csv.DictReader((out/'repr'/(name+'_profiler.csv')).open()))
        count=sum(int(r['count']) for r in pr if r['name']=='aten::_local_scalar_dense')
        delta=max(v['max_abs'] for r in a[name]['differences'] for v in r.values())
        lines.append(f"| {title} | {a[name]['mean_ms']:.3f} | {count} | {delta} |")
    lines+=['','불필요한 동기화 비용의 인과적 기여를 확인했습니다. 원복값의 차이는 실행 상태 변동도 있음을 보여주므로 이 소표본에서 정확한 절감 비율을 일반화하지 않습니다. 기존 53→2ms 차이는 컴파일의 효율만이 아니라 이 라이브러리 경로 회피도 포함합니다.',
      '', '## 2. W8A16의 느린 컴파일 패턴 선택','',
      '동일한 Int8WeightOnlyConfig 가중치와 BF16 활성값으로, 별도 프로세스에서 `_register_woq_lowerings` 등록만 생략했습니다. 두 조건 모두 fullgraph=True, dynamic=False, emulate_precision_casts=True이며 CUDA Graph는 사용하지 않았습니다. 내부 API 우회는 원인 확인용이며 배포용 수정이 아닙니다.',
      '', '| 조건 | 모델 시간 평균 (ms) | 3회 profiler의 weight_int8pack_mm 호출수 | eager 대비 patch 최대차이 |','|---|---:|---:|---:|']
    for name,count in [('woq_original',original_count),('woq_bypass',bypass_count)]:
        m=data[name];err=max(r['patch12']['max_abs'] for r in m['differences'])
        lines.append(f"| {name} | {m['mean_ms']:.3f} | {count} | {err:.6f} |")
    lines+=['','생성 코드의 weight_int8pack_mm 입력은 BF16입니다. CUDA 커널의 float 포인터 서명은 내부 처리 경로의 근거이며, 모델 전체가 FP32로 변경됐다는 뜻은 아닙니다. 원래 커널의 느린 시간은 측정 잡음만으로 설명할 수 없지만, INT8 가중치 방식 전체에 대한 일반적인 결론도 아닙니다.',
      '', '## 3. 특징 이상값: 전처리와 백본을 분리','',
      '기존 BF16 precise patch 최대오차 상위8개 + 균등8개 정상 프레임의 진단 표본입니다. 대표 성능 평균을 산정하기 위한 표본이 아닙니다. eager 반복과 eager 분할은 정확히 동일하며, CPU로 즉시 출력을 복사해 결과 버퍼 재사용을 배제했습니다.',
      '', '| 경로 | patch 최대 절대차이 | patch cosine 거리 평균 |','|---|---:|---:|']
    for name,title in [('eager_repeat','동일 eager 반복'),('eager_split','eager 전처리·백본 분리'),('whole_compile','전체 컴파일'),('body_compile_same_input','동일 전처리 입력 + 백본만 컴파일'),('prep_compile_only','전처리만 컴파일 + eager 백본')]:
        lines.append(f"| {title} | {max(r[name+'_patch12_max_abs'] for r in rows):.6f} | {sum(r[name+'_patch12_cosine_mean'] for r in rows)/len(rows):.8f} |")
    lines += ['', f"BF16 전처리 텐서의 최대 성분차이는 {max(r['prep_max_abs'] for r in rows):.7f}, 프레임별 달라진 성분 비율의 최댓값은 {max(r['prep_changed_fraction'] for r in rows)*100:.5f}%입니다. 작은 입력 차이가 일부 특징에서 크게 증폭됐으며, 전처리를 고정해도 백본 컴파일의 차이가 남았습니다. 어느 경로가 참값에 더 가까운지는 이 비교만으로 결정할 수 없습니다. FP32 고정 입력·중간층 검증이 추가로 필요합니다.",
      '', '## 해석 정정과 후속 조건','',
      '- 9-1의 최초 측정치는 폐기하지 않고 **구현 경로 진단 결과**로 유지합니다. 수치 동등성이 확보된 양자화 성능 비교 또는 배포 후보 확정 결과가 아닙니다.',
      '- INT8 느림에는 확인된 불필요한 라이브러리 처리와 부적합한 커널 선택이 포함됩니다. 두 경로를 정리한 뒤 동일한 전처리·정밀도·측정 범위로 대표384frame 재측정이 필요합니다.',
      '- 특징 최대오차 초과를 단순 반올림으로 무시하거나 탐지 성능 하락으로 확정하지 않습니다. 사전 허용선은 유지하며, 전체 VAD 정확도는 아직 평가하지 않았습니다.',
      '- 이번 타이밍은 입력을 GPU에 미리 올린 model-only, 정상16frame×3회입니다. 최초 JPEG포함384frame 표와 직접 비교하지 않습니다. desktop GPU 공유, A-B-A 변동, 작은 진단 표본이라는 제약이 있습니다.',
      '- 실제 테스트 라벨/학습/모델 선택은 사용하지 않았습니다. 원본과 패키지 파일은 보존했습니다. 프로파일링은 타이밍 측정 밖에서 실시했습니다.',
      '', '```bash', 'TORCHINDUCTOR_CACHE_DIR="$PWD/cache/stage9_audit/<kind>/inductor" TRITON_CACHE_DIR="$PWD/cache/stage9_audit/<kind>/triton" TORCHINDUCTOR_COMPILE_THREADS=2 .venv/bin/python scripts/diagnose_stage9.py --kind <kind>', '# kind: repr, numeric, woq_original, woq_bypass', '```', '', '원본 진단 디렉터리는 덮어쓰지 않습니다. 재실행하려면 별도 실행 디렉터리를 지정하도록 코드를 확장해야 합니다.']
    (out/'results.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'verification.json',{'state':'passed','diagnoses':required,'repr_features_exact':exact==0,'woq_original_calls':original_count,'woq_bypass_calls':bypass_count,'original_results_preserved':True,'test_labels_used':False,'numeric_equivalence_resolved':False,'diagnostic_source_sha256':digest(ROOT/'scripts/diagnose_stage9.py'),'report_source_sha256':digest(__file__)})
    root=ROOT/'runs/stage9/9-1/results.md';old=root.read_text();marker='\n## 사용자 지적 후 재검증 및 해석 정정\n'
    if marker not in old:
        root.write_text(old+marker+'\n아래 원인 분리 감사에서 불필요한 TorchAO 텐서 문자열 처리와 느린 WOQ 커널 선택을 확인했습니다. 최초 속도 표는 구현 경로 진단값이며, 유효한 최적 INT8 구현의 성능 비교로 해석하지 않습니다. 전처리/백본 컴파일 양쪽에서 특징 차이가 재현돼 수치 동등성은 미해결입니다. [재검증 결과·정정 상세](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/experiments/stage9_1_quant_compile/audit/results.md)\n')
    current=root.read_text()
    notice='> **해석 정정:** 사용자 지적 후 재검증에서 라이브러리의 불필요한 처리와 느린 커널 선택을 확인했습니다. 아래 최초 측정은 구현 진단값이며 최적 양자화 성능 비교가 아닙니다. [원인 분리 감사](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/experiments/stage9_1_quant_compile/audit/results.md)를 먼저 확인하세요.\n'
    if notice not in current:
        title,rest=current.split('\n',1)
        root.write_text(title+'\n\n'+notice+rest)
    print(json.dumps(load(out/'verification.json')))


if __name__=='__main__':main()
