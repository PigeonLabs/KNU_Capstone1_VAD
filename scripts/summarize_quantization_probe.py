"""Verify the normal-only quantization probe and write its bounded conclusion."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ipad.common import write_json
from ipad.stage5 import read_csv
from ipad.phase_routing import digest
import numpy as np


def main():
    root=ROOT/'runs/quantization_probe';variants=['bf16','int8','int4'];data=[];configs=[]
    for variant in variants:
        p=root/variant;r=json.loads((p/'completed.json').read_text());cfg=json.loads((p/'config.json').read_text());configs.append(cfg)
        assert r['state']=='diagnostic_complete' and cfg['test_labels_used'] is False
        assert r['quantized_linear_layers']==(0 if variant=='bf16' else 48)
        timings=read_csv(p/'timings.csv');features=read_csv(p/'feature_differences.csv')
        assert len(timings)==288 and len(features)==96
        assert [(v['video'],int(v['frame'])) for v in features]==[(v['video'],v['frame']) for v in cfg['samples']]
        assert np.isclose(np.mean([float(v['processing_ms']) for v in timings]),r['mean_ms'])
        assert all(np.isfinite(float(v['patch12_cosine_distance'])) for v in features)
        data.append(r)
    assert all(c['samples']==configs[0]['samples'] for c in configs)
    ops=json.loads((root/'int4/profiler_operators.json').read_text());assert 'aten::_weight_int4pack_mm' in ops
    baseline=data[0];lines=['# 8비트·4비트 DINOv2 양자화 예비 검증','',
        '**정상 영상 기반 구현 진단 완료. 전체 VAD 성능 실험은 아직 수행하지 않았습니다.** 고정 ViT-B/14, R01 정상 학습 영상 3개의 96프레임, batch 1·3회 반복. 동일 입력/전처리, JPEG 읽기·백본·GPU 동기화 포함. 위상 예측기·프로토타입·시간 검사는 제외합니다.','',
        '| 방식 | 백본 직렬화 크기 MiB | BF16 대비 감소 | GPU peak allocated MiB | 평균 처리 ms | p95 ms | 평균 patch cosine 차이 |','|---|---:|---:|---:|---:|---:|---:|']
    for r in data:
        reduction=(1-r['backbone_serialized_bytes']/baseline['backbone_serialized_bytes'])*100
        lines.append(f"| {r['variant']} | {r['backbone_serialized_bytes']/2**20:.2f} | {reduction:.1f}% | {r['peak_allocated_mib']:.2f} | {r['mean_ms']:.3f} | {r['p95_ms']:.3f} | {r['patch_cosine_distance_mean']:.6f} |")
    lines+=['','## 확인된 사항','',
        '- INT8·INT4 모두 48개 Linear 가중치의 실제 변환과 유한한 출력을 확인. 정규화/attention/patch embedding은 낮은 비트로 바꾸지 않았습니다.',
        '- INT8은 저비트 가중치 저장 후 BF16 행렬곱을 사용하는 경로입니다. 전체 8비트 정수 연산 가속으로 해석하지 않습니다. INT4는 profiler에서 aten::_weight_int4pack_mm/tinygemm packed weight kernel을 확인했습니다.',
        '- 현재 eager 실행은 압축과 GPU allocated 메모리 절감에 성공했으나 BF16보다 느립니다. torch.compile 미적용이며 컴파일·활성값 양자화 최적화 결과를 미리 주장하지 않습니다.',
        '- INT4 group128에서 특징 변화가 커졌지만 정상 96프레임의 특징 차이만으로 AUROC 손실을 추정할 수 없습니다. INT8도 이상탐지 성능 유지가 검증된 것은 아닙니다.',
        '- 짧은 정상 프레임 진단으로 end-to-end VAD FPS/30FPS 마감시간 보장/다른 장비 실시간을 주장하지 않습니다. 다른 프로세스와 GPU 클럭의 영향이 가능합니다.',
        '- 사전학습 가중치·원본·기존 결과를 보존합니다. 새로운 전체 patch 캐시는 없습니다. TorchAO 0.17.0은 별도 cache 경로에 설치했고 기존 PyTorch는 변경하지 않았습니다. 원본·패키지·가중치는 GitHub에 게시하지 않습니다.','',
        '## 권장 본 실험 범위 (아직 미실행)','',
        '1. 고정 DINOv2 BF16 ↔ INT8 weight-only ↔ INT8 weight/activation. 양자화된 백본의 특징으로 프로토타입·위상 예측기를 재구축하고 정상 보정을 다시 수행. 기존 위상 예측기·메모리를 그대로 사용하는 직접 변환 결과도 구분해 기록.',
        '2. INT4 group128/32 및 정상 특징 오차로 사전 선정한 민감층 BF16 혼합. 테스트 AUROC로 비트 수·층을 선택하지 않으며, 메모리 압축 효과와 특징 손실을 분리.',
        '3. R01–R04 동일 분할·유효 프레임·3개 seed의 AUROC/AUPRC, 활성 FPR, 구간 recall 및 실제 batch 1·30 FPS 측정. compile 최적화는 BF16 대조군에도 동일하게 적용하고 초기 compile 시간을 별도로 보고. 전체 가중치의 중복 저장을 피하고 10 GiB 안전마진 유지.','',
        '## 재현','',
        '```bash',
        'uv pip install --python .venv/bin/python --target cache/quantization/torchao017 --no-deps --no-cache torchao==0.17.0',
        '.venv/bin/python scripts/probe_quantization.py --variant bf16',
        '.venv/bin/python scripts/probe_quantization.py --variant int8',
        '.venv/bin/python scripts/probe_quantization.py --variant int4',
        '.venv/bin/python scripts/summarize_quantization_probe.py','```','',
        '[TorchAO公式推론文서](https://docs.pytorch.org/ao/stable/workflows/inference.html) · [예비 실행 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/quantization_probe.md)']
    (root/'results.md').write_text('\n'.join(lines).replace('TorchAO公式推론文서','TorchAO 공식 추론 문서')+'\n')
    write_json(root/'verification.json',{'passed':True,'variants':variants,'normal_frames':96,'timed_calls_per_variant':288,'test_evaluation':False,'same_input_hashes':True,'packed_int4_op_verified':True,
        'source_hashes':{str(p.relative_to(ROOT)):digest(p) for p in [ROOT/'scripts/probe_quantization.py',ROOT/'scripts/summarize_quantization_probe.py',ROOT/'docs/quantization_probe.md']}})
    print((root/'results.md').read_text())


if __name__=='__main__':main()
