# 9단계 사전 진단 — INT8·INT4 DINOv2 양자화

**정상 영상 기반 구현 진단 완료. 전체 VAD 성능 실험은 아직 수행하지 않았습니다.** 고정 ViT-B/14, R01 정상 학습 영상 3개의 96프레임, batch 1·3회 반복. 동일 입력/전처리, JPEG 읽기·백본·GPU 동기화 포함. 위상 예측기·프로토타입·시간 검사는 제외합니다.

| 방식 | 백본 직렬화 크기 MiB | BF16 대비 감소 | GPU peak allocated MiB | 평균 처리 ms | p95 ms | 평균 patch cosine 차이 |
|---|---:|---:|---:|---:|---:|---:|
| bf16 | 165.20 | 0.0% | 189.20 | 3.909 | 4.294 | 0.000000 |
| int8 | 84.47 | 48.9% | 112.48 | 5.697 | 6.225 | 0.000950 |
| int4 | 55.81 | 66.2% | 82.15 | 6.023 | 6.718 | 0.126972 |

## 확인된 사항

- INT8·INT4 모두 48개 Linear 가중치의 실제 변환과 유한한 출력을 확인. 정규화/attention/patch embedding은 낮은 비트로 바꾸지 않았습니다.
- INT8은 저비트 가중치 저장 후 BF16 행렬곱을 사용하는 경로입니다. 전체 8비트 정수 연산 가속으로 해석하지 않습니다. INT4는 profiler에서 aten::_weight_int4pack_mm/tinygemm packed weight kernel을 확인했습니다.
- 현재 eager 실행은 압축과 GPU allocated 메모리 절감에 성공했으나 BF16보다 느립니다. torch.compile 미적용이며 컴파일·활성값 양자화 최적화 결과를 미리 주장하지 않습니다.
- INT4 group128에서 특징 변화가 커졌지만 정상 96프레임의 특징 차이만으로 AUROC 손실을 추정할 수 없습니다. INT8도 이상탐지 성능 유지가 검증된 것은 아닙니다.
- 짧은 정상 프레임 진단으로 end-to-end VAD FPS/30FPS 마감시간 보장/다른 장비 실시간을 주장하지 않습니다. 다른 프로세스와 GPU 클럭의 영향이 가능합니다.
- 사전학습 가중치·원본·기존 결과를 보존합니다. 새로운 전체 patch 캐시는 없습니다. TorchAO 0.17.0은 별도 cache 경로에 설치했고 기존 PyTorch는 변경하지 않았습니다. 원본·패키지·가중치는 GitHub에 게시하지 않습니다.

## 후속 실험 범위와 현재 상태

1. **정상 데이터 실행 최적화:** 9단계 후속 실험에서 R01–R04의 정상 384프레임으로 BF16, W8A16, W8A8, W4 native, W4 packed를 비교했습니다. BF16에도 같은 eager·CUDA Graph·compile·autotune 탐색을 적용했습니다. [전체 결과](../stage9_1_quant_compile/optimized/results.md)
2. **커널 병목 진단:** 실제 DINOv2 선형층의 복원 비용, 토큰 수, 캐시 상태와 Nsight 지표를 비교했습니다. [전체 결과](../stage9_1_quant_compile/kernel_study/results.md)
3. **남은 검증:** 양자화 백본에 맞춘 위상 예측기·프로토타입 재구축, 분리된 정상 보정, R01–R04의 VAD AUROC/AUPRC·오탐·미탐 평가는 수행하지 않았습니다. 이 결과가 확보되기 전까지 양자화가 이상탐지 성능을 유지한다고 결론내리지 않습니다.

## 재현

```bash
uv pip install --python .venv/bin/python --target cache/quantization/torchao017 --no-deps --no-cache torchao==0.17.0
.venv/bin/python scripts/probe_quantization.py --variant bf16
.venv/bin/python scripts/probe_quantization.py --variant int8
.venv/bin/python scripts/probe_quantization.py --variant int4
.venv/bin/python scripts/summarize_quantization_probe.py
```

[TorchAO 공식 추론 문서](https://docs.pytorch.org/ao/stable/workflows/inference.html) · [예비 실행 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/quantization_probe.md)
