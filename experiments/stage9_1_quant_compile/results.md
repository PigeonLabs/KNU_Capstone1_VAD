# 9-1 실제 양자화 연산 경로와 컴파일 최적화

9단계는 양자화이며 기존 8단계 LoRA와 별개입니다. 고정 ViT-B/14, R01–R04 정상 학습 영상의 동일 384프레임, batch 1, 3회 반복. 아래 시간은 JPEG 읽기·전처리·전송·백본·CUDA 동기화를 포함하며 위상 예측기·메모리·경보는 제외합니다. 전체 VAD 성능과 AUROC는 이번 단계에서 측정하지 않았습니다.

| 경로 | 실행 | 평균 ms | p95 ms | 해당 eager 대비 속도 | 같은 실행 BF16 대비 속도 | GPU peak MiB | 초기 호출 초 | 수치 검사 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| bf16 | eager | 3.918 | 4.318 | 1.00× | 1.00× | 188.2 | 0.00 | 통과 |
| w8a16 | eager | 5.442 | 5.913 | 1.00× | 0.72× | 111.5 | 0.00 | 통과 |
| w8a8 | eager | 53.427 | 56.306 | 1.00× | 0.07× | 125.1 | 0.05 | 통과 |
| w4a16 | eager | 5.911 | 6.585 | 1.00× | 0.66× | 81.2 | 0.01 | 통과 |
| bf16 | default | 2.277 | 2.585 | 1.72× | 1.00× | 189.7 | 8.42 | 기준 초과 |
| w8a16 | default | 41.363 | 42.253 | 0.13× | 0.06× | 114.0 | 8.25 | 기준 초과 |
| w8a8 | default | 3.316 | 3.552 | 16.11× | 0.69× | 116.0 | 12.59 | 기준 초과 |
| w4a16 | default | 3.841 | 4.329 | 1.54× | 0.59× | 79.4 | 7.77 | 기준 초과 |
| bf16 | reduce-overhead | 1.482 | 1.532 | 2.64× | 1.00× | 173.4 | 6.45 | 기준 초과 |
| w8a16 | reduce-overhead | 44.650 | 45.452 | 0.12× | 0.03× | 94.3 | 7.81 | 기준 초과 |
| w8a8 | reduce-overhead | 1.975 | 2.284 | 27.05× | 0.75× | 94.3 | 12.83 | 기준 초과 |
| w4a16 | reduce-overhead | 3.710 | 4.150 | 1.59× | 0.40× | 65.9 | 7.80 | 기준 초과 |
| bf16 | reduce-overhead-precise | 1.494 | 1.593 | 2.62× | 1.00× | 173.4 | 6.52 | 기준 초과 |
| w8a16 | reduce-overhead-precise | 44.583 | 45.513 | 0.12× | 0.03× | 94.3 | 8.06 | 기준 초과 |
| w8a8 | reduce-overhead-precise | 2.028 | 2.144 | 26.34× | 0.74× | 94.3 | 13.90 | 기준 초과 |
| w4a16 | reduce-overhead-precise | 3.730 | 4.167 | 1.58× | 0.40× | 65.9 | 8.05 | 기준 초과 |

속도 배율은 클수록 빠릅니다. 양자화·컴파일 효과를 분리하기 위해 양자화 compiled를 BF16 eager와만 비교하지 않습니다. 초기 호출은 새 조건 캐시에서의 모델 compile 호출 비용이며 eager 참조 계산과 GPU 초기화 이후입니다. warmup 12회와 초기/정상 peak 메모리는 원 JSON에 별도 기록했습니다.

## 실제 연산 경로

| 조건 | 정수 GEMM 연산/커널 흔적 | packed INT4 흔적 | CUDA Graph 흔적 |
|---|---|---|---|
| bf16/eager | 이름에서 미확인 | 없음 | 이름에서 미확인 |
| w8a16/eager | 이름에서 미확인 | 없음 | 이름에서 미확인 |
| w8a8/eager | 확인 | 없음 | 이름에서 미확인 |
| w4a16/eager | 이름에서 미확인 | 확인 | 이름에서 미확인 |
| bf16/default | 이름에서 미확인 | 없음 | 이름에서 미확인 |
| w8a16/default | 이름에서 미확인 | 없음 | 이름에서 미확인 |
| w8a8/default | 확인 | 없음 | 이름에서 미확인 |
| w4a16/default | 이름에서 미확인 | 확인 | 이름에서 미확인 |
| bf16/reduce-overhead | 이름에서 미확인 | 없음 | 확인 |
| w8a16/reduce-overhead | 이름에서 미확인 | 없음 | 확인 |
| w8a8/reduce-overhead | 확인 | 없음 | 확인 |
| w4a16/reduce-overhead | 이름에서 미확인 | 확인 | 확인 |
| bf16/reduce-overhead-precise | 이름에서 미확인 | 없음 | 확인 |
| w8a16/reduce-overhead-precise | 이름에서 미확인 | 없음 | 확인 |
| w8a8/reduce-overhead-precise | 확인 | 없음 | 확인 |
| w4a16/reduce-overhead-precise | 이름에서 미확인 | 확인 | 확인 |

원시 kernel_evidence 필드의 단순 이름 검색은 CompiledFxGraph를 CUDA Graph로, weight_int8pack_mm을 정수 GEMM으로 오인할 수 있어 kernel_audit.json에서 다시 구분했습니다. 원 기록은 보존했습니다. 연산·커널 이름은 profiler.csv, 생성된 코드의 호출/자료형 발췌는 generated_kernel_evidence.json, 전체 로컬 생성물 SHA256은 generated_cache_hashes.json에 있습니다. profiler는 별도 5프레임 실행이며 시간 측정에 포함하지 않았습니다. 컴파일로 연산이 합쳐져 이름이 바뀐 경우 이름 미검출만으로 해당 정밀도 연산이 없다고 단정하지 않습니다.

## 특징 수치 변화

| 조건 | 동일 경로 eager 대비 patch cosine 차이 | 동일 eager 대비 patch 최대 오차 | BF16 eager 대비 patch cosine 차이 |
|---|---:|---:|---:|
| bf16/eager | 0.0000000 | 0.000000 | 0.000000 |
| w8a16/eager | 0.0000000 | 0.000000 | 0.001050 |
| w8a8/eager | 0.0000000 | 0.000000 | 0.006495 |
| w4a16/eager | 0.0000000 | 0.000000 | 0.130408 |
| bf16/default | 0.0003035 | 0.427217 | 0.000303 |
| w8a16/default | 0.0004407 | 0.364279 | 0.001043 |
| w8a8/default | 0.0048777 | 0.450719 | 0.006294 |
| w4a16/default | 0.0004809 | 0.380236 | 0.130315 |
| bf16/reduce-overhead | 0.0003035 | 0.427217 | 0.000303 |
| w8a16/reduce-overhead | 0.0004699 | 0.417678 | 0.001039 |
| w8a8/reduce-overhead | 0.0048777 | 0.450719 | 0.006294 |
| w4a16/reduce-overhead | 0.0004809 | 0.380236 | 0.130315 |
| bf16/reduce-overhead-precise | 0.0001454 | 0.411781 | 0.000145 |
| w8a16/reduce-overhead-precise | 0.0003581 | 0.320838 | 0.000999 |
| w8a8/reduce-overhead-precise | 0.0039925 | 0.459699 | 0.006510 |
| w4a16/reduce-overhead-precise | 0.0002500 | 0.369690 | 0.130396 |

수치 검사선은 동일 양자화 eager 대비 CLS와 patch 모두 평균 cosine 거리≤1e-3, 최대절대차이≤0.05입니다. BF16 대비 양자화 손실과 동일 양자화에서 compile에 의한 차이를 구분합니다. 정상 특징 수치 검사가 통과해도 이상탐지 AUROC·경보 동등성을 보장하지 않습니다.

수치 검사 기준 초과 조건: bf16_default, w8a16_default, w8a8_default, w4a16_default, bf16_reduce-overhead, w8a16_reduce-overhead, w8a8_reduce-overhead, w4a16_reduce-overhead, bf16_reduce-overhead-precise, w8a16_reduce-overhead-precise, w8a8_reduce-overhead-precise, w4a16_reduce-overhead-precise. 속도가 빨라도 동등한 대체 모델로 채택하지 않습니다.

## 해석과 재현

- 초기 12개 조건을 고정했고, 정상 특징 수치 차이를 관찰한 뒤 BF16 중간 반올림 보존 대조군 4개를 추가했습니다. 추가 조건은 reduce-overhead-precise이며 네 연산 경로에 동일하게 적용했습니다. calibration/test 영상을 사용하지 않았습니다. 모델 재학습과 전체 VAD 임계값 선택은 수행하지 않았습니다.
- BF16/weight-only 경로보다 INT8 정수 행렬곱이 빠르더라도 활성값 양자화·스케일 보정·커널 호출·입출력 비용을 포함한 전체 시간을 함께 비교합니다.
- default와 reduce-overhead를 같은 백본/입력에 적용했습니다. 후자는 CUDA Graph를 요청하며 실제 적용은 로그/커널 자료로 확인합니다. 실행 중 graph break/eager fallback은 허용하지 않았습니다.
- 모델 가중치 저장 크기와 compiler buffer/워크스페이스를 포함하는 runtime GPU peak는 다릅니다. compiled가 더 많은 메모리를 쓸 수 있습니다. GPU 구간시간도 커널 호출 사이의 빈 시간을 포함하므로 순수 연산시간 합계가 아닙니다.
- 현재 GPU, 정상 프레임, warm OS cache 가능 조건입니다. 실제 카메라·네트워크·전체 VAD·다른 장비의 실시간성을 주장하지 않습니다.
- compiler worker 수는2, torch CPU thread는8로 제한했습니다. 10 GiB 이하 중지, 자동 재개 금지, 원본/기존 결과 보존. 가중치/영상/컴파일 binary는 로컬에만 보존합니다.

```bash
.venv/bin/python scripts/run_stage9_1.py
.venv/bin/python scripts/summarize_stage9_1.py
```

[고정 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/stage9_protocol.md) · [TorchAO 연산 경로 문서](https://docs.pytorch.org/ao/stable/workflows/inference.html)

## 확인된 병목과 해석

- W8A8 eager의 별도 profiler 5프레임에서 local scalar 추출이 8,640회 관찰됐습니다. 설치된 torchao safe_int_mm은 eager 경로에서 input.__repr__()를 검사하며, CPU/GPU 동기화 비용을 유발할 수 있습니다. 정수 GEMM만으로 전체 실행 속도를 설명할 수 없습니다. 이 비용의 독립적인 인과 효과를 별도 ablation으로 정량화한 것은 아닙니다.
- W8A16 default의 GPU kernel 이벤트 시간 중 weight_int8pack_mm_kernel 비중은 48.9%였습니다. compiler가 선택한 이 weight-only 커널은 실제 W8A8 정수 GEMM과 별개이며, 현재 형상에서 지연이 집중됐습니다. GPU 이벤트 합계의 비중으로 전체 wall 시간의 인과 분해는 아닙니다.
- 컴파일은 연산 결합과 커널 선택, CUDA Graph에 따라 속도와 중간 반올림을 함께 바꿀 수 있습니다. 원 실험의 결과를 보존했고 수치 보존 대조군에서도 동일 허용선을 사용했습니다.
- 다음 평가에서는 같은 정상 데이터로 위상 head/메모리/보정을 정합한 뒤 AUROC·오탐·미탐을 확인해야 합니다. 이번 정상 특징 최대오차 초과를 곧바로 AUROC 저하량으로 해석하지 않습니다.
