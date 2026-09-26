# 9단계 후속 — INT8·INT4 최적화: 동일 전처리와 CUDA Graph 검증

기존 잘못된 커널 선택과 불필요한 CPU 동기화를 수정하고, BF16에도 똑같은 최적화 탐색을 적용했습니다. 이 결과는 정상 데이터에서의 실행 최적화입니다. 전체 이상탐지 AUROC·경보 성능은 평가하지 않았습니다.

## 검증 방법

- R01–R04 고정 정상384frame, seed0, batch1. eager FP32 RGB/resize/정규화를 모든 경로에 고정했습니다. 프레임별 BF16 백본 입력 SHA256이 모두 일치합니다.
- 5경로 × eager / eager CUDA Graph / compile CUDA Graph / max-autotune CUDA Graph =20개 pilot. 균등16frame에서 사전 수치선(동일 eager 대비 CLS/patch cosine평균≤1e-3, 최대절대차이≤.05)을 통과한 가장 빠른 모델 실행 방식을 고정한 뒤384frame×3회 검증했습니다. full 실패시 사전 지정 graph를 추가 확인합니다.
- FP32 원가중치+FP32입력(TF32off), BF16 eager, 동일 양자화 eager의 세 기준을 기록했습니다. 같은 eager와 일치해도 양자화 자체가 정확도를 보존한다는 의미는 아닙니다.
- 아래 total은 JPEG읽기·전처리·BF16변환·백본을 포함합니다. 전처리 후 동기화를 동일하게 적용해 model 구간과 분리했습니다. 기존 전처리까지 컴파일한 표와 직접 비교하지 않습니다.

## 전체384frame 수치검사를 통과한 경로

| 경로 | 실행 방식 | 백본 평균 ms | 전처리 포함 평균 ms | 전체 p95 ms | 가중치payload MiB | GPU peak MiB |
|---|---|---:|---:|---:|---:|---:|
| bf16 | graph | 3.306 | 3.848 | 4.213 | 165.14 | 203.37 |
| w4_native | graph | 5.196 | 5.763 | 6.371 | 55.73 | 76.45 |
| w4_packed | graph | 4.944 | 5.525 | 6.196 | 46.17 | 80.52 |
| w8a16 | graph | 3.748 | 4.298 | 4.707 | 84.38 | 121.11 |
| w8a8 | graph | 9.032 | 9.603 | 10.082 | 84.38 | 121.11 |

| 경로 | 동일 eager patch 최대차이 | FP32 대비 patch cosine 평균 | BF16 eager 대비 patch cosine 평균 |
|---|---:|---:|---:|
| bf16 | 0.000000 | 0.000444 | 0.000000 |
| w4_native | 0.000000 | 0.130292 | 0.130408 |
| w4_packed | 0.000000 | 0.130308 | 0.130425 |
| w8a16 | 0.000000 | 0.001243 | 0.001050 |
| w8a8 | 0.000000 | 0.006572 | 0.006495 |

## 가장 빠른 전체384frame 경로: 수치 실패 포함 진단

다음 표는 수치 기준과 무관하게 가장 빠른 normal pilot을 추가384frame에서 검증한 결과입니다. **수치 초과 경로는 동등한 대체 모델로 채택하지 않습니다.**

| 경로 | 방식 | 백본 ms | 총 ms | payload MiB | peak MiB | 수치 기준 |
|---|---|---:|---:|---:|---:|---|
| bf16 | autotune_graph | 0.930 | 1.474 | 165.14 | 174.67 | 초과 |
| w4_native | compile_graph | 3.265 | 3.821 | 55.73 | 66.01 | 초과 |
| w4_packed | autotune_graph | 1.093 | 1.642 | 46.17 | 53.83 | 초과 |
| w8a16 | compile_graph | 1.720 | 2.267 | 84.38 | 94.42 | 초과 |
| w8a8 | compile_graph | 1.575 | 2.122 | 84.38 | 94.42 | 초과 |

## 무엇을 최적화했나

- W8A16: Int8WeightOnlyConfig 가중치를 유지하고, 현재 GPU에서 느린 weight_int8pack_mm으로 치환하는 compiler 패턴을 프로세스 내부에서 비활성화했습니다. 단순 BF16 원가중치 복원 모델로 바꾼 것이 아닙니다.
- W8A8: 실제 INT8 정수 GEMM을 유지하고 FakeTensor 감지의 GPU 텐서 문자열 처리를 타입 검사로 교체했습니다. 설치된 라이브러리 파일은 변경하지 않았습니다.
- W4 native: 기존 tinygemm group128 커널에도 동일한 CUDA Graph와 autotune 기회를 적용했습니다.
- W4 packed: 원래 group128 INT4 q/scale/zero 값을 유지한 nibble packing. 48층 모두 padded tinygemm과 같은 양자화 값을 확인했습니다. 768→1024 패딩을 피하고 필요할 때 BF16으로 복원해 GEMM을 수행합니다. **INT4 저장/BF16 계산**이며 순수 INT4 Tensor Core 연산 가속으로 주장하지 않습니다.
- eager CUDA Graph는 기존 eager 연산을 캡처하여 CPU 호출 비용을 줄이고 compiler의 추가 수치 변경을 피하는 후보입니다. compile/max-autotune 실패나 수치 초과는 숨기지 않고 아래에 기록했습니다. 원본 경로와 패키지 파일은 보존했습니다.
- payload 절감과 실제GPU peak는 다릅니다. CUDA Graph가 복원용 임시 버퍼를 유지할 수 있으므로 압축모델 크기만으로 실행 메모리 절감을 주장하지 않습니다.

## 전체 탐색 기록

| 단계 | 경로 | 모드 | 백본 ms | 총 ms | 수치 기준 | patch 최대차이 | 첫 실행초 |
|---|---|---|---:|---:|---|---:|---:|
| pilot | bf16 | autotune_graph | 1.008 | 1.625 | 초과 | 0.086212 | 18.40 |
| pilot | bf16 | compile_graph | 1.033 | 1.596 | 초과 | 0.086212 | 6.08 |
| pilot | bf16 | eager | 3.340 | 3.900 | 통과 | 0.000000 | 0.00 |
| pilot | bf16 | graph | 3.316 | 3.866 | 통과 | 0.000000 | 0.03 |
| pilot | w4_native | autotune_graph | 3.268 | 3.826 | 초과 | 0.134805 | 8.99 |
| pilot | w4_native | compile_graph | 3.250 | 3.810 | 초과 | 0.134805 | 7.38 |
| pilot | w4_native | eager | 5.378 | 5.965 | 통과 | 0.000000 | 0.01 |
| pilot | w4_native | graph | 5.196 | 5.797 | 통과 | 0.000000 | 0.04 |
| pilot | w4_packed | autotune_graph | 1.134 | 1.706 | 초과 | 0.151299 | 19.65 |
| pilot | w4_packed | compile_graph | 1.263 | 1.817 | 초과 | 0.133015 | 9.64 |
| pilot | w4_packed | eager | 7.599 | 8.162 | 통과 | 0.000000 | 0.01 |
| pilot | w4_packed | graph | 5.025 | 5.583 | 통과 | 0.000000 | 0.05 |
| pilot | w8a16 | autotune_graph | 1.836 | 2.448 | 초과 | 0.211122 | 17.25 |
| pilot | w8a16 | compile_graph | 1.711 | 2.275 | 초과 | 0.211122 | 7.52 |
| pilot | w8a16 | eager | 4.800 | 5.355 | 통과 | 0.000000 | 0.00 |
| pilot | w8a16 | graph | 3.738 | 4.296 | 통과 | 0.000000 | 0.03 |
| pilot | w8a8 | autotune_graph | 1.620 | 2.178 | 초과 | 0.330381 | 21.37 |
| pilot | w8a8 | compile_graph | 1.596 | 2.155 | 초과 | 0.330381 | 13.66 |
| pilot | w8a8 | eager | 21.602 | 22.197 | 통과 | 0.000000 | 0.02 |
| pilot | w8a8 | graph | 9.032 | 9.750 | 통과 | 0.000000 | 0.13 |
| full | bf16 | autotune_graph | 0.930 | 1.474 | 초과 | 0.184604 | 18.89 |
| full | bf16 | graph | 3.306 | 3.848 | 통과 | 0.000000 | 0.03 |
| full | w4_native | compile_graph | 3.265 | 3.821 | 초과 | 0.369690 | 7.53 |
| full | w4_native | graph | 5.196 | 5.763 | 통과 | 0.000000 | 0.04 |
| full | w4_packed | autotune_graph | 1.093 | 1.642 | 초과 | 0.244626 | 20.22 |
| full | w4_packed | graph | 4.944 | 5.525 | 통과 | 0.000000 | 0.05 |
| full | w8a16 | compile_graph | 1.720 | 2.267 | 초과 | 0.297733 | 7.47 |
| full | w8a16 | graph | 3.748 | 4.298 | 통과 | 0.000000 | 0.03 |
| full | w8a8 | compile_graph | 1.575 | 2.122 | 초과 | 0.433021 | 13.57 |
| full | w8a8 | graph | 9.032 | 9.603 | 통과 | 0.000000 | 0.13 |

## 중간층·FP32 기준 진단

각 장면 첫 정상샘플4개에서 FP32/BF16/compiled 중간 출력을 비교했습니다. token 준비 이후 Transformer 블록에서도 수치 차이가 관찰됐습니다. 위치 임베딩을 eager에서 미리 계산한 대조군은 eager 출력을 바꾸지 않았지만 compiler 차이를 없애지는 못했습니다. 중간값 반환 자체가 fusion을 바꿀 수 있어 마지막 token 출력과 실사용 백본 경로의 결과를 동일하다고 간주하지 않습니다.

[중간층 차이 원자료](layer_diagnostic/layer_differences.csv) · [고정 위치 임베딩 대조군](layer_diagnostic/completed.json)

BF16 autotune 최초 시도의 compiler 초기화 구간에 별도 GPU 회귀검사가 겹쳤을 가능성을 발견해, 해당 기록을 restarts 아래 보존하고 모든 GPU 작업 종료 후 새 compiler cache로 단독 재측정했습니다. 표에는 재측정값만 사용합니다. [측정 감사](measurement_review.json)

## 제한 및 재현

- 이 결과만으로 이상탐지 성능 유지나 전체 VAD30FPS를 주장하지 않습니다. 중간층 진단은 추가 원인 자료이며 수치선이나 양자화 설정을 사후 변경하는 데 쓰지 않았습니다.
- 현재 RTX PRO 6000과 desktop GPU 공유 환경에서의 측정입니다. pilot 선택 편향을 줄이기 위해384frame에서 재측정했지만, 독립 날짜/장비 성능 보증은 아닙니다.
- 실제커널 profiler.csv, 입력/특징차이 feature_differences.csv, 3반복 timings.csv, 환경/config, compiler SHA 목록은 경로별 보존합니다. 원본영상·가중치·컴파일binary는 게시하지 않습니다.

```bash
.venv/bin/python scripts/run_stage9_optimized.py --suite
.venv/bin/python scripts/summarize_stage9_optimized.py
```

기존 결과가 있으면 덮어쓰기를 거부합니다. [사전 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/stage9_optimization_protocol.md)
