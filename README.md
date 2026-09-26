# 산업 공정 영상 이상탐지: IPAD 재현과 DINOv2 비교

R01–R04 실제 공정 영상으로 IPAD 논문 방법을 재현하고, DINOv2 특징·위상 메모리·온라인 추론·LoRA·양자화 경로를 단계별로 비교합니다. 합성 데이터는 사용하지 않습니다. 결과는 재현·진단·성능 평가의 범위를 구분해 보고합니다.

[단계별 진행](#단계별-진행) · [핵심 결과](#핵심-결과-요약) · [실행 방법](#실행-방법) · [평가 범위와 한계](#평가-범위와-한계) · [원 자료](#원-자료)

## 단계별 진행

| 단계 | 목적 | 상태 | 기록 |
|---|---|---|---|
| 1단계 | Swin-T + 주기 메모리 + 재구성 + 주기 검사 | 4개 장면 50 epochs 완료 · seed 0 | [전체 자료](experiments/stage1_reproduction/) |
| 2단계 | DINOv2 입력–복원 특징 비교 / 비재구성 prototype | 4개 장면 완료 · seed 0 | [전체 자료](experiments/stage2_dinov2/) |
| 1단계 추가 검증 | 메모리 제거 ablation | 4개 장면 완료 · seed 0 | [자료](experiments/stage1_memory_ablation/) |
| 3단계 | 위상 진단 및 불확실성을 고려한 메모리 선택 | R01–R04 × seed 0·1·2 및 시간 진단 완료 | [결과](experiments/stage3_phase_routing/results.md) |
| 4-1 | 외형·시간 검사 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage4_1_appearance_temporal/results.md) |
| 4-2 | 메모리 용량 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage4_2_memory_budget/results.md) |
| 4-3 | 전이·체류시간 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage4_3_process_prior/results.md) |
| 5-1 | 온라인 기준선 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage5_1_causal/results.md) |
| 5-2 | 백본·메모리 경량화 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage5_2_lightweight/results.md) |
| 5-3 | 30 FPS 재생·오탐·미탐 평가 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage5_3_streaming/results.md) |
| 6-1 | 정밀도·메모리 Pareto | 완료 · seed 0·1·2 | [결과·기록](experiments/stage6_1_pareto/results.md) |
| 6-2 | 정상 영상 보정 일반화 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage6_2_calibration/results.md) |
| 7-1 | 오탐·미탐 원인 분해 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage7_1_diagnostics/results.md) |
| 7-2 | causal 경보 규칙 비교 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage7_2_alerts/results.md) |
| 7-3 | 전체 영상 batch1 검증 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage7_3_full_stream/results.md) |
| 8-1 | LoRA 구현·학습 검증 | 완료 · seed 0 | [결과·기록](experiments/stage8_1_lora_validation/results.md) |
| 8-2 | 정상 영상 LoRA 비교 | 완료 · seed 0 | [결과·기록](experiments/stage8_2_lora_comparison/results.md) |
| 8-3 | 반복·병합 BF16 실시간 | 완료 · seed 0·1·2 | [결과·기록](experiments/stage8_3_lora_stream/results.md) |
| 9-1 | 양자화·커널 최적화 | 완료 · seed 0 | [결과·기록](experiments/stage9_1_quant_compile/results.md) |

## 핵심 결과 요약

| 실험 | 관찰 결과 | 상세 기록 |
|---|---|---|
| 1단계 논문 재현 | 네 장면 평균 AUROC **72.53%** (논문 기준 70.00%). R02 영상 12·13·14는 정렬 불일치로 제외했습니다. | [재현 결과와 차이](REPRODUCTION.md) |
| 2단계 DINOv2 | 비재구성 위상 무조건부 최근접 prototype의 seed 0 평균 AUROC는 78.37%였습니다. 단일 seed의 데이터셋 내 비교입니다. | [방법별 비교](experiments/stage2_dinov2/) |
| 3단계 위상 선택 | 정상 holdout에서 인접 위상 정확도는 장면별 83–96%였습니다. centered window 기반 테스트 결과는 오프라인 진단입니다. | [3-seed 분석](experiments/stage3_phase_routing/research_findings.md) |
| 4단계 시간 정보 | 외형+window 21은 80.97 ± 0.67% (3 seeds)였습니다. 결과는 같은 기존 테스트셋에서의 오프라인 비교입니다. | [통합 분석](experiments/stage4_summary/research_findings.md) |
| 5단계 온라인 전환 | 과거 프레임·외형+시간 기준선은 81.24 ± 0.12%였습니다. 정상 보정 분리와 인과 추론을 사용했습니다. | [통합 분석](experiments/stage5_summary/research_findings.md) |
| 6단계 효율 절충 | S_bf16_k5는 AUROC 79.85 ± 0.45%, capacity 260.6 FPS, peak 0.085 GiB의 관측 Pareto 후보였습니다. | [통합 분석](experiments/stage6_summary/research_findings.md) |
| 7단계 오경보 진단 | R01 B/k10에서 hysteresis는 오경보를 3.87→1.85회/정상 1,000프레임으로 줄였지만 구간 recall도 59.70→48.31%로 낮췄습니다. | [통합 분석](experiments/stage7_summary/research_findings.md) |
| 8단계 LoRA | 3-seed stream에서 frozen 81.08 ± 0.08%, anchored 80.97 ± 0.20% AUROC였습니다. 처리량 차이는 작고 정상 적응의 이득은 제한적입니다. | [통합 분석](experiments/stage8_summary/research_findings.md) |
| 9-1 양자화 | W4 packed payload는 46.17 MiB (BF16 165.14 MiB). eager 대비 컴파일 수치검사를 통과한 경로에서 백본은 4.944 ms (BF16 3.306 ms)였고, BF16 대비 patch cosine 차이는 0.1304였습니다. 전체 VAD AUROC는 측정하지 않았습니다. | [최적화](experiments/stage9_1_quant_compile/optimized/results.md) · [커널 진단](experiments/stage9_1_quant_compile/kernel_study/results.md) |

## 1단계 — 논문 방법론 재현

장면마다 독립 학습: 16프레임, 256×256, Video Swin-T, 200개 위상, 메모리 2,000개, window 5, Adam 1e-4, batch 8, 50 epochs, FP32, seed 0. 재구성·주기 점수를 장면별 정규화 후 같은 가중치로 결합합니다.

| 장면 | 논문 AUROC (%) | 구현 AUROC (%) | 차이 (pp) | 평가 프레임 |
|---|---:|---:|---:|---:|
| R01 | 84.40 | 88.13 | +3.73 | 3,400 |
| R02 | 75.40 | 80.25 | +4.85 | 7,478 |
| R03 | 43.50 | 46.89 | +3.39 | 11,682 |
| R04 | 76.70 | 74.83 | -1.87 | 7,793 |

장면별 AUROC 단순 평균: **72.53%** (논문 70.00%).

**해석 제한:** R02는 영상/라벨 길이가 다른 영상 12·13·14를 제외합니다. 공개 코드의 전체 파라미터는 263.48M으로 논문 표 35.9M과 다릅니다. 점수 결합 등 미기재 사항을 명시적 가정으로 보완했으므로 원 논문과 완전히 같은 조건의 우월성 증거로 해석하지 않습니다. [차이와 가정](REPRODUCTION.md)

## 2단계 — DINOv2 도입

- **A: 입력–복원 특징 비교** — 기존 IPAD checkpoint를 유지하고 frozen ViT-B/14의 CLS 및 6·12층 patch 특징 차이를 비교합니다.
- **B: 비재구성 prototype** — 16프레임 CLS 위상 MLP와 20개 위상 구간의 공간별 메모리를 사용합니다. 위상별 최대 10개 prototype, cosine NN 및 soft projection(온도 0.1)을 비교합니다.
- 조건부/무조건부 비교는 같은 정상 특징 표본과 같은 총 prototype 수를 사용합니다. 모든 방법은 동일한 유효 평가 프레임을 사용합니다.

| 방법 (AUROC %) | R01 | R02* | R03 | R04 | 평균 |
|---|---:|---:|---:|---:|---:|
| 원 재현: 픽셀+주기 | 88.13 | 80.25 | 46.89 | 74.83 | 72.53 |
| A: CLS | 91.86 | 77.77 | 57.61 | 72.92 | 75.04 |
| A: 최종층 patch | 92.14 | 70.30 | 53.08 | 74.39 | 72.48 |
| A: 다층 patch+주기 | 93.01 | 71.17 | 52.70 | 73.56 | 72.61 |
| B: 위상 조건부 NN | 83.77 | 84.93 | 55.55 | 80.89 | 76.29 |
| B: 위상 무조건부 NN | 86.24 | 85.57 | 62.34 | 79.33 | 78.37 |
| B: 위상 조건부 soft | 83.42 | 81.42 | 55.31 | 79.43 | 74.89 |
| B: 위상 무조건부 soft | 84.22 | 73.46 | 56.81 | 71.01 | 71.37 |

표의 방법은 모든 장면에서 같은 점수 정의를 사용합니다. 장면마다 가장 높은 변형을 골라 평균내지 않습니다. AUPRC, 모든 점수 변형, 원 점수·정답 CSV, R02 정렬 민감도는 단계별 폴더에 보존합니다. seed 0 단일 실행이며 통계적 유의성 주장이 아닙니다.

## 1단계 추가 검증: 메모리 제거

| 장면 | 원 모델 | 메모리 제거+주기 |
|---|---:|---:|
| R01 | 88.13 | 90.13 |
| R02 | 80.25 | 78.77 |
| R03 | 46.89 | 48.45 |
| R04 | 74.83 | 76.21 |

## 실행 방법

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
.venv/bin/python scripts/bootstrap_vendor.py
# 원 IPAD 데이터의 R01~R04를 IPAD_dataset/ 아래에 배치
.venv/bin/python -m ipad.data --cache
.venv/bin/python -m pytest -q
.venv/bin/python scripts/launch_suite.py --stage all
.venv/bin/python scripts/status.py
```

현재 실행 환경은 RTX PRO 6000 Blackwell 96GB, PyTorch 2.11.0+cu128입니다. 다른 GPU에서는 호환 환경과 소규모 검증을 먼저 확인합니다. [세부 명령](docs/RUNNING.md)

## 평가 범위와 한계

- experiments/에는 설정·실행 로그·프레임 단위 점수와 정답·평가지표를 단계별로 보관합니다. 원본 영상·가중치·특징 캐시 등 바이너리는 공개 저장소에 포함하지 않으며 SHA256 목록만 제공합니다.
- R02 영상 12·13·14는 프레임/라벨 길이 불일치 때문에 주 결과에서 제외합니다. 세부 민감도와 제외 근거는 단계별 기록에 있습니다.
- 1–4단계는 중앙 프레임과 테스트 구간 정규화를 쓰는 오프라인 비교입니다. 5단계 이후 온라인 결과는 정상 보정과 과거 입력을 사용하지만 실제 카메라·다른 GPU의 성능을 보장하지 않습니다.

[실행 기록 규칙](docs/EXPERIMENT_LOG_POLICY.md) · [코드–논문 차이](REPRODUCTION.md) · [3단계 결과](experiments/stage3_phase_routing/research_findings.md)

## 원 자료

- [IPAD 논문 v1](https://arxiv.org/abs/2404.15033v1) · [공식 코드](https://github.com/LJF1113/IPAD), commit `22764cbeeda3946303d236babdd2664fd6241b91`.
- [DINOv2 공식 구현](https://github.com/facebookresearch/dinov2), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- upstream 코드의 재배포 대신 출처·SHA256을 보존하고 bootstrap에서 원본을 내려받습니다.
