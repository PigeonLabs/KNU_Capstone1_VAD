# 산업 공정 영상 이상탐지: IPAD 재현과 DINOv2 비교

R01–R04 실제 공정 영상만 사용합니다. **1단계는 논문 방법론 재현, 2단계는 DINOv2 도입**입니다. 합성 데이터는 사용하지 않습니다. 1–7단계는 DINOv2 백본을 고정하며, 승인된 8단계에서는 정상 영상 기반 LoRA 적응을 비교합니다.

## 단계별 진행

| 단계 | 목적 | 상태 | 기록 |
|---|---|---|---|
| 1단계 | Swin-T + 주기 메모리 + 재구성 + 주기 검사 | 4개 장면 50 epochs 완료 · seed 0 | [전체 자료](experiments/stage1_reproduction/) |
| 2단계 | DINOv2 입력–복원 특징 비교 / 비재구성 prototype | 4개 장면 완료 · seed 0 | [전체 자료](experiments/stage2_dinov2/) |
| 1단계 추가 검증 | 메모리 제거 ablation | 4개 장면 완료 · seed 0 | [자료](experiments/stage1_memory_ablation/) |
| 3단계 | 위상 진단 및 불확실성을 고려한 메모리 선택 | R01–R04 × seed 0·1·2 및 시간 진단 완료 | [규약](docs/stage3_protocol.md) · [결과](experiments/stage3_phase_routing/) |
| 4-1 | 외형·시간 검사 | 완료 · seed 0·1·2 | [자료](experiments/stage4_1_appearance_temporal/) |
| 4-2 | 메모리 용량 | 완료 · seed 0·1·2 | [자료](experiments/stage4_2_memory_budget/) |
| 4-3 | 전이·체류시간 | 완료 · seed 0·1·2 | [자료](experiments/stage4_3_process_prior/) |
| 5-1 | 온라인 기준선 | 완료 · seed 0·1·2 | [자료](experiments/stage5_1_causal/) |
| 5-2 | 백본·메모리 경량화 | 완료 · seed 0·1·2 | [자료](experiments/stage5_2_lightweight/) |
| 5-3 | 30 FPS 재생·오탐·미탐 평가 | 완료 · seed 0·1·2 | [자료](experiments/stage5_3_streaming/) |
| 6-1 | 정밀도·메모리 Pareto | 완료 · seed 0·1·2 | [자료](experiments/stage6_1_pareto/) |
| 6-2 | 정상 영상 보정 일반화 | 완료 · seed 0·1·2 | [자료](experiments/stage6_2_calibration/) |
| 7-1 | 오탐·미탐 원인 분해 | 완료 · seed 0·1·2 | [자료](experiments/stage7_1_diagnostics/) |
| 7-2 | causal 경보 규칙 비교 | 완료 · seed 0·1·2 | [자료](experiments/stage7_2_alerts/) |
| 7-3 | 전체 영상 batch1 검증 | 완료 · seed 0·1·2 | [자료](experiments/stage7_3_full_stream/) |
| 8-1 | LoRA 구현·학습 검증 | 완료 · seed 0 | [자료](experiments/stage8_1_lora_validation/) |
| 8-2 | 정상 영상 LoRA 비교 | 완료 · seed 0 | [자료](experiments/stage8_2_lora_comparison/) |
| 8-3 | 반복·병합 BF16 실시간 | 완료 · seed 0·1·2 | [자료](experiments/stage8_3_lora_stream/) |

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

### 1단계 추가 검증: 메모리 제거

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

## 자료와 기록 정책

- `experiments/`: 단계별 설정·epoch 이력·실행 로그·프레임별 정답/점수·평가지표·정렬 민감도.
- `artifacts.jsonl`: 로컬 원본/모델/특징 파일의 경로·크기·SHA256. **바이너리는 GitHub에 업로드하지 않았습니다.**
- 과거 원 모델 stdout은 25배치 간격입니다. 이번 기록 정책 이후의 학습은 매 배치 JSONL을 추가합니다. 기록하지 않은 과거 값을 복원하지 않습니다.
- 최신 사용자 지시에 따라 승인된 실험 전체 완료 시 결과·로그·해시를 main에 자동 게시합니다. 바이너리를 제외하고 원격 변경을 강제로 덮어쓰지 않습니다.
- 디스크 여유가 **10 GiB 이하**가 되면 이 프로젝트의 실험을 일시중지하고 보고합니다. 자동 재개하지 않습니다.
- 메모리 제거 실험은 1단계 추가 검증입니다. 3단계는 2026-09-25 승인받아 정상 위상 진단과 routing 비교부터 진행합니다.
- 1~4단계는 테스트 전체 정규화와 미래 프레임을 사용하는 오프라인 평가입니다. 5단계는 정상 데이터에서 고정한 보정과 과거 입력으로 온라인 평가하며, 실제 카메라나 다른 장비 성능으로 일반화하지 않습니다.
[기록 규칙](docs/EXPERIMENT_LOG_POLICY.md) · [코드–논문 차이](REPRODUCTION.md) · [3단계 제안](docs/stage3_proposal.md)

## 원 자료

- [IPAD 논문 v1](https://arxiv.org/abs/2404.15033v1) · [공식 코드](https://github.com/LJF1113/IPAD), commit `22764cbeeda3946303d236babdd2664fd6241b91`.
- [DINOv2 공식 구현](https://github.com/facebookresearch/dinov2), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- upstream 코드의 재배포 대신 출처·SHA256을 보존하고 bootstrap에서 원본을 내려받습니다.

## 3단계 — 위상 진단과 선택 방식 비교

[연구 해석·3-seed 요약·시간 진단](experiments/stage3_phase_routing/research_findings.md) · [사전 고정 규약](docs/stage3_protocol.md)

단계별 완료 결과입니다. seed 0의 기존 hard/unconditional 점수와 2단계 점수 일치를 검증했습니다.
테스트 결과로 변형을 선택하지 않으며, 주 후보는 사전 지정한 top3 확률 가중 거리입니다.

| 장면/seed | 무조건부 | 기존 hard | 인접 NN | top3 NN | top3 가중 | fallback | random3 | 전체 조건부 bank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R01/seed0 | 86.24 | 83.77 | 81.49 | 83.37 | 86.85 | 86.85 | 81.04 | 85.43 |
| R01/seed1 | 86.33 | 85.05 | 81.76 | 83.48 | 86.16 | 86.16 | 79.79 | 85.30 |
| R01/seed2 | 85.92 | 84.10 | 82.36 | 84.48 | 86.81 | 86.81 | 81.86 | 85.75 |
| R02/seed0 | 85.57 | 84.93 | 85.40 | 85.76 | 86.30 | 84.24 | 57.86 | 85.32 |
| R02/seed1 | 85.37 | 85.46 | 85.40 | 85.75 | 86.60 | 84.20 | 56.42 | 85.36 |
| R02/seed2 | 85.37 | 84.82 | 85.19 | 85.21 | 86.73 | 83.83 | 58.71 | 85.62 |
| R03/seed0 | 62.34 | 55.55 | 57.32 | 58.01 | 55.53 | 55.17 | 51.47 | 59.94 |
| R03/seed1 | 61.31 | 57.27 | 58.17 | 58.21 | 56.41 | 55.83 | 51.49 | 59.79 |
| R03/seed2 | 60.85 | 54.69 | 57.83 | 57.89 | 55.15 | 54.16 | 50.93 | 59.24 |
| R04/seed0 | 79.33 | 80.89 | 80.53 | 79.95 | 80.57 | 79.33 | 63.24 | 78.93 |
| R04/seed1 | 78.81 | 81.01 | 80.51 | 79.79 | 80.49 | 78.81 | 63.27 | 78.50 |
| R04/seed2 | 79.20 | 81.55 | 81.20 | 80.39 | 81.06 | 79.20 | 63.19 | 78.96 |

| 정상 holdout | 20-bin 정확도 | ±1 정확도 | 원형 MAE(bin) | cutoff 활성 |
|---|---:|---:|---:|---|
| R01/seed0 | 59.70% | 93.49% | 0.50 | True |
| R01/seed1 | 58.16% | 92.21% | 0.56 | True |
| R01/seed2 | 59.30% | 93.02% | 0.52 | True |
| R02/seed0 | 76.99% | 93.11% | 0.33 | True |
| R02/seed1 | 76.84% | 93.23% | 0.33 | True |
| R02/seed2 | 76.82% | 93.17% | 0.34 | True |
| R03/seed0 | 64.87% | 95.85% | 0.42 | True |
| R03/seed1 | 63.52% | 96.17% | 0.43 | True |
| R03/seed2 | 63.76% | 95.47% | 0.43 | True |
| R04/seed0 | 50.61% | 84.03% | 0.71 | False |
| R04/seed1 | 50.34% | 83.66% | 0.73 | False |
| R04/seed2 | 49.97% | 83.24% | 0.74 | False |

원시 CSV에는 frame ID, 예측 확률, 모든 비교 점수와 정답이 포함됩니다. 세부 지표·AUPRC·R02 민감도는 장면/seed별 JSON에 있습니다.
정상 holdout의 상대 위치는 진단용 참조입니다. 테스트 정답 위상을 사용하지 않습니다. centered window와 테스트 전체 정규화를 사용하므로 온라인 실시간 결과가 아닙니다.


## 4-1 추가 실험

[전체 기록](experiments/stage4_1_appearance_temporal/) · [고정 규약](docs/stage4_protocol.md)

# 4-1 실험 결과

완료 실행 단위: 12. R01–R04 / seed 0·1·2.
테스트 결과로 설정을 선택하지 않았으며 사전 규약의 모든 변형을 보고한다.

| k | 방법 | seed0 macro | seed1 macro | seed2 macro | 평균 ± seed 표준편차 |
|---|---|---:|---:|---:|---:|
| - | unconditional | 78.58 | 78.15 | 78.05 | 78.26 ± 0.28 |
| - | time5 | 51.61 | 52.10 | 52.52 | 52.07 ± 0.45 |
| - | time21 | 69.58 | 69.56 | 69.05 | 69.40 ± 0.30 |
| - | appearance_time5 | 77.67 | 77.53 | 77.78 | 77.66 ± 0.13 |
| - | appearance_time21 | 81.75 | 80.63 | 80.54 | 80.97 ± 0.67 |

주 후보 appearance_time21 - 무조건부 기준선의 seed0 paired video bootstrap 차이95% 구간: [1.16, 5.91] pp.
window21로 인한 경계 제외 이후 모든 방법이 동일한 평가 frame/video/label ID를 사용한다. 이전 3단계 전체 support 결과와 직접 차감하지 않는다.

세부 AUPRC·장면별 수치·원 점수/정답·설정·해시·R02 민감도·실행 로그는 각 실행 폴더에 보존한다. centered clip과 테스트 전체 정규화를 사용하는 오프라인 비교이다. R02 불일치 영상12·13·14는 주 결과에서 제외한다.


## 4-2 추가 실험

[전체 기록](experiments/stage4_2_memory_budget/) · [고정 규약](docs/stage4_protocol.md)

# 4-2 실험 결과

완료 실행 단위: 48. R01–R04 / seed 0·1·2.
테스트 결과로 설정을 선택하지 않았으며 사전 규약의 모든 변형을 보고한다.

| k | 방법 | seed0 macro | seed1 macro | seed2 macro | 평균 ± seed 표준편차 |
|---|---|---:|---:|---:|---:|
| 1 | unconditional | 74.47 | 73.19 | 74.13 | 73.93 ± 0.66 |
| 1 | legacy_hard | 72.90 | 73.69 | 72.87 | 73.15 ± 0.46 |
| 1 | top3_weighted | 74.21 | 74.22 | 74.28 | 74.24 ± 0.04 |
| 2 | unconditional | 76.24 | 75.09 | 75.74 | 75.69 ± 0.58 |
| 2 | legacy_hard | 74.32 | 74.92 | 74.31 | 74.51 ± 0.35 |
| 2 | top3_weighted | 75.53 | 75.52 | 75.67 | 75.57 ± 0.08 |
| 5 | unconditional | 77.76 | 77.10 | 77.02 | 77.29 ± 0.41 |
| 5 | legacy_hard | 75.73 | 75.97 | 75.63 | 75.78 ± 0.18 |
| 5 | top3_weighted | 76.92 | 76.39 | 76.69 | 76.67 ± 0.27 |
| 10 | unconditional | 78.37 | 77.95 | 77.83 | 78.05 ± 0.28 |
| 10 | legacy_hard | 76.29 | 77.20 | 76.29 | 76.59 ± 0.52 |
| 10 | top3_weighted | 77.31 | 77.42 | 77.44 | 77.39 ± 0.07 |

## 메모리와 효율 (seed0 측정)

| 장면 | k | 위치당 총 prototype | bank MiB | 무조건부 cache FPS | 무조건부 DINO 포함 FPS | hard DINO 포함 FPS | top3 DINO 포함 FPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| R01 | 1 | 19 | 9.02 | 118976.8 | 458.9 | 456.4 | 454.8 |
| R01 | 10 | 190 | 90.18 | 50146.6 | 454.9 | 453.5 | 436.1 |
| R01 | 2 | 38 | 18.04 | 118380.9 | 454.5 | 453.2 | 448.9 |
| R01 | 5 | 95 | 45.09 | 89578.9 | 450.5 | 448.2 | 440.2 |
| R02 | 1 | 20 | 9.49 | 115519.9 | 457.3 | 456.0 | 453.1 |
| R02 | 10 | 200 | 94.92 | 46397.0 | 452.2 | 453.2 | 435.5 |
| R02 | 2 | 40 | 18.98 | 114862.1 | 455.2 | 454.6 | 449.0 |
| R02 | 5 | 100 | 47.46 | 82187.8 | 452.2 | 450.5 | 440.3 |
| R03 | 1 | 20 | 9.49 | 118038.1 | 455.0 | 451.1 | 449.2 |
| R03 | 10 | 200 | 94.92 | 45115.3 | 449.2 | 448.9 | 432.7 |
| R03 | 2 | 40 | 18.98 | 115841.6 | 452.6 | 449.7 | 446.9 |
| R03 | 5 | 100 | 47.46 | 81336.7 | 450.1 | 447.5 | 438.2 |
| R04 | 1 | 20 | 9.49 | 108453.9 | 453.4 | 450.7 | 447.0 |
| R04 | 10 | 200 | 94.92 | 45108.6 | 447.4 | 447.1 | 428.7 |
| R04 | 2 | 40 | 18.98 | 115757.3 | 450.4 | 451.5 | 445.1 |
| R04 | 5 | 100 | 47.46 | 81145.0 | 448.5 | 447.1 | 437.0 |

FPS는 warm-up 후 decoded RAM 입력 측정이다. 카메라·디스크 I/O는 포함하지 않으며 실제 공장/엣지 장치 streaming 성능이 아니다. cached 후보 제한 matching과 DINO 포함 측정은 분리한다. GPU peak는 비교 구현 전체 peak이며 단일 방법의 배포 peak가 아니다.
같은 k에서 조건부·무조건부 bank는 같은 표본과 총 prototype 수를 사용한다. k10의 기존 점수 일치 여부는 baseline_equivalence.json에 있다.

세부 AUPRC·장면별 수치·원 점수/정답·설정·해시·R02 민감도·실행 로그는 각 실행 폴더에 보존한다. centered clip과 테스트 전체 정규화를 사용하는 오프라인 비교이다. R02 불일치 영상12·13·14는 주 결과에서 제외한다.


## 4-3 추가 실험

[전체 기록](experiments/stage4_3_process_prior/) · [고정 규약](docs/stage4_protocol.md)

# 4-3 실험 결과

완료 실행 단위: 12. R01–R04 / seed 0·1·2.
테스트 결과로 설정을 선택하지 않았으며 사전 규약의 모든 변형을 보고한다.

| k | 방법 | seed0 macro | seed1 macro | seed2 macro | 평균 ± seed 표준편차 |
|---|---|---:|---:|---:|---:|
| - | unconditional | 78.58 | 78.15 | 78.05 | 78.26 ± 0.28 |
| - | transition_kl | 46.26 | 46.26 | 45.57 | 46.03 ± 0.40 |
| - | duration | 58.67 | 58.01 | 58.89 | 58.52 ± 0.46 |
| - | process | 55.64 | 54.84 | 55.05 | 55.18 ± 0.42 |
| - | phase_entropy | 64.90 | 64.93 | 65.12 | 64.99 ± 0.12 |
| - | appearance_transition | 60.49 | 59.99 | 59.76 | 60.08 ± 0.37 |
| - | appearance_duration | 75.59 | 75.21 | 75.56 | 75.45 ± 0.21 |
| - | appearance_process | 71.84 | 70.94 | 71.03 | 71.27 ± 0.50 |
| - | appearance_time21 | 81.75 | 80.63 | 80.54 | 80.97 ± 0.67 |

주 후보 appearance_process - 무조건부 기준선의 seed0 paired video bootstrap 차이95% 구간: [-10.74, -2.35] pp.
window21로 인한 경계 제외 이후 모든 방법이 동일한 평가 frame/video/label ID를 사용한다. 이전 3단계 전체 support 결과와 직접 차감하지 않는다.

전이 분포는 정상 학습 영상의 모델 예측으로 학습했다. 체류시간에서는 영상 시작/끝의 잘린 run을 제외한다. 위상 jitter와 in-sample prediction 분포 차이가 한계이며 실제 이상 유형 분류나 독립 정상 오탐률을 주장하지 않는다.

세부 AUPRC·장면별 수치·원 점수/정답·설정·해시·R02 민감도·실행 로그는 각 실행 폴더에 보존한다. centered clip과 테스트 전체 정규화를 사용하는 오프라인 비교이다. R02 불일치 영상12·13·14는 주 결과에서 제외한다.


## 4단계 통합 해석

[세 실험의 통합 보고서와 최종 검산](experiments/stage4_summary/research_findings.md)

## 5-1 온라인·경량화 실험

[전체 기록](experiments/stage5_1_causal/) · [고정 규약](docs/stage5_protocol.md)

# 5-1 실험 결과

R01–R04,seed0·1·2. 정상80% 학습/20% 고정 보정. 테스트 정답은 학습·보정·임계값 선택에 사용하지 않았다.

| 모델 | 점수 | seed0 | seed1 | seed2 | 평균 ± 표준편차 AUROC (%) |
|---|---|---:|---:|---:|---:|
| B/k10 | centered_testnorm | 80.21 | 80.91 | 80.97 | 80.70 ± 0.42 |
| B/k10 | centered_fixed | 80.38 | 80.64 | 80.66 | 80.56 ± 0.15 |
| B/k10 | causal_appearance | 78.73 | 78.70 | 78.50 | 78.64 ± 0.12 |
| B/k10 | causal_combined | 81.29 | 81.11 | 81.33 | 81.24 ± 0.12 |

공통 frame35..N-18에서 비교한다. stage4와 정상 학습 범위·타깃·보정 규칙이 달라 직접 차감하지 않는다.

장면별 AUROC/AUPRC,full causal support,프레임 점수,보정 median/q99.5/threshold,매배치 loss/gradient와 checkpoint 검증은 각 실행 폴더에 있다. R02영상12/13/14는 주 결과에서 제외하고 ±1 민감도를 보존한다.

causal 입력은 현재까지16프레임,시간 검사는 과거21개 예측이며 첫 점수는 frame35이다. centered 비교군만 미래17프레임을 사용한다. 온라인 점수 보정은 정상 holdout에서 고정했고 테스트 중 업데이트하지 않는다.


## 5-2 온라인·경량화 실험

[전체 기록](experiments/stage5_2_lightweight/) · [고정 규약](docs/stage5_protocol.md)

# 5-2 실험 결과

R01–R04,seed0·1·2. 정상80% 학습/20% 고정 보정. 테스트 정답은 학습·보정·임계값 선택에 사용하지 않았다.

| 모델 | 점수 | seed0 | seed1 | seed2 | 평균 ± 표준편차 AUROC (%) |
|---|---|---:|---:|---:|---:|
| B/k10 | causal_combined | 81.29 | 81.11 | 81.33 | 81.24 ± 0.12 |
| B/k5 | causal_combined | 80.95 | 80.61 | 80.75 | 80.77 ± 0.17 |
| S/k10 | causal_combined | 80.27 | 80.01 | 79.92 | 80.07 ± 0.18 |
| S/k5 | causal_combined | 80.10 | 79.51 | 79.33 | 79.65 ± 0.41 |

공통 frame35..N-18에서 비교한다. stage4와 정상 학습 범위·타깃·보정 규칙이 달라 직접 차감하지 않는다.

장면별 AUROC/AUPRC,full causal support,프레임 점수,보정 median/q99.5/threshold,매배치 loss/gradient와 checkpoint 검증은 각 실행 폴더에 있다. R02영상12/13/14는 주 결과에서 제외하고 ±1 민감도를 보존한다.

메모리 k5는 k10의 정확히 절반 prototype이며 실제 점유 bin 수에 따라 총개수가 달라진다. backbone 변경 시 head·memory·normal calibration을 재구축했다. B/k10은5-1 결과를 재사용했다.


## 5-3 온라인·경량화 실험

[전체 기록](experiments/stage5_3_streaming/) · [고정 규약](docs/stage5_protocol.md)

# 5-3 실험 결과

R01–R04,seed0·1·2. 정상80% 학습/20% 고정 보정. 테스트 정답은 학습·보정·임계값 선택에 사용하지 않았다.

| 모델 | 정상 frame FPR (%) | 오경보/정상1000frame | 구간 탐지율 (%) | 탐지 구간 지연 중앙값 (frame) |
|---|---:|---:|---:|---:|
| B/k10 | 18.01 | 3.87 | 59.70 | 49.38 |
| B/k5 | 18.02 | 3.74 | 59.38 | 47.46 |
| S/k10 | 16.66 | 3.53 | 57.93 | 50.92 |
| S/k5 | 16.71 | 3.50 | 56.53 | 55.50 |

위 표는 장면·seed별 지표의 단순평균이다. 지연은 탐지된 구간만의 중앙값을 평균한 값이며 미탐·coldstart 구간 수는개별 operation.json에 함께 기록한다. 고장 유형별 정확도나 독립 사건 정답을 의미하지 않는다.

## 실제 batch1 측정 (seed0)

| 장면 | 모델 | capacity FPS | processing p95 ms | paced E2E p95 ms | paced deadline miss (%) | max queue ms | peak allocated GiB |
|---|---|---:|---:|---:|---:|---:|---:|
| R01 | B/k10 | 129.6 | 8.23 | 8.61 | 0.00 | 0.91 | 0.544 |
| R01 | B/k5 | 130.7 | 8.13 | 8.83 | 0.00 | 1.34 | 0.456 |
| R01 | S/k10 | 226.1 | 4.98 | 5.83 | 0.00 | 2.75 | 0.198 |
| R01 | S/k5 | 228.2 | 4.92 | 6.83 | 0.00 | 1.21 | 0.155 |
| R02 | B/k10 | 129.3 | 8.24 | 8.71 | 0.00 | 1.58 | 0.554 |
| R02 | B/k5 | 130.4 | 8.18 | 8.46 | 0.00 | 0.72 | 0.461 |
| R02 | S/k10 | 225.3 | 4.98 | 6.44 | 0.00 | 1.42 | 0.203 |
| R02 | S/k5 | 226.7 | 4.95 | 5.51 | 0.00 | 1.87 | 0.157 |
| R03 | B/k10 | 128.2 | 8.29 | 8.97 | 0.00 | 0.38 | 0.554 |
| R03 | B/k5 | 129.3 | 8.21 | 8.73 | 0.00 | 2.83 | 0.461 |
| R03 | S/k10 | 223.4 | 5.03 | 5.96 | 0.00 | 0.53 | 0.203 |
| R03 | S/k5 | 224.4 | 5.02 | 6.12 | 0.00 | 0.49 | 0.157 |
| R04 | B/k10 | 128.6 | 8.29 | 8.72 | 0.00 | 0.95 | 0.554 |
| R04 | B/k5 | 129.8 | 8.22 | 8.58 | 0.00 | 2.61 | 0.461 |
| R04 | S/k10 | 225.4 | 5.00 | 6.21 | 0.00 | 1.77 | 0.203 |
| R04 | S/k5 | 225.7 | 4.99 | 6.13 | 0.00 | 1.69 | 0.157 |

현재 RTX PRO6000,FP32,batch1,원본 JPEG read/decode 포함. capacity3회/30FPS paced replay1회,각 영상최대256frame. OS page cache가 warm일 수 있으며 카메라·네트워크 지연은 미포함이다. 실제 촬영 FPS가 확인된 데이터는 아니므로30FPS는 도착률 시나리오이다. 경보 정확도는 전체 causal test cache 평가이며 위 짧은 replay와 구분한다.

raw latency·queue·직접 경보·stream/cache 비교·frame별 경보 상태·미탐을 포함한 구간별 지연은 각 실행 폴더에 있다.


## 5단계 통합 해석

[온라인·경량화·실시간 평가 통합 결과](experiments/stage5_summary/research_findings.md)

## 6-1 추가 실험

[전체 기록](experiments/stage6_1_pareto/) · [고정 규약](docs/stage6_protocol.md)

# 6-1 실험 결과

R01–R04, seed 0·1·2. 정상 80% 학습 / 20% 보정 분리와 causal frame 규약 유지. 기존 테스트셋에서 추가 탐색한 결과이며 독립 데이터 일반화 증거가 아니다.

| 구성 | 평균 AUROC ± seed SD | 기준선 차이 pp | capacity FPS | steady p95 평균 ms | peak allocated 최대 GiB | 관측 Pareto |
|---|---:|---:|---:|---:|---:|---|
| B_fp32_k10 | 81.24 ± 0.12 | +0.00 | 129.6 | 8.23 | 0.554 | 예 |
| B_fp32_k5 | 80.77 ± 0.17 | -0.47 | 130.3 | 8.17 | 0.460 | - |
| B_fp32_k2 | 79.81 ± 0.11 | -1.43 | 132.0 | 8.10 | 0.404 | - |
| B_bf16_k10 | 81.06 ± 0.06 | -0.18 | 237.7 | 4.68 | 0.283 | 예 |
| B_bf16_k5 | 80.60 ± 0.10 | -0.65 | 239.7 | 4.63 | 0.238 | 예 |
| B_bf16_k2 | 79.69 ± 0.07 | -1.55 | 240.4 | 4.61 | 0.213 | - |
| S_fp32_k10 | 80.07 ± 0.18 | -1.17 | 227.3 | 4.87 | 0.203 | - |
| S_fp32_k5 | 79.65 ± 0.41 | -1.59 | 227.1 | 4.93 | 0.157 | - |
| S_fp32_k2 | 78.65 ± 0.53 | -2.59 | 228.3 | 4.90 | 0.128 | - |
| S_bf16_k10 | 80.29 ± 0.24 | -0.95 | 254.9 | 4.43 | 0.109 | 예 |
| S_bf16_k5 | 79.85 ± 0.45 | -1.39 | 260.6 | 4.23 | 0.085 | 예 |
| S_bf16_k2 | 78.81 ± 0.54 | -2.43 | 261.0 | 4.22 | 0.071 | 예 |

정확도 최대 / steady-state 처리 p95 최소 / peak allocated VRAM 최소의 비지배 집합이다.12개 후보 내 관측 결과이며 전역 최적·통계적으로 확정된 우월성이 아니다.
지연 차이5%를 동률로 보는 보조 집합: B_fp32_k10, B_bf16_k10, B_bf16_k5, S_bf16_k10, S_bf16_k5, S_bf16_k2.
FPS는 이번 실행에서 모든 후보를 재측정했다. 현재 GPU,원본 JPEG read/decode,batch1,capacity3회/30FPS paced1회,최대256frames. 카메라/네트워크 제외,OS cache가 warm일 수 있다. 지연과 memory는 seed0,정확도는3seed이다.
BF16은 실제 backbone/head/bank dtype 변환과 재추출 평가이며 RGB/feature normalization/거리 누산은FP32이다. 새 전체 feature cache를 저장하지 않았고 영상별 추출 해시와 raw score를 보존했다.
FP32 k10/k5 기준선 점수 등가성, k2 정상 sample pool 동등성,각 precision의 정상 보정,stream/batched 추출 오차는 실행 폴더에 있다.

R02영상12·13·14는 주 결과에서 제외하고±1정렬 민감도를 보존했다. 모든 원본·기존 결과를 유지하고 바이너리는 로컬에 보존한다.


## 6-2 추가 실험

[전체 기록](experiments/stage6_2_calibration/) · [고정 규약](docs/stage6_protocol.md)

# 6-2 실험 결과

R01–R04, seed 0·1·2. 정상 80% 학습 / 20% 보정 분리와 causal frame 규약 유지. 기존 테스트셋에서 추가 탐색한 결과이며 독립 데이터 일반화 증거가 아니다.

| anchor | 보정·임계값 | AUROC | active 경보 FPR (%) | 구간 recall (%) | 오경보 / 정상1000frame | 탐지 구간 지연 중앙값 평균(frame) |
|---|---|---:|---:|---:|---:|---:|
| B/k10 | baseline | 81.24 | 16.86 | 59.70 | 3.87 | 49.38 |
| B/k10 | balanced_fixed | 81.20 | 16.97 | 58.19 | 3.77 | 50.12 |
| B/k10 | balanced_cv | 81.20 | 12.77 | 61.10 | 4.38 | 50.21 |
| B/k10 | phase_mean_fixed | 79.41 | 29.88 | 82.78 | 6.67 | 26.58 |
| B/k10 | phase_mean_cv | 79.41 | 19.69 | 79.75 | 6.47 | 29.46 |
| B/k10 | phase_max_fixed | 77.53 | 31.99 | 77.86 | 5.70 | 20.67 |
| B/k10 | phase_max_cv | 77.53 | 27.48 | 76.02 | 6.71 | 23.08 |
| S/k5 | baseline | 79.65 | 15.56 | 56.53 | 3.50 | 55.50 |
| S/k5 | balanced_fixed | 79.68 | 15.58 | 55.89 | 3.51 | 59.88 |
| S/k5 | balanced_cv | 79.68 | 15.91 | 52.01 | 3.80 | 71.50 |
| S/k5 | phase_mean_fixed | 77.88 | 22.94 | 73.05 | 8.87 | 23.04 |
| S/k5 | phase_mean_cv | 77.88 | 16.20 | 66.61 | 5.57 | 35.83 |
| S/k5 | phase_max_fixed | 76.45 | 28.73 | 74.85 | 7.57 | 17.54 |
| S/k5 | phase_max_cv | 76.45 | 19.10 | 67.98 | 6.39 | 18.67 |

## 정상 CV

| anchor / 보정 | CV 오탐 제약 충족 실행 |
|---|---:|
| B_k10/balanced | 7/12 |
| B_k10/phase_mean | 4/12 |
| B_k10/phase_max | 4/12 |
| S_k5/balanced | 12/12 |
| S_k5/phase_mean | 5/12 |
| S_k5/phase_max | 2/12 |

CV는 정상 validation 영상을 하나씩 제외하고 나머지로 보정한다. 평균active FPR≤1%,최대영상FPR≤5%를 만족하는 가장 낮은q를 선택했다. 실패한 실행은q=.999 fallback이며 오탐 보장을 주장하지 않는다.
주 비교는 phase_mean_cv 대 baseline. fixed는 영상 균형 q99.5,cv는 정상 영상만으로 고른q. 영상/seed별 결과·CV 분할과 전q 후보·보정 통계·정답/경보 frame·미탐/경보선행/coldstart 구간은 보존했다.
탐지율은 정답1 연속구간 기준이며 이미 활성화된 경보도 포함한다. 지연은 탐지된 구간만의 값이고 미탐은 별도로 기록한다. 같은 테스트셋을 반복 관찰했으므로 새로운 환경 일반화가 입증된 것은 아니다.

R02영상12·13·14는 주 결과에서 제외하고±1정렬 민감도를 보존했다. 모든 원본·기존 결과를 유지하고 바이너리는 로컬에 보존한다.


## 6단계 통합 해석

[파레토·일반화 통합 결과](experiments/stage6_summary/research_findings.md)

## 7-1 실험

[전체 기록](experiments/stage7_1_diagnostics/) · [규약](docs/stage7_protocol.md)

# 7-1 결과

R01–R04, seed 0·1·2. 기존 테스트셋 사후 진단/추가 탐색이며 독립 일반화 검증이 아니다.

| 모델 | 점수 | AUROC | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame |
|---|---|---:|---:|---:|---:|
| B/k10 | appearance_score | 78.64 | 17.69 | 43.30 | 3.78 |
| B/k10 | temporal_score | 70.18 | 0.57 | 27.01 | 0.89 |
| B/k10 | combined_score | 81.24 | 16.86 | 59.70 | 3.87 |
| S/k5 | appearance_score | 77.05 | 17.07 | 35.93 | 4.02 |
| S/k5 | temporal_score | 66.46 | 0.36 | 30.36 | 0.75 |
| S/k5 | combined_score | 79.65 | 15.56 | 56.53 | 3.50 |

| 장면 B/k10 | 외형 FPR (%) | 시간 FPR (%) | 결합 FPR (%) | 30frame 이상 지속하는 오경보 프레임 비율 (%) |
|---|---:|---:|---:|---:|
| R01 | 62.61 | 0.29 | 59.83 | 91.90 |
| R02 | 1.07 | 0.01 | 0.13 | 0.00 |
| R03 | 4.60 | 1.19 | 4.16 | 48.58 |
| R04 | 2.47 | 0.81 | 3.31 | 0.00 |

외형·시간 각각 정상 보정 q99.5 및 3연속 경보를 사용한다. 각 component의 임계값이 달라 단독 성능 차이를 인과적 기여도라고 단정하지 않는다. 30frame은 길이 구분 기준이며 원본 촬영FPS를 가정하지 않는다.
영상별 정상 보정/테스트 정상/테스트 이상 score 분위수, 정상 상대위상 오차, 모든 오경보 정상 구간 길이, 미탐 구간의 threshold 초과 여부를 보존했다. 테스트 라벨을 이용한 통계는 사후 진단이며 설정 선택용 검증 성능이 아니다.


## 7-2 실험

[전체 기록](experiments/stage7_2_alerts/) · [규약](docs/stage7_protocol.md)

# 7-2 결과

R01–R04, seed 0·1·2. 기존 테스트셋 사후 진단/추가 탐색이며 독립 일반화 검증이 아니다.

| anchor | 방법 | AUROC | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame | 탐지 지연 중앙값 평균(frame) |
|---|---|---:|---:|---:|---:|---:|
| B/k10 | baseline | 81.24 | 16.86 | 59.70 | 3.87 | 49.38 |
| B/k10 | raw_cv | 81.24 | 11.74 | 54.37 | 3.75 | 47.25 |
| B/k10 | ewma_cv | 81.84 | 17.07 | 49.68 | 2.43 | 42.33 |
| B/k10 | hysteresis_cv | 81.24 | 14.35 | 48.31 | 1.85 | 47.90 |
| B/k10 | ewma_hysteresis_cv | 81.84 | 18.54 | 49.04 | 1.93 | 40.00 |
| S/k5 | baseline | 79.65 | 15.56 | 56.53 | 3.50 | 55.50 |
| S/k5 | raw_cv | 79.65 | 15.22 | 47.52 | 3.10 | 81.33 |
| S/k5 | ewma_cv | 80.36 | 16.39 | 36.22 | 2.41 | 83.38 |
| S/k5 | hysteresis_cv | 79.65 | 15.71 | 42.98 | 1.76 | 90.29 |
| S/k5 | ewma_hysteresis_cv | 80.36 | 17.55 | 36.54 | 1.81 | 83.19 |

주 비교는 ewma_hysteresis_cv 대 baseline. alpha=.2, 해제 임계값=진입의.7배, 진입3연속/해제3연속. raw_cv는 정상 CV로 threshold만 바꾸는 대조군이다. 임계값은 정상 영상 leave-one-out CV로 고르고 모든 후보 결과를 기록했다.

| anchor/규칙 | 정상 CV 제약 충족 |
|---|---:|
| B_k10/raw | 6/12 |
| B_k10/ewma | 3/12 |
| B_k10/hysteresis | 6/12 |
| B_k10/ewma_hysteresis | 2/12 |
| S_k5/raw | 12/12 |
| S_k5/ewma | 8/12 |
| S_k5/hysteresis | 7/12 |
| S_k5/ewma_hysteresis | 3/12 |

정상 CV 제약은 평균 활성FPR≤1%, 영상최대≤5%, 평균 오경보≤정상1000frame당1회이다. 미충족 시q=.999 fallback이며 보장으로 표현하지 않는다. EWMA 점수의 AUROC도 표시하지만 경보 횟수 감소가 정확도·미탐 개선을 의미하지 않는다. 탐지 지연은 탐지된 구간에 한정되며 미탐·coldstart·선행경보는 별도 파일에 보존했다.


## 7-3 실험

[전체 기록](experiments/stage7_3_full_stream/) · [규약](docs/stage7_protocol.md)

# 7-3 결과

R01–R04, seed 0·1·2. 기존 테스트셋 사후 진단/추가 탐색이며 독립 일반화 검증이 아니다.

| 정밀도 | batch1 AUROC ± seed SD | batch 추출 기준선 AUROC | 활성 FPR (%) | 구간 recall (%) |
|---|---:|---:|---:|---:|
| fp32 | 81.24 ± 0.12 | 81.24 | 16.85 | 59.70 |
| bf16 | 81.08 ± 0.10 | 81.06 | 16.47 | 60.34 |
| bf16_mixed | 81.07 ± 0.11 | 81.06 | 16.43 | 60.34 |

| 정밀도 | 전체영상 capacity FPS 평균 | 가장 긴 영상 paced E2E p95 최댓값(ms) | 최대 기한 초과율(%) | peak allocated 최대 GiB | 실제 단일모델 seed0 AUROC |
|---|---:|---:|---:|---:|---:|
| fp32 | 129.1 | 9.03 | 0.00 | 0.553 | 81.29 |
| bf16 | 238.0 | 6.44 | 0.00 | 0.288 | 81.06 |
| bf16_mixed | 237.6 | 6.21 | 0.00 | 0.300 | 81.05 |

FP32/BF16 모두 정상 보정·전체 테스트를 원본 JPEG에서 batch1로 다시 추출했다. bf16_mixed는 백본·bank BF16에 head만 FP32로 바꾼 대조이다. mixed의 batch 기준선 열은6단계 전체 BF16이며 동일한 mixed batch 실험이 아니다.
정확도 패스의 공유 추출 시간은 속도 측정에서 제외했다. 별도 단일 모델 seed0로 전체 유효 테스트 capacity1회와 장면별 가장 긴 영상 전체30FPS 재생을 측정했다. 실제 카메라/네트워크 지연과 새 환경 일반화는 포함하지 않는다.
normal_calibration_lovo.json에는 보정 영상 하나씩 제외한 모든 통계·임계값·heldout 정상 경보·테스트 민감도를 기록했다. 모델 재학습 분할 검증이 아닌 보정 집합 구성 민감도이다. R02 12·13·14 제외/±1 민감도 유지.


## 7단계 통합 결과

[원인 분석·경보·전체 스트림 검증](experiments/stage7_summary/research_findings.md)

## 8-1 LoRA 실험

[전체 기록](experiments/stage8_1_lora_validation/) · [규약](docs/stage8_protocol.md)

R01 정상 영상의 구현·학습 진단 완료. 테스트 성능 결과가 아닙니다.

추가 학습 파라미터 **49,152개**. 초기/복원 최대오차 0/0, FP32병합 최대오차 4.92e-07. 기존 가중치 불변·gradient 차단·유한 손실 확인.

고정 소수 프레임 진단 loss: 첫5회 0.000981 → 마지막5회 0.000912. 진단 peak allocated 1.004 GiB.

| 방법 | 표본 | epoch | 첫/마지막 epoch loss | 학습·검증 시간(초) | GPU peak GiB |
|---|---:|---:|---:|---:|---:|
| consistency | 1728 | 10 | 0.000948 / 0.000007 | 37.0 | 2.582 |
| anchored | 1728 | 10 | 0.001246 / 0.001159 | 36.8 | 2.552 |

손실 정의가 다르므로 두 방법의 총 loss 크기로 우열을 비교하지 않습니다. 고정10epoch를 사용하며 테스트를 확인한 checkpoint 선택이 없습니다.
R01 두 실행 평균 36.9초. 다른 장면의 표본 수·특징 재추출·head/메모리 학습·전체 스트림 시간을 포함하지 않는 측정입니다.


## 8-2 LoRA 실험

[전체 기록](experiments/stage8_2_lora_comparison/) · [규약](docs/stage8_protocol.md)

R01–R04, seed 0, fp32_batch32. 장면별 단순 평균 후 seed 평균±표준편차. 주 후보는 사전 고정 anchored이며 테스트 최상 모델을 고르지 않습니다.

| 방법 | AUROC (%) | AUPRC (%) | 외형 AUROC (%) | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 81.30 | 76.07 | 78.73 | 16.85 | 58.39 | 3.69 |
| consistency | 73.25 | 63.58 | 75.04 | 17.55 | 40.12 | 4.24 |
| anchored | 81.03 | 75.56 | 78.62 | 16.79 | 60.66 | 3.85 |

anchored − frozen: AUROC -0.27pp, 활성 FPR -0.06pp, 구간 recall +2.27pp.

정상 오탐시간·발생횟수·미탐을 함께 해석합니다. 구간 recall은 시작 전부터 켜진 경보를 포함하고 지연은 탐지된 구간만 계산하므로 이 수치만으로 운영 신뢰성 향상을 주장하지 않습니다. R02 12/13/14 제외 및 동일 유효 프레임 유지. 이미 관찰한 테스트의 후속 탐색이며 새 환경 검증이 아닙니다.


## 8-3 LoRA 실험

[전체 기록](experiments/stage8_3_lora_stream/) · [규약](docs/stage8_protocol.md)

R01–R04, seed 0,1,2, bf16_batch1. 장면별 단순 평균 후 seed 평균±표준편차. 주 후보는 사전 고정 anchored이며 테스트 최상 모델을 고르지 않습니다.

| 방법 | AUROC (%) | AUPRC (%) | 외형 AUROC (%) | 활성 FPR (%) | 구간 recall (%) | 오경보/정상1000frame |
|---|---:|---:|---:|---:|---:|---:|
| frozen | 81.08 ± 0.08 | 75.92 ± 0.34 | 78.62 ± 0.13 | 16.47 ± 0.08 | 60.02 ± 2.53 | 3.78 ± 0.08 |
| consistency | 67.24 ± 2.97 | 57.72 ± 2.32 | 68.65 ± 2.53 | 1.36 ± 0.54 | 33.18 ± 1.26 | 3.79 ± 2.31 |
| anchored | 80.97 ± 0.20 | 75.68 ± 0.57 | 78.55 ± 0.11 | 16.32 ± 0.07 | 60.46 ± 0.68 | 3.60 ± 0.10 |

anchored − frozen: AUROC -0.11pp, 활성 FPR -0.15pp, 구간 recall +0.44pp.

정상 오탐시간·발생횟수·미탐을 함께 해석합니다. 구간 recall은 시작 전부터 켜진 경보를 포함하고 지연은 탐지된 구간만 계산하므로 이 수치만으로 운영 신뢰성 향상을 주장하지 않습니다. R02 12/13/14 제외 및 동일 유효 프레임 유지. 이미 관찰한 테스트의 후속 탐색이며 새 환경 검증이 아닙니다.

| 방법 | capacity FPS (장면 평균) | paced p95 최대(ms) | peak allocated 최대 GiB | 최대 deadline miss (%) |
|---|---:|---:|---:|---:|
| frozen | 236.5 | 6.28 | 0.294 | 0.000 |
| consistency | 236.6 | 6.30 | 0.292 | 0.000 |
| anchored | 237.4 | 6.41 | 0.292 | 0.000 |

실측: seed0, 단일모델·전체 유효 JPEG capacity1회·각 장면 최장영상 전체30FPS paced1회. 카메라/네트워크 제외, OS 캐시와 다른 프로세스 영향 가능. LoRA 병합은 추가분기를 제거하며 백본 경량화를 뜻하지 않습니다.


## 8단계 통합 결과

[정상 적응·오탐·미탐·실시간 검증](experiments/stage8_summary/research_findings.md)

## 8비트·4비트 양자화 예비 진단

정상 96프레임의 구현 진단을 완료했습니다. 전체 AUROC·실시간 VAD 평가는 아직 수행하지 않았습니다. [규약](docs/quantization_probe.md) · [전체측정·로그](experiments/quantization_probe/)

# 8비트·4비트 DINOv2 양자화 예비 검증

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

## 권장 본 실험 범위 (아직 미실행)

1. 고정 DINOv2 BF16 ↔ INT8 weight-only ↔ INT8 weight/activation. 양자화된 백본의 특징으로 프로토타입·위상 예측기를 재구축하고 정상 보정을 다시 수행. 기존 위상 예측기·메모리를 그대로 사용하는 직접 변환 결과도 구분해 기록.
2. INT4 group128/32 및 정상 특징 오차로 사전 선정한 민감층 BF16 혼합. 테스트 AUROC로 비트 수·층을 선택하지 않으며, 메모리 압축 효과와 특징 손실을 분리.
3. R01–R04 동일 분할·유효 프레임·3개 seed의 AUROC/AUPRC, 활성 FPR, 구간 recall 및 실제 batch 1·30 FPS 측정. compile 최적화는 BF16 대조군에도 동일하게 적용하고 초기 compile 시간을 별도로 보고. 전체 가중치의 중복 저장을 피하고 10 GiB 안전마진 유지.

## 재현

```bash
uv pip install --python .venv/bin/python --target cache/quantization/torchao017 --no-deps --no-cache torchao==0.17.0
.venv/bin/python scripts/probe_quantization.py --variant bf16
.venv/bin/python scripts/probe_quantization.py --variant int8
.venv/bin/python scripts/probe_quantization.py --variant int4
.venv/bin/python scripts/summarize_quantization_probe.py
```

[TorchAO 공식 추론 문서](https://docs.pytorch.org/ao/stable/workflows/inference.html) · [예비 실행 규약](https://github.com/PigeonLabs/KNU_Capstone1_VAD/blob/main/docs/quantization_probe.md)


## 9-1 양자화 연산·컴파일 최적화

[전체 기록](experiments/stage9_1_quant_compile/) · [규약](docs/stage9_protocol.md)

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
