# 산업 공정 영상 이상탐지: IPAD 재현과 DINOv2 비교

R01–R04 실제 공정 영상만 사용합니다. **1단계는 논문 방법론 재현, 2단계는 DINOv2 도입**입니다. 합성 데이터와 LoRA 전이는 이번 실험에서 제외합니다.

## 단계별 진행

| 단계 | 목적 | 상태 | 기록 |
|---|---|---|---|
| 1단계 | Swin-T + 주기 메모리 + 재구성 + 주기 검사 | 4개 장면 50 epochs 완료 · seed 0 | [전체 자료](experiments/stage1_reproduction/) |
| 2단계 | DINOv2 입력–복원 특징 비교 / 비재구성 prototype | 4개 장면 완료 · seed 0 | [전체 자료](experiments/stage2_dinov2/) |
| 1단계 추가 검증 | 메모리 제거 ablation | 실행 결과가 생기면 단계별 추가 기록 | [자료](experiments/stage1_memory_ablation/) |
| 3단계 제안 | 위상 조건의 유효성과 위상 추정 오차 분리 | 제안 상태 · 미실행 | [실험안](docs/stage3_proposal.md) |

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
- 실험 완료 단위마다 main에 commit/push하며 원격 변경은 강제로 덮어쓰지 않습니다. 공개 clone에서는 자동 push가 기본 비활성화됩니다.
- 디스크 여유가 **10 GiB 이하**가 되면 이 프로젝트의 실험을 일시중지하고 보고합니다. 자동 재개하지 않습니다.
- 메모리 제거 실험은 1단계 추가 검증입니다. 3단계 제안은 사용자 선택 전까지 실행하지 않습니다.
- 테스트 전체 정규화와 미래 프레임을 포함하는 centered window를 사용하므로 온라인/인과적 실시간 성능 주장이 아닙니다.
[기록 규칙](docs/EXPERIMENT_LOG_POLICY.md) · [코드–논문 차이](REPRODUCTION.md) · [3단계 제안](docs/stage3_proposal.md)

## 원 자료

- [IPAD 논문 v1](https://arxiv.org/abs/2404.15033v1) · [공식 코드](https://github.com/LJF1113/IPAD), commit `22764cbeeda3946303d236babdd2664fd6241b91`.
- [DINOv2 공식 구현](https://github.com/facebookresearch/dinov2), commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- upstream 코드의 재배포 대신 출처·SHA256을 보존하고 bootstrap에서 원본을 내려받습니다.
