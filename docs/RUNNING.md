# IPAD 실제 장면 재현 및 DINOv2 비교

R01–R04만 사용하는 연구 파이프라인입니다. 원본 `IPAD_dataset`과 PDF는 수정하지 않습니다.
합성 데이터 학습·LoRA 전이는 포함하지 않습니다. 학습이 끝나기 전에는 논문 성능 재현을 주장하지 않습니다.

## 실행

```bash
# 이미 만들어진 프로젝트 전용 환경
.venv/bin/python -m pytest -q
.venv/bin/python -m ipad.data --cache

# 전체 순차 실험: 원 방법 4개 장면 → DINOv2 A/B → 메모리 제거 실험
.venv/bin/python scripts/run_suite.py --stage all

# 독립 프로세스로 실행 (같은 작업을 중복 실행하지 마세요)
.venv/bin/python scripts/launch_suite.py --stage all
.venv/bin/python scripts/status.py

# 단계별 실행, 이미 완료된 단계는 건너뜁니다.
.venv/bin/python scripts/run_suite.py --stage paper
.venv/bin/python scripts/run_suite.py --stage extensions
.venv/bin/python scripts/run_suite.py --stage ablations

# 상태 확인
cat runs/suite_status.json
cat runs/paper/R01/seed0/progress.json
tail -f runs/paper_R01_s0.log

# 최신 결과 보고서 생성 (학습 중이면 미완료로 표시)
.venv/bin/python -m ipad.report
```

실행기는 동시에 하나만 실행할 수 있습니다. 종료한 실험은 같은 명령으로 다시 시작하면 마지막 5-epoch
체크포인트에서 재개합니다. 컴퓨터가 꺼지거나 절전되면 계산은 계속되지 않습니다. OS 서비스나 주기적 자동화는 설치하지 않습니다.
학습 중 모델 상태와 Adam 상태를 저장하므로 재개 시 학습률·위상 축 등의 설정을 바꾸면 오류를 냅니다.

## 개별 실행

```bash
.venv/bin/python -m ipad.train --scene R01 --output runs/paper/R01/seed0
.venv/bin/python -m ipad.evaluate --checkpoint runs/paper/R01/seed0/model.pt
.venv/bin/python -m ipad.evaluate --checkpoint runs/paper/R01/seed0/model.pt --dino
.venv/bin/python -m ipad.features --scene R01
.venv/bin/python -m ipad.prototype --scene R01 --output runs/prototype/R01/seed0
```

기본 재현은 16프레임, 256×256, batch 8, Adam 1e-4, 50 epochs, FP32, seed 0입니다.
`--smoke-steps`, `--steps-per-epoch`는 실행 검증 전용이며 완전한 재현 결과로 표시되지 않습니다.
`--subset train`은 영상 단위로 고정한 80% 개발 학습 split, `--subset all`은 최종 비교용 전체 정상 학습입니다.
공식 설정을 그대로 고정한 본 학습에는 테스트 성능에 의한 체크포인트 선택을 사용하지 않습니다.
프로토타입 위상 분류기는 별도 80/20 정상 영상 검증 후 전체 정상 영상으로 재학습합니다.

## 구현 및 출력

| 코드 | 역할 |
|---|---|
| `ipad/data.py` | 숫자순 프레임, 16프레임 클립, 정상 영상 분할, 데이터 감사·캐시 |
| `ipad/model.py` | 공식 Swin/I3D 계열 구조, 배치별 주기 메모리, 위상 분류 |
| `ipad/train.py` | 학습·FP32 검증·checkpoint 재개 |
| `ipad/evaluate.py`, `ipad/metrics.py` | 동일 프레임 정렬, offline scene-wide 점수, R02 민감도 |
| `ipad/features.py` | 고정 DINOv2, CLS/6·12층 patch 특징, 디스크 캐시 |
| `ipad/prototype.py` | 시간 순서를 보존한 위상 MLP, 공간별 prototype, NN/soft projection |
| `ipad/report.py` | 완료 상태와 논문 대비 성능표 |

- `reports/data_audit.json`: 모든 실제 장면 이미지의 디코딩 검사와 라벨 불일치.
- `reports/cuda_check.json`, `reports/pytest.txt`: CUDA 및 회귀 검증.
- `runs/smoke/fp32/smoke.json`: 실제 클립 학습·체크포인트 동등성 검증.
- `runs/paper/<scene>/seed0/`: 설정, epoch 로그, `last.pt`, 최종 `model.pt`.
- `evaluation/scores.csv`: scene/video/frame, 정답, 재구성·위상 점수.
- `evaluation_dino/scores.csv`: 같은 모델·프레임의 DINOv2 특징 비교.
- `runs/prototype/<scene>/seed0/`: 위상 모델·메모리·정량 결과.
- `reports/results_seed0.md`: 전체 실행이 끝난 뒤 방법별 비교표.

원 학습 정상 데이터 50,642프레임, 테스트 33,462프레임입니다. R02의 불일치 3개 영상은
주 평가에서 제외하고 별도의 공통 길이/±1 label offset 결과를 저장합니다. 전체 평균은 이 제한을 명시합니다.

## 재현의 한계

논문과 공개 코드가 완전히 일치하지 않습니다. 특히 공개 위상 head를 유지하면 263,478,713개 파라미터로,
논문 표의 35.9M과 다릅니다. 상세 선택·근거는 `REPRODUCTION.md`에 기록했습니다.

위상 검사 5개 window와 16프레임 중앙 재구성은 미래 프레임을 사용합니다.
scene-wide test normalization은 논문 비교용 offline 처리이며, 온라인 임계값 또는 실시간 지연의 검증이 아닙니다.
DINOv2의 사전학습과 원 IPAD random initialization의 정보 차이도 성능 해석에서 고려해야 합니다.

## 환경 및 출처

현재 환경의 정확한 버전은 `requirements.lock.txt`에 저장됩니다. 재설치 시 GPU/CUDA 호환성을 먼저 확인합니다.

```bash
uv venv .venv --python 3.13
uv pip install --python .venv/bin/python torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r requirements.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

- IPAD: https://github.com/LJF1113/IPAD, commit `22764cbeeda3946303d236babdd2664fd6241b91`.
- DINOv2: https://github.com/facebookresearch/dinov2, commit `7764ea0f912e53c92e82eb78a2a1631e92725fc8`.
- 원 Swin/decoder 파일은 `ipad/vendor/PROVENANCE.json`의 SHA256으로 검증하며 수정하지 않았습니다.
- 공식 IPAD 저장소에서 별도의 라이선스 파일은 확인하지 못했습니다. 포함된 코드의 저작권은 원 저자에게 있습니다.
  DINOv2 소스는 해당 저장소의 라이선스를 함께 캐시합니다.
