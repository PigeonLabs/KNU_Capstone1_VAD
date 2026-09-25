# 8단계: 정상 영상 기반 DINOv2 LoRA 적응

2026-09-25 사용자가 8-1·8-2·8-3 실행 승인. R01–R04, 기존 정상 train80%/calibration20% 영상 분할을 유지한다. 기존 테스트를 이미 관찰한 후속 탐색이며 새 환경에 대한 독립 검증으로 주장하지 않는다. 테스트 정답으로 방법·epoch·rank·손실 가중치를 선택하지 않는다. 원본/기존 결과 보존, 10 GiB 이하 중지 및 자동 재개 금지, 완료 단계별 코드·분석·점수·로그·SHA256만 main 자동 게시. 예약 모니터 생성 없음.

## 사전 고정 방법

장면별 독립 ViT-B/14, 입력252, 마지막4블록(index8..11)의 fused qkv 중 Q/V에만 독립 rank4/alpha8 LoRA. K·bias·기존 가중치는 고정한다. LoRA A=Kaiming, B=0; 추가49152개 파라미터. 원본 pretrained/source commit은 기존 단계와 같다. 모델은 eval 모드로 stochastic depth/dropout 없이 adapter만 gradient 활성화. 기존 inference_mode 추출 경로와 학습 경로 분리.

세 방법: frozen(고정 DINO), consistency(증강 일관성), anchored(일관성+원본 특징 보존, 주 후보). 같은 메모리 크기k10/총최대200, 동일 seed별 정상 표본, head구조/학습법, 정상 보정과 경보 규칙.

학습 표본은 각 train 영상의 균등 시간 구간에서 최대64프레임(구간 내 seed별 무작위1개) 고정; 모든 epoch에서 같은 표본 순서만 shuffle. 정상 cal 영상은 학습에 사용하지 않는다. RGB 밝기 배율/대비 배율 각각 Uniform[.9,1.1], 이미지 평균 기준 대비, [0,1]clip; 위치·시간 불변성은 강요하지 않는다. 강한 색 변화/crop/flip 없음. 약한 광도 증강도 실제 모든 이상 의미를 보존한다는 보장은 없다.

Lcons=mean(1-cos(student(original) patch,student(augmented) patch)). Lanchor=mean(1-cos(student(original) patch,frozen_teacher(original) patch)). 두 student 분기에 gradient, teacher는 gradient 없음. anchored는 Lcons+1.0*Lanchor, consistency는 Lcons. 동일 가중치를 teacher 모드(LoRA비활성)로 읽어 별도 full teacher 체크포인트를 만들지 않는다. 정상 특징 수렴이 이상 분리 개선을 보장하지 않으며, patch 분산·원본과 cosine변화도 기록한다.

AdamW lr1e-4 weight_decay.01, batch32,10epochs, gradient clip1.0. BF16 autocast에서 base/LoRA masterweights FP32, loss/정규화 FP32, GradScaler불필요. 고정10epoch 최종 adapter 사용(테스트/보정셋 best checkpoint 선택 없음). 학습 history/매batchloss/gradient/표본/매epochadapter와 optimizer 상태 저장. epoch별 정상 진단은 train내 고정 소수 표본만 사용. 실패시 부분결과 보존하고 자동 덮어쓰기 금지.

## 8-1 구현·학습 검증

R01 train seed0: LoRA0일 때 frozen일치, K불변, basegradient없음/adaptergradient유한, 소수 정상프레임 반복학습, checkpoint복원과 FP32merge 출력 검산. merge 허용오차 CLS/patch 최대2e-5. 실제 시간/GPU peak/디스크량 측정 후 실행예상 갱신. R01 seed0 두 adapter를 고정10epoch 학습하여 8-2에서 그대로 사용한다. 테스트 AUROC를 읽기 전에 freeze.json 생성. 미니배치 감소 확인은 학습 동작 진단이지 일반화 증거가 아니다.

## 8-2 seed0 전체 비교

세 방법 × 네 장면. train 원본 특징을 다시 추출하여 CLS만 소형 로컬 저장, 메모리 후보 patch는 RAM에만 유지(최대512/위상bin, 기존20bin표본선택). 같은 공간 위치에서 spherical kmeans20iterations; FP16저장 후 FP32정규화. 위상 head는 과거16CLS→512→200, Adam1e-4/50epochs/batch128; 기존train frame15..N-8 그대로. 새 특징에 맞춰 head/bank 전부 재학습하고 frozen도 같은 경로 재구축. 전체 patch 캐시 신규 저장 없음.

FP32병합 모델로 batch32 정상cal/전체test 추출. 영상 경계별 과거16CLS,21phase reset. 정상 주기중앙값, component median/q99.5, score평균, 정상combined q99.5 threshold 및3연속 경보. 공통 frame35..N-18/운영35..N-1, R02 길이불일치12/13/14제외 및 공통길이±1민감도. 외형/시간/결합 AUROC·AUPRC, 활성FPR,구간recall,선행경보,지연(탐지된 구간만),오경보 횟수 모두 기록. 세 방법 전체 공개하며 테스트 최상 변형을 선택하지 않는다.

## 8-3 반복·병합 BF16 실시간

설정 변경 없이 seed1/2의 세 방법 재학습/FP32평가. 모든12scene-seed 조합×3방법의 merged BF16 백본/head/bank를 실제 JPEG batch1로 정상cal와 전체test 재평가한다. RGB·특징정규화·cosine누산 FP32. 해당 배치1 정상cal로 임계값 재적합. 시간/점수/경보 상태 매영상 reset, 미래프레임 미사용. seed0의 세방법×4장면은 단일모델 전체 유효test capacity1회 및 장면별 최장 유효영상 전체30FPS paced1회 측정. 정상train36frame warmup, JPEGread/decode 포함, 카메라/네트워크 제외, 다른프로세스 종료 없음. BF16배치1 정확도와 전용벤치마크 점수/경보 일치 검산. 모델을 작게 만든 실험이나 타장비 실시간 보장으로 주장하지 않는다.

## 판정

주 후보anchored가 frozen대비 AUROC·구간recall을 유지하면서 활성FPR을 줄이는지 보고한다. 장면별/seed별 수치와 평균·표준편차, paired차이를 모두 공개. consistency대비anchor효과 분리. 개선 없거나 특징 붕괴/미탐 증가도 결과다. 정상cal leave-one-video-out 민감도는 새 모델학습 분할 일반화가 아니다. 완전한새공정 일반화/유형별 성능 주장은 하지 않는다.
