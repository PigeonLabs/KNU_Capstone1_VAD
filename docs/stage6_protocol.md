# 6-1·6-2 사전 고정 실험 규약

2026-09-25 사용자 실행 승인. R01–R04,seed0/1/2,기존 정상80% 모델 적합/20% 보정 분리를 유지한다. 현재 RTX PRO6000,단일30FPS,batch1이 목표이다. 승인된 각 실험 완료 후 코드·한국어 README·분석·로그·점수·해시만 main에 자동 게시한다. 원본/기존 결과/바이너리 보존,10GiB 이하 중지·보고·자동재개 금지. 예약 모니터를 만들지 않는다.

## 6-1: 제한된 후보군에서의 파레토 비교

후보군은 backbone B/S × 메모리 k10/k5/k2 × precision FP32/BF16의12개이다. k는 점유 위상 수에 곱하는 prototype 수이며 실제 bank 크기를 보고한다. head는5단계 정상80% causal head를 고정한다. k10/k5 bank는5단계 재사용,k2만 동일 정상 sample pool/seed/kmeans20회로 적합한다. backbone·head 재학습 없음. 모델과 bank의 BF16 변환은 사후 양자화이며 BF16에 맞춘 추가 학습은 하지 않는다.

BF16에서 backbone/head/bank 가중치를 실제 BF16으로 적재한다. RGB 전처리와 feature L2 normalization은FP32, cosine matching은 BF16 입력/FP32 누산·출력(torch.bmm out_dtype=float32),시간 잔차·점수 보정은기존 정밀도를 유지한다. BF16 backbone을 실제 실행해 정상 validation·test 특징을 다시 계산하고 해당 precision에서 정상 보정·임계값을 다시 산출한다. FP32 특징을 단순 반올림해 BF16 모델 정확도로 대신하지 않는다. 특징은 영상 단위 RAM/GPU에서 처리하고 전체 feature checksum·source/model hash·raw frame score를 저장하며 새 전체 특징 캐시는 디스크에 쓰지 않는다.

FP32 k10/k5 점수·보정은 기존5단계 재사용하며 평가 frame/label 일치를 검산한다. FP32 k2는 기존FP16 저장 cache 기반으로 평가한다. BF16 output은 FP32 정규화 feature를 추가FP16 저장 없이 평가한다. 이 차이와 batched 특징 추출/단일frame 검산을 기록한다. 공통 frame35..N-18,운영 점수 frame35..N-1. R02 12/13/14 주평가 제외,±1 정렬 민감도 보존.

속도는12설정 모두 같은 이번 실행에서 새로 측정한다(seed0,4장면). 장면 첫 유효 영상의 원본 JPEG 최대256frames,capacity3회와30FPS paced replay1회. 같은 decode/preprocess/causal head/time/alarm 경로,영상별 reset,35frame coldstart,카메라/네트워크 제외. 다른본프로젝트GPU작업과병행하지 않는다. batch1 steady-state p95 처리지연,capacityFPS,paced E2E p95/기한초과,peak allocated/reserved VRAM,bank/model bytes를 저장한다.

관측 파레토 집합의 주 목적은3-seed macro AUROC 최대화 / 장면별 steady-state 처리 p95 평균 최소화 / 장면 최대 allocated VRAM 최소화이다. 모든 목적이 같거나 낫고 하나 이상 엄격히 나으면 지배한다고 정의한다. 측정 잡음 민감도를 위해 지연 차이5% 이내를 동률로 보는 보조 분석을 별도로 기록한다. frontier는 이12개 후보와현재테스트셋에서의 관측 결과이며 전역 최적·독립 generalization 증거가 아니다. 기존기준선 대비1pp 이내 손실/30FPS 조건도 표시하되 테스트 최고만 선택해새성능으로포장하지 않는다.

## 6-2: 정상 영상 간 보정 일반화와 경보 개선

6-1 test 점수로 선택하지 않고 anchor를 B/k10/FP32와 S/k5/FP32로 미리 고정한다. 정상 모델/메모리는5단계 그대로 사용한다. 검증 영상20%만 보정에 사용하고 테스트·초기 테스트 구간을 정상이라고 가정하여 적응하지 않는다.

기존 pooled median/q99.5 scaling,component 평균,pooled q99.5 threshold,3연속 경보를 기준선으로 보존한다.

세 보정 후보: (1) video-balanced global + 평균 결합,(2) video-balanced phase shrinkage + 평균 결합(주 후보),(3) 동일 phase shrinkage + max 결합. 각 영상의 총 가중치가 같도록 weighted empirical quantile를 정의하고 median/q99.5에서 scaling=max(0,(x-median)/max(q99.5-median,1e-8))을 사용한다. 위상은 모델이 예측한200class를20bin으로 매핑하며 정답 위상을 입력하지 않는다. 현재bin±1의 circular 이웃 정상 데이터가100frame 이상/2영상 이상이면 local median/q99.5를 global 통계와 n/(n+200)로 혼합,부족하면 global fallback한다. 학습된 통계는 테스트 도중 고정한다.

각 후보에 두 threshold 규칙을 모두 보고한다: fixed video-balanced q99.5 / normal-only leave-one-validation-video-out CV 선택. CV는 나머지 정상 영상에서 scaling과threshold를 적합하고 제외한 영상의3연속 active-alarm FPR를 평가한다. q 후보[.95,.975,.99,.995,.999] 중 fold 평균 FPR≤1%,최대 fold FPR≤5%를 만족하는 가장 낮은q를 선택한다. 조건을 만족하는 후보가 없으면 q=.999를 사용하되 CV 제약 실패를 명시하고 오탐 보장을 주장하지 않는다. 최종 보정은 모든 정상 validation 영상으로 적합한다. 주 비교는 phase-mean CV vs 기존 pooled baseline이며 모든7개 변형을 공개한다.

이 절차는 상관된 영상 frame에 대한 empirical CV이며 distribution-free/conformal coverage 보장이 아니다. 소수 정상영상 간 일반화와이미여러번관찰한테스트셋의추가탐색을 구분한다. AUROC/AUPRC,frame FPR/active-alarm FPR,오경보 episode,구간 recall/미탐/coldstart/경보선행/탐지지연을 함께 보고한다. 한쪽만 개선되면 동시개선이라고 하지 않는다. 같은 모델·같은 frame support로 비교하고 R02±1민감도 보존한다. 정상 calibration 영상별 component 분포 및 CV 실패를 남긴다.

## 검증·산출물

정상 split/hash 보존,FP32 기준선 등가성,BF16 실제 dtype/FP32 누산,유한한 점수,새 k2 bank sample 동등성,정상 CV train/heldout 분리,예측위상만 사용,미래frame 독립,파레토 지배관계 테스트. 6-1 전체 정확도·재측정 지연표·관측 frontier,6-2 CV 조건·전 변형 경보 비교표,프레임별 점수·정답·실행 명령·실패/재시도·환경/source/model/hash를 보존한다. S/B의 새 feature cache를 추가로 만들지 않아 예상 추가 디스크 사용은 분석/모델/게시를 합해수GiB 이내이다.
