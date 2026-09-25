# 5-1·5-2·5-3 사전 고정 실험 규약

2026-09-25 사용자가 세 실험 실행과 완료 실험별 main 자동 게시를 승인했다. 목표 장비는 현재 RTX PRO 6000 Blackwell 96GB, 단일 입력 30 FPS, batch 1로 명시적으로 확정했다. 다른 장비·현장 카메라 성능으로 일반화하지 않는다. R01–R04만 사용한다. 디스크 여유 10 GiB 이하에서는 중단·보고하며 자동 재개·자료 삭제하지 않는다.

## 공통 데이터와 선택 규칙

seed 0/1/2, 기존 파일 단위 정상 80/20 분할을 고정한다. 모든 5단계 모델과 메모리는 80%에만 적합한다. 20% 정상 영상은 점수 보정과 경보 임계값 산출에만 사용하며 전체 데이터 재학습하지 않는다. 이 변경 때문에 기존 4단계 80.97%와 직접 차감하여 경량화/온라인 손실이라고 주장하지 않는다. 정상 검증셋에서 임계값을 맞춘 오탐률은 독립 검증 성능이 아니다. 테스트 정상 구간의 오탐을 별도로 보고한다.

DINOv2 frozen B/14 또는 S/14,252x252,RGB/ImageNet normalization,CLS 및12층 patch L2 normalization. B 특징은 기존 FP16 캐시를 재사용하며 S는 동일 pinned source commit에서 다시 추출한다. 새 cache도 FP16/추출 FP32로 고정한다. backbone checkpoint/source/cache 및 학습 분할·코드·환경의 해시를 보존한다. 새 특징 캐시 예상19.6GiB, 기존 자료 삭제 없음.

학습 head는16 CLS ->512 ReLU ->200 phase, Adam1e-4,50epochs,batch128이다. 정답은 입력의 예측 대상 프레임 t*200/N이다. 길이는 정상 학습 위상 타깃에서만 사용하며 테스트 위상 입력에는 사용하지 않는다. 두 head를 동일한 정상 학습 target frame t=15..N-8에서 비교한다. causal 입력은[t-15,t],centered 입력은[t-8,t+7]. 테스트 causal은 t=15..N-1,centered는 t=15..N-8에서 예측한다. 모든 t는0-based frame ID이다.

메모리는 무조건부 공간 위치별 cosine NN. 위상 조건 routing은 이번 범위에 추가하지 않는다. 정상 train target frame pool을20개 상대 위상으로 나누고 각 구간 최대512개를 seed로 선택한다. 이 pool은 backbone/용량 간 동일하다. 공간별 spherical k-means20회. K는 점유 위상 수*k(k=10/5); 최대200/100이며 실제 개수·바이트·선택 frame을 기록한다. 정상 주기 기준은 train80 영상 길이 중앙값. 원본 영상 사이 window를 만들지 않는다.

## 5-1 온라인 기준선

B/14,k10. centered head와 causal head 각각 고정50epochs 학습. 외형 점수는 같은 bank/같은 프레임으로 계산한다.

시간 검사는 위상 잔차의 circular absolute mean: centered window21 offsets[-10,+10],causal window21 offsets[-20,0],현재 예측 위상을 기준으로 정상 속도와의 차이를 측정한다. causal 첫 유효 점수 frame35(15+20). causal이 미래 정보에 독립적인지 prefix perturbation 테스트한다.

고정 점수 보정은 각 raw component의 정상 validation median,q99.5를 사용해 max(0,(x-median)/max(q99.5-median,1e-8)); 상한 clipping은 하지 않는다. 외형·시간 동일 가중 평균. 프레임 경보 임계값은 정상 validation의 결합 점수 q99.5이며 엄격한 > 비교. 실제 경보는 임계값3프레임 연속 초과 시 발생하며 초과 episode당1회, 아래로 내려가면 reset한다. 테스트 값으로 보정/임계값을 수정하지 않는다.

비교: centered+test minmax(기존 평가 형태에 가까운 보조군),centered+고정 정상 보정,causal+고정 정상 보정 및 causal 외형 단독. 공통 평가 frame t=35..N-18. 온라인 운영 평가는 전체 causal 유효 frame35..N-1도 별도 보존한다. centered head부터 다시 학습하는 통제 실험이며 기존4-1 결과를 재현했다고 부르지 않는다. R02영상12/13/14 주평가 제외,공통길이±1 민감도 보존.

## 5-2 백본·메모리 축소

B/k10(5-1 재사용),B/k5,S/k10,S/k5의2x2 비교. S head 재학습,메모리 재구축,보정/임계값 재산출. B/k5는 B head 재사용. 정확도 평가는 위5-1과 동일 공통 frame/label. full causal B/k10 대비 macro AUROC 손실1pp 이내를 사전 실용성 기준으로만 사용하며 통계적 비열등성을 주장하지 않는다. 장면별·seed별 결과도 모두 보고한다. 해상도/INT8/추가 학습/적응형 추출 등은 추가하지 않는다.

## 5-3 경보와 실제 단일 프레임 실행

모든2x2 모델·4장면·3seed의 causal full support에서 AUROC/AUPRC,고정 threshold frame FPR,3연속 경보 episode 빈도,정답1 연속구간 탐지율/미탐률·최초경보 지연(frame)을 보고한다. 경보가 없으면 delay에서 누락시키지 않고 미탐으로 별도 집계한다. 정답1 연속구간은 프레임 라벨로 구성한 segment이며 실제 고장 유형/독립사건 정답이 아니다. cold start와 겹치는 구간은 별도 집계하고 주 지연 통계에서 제외한다. 30 FPS 환산 seconds는 가정이며 원본 촬영 FPS 검증값이 아니다. 정상 frame1000개당 오경보와 가정30FPS 분당 환산을 모두 표시한다.

성능 benchmark는 seed0,4장면,4구성,각 장면 첫 유효 테스트 영상 최대256연속 원본 JPEG. batch1,FP32,디스크 읽기·JPEG decode·256resize·RGB252 preprocess·DINO·past CLS head·matching·time score·고정 threshold·alarm까지 측정한다. 연산 warmup은 정상 train 영상16프레임을 사용하고 테스트 상태는 reset한다. 단일 구성만GPU에 적재해 peak allocated/reserved VRAM을 각각 측정한다. 배포 model/bank bytes와 전체 프로세스 VRAM은 구분한다. OS disk page cache가 warm일 수 있음도 표시한다.

(a) capacity3회 반복: sleep없이 실제batch1 처리,p50/p95/max latency 및 total FPS 측정. (b)30FPS paced replay1회: 미래 프레임을 읽지 않고 t/30 도착 시각까지 기다린 뒤 파일 read부터 순차 처리. 각 frame의 processing time,대기시간,end-to-end delay,33.3ms deadline miss를 기록한다. queue 적체 추세/최대대기 포함. 실제 카메라·인코더·네트워크 지연은 미포함. 일반 사용자GPU 프로세스는 중지하지 않으며 시작 시 nvidia-smi 기록. 성능 측정 중 별도 본 프로젝트 GPU 실험을 병행하지 않는다.

스트리밍 raw score와 FP16 cached 평가 score의 차이는 비교하되 batch/FP16 양자화로 완전동일을 요구하지 않는다. cached quantized reference와 live 단일 frame 결과를 동일 decode/preprocess로 검산한다. 실시간 경보 지표는5-1/2 cached full-test metric이며 benchmark256 frame의 직접 경보 결과는 별도 로그로 보존한다.

## 완료 및 게시

각 stage 설정·명령·매배치 loss/gradient·환경/source/모델 SHA256·frame ID/정답/점수·실패/재시도·훈련 시간·성능을 보존한다. checkpoint restore inference consistency,finite loss/gradient,split disjointness,causality,frame alignment,alarm coldstart/event evaluation,readme aggregate 검산 필수. 세 실험 전체 완료 후 통합 해석을 제공한다. 실험별 완료 때 분석/code/log/score/hash만 main에 자동 게시하며 바이너리는 로컬에 보존한다. 예약 자동화는 만들지 않는다.
