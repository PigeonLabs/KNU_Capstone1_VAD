# 4-1 / 4-2 / 4-3 사전 고정 실험 규약

2026-09-25 사용자 승인. 세 실험을 순서대로 수행하고 각 실험 전체 완료 시 분석 자료·코드·로그·점수·해시 목록을 main에 자동 게시한다. 원본·기존 결과·바이너리는 보존하며 바이너리는 게시하지 않는다. 10GiB 이하에서 중단하고 자동 재개하지 않는다. 새 예약 모니터는 만들지 않는다.

## 공통

R01–R04, seed 0/1/2, 기존 frozen DINOv2와 3단계 전체 정상 학습 head를 사용한다. 모델·특징 재선택 및 테스트 AUROC에 따른 가중치 선택은 하지 않는다. 3단계 raw score/모델의 SHA256과 실행 코드 SHA256을 기록한다. 이상 유형 정답은 추가 확보하지 않았으므로 실제 유형별 성능을 주장하지 않는다.

오프라인 논문 비교 규약을 유지한다: 중심 16프레임 clip, 장면 테스트 전체 min-max 정규화, 점수 결합은 동일 가중 평균. 온라인/독립 검증 결과로 해석하지 않는다. R02 영상12/13/14는 주 결과 제외하고 common length ±1 민감도를 별도 기록한다. 4-1/4-3은 window21 때문에 3단계보다 경계 프레임이 더 제외된다. 해당 실험의 모든 기준선도 같은 교집합 프레임에서 다시 평가한다. 4-2는 기존 3단계 유효 프레임을 유지한다. 서로 다른 평가 support의 수치를 직접 차감하지 않는다.

## 4-1 외형 / 시간 검사 분리

기존 무조건부 NN, 기존 hard, top3 weighted를 기준선으로 보존한다. 위상 진행 잔차 window5/21, 외형+window5, 외형+window21을 비교한다. 주 후보는 무조건부 외형 + window21의 동일 가중 평균이다. 정상 주기는 전체 정상 영상 길이 중앙값이며 기존 phase는 clip 시작 위치를 나타낸다. 영상 간 window를 만들지 않는다. 각 프레임의 정답·원 점수·정규화 점수와 AUROC/AUPRC를 기록한다.

## 4-2 메모리 용량 / 효율

위상별 prototype k=1,2,5,10. 같은 정상 표본 선택(seed 고정, phase당 최대512)과 k-means20회를 유지한다. 무조건부 bank는 조건부와 정확히 같은 총개수/정상 표본을 사용한다. R01처럼 점유 bin이19개면 실제 총개수를 기록한다. k10은 기존 bank 재사용으로 기준선 일치를 확인한다. phase head는 재학습하지 않는다.

비교 점수: 무조건부 NN, 기존 hard, top3 weighted. full bank 대비 AUROC 감소1pp 이내는 사전 정의한 실용성 기준이며 기대 성능을 보장하지 않는다. 각 scene/seed/k의 FP16 bank byte 수와 GPU peak를 기록한다. 모든 bank를 로컬에 보존한다.

효율 측정은 seed0의 각 장면·k에 대해 같은 첫 테스트 영상의 최대256 연속 프레임을 사용한다. warm-up 후 CUDA synchronize를 포함한 반복 측정을 한다. (a) cache 입력 matching (b) RGB 전처리·DINOv2 추출·16 CLS head·matching을 포함한 측정을 분리한다. hard/top3는 실제 후보 bank만 gather하여 비교한다. 실제 추출을 포함하되 이미 디코딩한 프레임의 RAM 입력 측정으로 명시하고 카메라/디스크 I/O·실제 공장 streaming/edge 성능을 주장하지 않는다. 10회의 cached 반복, 3회의 DINO 포함 반복을 사용한다.

## 4-3 정상 위상 전이 / 체류시간

새 backbone 없이 정상 학습 영상에서 예측한20-bin posterior를 이용한다. 영상 경계를 넘지 않는 인접 posterior outer product로 soft transition count를 구하고 모든 edge에 총1의 균일 Dirichlet pseudocount를 더해 row normalize한다. 추론 시 이전 posterior와 transition으로 현재 posterior를 예측한다.

비교: predictive negative log overlap, KL(current posterior || predicted posterior), 현재 posterior entropy(불확실성 대조군), 체류시간 right-tail surprise. 체류시간은 normal argmax bin의 완결된 내부 run 길이 분포로 구하며 첫/마지막 censored run을 제외한다. 해당 bin 자료가 없으면 전체 내부 run 분포를 사용한다. 전이·체류시간은 smoothing으로 위반을 수정하기 전에 평가한다. runtime의 run age만 사용하며 미래 run 종료를 보지 않는다. 전체 system은 기존 centered clip을 쓰므로 causal 주장은 하지 않는다.

주 시간 후보는 KL과 체류시간 점수의 장면별 정규화 후 동일 가중 평균이며, 외형 점수와 다시1:1로 결합한다. transition only, duration only, combined 및 각 외형 결합을 모두 비교한다. 4-1 window21과 같은 프레임 support에서 비교한다. 위상 jitter와 training-prediction 분포의 한계를 보고하고 정상 holdout/실제 이상 유형의 독립 성능으로 포장하지 않는다.

## 보고

장면별/seed별 AUROC·AUPRC, 4장면 macro 평균과 3-seed 표준편차, 정확한 평가 프레임 수를 기록한다. 4-1/4-3 주 후보는 seed0 영상 단위 paired bootstrap 2000회로 기준선과의 차이 구간을 계산한다. 4-2는 저장 메모리·matching·DINO 포함 시간의 정확도-효율 절충을 보고한다. 실패/재시도와 설정 고정 이후 변경은 별도 로그로 남긴다. 실험 간 최고 변형을 골라 새로운 최적 성능처럼 보고하지 않는다.
