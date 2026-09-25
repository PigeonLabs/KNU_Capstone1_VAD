# 3단계 사전 고정 실험 규약

2026-09-25 사용자 승인. 3A 정상 위상 진단과 3B 매칭 비교부터 수행한다. R01–R04만 사용하며 원본과 1·2단계 결과를 보존한다. 이 문서의 SHA256을 실행 설정에 기록한다.

- seed 0에서 기존 전체 정상 학습 위상 head와 prototype을 그대로 재사용한다. 정상 holdout 진단은 기존 영상 단위 80/20 분할을 유지하고, 80%에서 head와 메모리를 별도 학습한다. 따라서 기존 전체 학습 메모리로 holdout을 평가하지 않는다.
- 입력·출력 위상은 clip 시작 위치, 비교 patch는 start+8이다. 위상은 200-class 확률을 20개 연속 bin으로 합산한다. 기존 hard 방식은 200-class argmax를 20-bin으로 변환하므로 별도 기준선으로 보존한다.
- 고정 비교: 무조건부 NN, 기존 hard NN, 합산 posterior hard NN, 원형 인접 ±1 bin NN, posterior top3 NN, posterior top3 거리 가중평균, confidence fallback, 무작위 3-bin NN, 인접 방식과 같은 후보 bin 수의 무작위 NN, 조건부 bank 전체 NN.
- 주 방법 후보는 **posterior top3 거리 가중평균**이다. patch별 최근접 prototype 거리를 bin별로 계산한 뒤 공간 평균과 top3 확률 가중평균을 적용한다. prototype 벡터를 혼합하는 2단계 soft projection과 다르다. 나머지는 사전 지정 비교이며 테스트 최고 성능으로 후보를 바꾸지 않는다.
- confidence fallback은 정상 holdout에서 20-bin ±1 정확도 90% 이상, coverage 10% 이상을 만족하는 가장 낮은 confidence cutoff를 사용한다. cutoff 후보는 confidence의 0~90 percentile, 5 percentile 간격. 없으면 fallback을 모두 무조건부로 설정한다. 전체 정상 refit 후 신뢰도 분포가 달라질 수 있으므로 이 변형은 탐색적 비교로 표시한다.
- 정상 진단: 20-bin 정확도, ±1 정확도, 원형 MAE, confusion, confidence 구간별 정확도, bin별 prototype 수, 참조 위상/예측 위상/다른 위상/무조건부 정상 거리. 정상 상대 위치는 진단용 참조일 뿐 이상 테스트의 oracle이 아니다.
- 같은 conditional bank에서 routing을 비교하고, random 후보 수 대조군 및 conditional 전체 bank를 사용해 후보 증가/클러스터링 효과를 분리한다. bin당 prototype 수가 다르면 후보 prototype 수 차이도 보고한다. 무조건부 bank는 같은 정상 표본과 총 prototype 수를 사용한다.
- 모든 테스트 방법은 기존 평가 frame/video/label ID와 일치해야 한다. seed 0 hard/unconditional 점수와 2단계 원 점수의 허용 최대 오차는 2e-6이다. R02 불일치 3개 영상은 주 결과 제외, ±1 common-length 민감도 별도 저장.
- 프레임 AUROC/AUPRC, 개별 점수와 기존 주기 점수 결합을 모두 기록한다. centered clip/window와 테스트 전체 정규화는 유지하며 온라인 실시간 성능으로 주장하지 않는다.
- 진단 후 설정을 변경하지 않고 seed 1/2로 반복할 경우 별도 실행 기록을 남긴다. 이전 테스트 결과를 이미 본 연구이므로 확인적 독립 검증이라고 주장하지 않는다. 영상 단위 bootstrap을 사용하고 프레임 iid bootstrap은 사용하지 않는다.
- 시간 변형 실험은 3A/B 결과 확인 후 수행할 후속 단계이다. 원본이 아닌 정상 holdout의 파생 진단만 사용하며 실제 이상 유형별 정확도를 대체하지 않는다.
- 매 학습 배치 ID/loss/gradient, 단계 event, 실행 환경, 코드·입력 모델 해시, 프레임별 확률/점수/정답과 결과를 보존한다. 바이너리 대신 분석 자료와 SHA256 목록만 main에 게시한다. 10GiB 이하에서 중단하며 자동 재개하지 않는다.


## 게시 정책 변경 (2026-09-25)

사용자 요청에 따라 실험을 완료하고 결과를 보고한 후, 사용자가 해당 묶음의 게시를 승인하면 GitHub main에 한 번에 올린다. 이전 단계별 자동 push 지시는 이 규칙으로 대체한다. 승인 전에는 로컬 로그·스냅샷·커밋을 보존한다.
