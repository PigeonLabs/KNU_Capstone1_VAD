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
