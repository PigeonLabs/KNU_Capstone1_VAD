# 9단계 — INT8·INT4 동등 최적화 규약

이 후속 실험은 INT8·INT4 경로에 BF16과 같은 최적화 탐색 예산을 적용한다. 기존 실험과 감사 결과를 보존하고 별도 `optimized` 디렉터리에 기록한다. 테스트 라벨, 전체 AUROC, 재학습은 범위에 포함하지 않는다.

## 고정 조건

R01–R04, 기존 동일 정상384frame, seed0, batch1, 입력252. RGB 변환·bilinear resize·정규화는 모든 조건에 같은 eager FP32 연산으로 고정하고 BF16 변환 뒤 백본만 최적화한다. 입력 BF16 bytes SHA256을 프레임별 대조한다. 기준 FP32는 원래 FP32 pretrained weights와 FP32 전처리, TF32 비활성화. BF16 및 해당 양자화 eager 기준을 별도로 보존해 양자화 오차와 실행 방식 오차를 구분한다. 특징은 RAM에서만 보존하고 새 전체 patch 디스크 캐시를 만들지 않는다.

5경로 × 4실행 방식의 20개 pilot 조건:
- BF16, W8A16(느린 WOQ lowering 프로세스 내 비활성화), W8A8(FakeTensor 문자열 검사→타입 검사), W4 native(group128 tinygemm), W4 packed(group128 동일 q/scale/zero를 두 nibble/byte에 저장하고 일시적 BF16 복원 후 GEMM). W4 packed는 native 패딩된 양자화와 원래 영역 값이 같은지48층 모두 검증한다. W4 packed는 INT4 저장/BF16 연산이며 순수 INT4 산술로 표시하지 않는다.
- eager, eager CUDA Graph, compile+CUDA Graph, max-autotune+CUDA Graph. 이전 max-autotune 제외는 이 후속 승인에 따라 해제한다. 모든 경로에 같은 탐색 예산 적용; CPU8thread, compile2worker, backendATEN/TRITON, fullgraph=True,dynamic=False,중간반올림보존True. 설치 패키지 파일은 수정하지 않는다.

pilot은 기존384frame에서 균등16frame, 3반복. 같은 eager 대비 CLS/patch 평균 cosine거리≤1e-3, 최대절대차이≤.05를 모두 통과한 조건 중 model 평균 시간이 가장 짧은 방식을 경로별 선택한다. 테스트 정보를 쓰지 않으며 선택은 full384 평가 전에 JSON으로 고정한다. 선택 경로를384frame×3회 재측정한다. full에서 실패하면 사전 지정 eager CUDA Graph를 full로 검증한다. 실패/느린 결과/수치 초과를 모두 남기며 임계값은 완화하지 않는다. 유효 후보가 없으면 배포 가능으로 표시하지 않는다.

각 프로세스는 별도캐시. 컴파일/캡처 startup비용과 정상상태시간 구분. model만 시간과 JPEG+FP32전처리+BF16변환 포함 총시간을 별도로 측정; 전처리 뒤 CUDA synchronize를 모든 조건에 동일 적용하여 구간을 분리한다. 기존 전처리 포함 compiler와 직접 speedup 계산하지 않는다. profiler는 별도3회. kernel이름/원측정/특징차이/환경/코드/가중치/입력/캐시hash를 보존한다. GPU peak allocated/reserved와 가중치payload를 별도기록하며 CUDA Graph의 일시적 복원 버퍼 유지비용도 peak에 반영한다.

같은 백본 입력에서 FP32·BF16·컴파일 중간층의 추가 소규모 진단으로 수치 오차 위치를 확인한다. 이 진단으로 사후 양자화 설정을 고르지 않는다.

여유10GiB 이하 SIGSTOP, 자동재개없음. 실험 runner/자식/컴파일러는 disk_guard 대상. 원본/기존결과 삭제없음. 완료시 한국어README·분석·로그·해시만 main 자동게시한다.

## 빠른 경로의 대표 표본 진단 추가

20개 pilot에서 모든 compile 경로가 고정수치선을 넘었으므로, 수치통과 기반 graph 선택은 그대로 유지한다. 별도로 각 표현의 pilot 최저 model평균시간 경로를 수치 통과 여부와 무관하게 normal384 평가 전에 frozen_fast_diagnostic.json에 고정하여 전체384frame에서 진단한다. 이는 작은 pilot 시간만으로 가속량을 주장하지 않기 위한 측정이며 배포 선택이나 임계값 완화가 아니다. 모든 오차/실패를 유지한다. 마지막 중간층 진단은 각 장면 첫 정상샘플1개씩4개, FP32/BF16/compiled trace 및 위치임베딩 eager 사전계산 대조군을 포함한다. 중간값 반환은 compiler fusion에 영향을 줄 수 있어 실사용 시간으로 해석하지 않는다.
