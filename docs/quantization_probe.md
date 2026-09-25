# DINOv2 INT8/INT4 예비 실행 규약

사용자 요청: 8비트·4비트 경량화 실험 가능성 확인. 이번 실행은 정상 프레임만 사용하는 구현/자원 진단이며 4장면 AUROC 또는 최종 실시간 검증이 아니다. 단계9 본 실험 범위는 별도 결과 제안으로 제시한다.

- 고정 ViT-B/14의 48개 attention/MLP Linear 가중치만 양자화. 원본 checkpoint 불변. LoRA 미적용. 입력252, RGB/특징정규화 FP32, 활성값·LayerNorm·patch embedding·attention 연산 BF16 유지. head/prototype는 이번 백본 단독 진단에 포함하지 않음.
- 비교: BF16, INT8 weight-only(per output channel), INT4 weight-only(group128, tile_packed_to_4d, tinygemm). TorchAO0.17.0을 cache/quantization/torchao017에 --no-deps로 별도 설치하고 기존torch2.11.0+cu128 환경을 변경하지 않는다. CUDA실제forward 및 profiler 연산 이름으로 변환/커널 경로 확인. fake quantization 결과를 packed INT4 결과로 표시하지 않는다.
- R01 정상 train 영상 앞3개의 균등 위치32frame씩 총96frame. 정상cal/test 영상 및 이상 라벨은 사용하지 않는다. 같은 frame/순서/seed0. JPEG→256BGR→252RGB를 기존과 동일하게 적용한다.
- 각 구성 별도 프로세스. 원본 BF16 특징을 CPU RAM에 임시 보관해 동일 frame CLS/patch cosine 차이·유한값 확인. warmup8회 이후 3회 반복 single-frame 읽기/전처리/백본/CUDA동기화 시간을 기록한다. 참조 특징 복사/비교/양자화·직렬화 시간은 속도에서 제외한다. 초기사이즈변환/커널초기화도 warmup밖에서 수행한다. 짧은 정상프레임 처리시간으로 전체VAD throughput/30FPS 연속 실시간 보장을 주장하지 않는다.
- 실제 양자화 tensor payload와 state_dict 직렬화 크기, peak CUDA allocated/reserved, 처리p50/p95, 특징 변화 기록. 모델 크기 감소비는 BF16기준이며 전체GPU메모리 감소비와 구별한다. 파라미터 수·토큰수 감소 실험이 아니다.
- 실패도 failure.json/traceback에 보존. 10GiB 이하중지, 자동재개 없음. 신규전체patch cache 없음. 원본/기존결과 보존. 로그·수치·코드·SHA만 게시하고 패키지/가중치/영상은 공개하지 않는다.
