# 9-1 후속: 선형층 커널·캐시 병목 진단

사용자는 “다음 실험 수행해줘”로 실제 선형층별 BF16/INT4 커널, 압축해제 비용, 메모리 대역폭과 Tensor Core 활용률 진단을 승인했다. 기존 결과를 보존하고 kernel_study 하위에 기록한다. 새 VAD 학습이나 테스트 AUROC 평가는 포함하지 않는다.

## 고정 설계

DINOv2 ViT-B/14 block0의 qkv (K768,N2304), attention projection (768,768), fc1 (768,3072), fc2 (3072,768). R01–R04 각 기존 정상train 샘플의 첫 프레임에서 실제 입력 활성값을 capture한다. numerical 검증은 네 장면 모두, 타이밍은 R01 같은 입력을 사용한다. 전체 모델의 모든48층을 대표한다고 주장하지 않는다.

토큰 M=1,16,64,325는 각 실제 활성값의 앞 M개를 사용한다. M325가 실제 단일 이미지 조건이며 나머지는 작업 크기의 인과 진단용이다. 작은 M을 LLM 전체 모델 결과로 일반화하지 않는다.

6경로: BF16 GEMM+bias, 기존 native tinygemm INT4, 현재 packed INT4 복원+GEMM, 동일 packed 가중치의 복원만, 사전 복원한 BF16 가중치의 GEMM만(메모리 절약 없는 비용 분리 대조군), 직접 fused Triton INT4 unpack/dequant+BF16 GEMM. 마지막은 INT4 산술이 아닌 W4A16 커널이다. 같은 group128 q/scale/zero를 유지한다. fused는 사전 고정6개tile설정을 정상활성값에서 튜닝한다. 각 후보는 네장면의 동일 복원 가중치 참조 대비 relativeL2≤.01 및 평균cosine거리≤1e-3를 통과해야 한다. 비교값/실패를 모두 보존한다. 출력 특징검사 .05 규칙과 raw linear 오차는 서로 다른 척도임을 명시한다.

모든 torch 경로는 max-autotune, emulate_precision_casts=True, fullgraph=True, dynamic=False, CUDA Graph는 외부에서 동일하게 적용한다. CPU8thread/compiler2worker/seed0 고정. 초기 compile/tune과 정상상태 시간을 구분한다. 부동소수점 환경은 기존BF16 reduction False, TF32off. 모델의 양자화 오차와 kernel 구현 오차를 분리한다.

warm-cache: CUDA Graph32회 연속호출 ×7 round, GPU event로 나누어 단일 호출 시간을 구한다. cache-evicted: GPU L2크기보다 큰256MiB 버퍼를 zero로 덮은 뒤 별도event구간에서 single-call graph를 측정해21회 비교한다. flush 시간은 제외한다. 캐시 완전 비움이 하드웨어 카운터로 증명되지 않으면 'eviction 시도'로 표기한다. 모든 후보를 고정seed로 섞어 라운드별 측정해 순서 영향을 줄인다. 타이밍은 해당linear만 포함하며 JPEG/전체VAD와 비교하지 않는다. M1·325는 별도 profiler로 커널을 기록한다. 개별 커널 입력/출력/weight byte에서 계산하는 bandwidth는 실측 DRAM bandwidth가 아니다.

Nsight Compute의 GPU 카운터 권한을 실제작업에서 시험한다. 가능하면 duration/DRAM bytes·throughput/SM throughput/Tensor pipe 활용률을 수집하고 cache policy 및 profiler replay를 기록한다. 관리자전용 등으로 불가능하면 오류를 보존하고 권한을 사용자에게 요청한다. nvidia-smi GPU-util을 Tensor Core 활용률로 대체하지 않으며, 카운터 없이 compute-bound/memory-bound를 확정하지 않는다. 드라이버·보안 설정은 자동 변경하지 않는다.

원본 데이터·결과 보존. 10GiB 이하 프로젝트 프로세스 중지, 자동재개없음. 새전체feature cache 없음. 분석·코드·로그·해시만 main 자동게시, 프로파일러 binary 리포트/캐시는 로컬 유지. Nsight 분석은 별도 실행으로 타이밍을 오염시키지 않는다.
