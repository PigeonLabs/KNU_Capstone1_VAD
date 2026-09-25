# 9단계 양자화 / 9-1 실제 연산 경로와 컴파일 최적화

2026-09-26 사용자가 이번 실험을 9-1로 지정했다. 8단계는 기존 LoRA 실험이며 변경하지 않는다. 이전 정상96frame 양자화 probe는 9단계의 사전 진단이다. 이번 승인 범위는 9-1: 연산 경로/컴파일/자원/정상 특징 수치 검증이다. 전체 테스트 AUROC와 새로운 모델 선택은 이 단계에 포함하지 않는다.

## 사전 고정 설계

고정 DINOv2 ViT-B/14, 252 입력, batch1, seed0. R01–R04 각각 기존 정상train 영상 앞3개에서 균등32frame씩(총384frame)을 고정한다. calibration/test 영상·이상 라벨을 사용하지 않는다. JPEG→256BGR→252RGB와 FP32 정규화를 유지한다. 위상head·prototype·시간검사는 백본 경로 비교에서 제외한다.

4가지 연산 경로 × 3가지 실행 방식 = 12조건:
- BF16 기준선.
- W8A16: Int8WeightOnlyConfig(version2), weight INT8 per-row, 활성값 BF16. 실제 BF16변환/matmul 경로를 profiler로 확인.
- W8A8: Int8DynamicActivationInt8WeightConfig(version2), weight INT8 per-row 및 활성값 INT8 per-token. 실제 integer GEMM/커널을 확인하고 이름만으로 INT8가속을 주장하지 않는다.
- W4A16: Int4WeightOnlyConfig(group128,tile_packed_to_4d,tinygemm), 활성값 BF16. packed INT4 커널과768→1024 패딩 기록.
- 각각 eager, torch.compile(mode=default), torch.compile(mode=reduce-overhead). backend=inductor, fullgraph=True, dynamic=False. BF16 대조군에도 같은 컴파일을 적용한다. graph break/eager fallback은 허용하지 않고 실패로 기록한다. max-autotune은 이번 범위에 포함하지 않는다.

TorchAO0.17.0/torch2.11.0+cu128 고정. quantize_의 set_inductor_config=False로 모든 조건의 전역 compiler설정을 같게 유지. compile_threads=2, CPUtorchthreads8, compiler캐시는 조건별새로분리하고기존파일삭제없음. CUDA graph marker는 reduce-overhead의 호출마다 사용하며 출력은 다음 호출 전에 CPU로 보존하거나 폐기한다. 모델/shape/정밀도 변경으로 실행 중 재컴파일이 발생하면 기록한다.

각 조건은 별도 프로세스에서 순차 실행한다. 같은 프로세스에서 BF16 eager 및 해당양자화 eager 특징을 CPU RAM에 임시 저장하여 compiled 출력과 비교한다. 전체 특징캐시나 중복가중치파일은 새로 저장하지 않는다. 첫 compiler wrapper 생성+첫호출시간 및 12회warmup비용을 별도 기록; 신규조건 캐시의첫호출이지GPU드라이버전체cold start는아님. 정상384frame×3회반복으로 JPEG읽기/전처리/전송, 백본wall시간, GPU event 구간시간, 전체처리mean/p50/p95 측정. profiler는 시간측정 후 별도5frame에만실행한다. CUDA peak allocated/reserved와 직접저장payload/serialized크기는 구분한다. compile영구buffer가모델크기를늘리는경우도공개한다.

동일정밀도eager대비compiled의 CLS/patch cosine거리와최대절대차이, BF16 eager대비변화를같은frame에서기록. finite 필수. 고정수치검사선: 동일양자화eager대비평균cosine차이≤1e-3 및최대절대차이≤.05; 초과시속도를보고하더라도수치검사실패를명시하고동등성/배포채택을주장하지않는다. 정상특징동등성은AUROC나경보동등성을보장하지않는다. CUDA커널이름/호출수/CPU·device시간과생성코드SHA를보존한다. kernel 종류별합산은불완전한진단이며end-to-end절감의인과분해로단정하지않는다.

## 안전·재현·게시

원본·기존결과보존. 여유10GiB이하프로젝트실험SIGSTOP/latched pause, 자동재개금지. compiler하위프로세스도disk_guard감시대상. 실패조건은config/traceback/log보존하고다른조건의독립실행은가능하나실패결과를완료로숨기지않는다. protocol·코드·환경·가중치·입력hash,명령,진행상태기록. 완료후코드·한국어결과·원측정CSV·커널자료·해시만 main 자동게시. 생성native커널binary/가중치/이미지/패키지는로컬유지. 예약모니터추가없음.

해석은현재GPU의단일프레임백본처리와정상특징수치에한정한다. 카메라/네트워크/전체VAD30FPS/독립테스트성능/다른장비로일반화하지않는다. 설정을테스트AUROC로바꾸지않는다.

## 정상 특징 진단 후 추가된 수치 보존 대조군

초기12조건은 그대로 보존한다. BF16/default에서 평균patch cosine차이0.000303, 최대절대차이0.427로고정수치선(.05)을넘었다. 설치된Inductor config의설명상fusion이BF16 downcast/upcast를생략할수있다. 테스트라벨없이원인분리를위해4경로모두reduce-overhead-precise 조건을추가한다: fullgraph=True,dynamic=False,options={triton.cudagraphs:True,emulate_precision_casts:True}. 기존reduce-overhead 대비해당option만변경하며기존허용선은바꾸지않는다. 성공을가정하지않고수치·속도·메모리 모두공개한다. 원규약과실행코드는source_versions에원해시와함께보존한다. 기본12조건을완료한뒤추가4조건을순차실행한다.

## 사용자 이상값 지적 후 원인 분리 감사 (9-1 추가 진단)

기존16조건/허용선/결과는 변경하지 않는다. 정상384frame 중 균등16frame으로 W8A8 safe_int_mm의 텐서 repr 검사를 타입 검사로만 바꾼 A-B-A 대조 실험을 수행한다. 라이브러리 파일은 변경하지 않고 프로세스 내부 함수만 임시 교체하며 원복한다. W8A16은 동일16frame에서 원래 WOQ lowering과 해당 패턴 등록만 끈 별도 프로세스를 비교한다. FP32 kernel 이름은 내부 구현 자료형과 모델 활성값 자료형을 구분한다.

특징 차이는 기존 BF16 precise 최대오차 상위8frame과 균등8frame을 대상으로 eager 반복, eager 전처리/백본 분리, 전처리만 compile, 동일 전처리 텐서에서 백본만 compile, 전체 compile을 비교한다. 이 표본은 원인 진단용 편향 표본이며 성능 대표 표본이 아니다. CPU feature를 즉시 보존해 출력 버퍼 재사용을 배제한다. 입력은 미리 GPU에 올린 model-only timing을 사용하며 기존JPEG포함시간과 직접 비교하지 않는다. 모든 조건 동일 seed0, batch1, CPU8thread, compiler2worker. 테스트 라벨/전체AUROC/모델선택 없음. 원본 결과는 보존하고 실패와 재시작도 기록한다.
