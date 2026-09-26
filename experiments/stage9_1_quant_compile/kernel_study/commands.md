# 실행 및 재계측 기록

1. 일반 계정의 Nsight probe: ERR_NVGPUCTRPERM. `counter_permission_probe.log` 보존.
2. `.venv/bin/python -m pytest -q tests/test_stage9_kernel.py`: 1 passed.
3. `TORCHINDUCTOR_CACHE_DIR="$PWD/cache/stage9_kernel/inductor" TRITON_CACHE_DIR="$PWD/cache/stage9_kernel/triton" TORCHINDUCTOR_COMPILE_THREADS=2 TORCHINDUCTOR_FREEZING=0 .venv/bin/python scripts/stage9_kernel_study.py`: 16 cases 완료.
4. 사용자 승인 후 `bash scripts/collect_stage9_counters.sh`: 독립 autotune cache의 탐색 계측. 최종 표에서 제외.
5. 원 벤치마크의 inductor/triton cache를 `cache/stage9_kernel_counters_frozen`으로 복사. `frozen_counter_cache.json`에 SHA256 기록.
6. `bash scripts/collect_stage9_counters_frozen.sh`: 최종 관리자 계측, 새로운 AUTOTUNE 없음. 드라이버 설정 변경 없음.
7. `ncu --import cache/stage9_kernel_counters_frozen/profile.ncu-rep --csv --page raw --print-nvtx-rename kernel`의 stdout을 `hardware_raw.csv`로 보존. `--page details --print-details all` 출력은 `hardware_details.csv`로 보존.
8. `.venv/bin/python scripts/summarize_stage9_kernel.py`: 건수·수치·단위·소스 hash 검증, 결과표·그림 생성.

명령의 ncu 절대경로는 `/opt/nvidia/nsight-compute/2026.1.1/ncu`입니다. 단계별 로그와 설정을 함께 게시하며 바이너리 프로파일 및 컴파일 캐시는 로컬에 유지합니다. 일반 벤치마크 completed.json의 별도 카운터 필요 표기는 당시 상태이며, 최종 완료는 verification.json과 final_provenance.json을 참조합니다.
