#!/usr/bin/env bash
# Replay the original uninstrumented autotune cache; preserve the first capture.
set -euo pipefail
cd /home/jeong/Desktop/IPAD
out=runs/stage9/9-1/kernel_study
if [[ $EUID -ne 0 ]]; then
  exec sudo -- /bin/bash "$0"
fi
[[ -f "$out/completed.json" && ! -f runs/disk_pause.json ]]
[[ ! -f "$out/hardware_frozen.csv" ]]
[[ -d cache/stage9_kernel_counters_frozen/inductor ]]
.venv/bin/python scripts/disk_guard.py > "$out/frozen_disk_guard.log" 2>&1 &
guard_pid=$!
trap 'kill "$guard_pid" 2>/dev/null || true' EXIT
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/stage9_kernel_counters_frozen/inductor"
export TRITON_CACHE_DIR="$PWD/cache/stage9_kernel_counters_frozen/triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_FREEZING=0
/opt/nvidia/nsight-compute/2026.1.1/ncu \
  --profile-from-start off --clock-control none --cache-control none \
  --nvtx --print-nvtx-rename kernel \
  --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  --export "$PWD/cache/stage9_kernel_counters_frozen/profile" \
  --csv .venv/bin/python scripts/stage9_kernel_study.py --counters \
  > "$out/hardware_frozen.csv" 2> "$out/hardware_frozen_stderr.log"
/opt/nvidia/nsight-compute/2026.1.1/ncu --import "$PWD/cache/stage9_kernel_counters_frozen/profile.ncu-rep" \
  --csv --print-nvtx-rename none > "$out/hardware_kernel_names.csv" 2> "$out/hardware_import.log"
printf '동결된 커널 캐시의 하드웨어 카운터 측정 완료\n'
