#!/usr/bin/env bash
# User-invoked hardware counter capture; no driver/security configuration changes.
set -euo pipefail
cd /home/jeong/Desktop/IPAD
out=runs/stage9/9-1/kernel_study
if [[ $EUID -ne 0 ]]; then
  while [[ ! -f "$out/completed.json" ]]; do
    if [[ -f runs/disk_pause.json || -f "$out/failure.json" ]]; then
      echo '실험 중지/실패 기록을 먼저 확인해야 합니다.'; exit 1
    fi
    sleep 3
  done
  exec sudo -- /bin/bash "$0"
fi
[[ ! -f runs/disk_pause.json ]]
[[ ! -f "$out/hardware_counters.csv" ]]
.venv/bin/python scripts/disk_guard.py > "$out/counter_disk_guard.log" 2>&1 &
guard_pid=$!
trap 'kill "$guard_pid" 2>/dev/null || true' EXIT
export TORCHINDUCTOR_CACHE_DIR="$PWD/cache/stage9_kernel_counters/inductor"
export TRITON_CACHE_DIR="$PWD/cache/stage9_kernel_counters/triton"
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_FREEZING=0
/opt/nvidia/nsight-compute/2026.1.1/ncu \
  --profile-from-start off --clock-control none --cache-control none \
  --nvtx --print-nvtx-rename kernel \
  --section SpeedOfLight --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis \
  --metrics dram__bytes_read.sum,dram__bytes_write.sum \
  --csv .venv/bin/python scripts/stage9_kernel_study.py --counters \
  > "$out/hardware_counters.csv" 2> "$out/hardware_counters_stderr.log"
printf 'Nsight Compute 하드웨어 카운터 측정 완료\n'
