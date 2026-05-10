#!/usr/bin/env bash
# Capture kernel timing + metrics for matmul. Uses Nsight Compute (ncu) on
# modern GPUs (CC >= 7.5) and falls back to nvprof on older ones.
# Output: matrix_mult_cuda_perf.txt
set -e
cd "$(dirname "$0")"
make -s

OUT=matrix_mult_cuda_perf.txt
: > "$OUT"

PROFILER=""
if command -v ncu >/dev/null 2>&1; then
    PROFILER="ncu"
elif command -v nvprof >/dev/null 2>&1; then
    PROFILER="nvprof"
fi

if [ -z "$PROFILER" ]; then
    echo "No profiler found (need ncu or nvprof)." | tee -a "$OUT"
    exit 1
fi
echo "Using profiler: $PROFILER" | tee -a "$OUT"
echo "" >> "$OUT"

run_profile() {
    local n=$1
    local bs=$2
    echo "================================================================" >> "$OUT"
    echo ">>> profile: N=$n | block $bs ($PROFILER)" >> "$OUT"
    echo "================================================================" >> "$OUT"
    if [ "$PROFILER" = "ncu" ]; then
        ncu --set basic --target-processes all --print-summary per-kernel \
            ./matrix_mult_cuda $n $bs >> "$OUT" 2>&1 || true
    else
        nvprof --print-gpu-trace --metrics achieved_occupancy,gld_efficiency,gst_efficiency,shared_load_throughput \
            ./matrix_mult_cuda $n $bs >> "$OUT" 2>&1 || true
    fi
    echo "" >> "$OUT"
}

for n in 256 512 1024 2048; do
  for bs in 8 16 32; do
    run_profile $n $bs
  done
done

if command -v nsys >/dev/null 2>&1; then
    echo "================================================================" >> "$OUT"
    echo ">>> nsys timeline: N=2048 | block 16" >> "$OUT"
    echo "================================================================" >> "$OUT"
    nsys profile --stats=true --force-overwrite=true -o /tmp/nsys_matmul \
        ./matrix_mult_cuda 2048 16 >> "$OUT" 2>&1 || true
    rm -f /tmp/nsys_matmul.*
fi

echo "Wrote $OUT"
