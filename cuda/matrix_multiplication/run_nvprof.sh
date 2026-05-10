#!/usr/bin/env bash
# Capture nvprof / nsys profiling traces for matmul.
# Output: matrix_mult_cuda_perf.txt
set -e
cd "$(dirname "$0")"
make -s

OUT=matrix_mult_cuda_perf.txt
: > "$OUT"

run_profile() {
    local n=$1
    local bs=$2
    echo "================================================================" >> "$OUT"
    echo ">>> profile: N=$n | block $bs" >> "$OUT"
    echo "================================================================" >> "$OUT"
    if command -v nvprof >/dev/null 2>&1; then
        nvprof --print-gpu-trace --metrics achieved_occupancy,gld_efficiency,gst_efficiency,shared_load_throughput \
            ./matrix_mult_cuda $n $bs >> "$OUT" 2>&1 || true
    else
        nsys profile --stats=true --force-overwrite=true -o /tmp/nsys_$$ \
            ./matrix_mult_cuda $n $bs >> "$OUT" 2>&1 || true
        rm -f /tmp/nsys_$$.*
    fi
    echo "" >> "$OUT"
}

for n in 256 512 1024 2048; do
  for bs in 8 16 32; do
    run_profile $n $bs
  done
done

echo "Wrote $OUT"
