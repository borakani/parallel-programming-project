#!/usr/bin/env bash
# Capture per-kernel timing, occupancy and memory metrics with nvprof.
# Output: image_proc_cuda_perf.txt
# Note: on newer GPUs (CC >= 7.5) nvprof is replaced by `nsys` / `ncu`.
#       The script tries nvprof first and falls back to nsys profile.
set -e
cd "$(dirname "$0")"
make -s

OUT=image_proc_cuda_perf.txt
: > "$OUT"

run_profile() {
    local img=$1
    local bs=$2
    echo "================================================================" >> "$OUT"
    echo ">>> profile: $img | block $bs" >> "$OUT"
    echo "================================================================" >> "$OUT"
    if command -v nvprof >/dev/null 2>&1; then
        nvprof --print-gpu-trace --metrics achieved_occupancy,gld_efficiency,gst_efficiency,shared_load_throughput \
            ./image_processing_cuda ../../images/$img $bs >> "$OUT" 2>&1 || true
    else
        nsys profile --stats=true --force-overwrite=true -o /tmp/nsys_$$ \
            ./image_processing_cuda ../../images/$img $bs >> "$OUT" 2>&1 || true
        rm -f /tmp/nsys_$$.*
    fi
    echo "" >> "$OUT"
}

for img in 512x512.pgm 1024x1024.pgm 7680x4320.pgm; do
  for bs in 8 16 32; do
    run_profile $img $bs
  done
done

echo "Wrote $OUT"
