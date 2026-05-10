#!/usr/bin/env bash
# Capture kernel timing + metrics. Uses Nsight Compute (ncu) on modern GPUs
# (CC >= 7.5, e.g. T4/A100/RTX 30xx) and falls back to nvprof on older ones.
# Also runs nsys for an end-to-end timeline trace.
# Output: image_proc_cuda_perf.txt
set -e
cd "$(dirname "$0")"
make -s

OUT=image_proc_cuda_perf.txt
: > "$OUT"

# Pick the right profiler. nvprof rejects CC>=7.5 even though the binary
# exists, so prefer ncu when available.
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
    local img=$1
    local bs=$2
    echo "================================================================" >> "$OUT"
    echo ">>> profile: $img | block $bs ($PROFILER)" >> "$OUT"
    echo "================================================================" >> "$OUT"
    if [ "$PROFILER" = "ncu" ]; then
        # --set basic: occupancy, throughput, memory metrics (fast, ~1-2x overhead)
        # --target-processes all to capture the child too if any.
        ncu --set basic --target-processes all --print-summary per-kernel \
            ./image_processing_cuda ../../images/$img $bs >> "$OUT" 2>&1 || true
    else
        nvprof --print-gpu-trace --metrics achieved_occupancy,gld_efficiency,gst_efficiency,shared_load_throughput \
            ./image_processing_cuda ../../images/$img $bs >> "$OUT" 2>&1 || true
    fi
    echo "" >> "$OUT"
}

for img in 512x512.pgm 1024x1024.pgm 7680x4320.pgm; do
  for bs in 8 16 32; do
    run_profile $img $bs
  done
done

# Bonus: nsys end-to-end timeline for one representative run.
if command -v nsys >/dev/null 2>&1; then
    echo "================================================================" >> "$OUT"
    echo ">>> nsys timeline: 7680x4320.pgm | block 16" >> "$OUT"
    echo "================================================================" >> "$OUT"
    nsys profile --stats=true --force-overwrite=true -o /tmp/nsys_imgproc \
        ./image_processing_cuda ../../images/7680x4320.pgm 16 >> "$OUT" 2>&1 || true
    rm -f /tmp/nsys_imgproc.*
fi

echo "Wrote $OUT"
