#!/usr/bin/env bash
# Run all images x all block sizes, capture stdout into image_proc_cuda_results.txt.
set -e
cd "$(dirname "$0")"
make -s

OUT=image_proc_cuda_results.txt
: > "$OUT"

for img in 512x512.pgm 1024x1024.pgm 7680x4320.pgm; do
  for bs in 8 16 32; do
    echo "================================================================" >> "$OUT"
    echo ">>> $img | block $bs" >> "$OUT"
    echo "================================================================" >> "$OUT"
    ./image_processing_cuda ../../images/$img $bs >> "$OUT"
    echo "" >> "$OUT"
  done
done

echo "Wrote $OUT"
