#!/usr/bin/env bash
# Run the matmul sweep, capture stdout into matrix_mult_cuda_results.txt.
set -e
cd "$(dirname "$0")"
make -s

OUT=matrix_mult_cuda_results.txt
: > "$OUT"

for n in 256 512 1024 2048; do
  for bs in 8 16 32; do
    echo "================================================================" >> "$OUT"
    echo ">>> N=$n | block $bs" >> "$OUT"
    echo "================================================================" >> "$OUT"
    ./matrix_mult_cuda $n $bs >> "$OUT"
    echo "" >> "$OUT"
  done
done

echo "Wrote $OUT"
