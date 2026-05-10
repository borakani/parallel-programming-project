# CUDA Matrix Multiplication

Tiled matrix multiplication on the GPU using shared memory.

## Build
```
make
```
Requires `nvcc` (CUDA Toolkit).

## Run
```
./matrix_mult_cuda <N> <block_size>
```
`block_size` ∈ {8, 16, 32}. Each block is a `block_size × block_size` tile that cooperatively loads sub-matrices of A and B into shared memory.

Example:
```
./matrix_mult_cuda 1024 16
```

Sweep:
```
make run-all
```

## Output
```
=== Matrix Multiplication | Block: <bs> ===
Matrix size: NxN
Time:        0.XXXX seconds
Energy:      X.XXXX Joules
GFLOPS:      X.XXXX
GFLOPS/W:    X.XXXX
```

## Metrics
- **Time**: kernel time via `cudaEvent` (seconds).
- **Energy**: average GPU power (`nvidia-smi --query-gpu=power.draw`, sampled before/after kernel) × time → Joules.
- **GFLOPS**: `2·N³ / (t·1e9)`.
- **GFLOPS/W**: `GFLOPS / power_W`.

## Notes
- Uses `double` precision — same as the sequential and OpenMP baselines.
- Same `srand(42)` seed as the baselines so inputs are reproducible across implementations.
