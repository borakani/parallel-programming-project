# CUDA Implementations

GPU implementations for both project workloads:

| Folder | Workload | Kernel(s) |
|---|---|---|
| `image_processing/` | Gaussian Blur (5×5) + Sobel Edge Detection (3×3) | shared-memory tiled, halo-padded |
| `matrix_multiplication/` | Dense N×N matrix multiply | shared-memory tiled |

Both binaries take `<block_size>` as a CLI argument so the same build can be benchmarked at 8×8, 16×16 and 32×32 thread blocks.

## Grid / Block configuration

For a 2-D workload of size W×H (or N×N) with a square thread block of edge length BS:

```
dim3 block(BS, BS);                              // BS² threads per block
dim3 grid((W + BS - 1) / BS, (H + BS - 1) / BS); // ceil-div, covers borders
```

Each thread maps 1-to-1 to one output pixel / one output cell. Block sizes:

- **8×8 (64 threads)** — undersubscribes warps (2 warps/block); generally limits occupancy on modern GPUs.
- **16×16 (256 threads)** — balanced default; 8 warps/block, good occupancy on Pascal/Turing/Ampere.
- **32×32 (1024 threads)** — maximum threads/block; high register pressure can spill, but maximizes shared-memory reuse for stencils.

The sweep scripts (`run_benchmark.sh`) measure all three so the report can pick the best by GFLOPS/W per workload size.

## Shared memory tuning

### Image processing
Both Gaussian and Sobel are stencil operations: each output pixel reads a small neighborhood. Without shared memory, a 5×5 Gaussian re-reads each input pixel up to 25× from global memory. The kernels load a `(BS + 2·HALO)²` tile cooperatively into `__shared__` once per block, then every thread reads only from on-chip memory.

| Kernel | Halo | Shared tile | Bytes (BS=16) | Bytes (BS=32) |
|---|---|---|---|---|
| Gaussian (5×5) | 2 | (BS+4)² | 1 600 | 5 184 |
| Sobel (3×3)    | 1 | (BS+2)² | 1 296 | 4 624 |

All sizes are well under the 48 KB / 100 KB per-block shared memory limits, so occupancy is not constrained by smem.

### Matrix multiplication
Classical tiled matmul. Each thread block computes one BS×BS output tile by streaming through K-tiles of A and B in shared memory. Reuse factor = BS, so global-memory bandwidth requirement drops linearly with the tile size.

The tile size is a non-type template parameter so a single source produces specialized kernels for BS = 8 / 16 / 32 with `#pragma unroll` over the inner dot-product loop.

## Benchmarking + profiling

On a CUDA-capable host:

```bash
# image processing — runtime + energy across all sizes × all block sizes
cd cuda/image_processing
./run_benchmark.sh        # writes image_proc_cuda_results.txt
./run_nvprof.sh           # writes image_proc_cuda_perf.txt (nvprof or nsys)

# matrix multiplication
cd ../matrix_multiplication
./run_benchmark.sh        # writes matrix_mult_cuda_results.txt
./run_nvprof.sh           # writes matrix_mult_cuda_perf.txt
```

`*_results.txt` files mirror the convention used by the OpenMP branch and are consumed by the scheduler / report stage.

## Metrics

| Metric | Source | Formula |
|---|---|---|
| Time | `cudaEventElapsedTime` around the kernel launch | seconds |
| Power | `nvidia-smi --query-gpu=power.draw` sampled before and after each kernel, averaged | watts |
| Energy | Power × Time | joules |
| GFLOPS — Gaussian | `W·H·50 / (t·1e9)` |
| GFLOPS — Sobel | `W·H·36 / (t·1e9)` |
| GFLOPS — MatMul | `2·N³ / (t·1e9)` |
| GFLOPS/W | `GFLOPS / power_W` |

Output formatting matches the sequential and OpenMP implementations so the scheduler can parse all three.
