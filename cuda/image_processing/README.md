# CUDA Image Processing

Gaussian Blur (5×5) + Sobel Edge Detection on the GPU.

## Build
```
make
```
Requires `nvcc` (CUDA Toolkit).

## Run
```
./image_processing_cuda <image.pgm> <block_size>
```
`block_size` ∈ {8, 16, 32}.

Example:
```
./image_processing_cuda ../../images/512x512.pgm 16
```

Run the full sweep:
```
make run-all
```

## Output
Each run prints:
```
=== Gaussian Blur | Block: <bs> ===
Image size:  WxH
Time:        0.XXXX seconds
Energy:      X.XXXX Joules
GFLOPS:      X.XXXX
GFLOPS/W:    X.XXXX

=== Sobel Edge Detection | Block: <bs> ===
...
```
And writes `output_<W>x<H>_blurred.pgm` / `output_<W>x<H>_edges.pgm`.

## Metrics
- **Time**: kernel time via `cudaEvent` (seconds).
- **Energy**: average GPU power (`nvidia-smi --query-gpu=power.draw`, sampled before/after kernel) × time → Joules.
- **GFLOPS**: Gaussian = `W·H·50 / (t·1e9)`, Sobel = `W·H·36 / (t·1e9)`.
- **GFLOPS/W**: `GFLOPS / power_W`.
