#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                          \
    cudaError_t _e = (call);                                           \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                __FILE__, __LINE__, cudaGetErrorString(_e));           \
        exit(1);                                                       \
    }                                                                  \
} while (0)

/* Tiled matrix multiply with shared memory.
 * BS is the block (and tile) edge length, set as a template parameter so
 * the same code works for 8x8, 16x16 and 32x32 blocks. */
template <int BS>
__global__ void matmul_tiled(const double *A, const double *B, double *C, int N) {
    __shared__ double sA[BS][BS];
    __shared__ double sB[BS][BS];

    int row = blockIdx.y * BS + threadIdx.y;
    int col = blockIdx.x * BS + threadIdx.x;

    double acc = 0.0;
    int tiles = (N + BS - 1) / BS;

    for (int t = 0; t < tiles; t++) {
        int aCol = t * BS + threadIdx.x;
        int bRow = t * BS + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
        sB[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BS; k++) {
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = acc;
}

static double sample_gpu_power_watts() {
    FILE *p = popen("nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null", "r");
    if (!p) return -1.0;
    double w = -1.0;
    if (fscanf(p, "%lf", &w) != 1) w = -1.0;
    pclose(p);
    return w;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <block_size>\n", argv[0]);
        return 1;
    }

    int N  = atoi(argv[1]);
    int BS = atoi(argv[2]);
    if (BS != 8 && BS != 16 && BS != 32) {
        fprintf(stderr, "block_size must be 8, 16, or 32\n");
        return 1;
    }

    size_t bytes = (size_t)N * N * sizeof(double);
    double *A = (double*)malloc(bytes);
    double *B = (double*)malloc(bytes);
    double *C = (double*)malloc(bytes);
    if (!A || !B || !C) { fprintf(stderr, "Host alloc failed\n"); return 1; }

    /* Same seed as sequential for reproducibility. */
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }

    double *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B, bytes, cudaMemcpyHostToDevice));

    dim3 block(BS, BS);
    dim3 grid((N + BS - 1) / BS, (N + BS - 1) / BS);

    cudaEvent_t e_start, e_stop;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_stop));

    double p_before = sample_gpu_power_watts();
    CUDA_CHECK(cudaEventRecord(e_start));

    if (BS == 8)       matmul_tiled<8 ><<<grid, block>>>(dA, dB, dC, N);
    else if (BS == 16) matmul_tiled<16><<<grid, block>>>(dA, dB, dC, N);
    else               matmul_tiled<32><<<grid, block>>>(dA, dB, dC, N);

    CUDA_CHECK(cudaEventRecord(e_stop));
    CUDA_CHECK(cudaEventSynchronize(e_stop));
    double p_after = sample_gpu_power_watts();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));
    double time_sec = ms / 1000.0;

    double power_w = -1.0;
    if (p_before > 0 && p_after > 0) power_w = (p_before + p_after) * 0.5;
    else if (p_after > 0)  power_w = p_after;
    else if (p_before > 0) power_w = p_before;

    double energy_j = (power_w > 0) ? power_w * time_sec : -1.0;
    double flops    = 2.0 * (double)N * N * N;
    double gflops   = (flops / time_sec) / 1e9;
    double gflops_w = (power_w > 0) ? gflops / power_w : -1.0;

    printf("=== Matrix Multiplication | Block: %d ===\n", BS);
    printf("Matrix size: %dx%d\n", N, N);
    printf("Time:        %.4f seconds\n", time_sec);
    if (energy_j >= 0) printf("Energy:      %.4f Joules\n", energy_j);
    else               printf("Energy:      N/A\n");
    printf("GFLOPS:      %.4f\n", gflops);
    if (gflops_w >= 0) printf("GFLOPS/W:    %.4f\n", gflops_w);
    else               printf("GFLOPS/W:    N/A\n");

    CUDA_CHECK(cudaMemcpy(C, dC, bytes, cudaMemcpyDeviceToHost));

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);
    free(A); free(B); free(C);
    return 0;
}
