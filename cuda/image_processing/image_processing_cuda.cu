#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

typedef struct {
    int width, height;
    float *data;
} Image;

static Image *read_pgm(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }

    char line[1024];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return NULL; }
    if (line[0] != 'P' || line[1] != '5') {
        fprintf(stderr, "Not a binary PGM file\n");
        fclose(f); return NULL;
    }

    while (fgets(line, sizeof(line), f) && line[0] == '#');

    int w, h;
    sscanf(line, "%d %d", &w, &h);

    if (!fgets(line, sizeof(line), f)) { fclose(f); return NULL; }

    Image *img = (Image*)malloc(sizeof(Image));
    img->width = w;
    img->height = h;
    img->data = (float*)malloc((size_t)w * h * sizeof(float));

    unsigned char *buf = (unsigned char*)malloc((size_t)w * h);
    size_t nread = fread(buf, 1, (size_t)w * h, f);
    (void)nread;
    for (int i = 0; i < w * h; i++) img->data[i] = (float)buf[i];
    free(buf);
    fclose(f);
    return img;
}

static void write_pgm(const char *filename, const Image *img) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", filename); return; }
    fprintf(f, "P5\n%d %d\n255\n", img->width, img->height);
    unsigned char *buf = (unsigned char*)malloc((size_t)img->width * img->height);
    for (int i = 0; i < img->width * img->height; i++) {
        float v = img->data[i];
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        buf[i] = (unsigned char)v;
    }
    fwrite(buf, 1, (size_t)img->width * img->height, f);
    free(buf);
    fclose(f);
}

static void free_image(Image *img) {
    free(img->data);
    free(img);
}

__constant__ float d_GAUSS[25] = {
    1,  4,  7,  4, 1,
    4, 16, 26, 16, 4,
    7, 26, 41, 26, 7,
    4, 16, 26, 16, 4,
    1,  4,  7,  4, 1
};

/* Gaussian Blur — shared-memory tile with 2-pixel halo (5x5 stencil).
 * Each block loads a (BS+4)x(BS+4) tile cooperatively into shared memory,
 * then every thread reads only from shared memory. */
template <int BS>
__global__ void gaussian_blur_shared(const float *src, float *dst, int W, int H) {
    constexpr int HALO = 2;
    constexpr int TILE = BS + 2 * HALO;
    __shared__ float sm[TILE][TILE];

    int gx0 = blockIdx.x * BS - HALO;
    int gy0 = blockIdx.y * BS - HALO;

    int tid = threadIdx.y * BS + threadIdx.x;
    int total = TILE * TILE;
    for (int idx = tid; idx < total; idx += BS * BS) {
        int ly = idx / TILE;
        int lx = idx - ly * TILE;
        int gx = gx0 + lx;
        int gy = gy0 + ly;
        if (gx < 0) gx = 0; else if (gx >= W) gx = W - 1;
        if (gy < 0) gy = 0; else if (gy >= H) gy = H - 1;
        sm[ly][lx] = src[gy * W + gx];
    }
    __syncthreads();

    int x = blockIdx.x * BS + threadIdx.x;
    int y = blockIdx.y * BS + threadIdx.y;
    if (x >= W || y >= H) return;

    int sy = threadIdx.y + HALO;
    int sx = threadIdx.x + HALO;
    float acc = 0.0f;
    #pragma unroll
    for (int ky = -2; ky <= 2; ky++) {
        #pragma unroll
        for (int kx = -2; kx <= 2; kx++) {
            acc += sm[sy + ky][sx + kx] * d_GAUSS[(ky + 2) * 5 + (kx + 2)];
        }
    }
    dst[y * W + x] = acc / 273.0f;
}

/* Sobel Edge Detection — shared-memory tile with 1-pixel halo (3x3 stencil). */
template <int BS>
__global__ void sobel_shared(const float *src, float *dst, int W, int H) {
    constexpr int HALO = 1;
    constexpr int TILE = BS + 2 * HALO;
    __shared__ float sm[TILE][TILE];

    int gx0 = blockIdx.x * BS - HALO;
    int gy0 = blockIdx.y * BS - HALO;

    int tid = threadIdx.y * BS + threadIdx.x;
    int total = TILE * TILE;
    for (int idx = tid; idx < total; idx += BS * BS) {
        int ly = idx / TILE;
        int lx = idx - ly * TILE;
        int gx = gx0 + lx;
        int gy = gy0 + ly;
        if (gx < 0) gx = 0; else if (gx >= W) gx = W - 1;
        if (gy < 0) gy = 0; else if (gy >= H) gy = H - 1;
        sm[ly][lx] = src[gy * W + gx];
    }
    __syncthreads();

    int x = blockIdx.x * BS + threadIdx.x;
    int y = blockIdx.y * BS + threadIdx.y;
    if (x >= W || y >= H) return;

    int sy = threadIdx.y + HALO;
    int sx = threadIdx.x + HALO;

    float p00 = sm[sy-1][sx-1], p01 = sm[sy-1][sx], p02 = sm[sy-1][sx+1];
    float p10 = sm[sy  ][sx-1],                     p12 = sm[sy  ][sx+1];
    float p20 = sm[sy+1][sx-1], p21 = sm[sy+1][sx], p22 = sm[sy+1][sx+1];

    float gx = -p00 + p02 - 2.0f * p10 + 2.0f * p12 - p20 + p22;
    float gy = -p00 - 2.0f * p01 - p02 + p20 + 2.0f * p21 + p22;
    dst[y * W + x] = sqrtf(gx * gx + gy * gy);
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
        fprintf(stderr, "Usage: %s <image.pgm> <block_size>\n", argv[0]);
        return 1;
    }

    int BS = atoi(argv[2]);
    if (BS != 8 && BS != 16 && BS != 32) {
        fprintf(stderr, "block_size must be 8, 16, or 32\n");
        return 1;
    }

    Image *src = read_pgm(argv[1]);
    if (!src) return 1;

    int W = src->width, H = src->height;
    size_t bytes = (size_t)W * H * sizeof(float);
    long long pixels = (long long)W * H;

    Image *blurred = (Image*)malloc(sizeof(Image));
    blurred->width = W; blurred->height = H;
    blurred->data = (float*)malloc(bytes);

    Image *edges = (Image*)malloc(sizeof(Image));
    edges->width = W; edges->height = H;
    edges->data = (float*)malloc(bytes);

    float *d_src, *d_blur, *d_edges;
    CUDA_CHECK(cudaMalloc(&d_src,   bytes));
    CUDA_CHECK(cudaMalloc(&d_blur,  bytes));
    CUDA_CHECK(cudaMalloc(&d_edges, bytes));

    CUDA_CHECK(cudaMemcpy(d_src, src->data, bytes, cudaMemcpyHostToDevice));

    dim3 block(BS, BS);
    dim3 grid((W + BS - 1) / BS, (H + BS - 1) / BS);

    cudaEvent_t e_start, e_stop;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_stop));

    /* Warmup: pay CUDA context init / JIT cost outside the timed region
     * so the first measured kernel reflects steady-state performance. */
    if (BS == 8)       gaussian_blur_shared<8 ><<<grid, block>>>(d_src, d_blur, W, H);
    else if (BS == 16) gaussian_blur_shared<16><<<grid, block>>>(d_src, d_blur, W, H);
    else               gaussian_blur_shared<32><<<grid, block>>>(d_src, d_blur, W, H);
    CUDA_CHECK(cudaDeviceSynchronize());

    /* === Gaussian Blur === */
    double p_before = sample_gpu_power_watts();
    CUDA_CHECK(cudaEventRecord(e_start));
    if (BS == 8)       gaussian_blur_shared<8 ><<<grid, block>>>(d_src, d_blur, W, H);
    else if (BS == 16) gaussian_blur_shared<16><<<grid, block>>>(d_src, d_blur, W, H);
    else               gaussian_blur_shared<32><<<grid, block>>>(d_src, d_blur, W, H);
    CUDA_CHECK(cudaEventRecord(e_stop));
    CUDA_CHECK(cudaEventSynchronize(e_stop));
    double p_after = sample_gpu_power_watts();

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));
    double time_sec = ms / 1000.0;

    double power_w = -1.0;
    if (p_before > 0 && p_after > 0) power_w = (p_before + p_after) * 0.5;
    else if (p_after > 0) power_w = p_after;
    else if (p_before > 0) power_w = p_before;

    double energy_j = (power_w > 0) ? power_w * time_sec : -1.0;
    double gflops   = (pixels * 50.0) / (time_sec * 1e9);
    double gflops_w = (energy_j > 0) ? gflops / power_w : -1.0;

    printf("=== Gaussian Blur | Block: %d ===\n", BS);
    printf("Image size:  %dx%d\n", W, H);
    printf("Time:        %.4f seconds\n", time_sec);
    if (energy_j >= 0) printf("Energy:      %.4f Joules\n", energy_j);
    else               printf("Energy:      N/A\n");
    printf("GFLOPS:      %.4f\n", gflops);
    if (gflops_w >= 0) printf("GFLOPS/W:    %.4f\n\n", gflops_w);
    else               printf("GFLOPS/W:    N/A\n\n");

    /* === Sobel Edge Detection === */
    p_before = sample_gpu_power_watts();
    CUDA_CHECK(cudaEventRecord(e_start));
    if (BS == 8)       sobel_shared<8 ><<<grid, block>>>(d_blur, d_edges, W, H);
    else if (BS == 16) sobel_shared<16><<<grid, block>>>(d_blur, d_edges, W, H);
    else               sobel_shared<32><<<grid, block>>>(d_blur, d_edges, W, H);
    CUDA_CHECK(cudaEventRecord(e_stop));
    CUDA_CHECK(cudaEventSynchronize(e_stop));
    p_after = sample_gpu_power_watts();

    ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e_start, e_stop));
    time_sec = ms / 1000.0;

    power_w = -1.0;
    if (p_before > 0 && p_after > 0) power_w = (p_before + p_after) * 0.5;
    else if (p_after > 0) power_w = p_after;
    else if (p_before > 0) power_w = p_before;

    energy_j = (power_w > 0) ? power_w * time_sec : -1.0;
    gflops   = (pixels * 36.0) / (time_sec * 1e9);
    gflops_w = (energy_j > 0) ? gflops / power_w : -1.0;

    printf("=== Sobel Edge Detection | Block: %d ===\n", BS);
    printf("Image size:  %dx%d\n", W, H);
    printf("Time:        %.4f seconds\n", time_sec);
    if (energy_j >= 0) printf("Energy:      %.4f Joules\n", energy_j);
    else               printf("Energy:      N/A\n");
    printf("GFLOPS:      %.4f\n", gflops);
    if (gflops_w >= 0) printf("GFLOPS/W:    %.4f\n\n", gflops_w);
    else               printf("GFLOPS/W:    N/A\n\n");

    CUDA_CHECK(cudaMemcpy(blurred->data, d_blur,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(edges->data,   d_edges, bytes, cudaMemcpyDeviceToHost));

    char blur_out[64], edge_out[64];
    sprintf(blur_out, "output_%dx%d_blurred.pgm", W, H);
    sprintf(edge_out, "output_%dx%d_edges.pgm",   W, H);
    write_pgm(blur_out, blurred);
    write_pgm(edge_out, edges);

    cudaFree(d_src);
    cudaFree(d_blur);
    cudaFree(d_edges);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_stop);

    free_image(src);
    free_image(blurred);
    free_image(edges);
    return 0;
}
