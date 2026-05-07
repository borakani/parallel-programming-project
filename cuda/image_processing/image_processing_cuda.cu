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
    fread(buf, 1, (size_t)w * h, f);
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

__global__ void gaussian_blur_kernel(const float *src, float *dst, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.0f;
    #pragma unroll
    for (int ky = -2; ky <= 2; ky++) {
        int ny = y + ky;
        if (ny < 0) ny = 0;
        else if (ny >= H) ny = H - 1;
        #pragma unroll
        for (int kx = -2; kx <= 2; kx++) {
            int nx = x + kx;
            if (nx < 0) nx = 0;
            else if (nx >= W) nx = W - 1;
            acc += src[ny * W + nx] * d_GAUSS[(ky + 2) * 5 + (kx + 2)];
        }
    }
    dst[y * W + x] = acc / 273.0f;
}

__global__ void sobel_kernel(const float *src, float *dst, int W, int H) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float Gx[9] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
    float Gy[9] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};

    float gx = 0.0f, gy = 0.0f;
    #pragma unroll
    for (int ky = -1; ky <= 1; ky++) {
        int ny = y + ky;
        if (ny < 0) ny = 0;
        else if (ny >= H) ny = H - 1;
        #pragma unroll
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            if (nx < 0) nx = 0;
            else if (nx >= W) nx = W - 1;
            float px = src[ny * W + nx];
            int ki = (ky + 1) * 3 + (kx + 1);
            gx += px * Gx[ki];
            gy += px * Gy[ki];
        }
    }
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

    int block_size = atoi(argv[2]);
    if (block_size != 8 && block_size != 16 && block_size != 32) {
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

    dim3 block(block_size, block_size);
    dim3 grid((W + block_size - 1) / block_size,
              (H + block_size - 1) / block_size);

    cudaEvent_t e_start, e_stop;
    CUDA_CHECK(cudaEventCreate(&e_start));
    CUDA_CHECK(cudaEventCreate(&e_stop));

    /* === Gaussian Blur === */
    double p_before = sample_gpu_power_watts();
    CUDA_CHECK(cudaEventRecord(e_start));
    gaussian_blur_kernel<<<grid, block>>>(d_src, d_blur, W, H);
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

    printf("=== Gaussian Blur | Block: %d ===\n", block_size);
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
    sobel_kernel<<<grid, block>>>(d_blur, d_edges, W, H);
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

    printf("=== Sobel Edge Detection | Block: %d ===\n", block_size);
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
