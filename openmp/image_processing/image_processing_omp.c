#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define RAPL_PATH "/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"

static long long read_energy() {
    FILE *f = fopen(RAPL_PATH, "r");
    if (!f) return -1;
    long long val;
    fscanf(f, "%lld", &val);
    fclose(f);
    return val;
}

typedef struct {
    int width, height;
    float *data;
} Image;

Image *read_pgm(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", filename); return NULL; }

    char line[1024];

    fgets(line, sizeof(line), f);
    if (line[0] != 'P' || line[1] != '5') {
        fprintf(stderr, "Not a binary PGM file\n");
        fclose(f); return NULL;
    }

    while (fgets(line, sizeof(line), f) && line[0] == '#');

    int w, h;
    sscanf(line, "%d %d", &w, &h);

    fgets(line, sizeof(line), f);

    Image *img = malloc(sizeof(Image));
    img->width  = w;
    img->height = h;
    img->data   = malloc(w * h * sizeof(float));

    unsigned char *buf = malloc(w * h);
    fread(buf, 1, w * h, f);
    for (int i = 0; i < w * h; i++)
        img->data[i] = (float)buf[i];
    free(buf);
    fclose(f);
    return img;
}

void free_image(Image *img) {
    free(img->data);
    free(img);
}

void write_pgm(const char *filename, const Image *img) {
    FILE *f = fopen(filename, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", filename); return; }

    fprintf(f, "P5\n%d %d\n255\n", img->width, img->height);

    unsigned char *buf = malloc(img->width * img->height);
    for (int i = 0; i < img->width * img->height; i++) {
        float val = img->data[i];
        if (val < 0)   val = 0;
        if (val > 255) val = 255;
        buf[i] = (unsigned char)val;
    }
    fwrite(buf, 1, img->width * img->height, f);
    free(buf);
    fclose(f);
}

static const float GAUSS[5][5] = {
    {1,  4,  7,  4, 1},
    {4, 16, 26, 16, 4},
    {7, 26, 41, 26, 7},
    {4, 16, 26, 16, 4},
    {1,  4,  7,  4, 1}
};
static const float GAUSS_SUM = 273.0f;

void gaussian_blur(const Image *src, Image *dst, int num_threads) {
    int W = src->width, H = src->height;
    #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float acc = 0.0f;
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int ny = y + ky, nx = x + kx;
                    if (ny < 0) ny = 0;
                    if (ny >= H) ny = H - 1;
                    if (nx < 0) nx = 0;
                    if (nx >= W) nx = W - 1;
                    acc += src->data[ny * W + nx] * GAUSS[ky+2][kx+2];
                }
            }
            dst->data[y * W + x] = acc / GAUSS_SUM;
        }
    }
}

static const float Gx[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1}
};

static const float Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

void sobel_edge(const Image *src, Image *dst, int num_threads) {
    int W = src->width, H = src->height;
    #pragma omp parallel for collapse(2) num_threads(num_threads) schedule(static)
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ny = y + ky, nx = x + kx;
                    if (ny < 0) ny = 0;
                    if (ny >= H) ny = H - 1;
                    if (nx < 0) nx = 0;
                    if (nx >= W) nx = W - 1;
                    float px = src->data[ny * W + nx];
                    gx += px * Gx[ky+1][kx+1];
                    gy += px * Gy[ky+1][kx+1];
                }
            }
            dst->data[y * W + x] = sqrtf(gx*gx + gy*gy);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <image.pgm> <num_threads>\n", argv[0]);
        return 1;
    }

    int num_threads = atoi(argv[2]);

    Image *src = read_pgm(argv[1]);
    if (!src) return 1;

    Image *blurred = malloc(sizeof(Image));
    blurred->width  = src->width;
    blurred->height = src->height;
    blurred->data   = malloc(src->width * src->height * sizeof(float));

    Image *edges = malloc(sizeof(Image));
    edges->width  = src->width;
    edges->height = src->height;
    edges->data   = malloc(src->width * src->height * sizeof(float));

    struct timespec t0, t1;
    long long e0, e1;
    double time_sec, energy_j, gflops, gflops_w;
    long long pixels = (long long)src->width * src->height;

    e0 = read_energy();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    gaussian_blur(src, blurred, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    e1 = read_energy();

    time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    energy_j = (e0 >= 0 && e1 >= 0) ? (e1 - e0) / 1e6 : -1.0;
    gflops   = (pixels * 50.0) / (time_sec * 1e9);
    gflops_w = (energy_j > 0) ? gflops / (energy_j / time_sec) : -1.0;

    printf("=== Gaussian Blur | Threads: %d ===\n", num_threads);
    printf("Image size:  %dx%d\n", src->width, src->height);
    printf("Time:        %.4f seconds\n", time_sec);
    if (energy_j >= 0) printf("Energy:      %.4f Joules\n", energy_j);
    else                printf("Energy:      N/A (run with sudo)\n");
    printf("GFLOPS:      %.4f\n", gflops);
    if (gflops_w >= 0) printf("GFLOPS/W:    %.4f\n\n", gflops_w);
    else                printf("GFLOPS/W:    N/A\n\n");

    e0 = read_energy();
    clock_gettime(CLOCK_MONOTONIC, &t0);
    sobel_edge(blurred, edges, num_threads);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    e1 = read_energy();

    time_sec = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9;
    energy_j = (e0 >= 0 && e1 >= 0) ? (e1 - e0) / 1e6 : -1.0;
    gflops   = (pixels * 36.0) / (time_sec * 1e9);
    gflops_w = (energy_j > 0) ? gflops / (energy_j / time_sec) : -1.0;

    printf("=== Sobel Edge Detection | Threads: %d ===\n", num_threads);
    printf("Image size:  %dx%d\n", src->width, src->height);
    printf("Time:        %.4f seconds\n", time_sec);
    if (energy_j >= 0) printf("Energy:      %.4f Joules\n", energy_j);
    else                printf("Energy:      N/A (run with sudo)\n");
    printf("GFLOPS:      %.4f\n", gflops);
    if (gflops_w >= 0) printf("GFLOPS/W:    %.4f\n\n", gflops_w);
    else                printf("GFLOPS/W:    N/A\n\n");

    char blur_out[64], edge_out[64];
    sprintf(blur_out, "output_%dx%d_blurred.pgm", src->width, src->height);
    sprintf(edge_out, "output_%dx%d_edges.pgm",   src->width, src->height);
    write_pgm(blur_out, blurred);
    write_pgm(edge_out, edges);

    free_image(src);
    free_image(blurred);
    free_image(edges);
    return 0;
}