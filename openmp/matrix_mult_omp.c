#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

// Read energy from Intel RAPL
double read_energy() {
    FILE *f = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
    if (!f) return -1.0;
    double energy;
    fscanf(f, "%lf", &energy);
    fclose(f);
    return energy / 1e6; // Convert to Joules
}

// Perform matrix multiplication C = A * B (OpenMP parallelized)
// Only i and j loops are parallelized with collapse(2).
// Parallelizing k would require reduction(+:C[i*N+j]) to avoid race conditions,
// but this creates a new thread team for each (i,j) pair — millions of thread
// creations — and forces synchronization at every reduction. Since collapse(2)
// already exposes N*N iterations to the thread pool, parallelizing k adds
// overhead without meaningful gain.
void matrix_multiply(double *A, double *B, double *C, int N) {
    #pragma omp parallel for schedule(static) collapse(2) // with collapse(2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int num_threads = omp_get_max_threads();

    // Allocate matrices
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)calloc(N * N, sizeof(double));

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Initialize matrices with random values
    srand(42);
    for (int i = 0; i < N * N; i++) {
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
    }

    // Record start time and energy
    struct timespec start, end;
    double energy_start = read_energy();
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Run matrix multiplication
    matrix_multiply(A, B, C, N);

    // Record end time and energy
    clock_gettime(CLOCK_MONOTONIC, &end);
    double energy_end = read_energy();

    // Calculate metrics
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    double energy_j = (energy_end > 0 && energy_start > 0) ? (energy_end - energy_start) : -1.0;
    double gflops = (2.0 * N * N * N) / (time_sec * 1e9);
    double gflops_per_watt = (energy_j > 0) ? (gflops / (energy_j / time_sec)) : -1.0;

    printf("Matrix size:  %dx%d\n", N, N);
    printf("Threads:      %d\n", num_threads);
    printf("Time:         %.4f seconds\n", time_sec);
    printf("Energy:       %.4f Joules\n", (energy_j > 0) ? energy_j : 0.0);
    printf("GFLOPS:       %.4f\n", gflops);
    printf("GFLOPS/W:     %.4f\n", gflops_per_watt);

    free(A);
    free(B);
    free(C);

    return 0;
}