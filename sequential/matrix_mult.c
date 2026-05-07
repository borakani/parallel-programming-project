#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

// Read CPU energy consumption from Intel RAPL (in Joules)
double read_energy() {
    //Intel RAPL, writes the energy which consumed by CPU to a file non-stop
    //The energy is taken from that file
    FILE *f = fopen("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj", "r");
    if (f == NULL) return -1.0;
    long long energy;
    fscanf(f, "%lld", &energy);
    fclose(f);
    // energy / 1e6 because we want to use joule but energy comes in mikrojoule format
    return (double)energy / 1e6;
}
// With i we calculate the rows of A and C
// With j we calculate the columns of B
// With k we go from 0 to N for each (i,j)
void matrix_multiply(double *A, double *B, double *C, int N){
    for(int i = 0; i < N ; i++){
        for(int j = 0; j < N ; j++){
            for(int k = 0; k < N; k++){
                C[i * N + j] += A[i * N + k] * B[k * N + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    if(argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    int N = atoi(argv[1]);
    printf("Matrix size: %dx%d\n", N,N);

    //Allocate memory for matrices A, B and C
    double *A = (double *)malloc(N * N * sizeof(double));
    double *B = (double *)malloc(N * N * sizeof(double));
    double *C = (double *)malloc(N * N * sizeof(double));

    if(A == NULL || B == NULL || C == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }

    //Fill A and b with randow values between 0 and 1
    srand(42);
    for(int i = 0; i < N * N; i++){
        // rand / RAND_MAX generates values between 0 and 1
        A[i] = (double)rand() / RAND_MAX;
        B[i] = (double)rand() / RAND_MAX;
        C[i] = 0.0;
    }

    // Record start time and energy
    // struct timespec is a special structure which keeps time in two different ways
    // first is tv_sec which keeps the whole seconds
    // and the second is tv_nsec which keeps the nanoseconds (1 second equals to 1 billion nanoseconds)
    struct timespec start, end;
    double energy_start = read_energy();
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    // Run matrix multiplication
    matrix_multiply(A, B, C, N);

    // Record end time and energy
    clock_gettime(CLOCK_MONOTONIC, &end);
    double energy_end = read_energy();

    // Calculate the elapsed time in seconds
    double time_sec = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    // Calculate energy consumed in Joules
    double energy_j = energy_end - energy_start;

    // FLOPS means "Floating Point Operations Per Second"
    // GFLOPS means "Giga Floating Point Operations Per Second" (FLOPS * 1 billion)
    // 2 * N^3 because for each cell in C there are 3 loops from 0 to N
    // and in each loop there are 2 operations made: 1 multiplication, 1 addition
    double flops = 2.0 * N * N * N;
    double gflops = (flops / time_sec) / 1e9;

    // Calculate GFLOPS/W (efficiency)
    double gflops_per_watt = (energy_j > 0) ? (gflops / (energy_j / time_sec)) : -1.0;
    
    // Print results
    printf("Time:        %.4f seconds\n", time_sec);
    printf("Energy:      %.4f Joules\n", energy_j);
    printf("GFLOPS:      %.4f\n", gflops);
    printf("GFLOPS/W:    %.4f\n", gflops_per_watt);

    free(A);
    free(B);
    free(C);

    return 0;

}