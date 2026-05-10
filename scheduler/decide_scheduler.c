#include "sche_policy.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define SHE_POPEN(a, b) _popen((a), (b))
#define SHE_PCLOSE(f) _pclose(f)
#else
#define SHE_POPEN(a, b) popen((a), (b))
#define SHE_PCLOSE(f) pclose(f)
#endif
static double nvsmi_draw(void) {
    FILE *fp;
    double w = -1.0;
#ifndef _WIN32
    fp = SHE_POPEN(
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits "
        "2>/dev/null",
        "r");
#else
    fp = SHE_POPEN(
        "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
        "r");
#endif
    if (!fp)
        return -1.0;
    if (fscanf(fp, "%lf", &w) != 1)
        w = -1.0;
    SHE_PCLOSE(fp);
    if (w <= 0.0 || !isfinite(w))
        return -1.0;
    return w;
}

int main(int argc, char **argv) {
    double budget = -1;
    double cpu_e = 0, gpu_e = 0, gpu_w = 0;
    int ok_c = 0, ok_g = 0, ok_w = 0, poll = 0, ex = 0;
    int i;
    SchePolicyDecision dec;
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
            errno = 0;
            budget = strtod(argv[++i], NULL);
            if (errno)
                budget = -1;
        } else if (strcmp(argv[i], "--cpu-energy") == 0 &&
                   i + 1 < argc) {
            errno = 0;
            cpu_e = strtod(argv[++i], NULL);
            ok_c = (errno == 0);
        } else if (strcmp(argv[i], "--gpu-energy") == 0 &&
                   i + 1 < argc) {
            errno = 0;
            gpu_e = strtod(argv[++i], NULL);
            ok_g = (errno == 0);
        } else if (strcmp(argv[i], "--gpu-power") == 0 &&
                   i + 1 < argc) {
            errno = 0;
            gpu_w = strtod(argv[++i], NULL);
            ok_w = (errno == 0 && gpu_w > 0);
        } else if (strcmp(argv[i], "--poll-gpu") == 0)
            poll = 1;
        else if (strcmp(argv[i], "--exit-code") == 0)
            ex = 1;
        else {
            fprintf(stderr, "bilinmeyen: %s\n", argv[i]);
            return 2;
        }
    }
    if (budget < 0) {
        fprintf(stderr, "--budget gerekli (W)\n");
        return 2;
    }
    if (poll)
        gpu_w = nvsmi_draw();
    ok_w = poll ? (gpu_w > 0) : ok_w;
    sche_policy_choose(cpu_e, ok_c, gpu_e, ok_g, gpu_w, ok_w, budget,
                      &dec);
    printf("%s\n", dec.dev == SCHE_DEV_GPU ? "GPU" : "CPU");

    puts(dec.reason);
    return ex ? (dec.dev == SCHE_DEV_GPU ? 0 : 1) : 0;
}
