#ifndef SCHE_METRIC_H
#define SCHE_METRIC_H

#include <stddef.h>

#define SCHE_METRIC_PHASE_SZ 24
#define SCHE_METRIC_BACKEND_SZ 28
#define SCHE_METRIC_THR_SZ 12
#define SCHE_METRIC_BLOCK_SZ 12
#define SCHE_METRIC_LABEL_SZ 96
#define SCHE_METRIC_WH_SZ 16
#define SCHE_METRIC_SAMPLES_SZ 16

typedef struct ScheMetricRow {
    char phase[SCHE_METRIC_PHASE_SZ];
    char backend[SCHE_METRIC_BACKEND_SZ];
    char threads[SCHE_METRIC_THR_SZ];
    char block[SCHE_METRIC_BLOCK_SZ];
    char label[SCHE_METRIC_LABEL_SZ];
    char width[SCHE_METRIC_WH_SZ];
    char height[SCHE_METRIC_WH_SZ];
    int has_time;
    double time_s;
    int has_energy;
    double energy_j;
    int has_gflops;
    double gflops;
    int has_gflops_w;
    double gflops_w;
    char samples[SCHE_METRIC_SAMPLES_SZ]; /* yazimda kullanilir */
} ScheMetricRow;

/* stdout metni -> satir dizisi; *pn satir sayisi; free: her satir + base */
ScheMetricRow *sche_metric_parse_stdout(const char *txt, size_t *pn);

void sche_metric_row_free_slice(ScheMetricRow *rows, size_t n);

/* N basarili kosum; her blok esit satir uzunlugunda olmali */
ScheMetricRow *sche_metric_average_runs(const ScheMetricRow **runs_rows,
                                        const size_t *runs_count,
                                        size_t n_runs,
                                        size_t *out_len);

#endif
