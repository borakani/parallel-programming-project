#include "sche_metric.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string.h>

static int row_identity_ok(const ScheMetricRow *base,
                           const ScheMetricRow *o) {
    return strcmp(base->phase, o->phase) == 0 &&
           strcmp(base->backend, o->backend) == 0 &&
           strcmp(base->threads, o->threads) == 0 &&
           strcmp(base->block, o->block) == 0 &&
           strcmp(base->width, o->width) == 0 &&
           strcmp(base->height, o->height) == 0;
}

ScheMetricRow *sche_metric_average_runs(const ScheMetricRow **runs_rows,
                                        const size_t *runs_count,
                                        size_t n_runs,
                                        size_t *out_len) {
    ScheMetricRow *mean = NULL;
    size_t j, nrow, k;

    *out_len = 0;
    if (!runs_rows || !runs_count || n_runs == 0 ||
        runs_count[0] == 0)
        return NULL;
    nrow = runs_count[0];
    for (k = 0; k < n_runs; k++) {
        if (!runs_rows[k] || runs_count[k] != nrow)
            return NULL;
    }

    mean = calloc(nrow, sizeof(ScheMetricRow));
    if (!mean)
        return NULL;

    for (j = 0; j < nrow; j++) {
        const ScheMetricRow *base = &runs_rows[0][j];
        double st = 0, se = 0, sg = 0, sg2 = 0;
        unsigned ct = 0, ce = 0, cg = 0, cgw = 0;
        ScheMetricRow *dst = &mean[j];

        memcpy(dst->phase, base->phase, sizeof(dst->phase));
        memcpy(dst->backend, base->backend,
               sizeof(dst->backend));
        memcpy(dst->threads, base->threads, sizeof(dst->threads));
        memcpy(dst->block, base->block, sizeof(dst->block));
        memcpy(dst->label, base->label, sizeof(dst->label));
        memcpy(dst->width, base->width, sizeof(dst->width));
        memcpy(dst->height, base->height, sizeof(dst->height));

        snprintf(dst->samples, sizeof(dst->samples), "%zu",
                 n_runs);

        for (k = 0; k < n_runs; k++) {
            const ScheMetricRow *r = &runs_rows[k][j];
            if (!row_identity_ok(base, r)) {
                free(mean);
                *out_len = 0;
                return NULL;
            }
            if (r->has_time) {
                st += r->time_s;
                ct++;
            }
            if (r->has_energy) {
                se += r->energy_j;
                ce++;
            }
            if (r->has_gflops) {
                sg += r->gflops;
                cg++;
            }
            if (r->has_gflops_w) {
                sg2 += r->gflops_w;
                cgw++;
            }
        }
        if (ct) {
            dst->has_time = 1;
            dst->time_s = st / ct;
        }
        if (ce) {
            dst->has_energy = 1;
            dst->energy_j = se / ce;
        }
        if (cg) {
            dst->has_gflops = 1;
            dst->gflops = sg / cg;
        }
        if (cgw) {
            dst->has_gflops_w = 1;
            dst->gflops_w = sg2 / cgw;
        }
    }

    *out_len = nrow;
    return mean;
}
