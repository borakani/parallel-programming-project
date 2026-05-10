#include "sche_csv.h"

#include <stdio.h>

void sche_metrics_csv_append(const char *path, const char *run_name,
                             const char *image_stem,
                             const ScheMetricRow *rows, size_t nrow) {
    FILE *fp = fopen(path, "ab+");
    size_t i;
    long sz;

    if (!fp)
        return;
    if (fseek(fp, 0L, SEEK_END) != 0) {
        fclose(fp);
        return;
    }
    sz = ftell(fp);

    if (sz == 0L) {
        fprintf(fp,
                "run_name,image_stem,phase,backend,threads,block,label,width,"
                "height,time_s,energy_j,gflops,gflops_w,samples\n");
    }

    for (i = 0; i < nrow; i++) {
        const ScheMetricRow *r = &rows[i];

        fprintf(fp, "%s,%s,%s,%s,%s,%s,%s,%s,%s,",
                run_name ? run_name : "",
                image_stem ? image_stem : "", r->phase, r->backend,
                r->threads, r->block, r->label, r->width, r->height);

        if (r->has_time)
            fprintf(fp, "%.10g,", r->time_s);
        else
            fprintf(fp, ",");

        if (r->has_energy)
            fprintf(fp, "%.10g,", r->energy_j);
        else
            fprintf(fp, ",");

        if (r->has_gflops)
            fprintf(fp, "%.10g,", r->gflops);
        else
            fprintf(fp, ",");

        if (r->has_gflops_w)
            fprintf(fp, "%.10g,", r->gflops_w);
        else
            fprintf(fp, ",");

        fprintf(fp, "%s\n", r->samples[0] ? r->samples : "");
    }

    fclose(fp);
}
