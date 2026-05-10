#ifndef SCHE_CSV_H
#define SCHE_CSV_H

#include "sche_metric.h"
#include <stddef.h>

void sche_metrics_csv_append(const char *path, const char *run_name,
                             const char *image_stem,
                             const ScheMetricRow *rows, size_t nrow);

#endif
