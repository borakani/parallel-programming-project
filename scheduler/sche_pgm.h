#ifndef SCHE_PGM_H
#define SCHE_PGM_H

#include <stddef.h>

typedef struct {
    int width;
    int height;
    unsigned char *pixels; /* malloc; boyut width*height */
} SchePgm;

/* P5 PGM yorum satiri atlanir; yalnizca maxval=255 */
int sche_pgm_read(const char *path, SchePgm *out);
void sche_pgm_free(SchePgm *p);

typedef struct {
    int ok;
    int max_abs_diff;
    double mean_abs_diff;
    double rmse;
    long differing_pixels;
} SchePgmCompare;

/* threshold: max mutlak piksel farki uyum kontrolunde */
SchePgmCompare sche_pgm_compare_files(const char *pa, const char *pb,
                                      int threshold);

#endif
