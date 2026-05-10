#include "sche_pgm.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *trim_line_inplace(char *s) {
    char *end;
    while (*s && isspace((unsigned char)*s))
        s++;
    if (!*s)
        return s;
    end = s + strlen(s) - 1;
    while (end > s && isspace((unsigned char)*end))
        end--;
    end[1] = '\0';
    return s;
}

static int read_ascii_line(FILE *f, char *buf, size_t bufsize) {
    if (!fgets(buf, (int)bufsize, f))
        return -1;
    return 0;
}

int sche_pgm_read(const char *path, SchePgm *out) {
    char line[1024];
    char *p;
    int w = 0, h = 0, maxv = 0;

    memset(out, 0, sizeof(*out));
    FILE *f = fopen(path, "rb");
    if (!f)
        return -1;

    for (;;) {
        if (read_ascii_line(f, line, sizeof line) < 0) {
            fclose(f);
            return -1;
        }
        p = trim_line_inplace(line);
        if (p[0] == '#' || strlen(p) == 0)
            continue;
        if (strcmp(p, "P5") != 0 && strncmp(p, "P5", 2) != 0) {
            fclose(f);
            return -1;
        }
        break;
    }
    for (;;) {
        if (read_ascii_line(f, line, sizeof line) < 0) {
            fclose(f);
            return -1;
        }
        p = trim_line_inplace(line);
        if (p[0] == '#' || strlen(p) == 0)
            continue;
        if (sscanf(p, "%d %d", &w, &h) != 2 || w <= 0 || h <= 0) {
            fclose(f);
            return -1;
        }
        break;
    }
    for (;;) {
        if (read_ascii_line(f, line, sizeof line) < 0) {
            fclose(f);
            return -1;
        }
        p = trim_line_inplace(line);
        if (p[0] == '#' || strlen(p) == 0)
            continue;
        maxv = atoi(p);
        if (maxv != 255) {
            fclose(f);
            return -1;
        }
        break;
    }

    size_t np = (size_t)w * (size_t)h;
    unsigned char *pix = malloc(np);
    if (!pix) {
        fclose(f);
        return -1;
    }
    size_t nr = fread(pix, 1, np, f);
    fclose(f);
    if (nr != np) {
        free(pix);
        return -1;
    }
    out->width = w;
    out->height = h;
    out->pixels = pix;
    return 0;
}

void sche_pgm_free(SchePgm *p) {
    if (!p)
        return;
    free(p->pixels);
    p->pixels = NULL;
    p->width = p->height = 0;
}

SchePgmCompare sche_pgm_compare_files(const char *pa, const char *pb,
                                      int threshold) {
    SchePgm A, B;
    SchePgmCompare r = {0};
    int i;
    if (sche_pgm_read(pa, &A) || sche_pgm_read(pb, &B)) {
        return r;
    }
    if (A.width != B.width || A.height != B.height) {
        sche_pgm_free(&A);
        sche_pgm_free(&B);
        return r;
    }
    size_t n = (size_t)A.width * (size_t)A.height;
    long max_d = 0;
    unsigned long sum_d = 0;
    unsigned long differing = 0;
    double sum_sq = 0.0;

    for (i = 0; i < (int)n; i++) {
        int da = A.pixels[i];
        int db = B.pixels[i];
        int d = da > db ? da - db : db - da;
        if ((long)d > max_d)
            max_d = d;
        sum_d += (unsigned long)d;
        sum_sq += (double)d * (double)d;
        if (d > threshold)
            differing++;
    }
    r.mean_abs_diff = n ? (double)sum_d / (double)n : 0.0;
    r.rmse = n ? sqrt(sum_sq / (double)n) : 0.0;
    r.max_abs_diff = (int)max_d;
    r.differing_pixels = differing;
    r.ok = (max_d <= threshold) ? 1 : 0;
    sche_pgm_free(&A);
    sche_pgm_free(&B);
    return r;
}
