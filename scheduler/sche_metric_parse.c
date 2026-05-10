#include "sche_metric.h"

#include "sche_compat.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int starts_three_eq(const char *s) {
    while (*s && isspace((unsigned char)*s))
        s++;
    return strncmp(s, "===", 3) == 0;
}

static int split_lines(const char *txt, char ***outs, size_t *nlines) {
    const char *p = txt;
    char **lines = NULL;
    size_t n = 0, cap = 0;

    *outs = NULL;
    *nlines = 0;
    if (!txt)
        return 0;

    for (;;) {
        const char *e = strchr(p, '\n');
        size_t ln = e ? (size_t)(e - p) : strlen(p);

        while (ln > 0 && (p[ln - 1] == '\r'))
            ln--;

        if (n >= cap) {
            cap = cap ? cap * 2 : 32;
            void *nx = realloc(lines, cap * sizeof(*lines));
            if (!nx)
                goto fail;
            lines = nx;
        }
        lines[n] = malloc(ln + 1);
        if (!lines[n])
            goto fail;
        memcpy(lines[n], p, ln);
        lines[n][ln] = '\0';
        n++;

        if (!e)
            break;
        p = e + 1;
    }
    *outs = lines;
    *nlines = n;
    return 0;

fail:
    for (size_t i = 0; i < n; i++)
        free(lines[i]);
    free(lines);
    return -1;
}

static void free_lines(char **lines, size_t n) {
    size_t i;
    for (i = 0; i < n; i++)
        free(lines[i]);
    free(lines);
}

static int hdr_parse(char *ln, char *phase, size_t psz, char *backend,
                     size_t bsz, char *threads, size_t tsz,
                     char *block, size_t bksz,
                     char *label, size_t lsz) {
    char *s = ln;
    while (*s && isspace((unsigned char)*s))
        s++;

    threads[0] = '\0';
    block[0] = '\0';
    snprintf(label, lsz, "%s", "");

    if (strstr(s, "Gaussian Blur"))
        snprintf(phase, psz, "gaussian");
    else if (strstr(s, "Sobel Edge Detection"))
        snprintf(phase, psz, "sobel");
    else
        return 0;

    {
        char *t = strstr(s, "Threads:");
        char *bk = strstr(s, "Block:");
        if (t && bk) {
            snprintf(backend, bsz, "unknown");
            snprintf(label, lsz, "ambiguous_tail");
            return 1;
        }
        if (t) {
            int tv = -1;
            char *colon = strchr(t, ':');
            if (colon)
                sscanf(colon + 1, "%d", &tv);
            if (tv >= 0)
                snprintf(threads, tsz, "%d", tv);
            snprintf(backend, bsz, "openmp");
            snprintf(label, lsz, "threads=%s", threads);
            return 1;
        }
        if (bk) {
            int bv = -1;
            char *colon = strchr(bk, ':');
            if (colon)
                sscanf(colon + 1, "%d", &bv);
            if (bv >= 0)
                snprintf(block, bksz, "%d", bv);
            snprintf(backend, bsz, "cuda");
            snprintf(label, lsz, "block=%s", block);
            return 1;
        }
        snprintf(backend, bsz, "sequential");
        snprintf(label, lsz, "sequential");
        return 1;
    }
}

static int ln_image_size(char *ln, ScheMetricRow *acc) {
    char *q = strstr(ln, "Image size:");
    int w, h;
    if (!q)
        return 0;
    q += strlen("Image size:");
    while (*q && isspace((unsigned char)*q))
        q++;
    if (sscanf(q, "%dx%d", &w, &h) != 2)
        return 0;
    snprintf(acc->width, sizeof(acc->width), "%d", w);
    snprintf(acc->height, sizeof(acc->height), "%d", h);
    return 1;
}

static int ln_double_after_colon(char *ln, const char *key, double *v,
                                int *ok) {
    char *q = strstr(ln, key);
    if (!q)
        return 0;
    q += strlen(key);
    if (*q == ':')
        q++;
    while (*q && isspace((unsigned char)*q))
        q++;
    if (strncmp(q, "N/A", 3) == 0) {
        *ok = 0;
        return 1;
    }
    errno = 0;
    *v = strtod(q, NULL);
    if (errno)
        return 0;
    *ok = 1;
    return 1;
}

static int ln_time(char *ln, ScheMetricRow *acc) {
    if (!strstr(ln, "Time:") || !strstr(ln, "seconds"))
        return 0;
    return ln_double_after_colon(ln, "Time:", &acc->time_s,
                                 &acc->has_time);
}

static void push_row(char *phase, char *backend, char *threads, char *block,
                     char *label, ScheMetricRow *acc,
                     ScheMetricRow **buf, size_t *cap,
                     size_t *len) {
    int ww, hh;
    if (!acc->has_time)
        return;
    ww = atoi(acc->width);
    hh = atoi(acc->height);
    if (ww <= 0 || hh <= 0)
        return;

    if (*len >= *cap) {
        size_t nc = *cap ? *cap * 2 : 8;
        void *nx = realloc(*buf, nc * sizeof(ScheMetricRow));
        if (!nx)
            return;
        *buf = nx;
        *cap = nc;
    }
    memset(&(*buf)[*len], 0, sizeof(ScheMetricRow));
    snprintf((*buf)[*len].phase, sizeof((*buf)[*len].phase), "%s", phase);
    snprintf((*buf)[*len].backend, sizeof((*buf)[*len].backend), "%s",
             backend);
    snprintf((*buf)[*len].threads, sizeof((*buf)[*len].threads), "%s",
             threads);
    snprintf((*buf)[*len].block, sizeof((*buf)[*len].block), "%s", block);
    snprintf((*buf)[*len].label, sizeof((*buf)[*len].label), "%s", label);
    snprintf((*buf)[*len].width, sizeof((*buf)[*len].width), "%s",
             acc->width);
    snprintf((*buf)[*len].height, sizeof((*buf)[*len].height), "%s",
             acc->height);
    (*buf)[*len].time_s = acc->time_s;
    (*buf)[*len].has_time = acc->has_time;
    (*buf)[*len].has_energy = acc->has_energy;
    (*buf)[*len].energy_j = acc->energy_j;
    (*buf)[*len].has_gflops = acc->has_gflops;
    (*buf)[*len].gflops = acc->gflops;
    (*buf)[*len].has_gflops_w = acc->has_gflops_w;
    (*buf)[*len].gflops_w = acc->gflops_w;

    (*len)++;
}

ScheMetricRow *sche_metric_parse_stdout(const char *txt, size_t *pn) {
    char **lines = NULL;
    size_t nl = 0, i = 0;
    ScheMetricRow *buf = NULL;
    size_t cap = 0, len = 0;

    if (pn)
        *pn = 0;
    if (!txt || split_lines(txt, &lines, &nl) < 0)
        return NULL;

    while (i < nl) {
        if (!starts_three_eq(lines[i])) {
            i++;
            continue;
        }

        char phase[SCHE_METRIC_PHASE_SZ];
        char backend[SCHE_METRIC_BACKEND_SZ];
        char threads[SCHE_METRIC_THR_SZ];
        char block[SCHE_METRIC_BLOCK_SZ];
        char label[SCHE_METRIC_LABEL_SZ];
        ScheMetricRow acc;

        if (!hdr_parse(lines[i], phase, sizeof(phase), backend,
                      sizeof(backend), threads, sizeof(threads), block,
                      sizeof(block), label,
                      sizeof(label))) {
            i++;
            continue;
        }

        memset(&acc, 0, sizeof(acc));
        acc.width[0] = '\0';
        acc.height[0] = '\0';

        i++;
        while (i < nl && !starts_three_eq(lines[i])) {
            ln_image_size(lines[i], &acc);
            ln_time(lines[i], &acc);

            if (strstr(lines[i], "Energy:")) {
                if (strstr(lines[i], "N/A"))
                    acc.has_energy = 0;
                else {
                    double ee;
                    int ok_e = 0;
                    ln_double_after_colon(lines[i], "Energy:",
                                         &ee, &ok_e);
                    if (ok_e) {
                        acc.has_energy = 1;
                        acc.energy_j = ee;
                    }
                }
            }
            if (strstr(lines[i], "GFLOPS/W")) {
                if (strstr(lines[i], "N/A"))
                    acc.has_gflops_w = 0;
                else {
                    double gw;
                    int pkg = 0;
                    if (ln_double_after_colon(lines[i],
                                              "GFLOPS/W",
                                              &gw, &pkg) &&
                        pkg) {
                        acc.has_gflops_w = 1;
                        acc.gflops_w = gw;
                    }
                }
            } else if (strstr(lines[i], "GFLOPS:")) {
                double g;
                int okg = 0;
                ln_double_after_colon(lines[i], "GFLOPS:", &g, &okg);
                if (okg) {
                    acc.gflops = g;
                    acc.has_gflops = 1;
                }
            }
            i++;
        }

        push_row(phase, backend, threads, block, label, &acc,
                 &buf, &cap, &len);
    }

    free_lines(lines, nl);
    if (pn)
        *pn = len;
    return buf;
}

void sche_metric_row_free_slice(ScheMetricRow *rows, size_t n) {
    (void)n;
    free(rows);
}
