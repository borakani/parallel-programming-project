#ifdef _WIN32
#include <stdio.h>
int main(void) {
    fputs("run_benchmark: WSL/Linux gerektirir.\n", stderr);
    return 2;
}
#else
#define _GNU_SOURCE
#include "sche_compat.h"
#include "sche_csv.h"
#include "sche_execposix.h"
#include "sche_fsutil.h"
#include "sche_metric.h"
#include "sche_pgm.h"
#include <errno.h>
#include <getopt.h>
#include <glob.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#define MAXL 512
#define MAXT 32

struct Opts {
    char repo[MAXL];
    char outroot[MAXL];
    char runname[128];
    char seq[MAXL];
    char omp[MAXL];
    char cuda[MAXL];
    int has_omp;
    int has_cuda;
    int threads[MAXT];
    int nthr;
    int blocks[MAXT];
    int nblk;
    int repeat;
    int thrpix;
    char **imgs;
    size_t nimg;
    char metrics[MAXL];
};

static int mkdir_p(char *path) {
    char *p;
    for (p = path + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            if (mkdir(path, 0755) != 0 && errno != EEXIST)
                return -1;
            *p = '/';
        }
    }
    return mkdir(path, 0755) != 0 && errno != EEXIST ? -1 : 0;
}

static void pjoin(char *o, size_t c, const char *a, const char *b) {
    snprintf(o, c, "%s/%s", a, b);
}

static int parse_int_csv(const char *s, int *o, int mx, int *n) {
    const char *p = s;
    *n = 0;
    while (*p) {
        char *ep;
        long v;

        while (*p == ' ' || *p == '\t' || *p == ',')
            p++;
        if (!*p)

            break;
        errno = 0;


        v = strtol(p, &ep, 10);

        if (errno || ep == p)
            return -1;

        o[(*n)++] = (int)v;
        if (*n >= mx)
            break;
        p = ep;
        if (*p == ',')
            p++;
    }


    return *n > 0 ? 0 : -1;
}


static void path_stem(const char *p, char *o, size_t cap) {
    const char *s = strrchr(p, '/');

    s = s ? s + 1 : p;
    snprintf(o, cap, "%s", s);


    {
        char *d;

        d = strrchr(o, '.');
        if (d)
            *d = '\0';
    }


}

static void free_img_list(struct Opts *o) {


    size_t i;

    for (i = 0; i < o->nimg; i++)

        free(o->imgs[i]);
    free(o->imgs);


    o->imgs = NULL;


    o->nimg = 0;

}

static int glob_imgs(struct Opts *o) {
    glob_t g;
    char pat[MAXL];


    size_t i;



    snprintf(pat, sizeof(pat), "%s/images/*.pgm", o->repo);

    if (glob(pat, 0, NULL, &g) != 0)


        return -1;

    o->imgs = calloc((size_t)g.gl_pathc, sizeof *o->imgs);
    if (!o->imgs)


        return -1;



    o->nimg = (size_t)g.gl_pathc;



    for (i = 0; i < o->nimg; i++)

        o->imgs[i] = strdup(g.gl_pathv[i]);
    globfree(&g);

    return o->nimg ? 0 : -1;

}


static int run_variant(const char *cwd, char **av, int rep,


                      ScheMetricRow **out_mean,

                      size_t *out_n, int *okc, int *lrc) {


    ScheMetricRow **blk;

    size_t *lens;



    const ScheMetricRow **pp;

    size_t *lc;

    size_t mlen = 0;



    ScheMetricRow *mean;



    int good = 0;

    int k;

    int j;

    blk = calloc((size_t)rep, sizeof *blk);

    lens = calloc((size_t)rep, sizeof *lens);



    *lrc = -1;

    *okc = 0;

    *out_n = 0;

    *out_mean = NULL;



    if (!blk || !lens)

        return -1;



    {

        char tmpm[MAXL];

        snprintf(tmpm, sizeof tmpm, "%s", cwd);

        if (mkdir_p(tmpm) != 0) {

            free(blk);

            free(lens);

            return -1;

        }

    }

    for (k = 0; k < rep; k++) {

        char sp[MAXL], se[MAXL];


        int st = 0;



        snprintf(sp, sizeof(sp), "%s/stdout_%02d.txt", cwd, k + 1);



        snprintf(se, sizeof(se), "%s/stderr_%02d.txt", cwd, k + 1);



        if (sche_execposix(cwd, (char *const *) av, sp, se, &st) !=


            0)



            continue;

        *lrc = st;



        if (st != 0)

            continue;



        {

            size_t z = 0;



            char *tx = sche_read_file(sp, &z);


            size_t nr = 0;



            ScheMetricRow *pr;


            if (!tx)

                continue;



            pr = sche_metric_parse_stdout(tx, &nr);

            free(tx);

            if (!pr || nr == 0)

                continue;

            blk[good] = pr;

            lens[good] = nr;



            good++;

        }

    }



    *okc = good;



    if (!good) {


        free(blk);

        free(lens);

        return -1;

    }

    for (j = 1; j < good; j++) {
        if (lens[j] != lens[0]) {
            int ii;
            for (ii = 0; ii < good; ii++)
                sche_metric_row_free_slice(blk[ii], lens[ii]);
            free(blk);
            free(lens);

            return -1;
        }


    }





    pp = malloc((size_t) good * sizeof *pp);

    lc = malloc((size_t) good * sizeof *lc);

    for (j = 0; j < good; j++) {

        pp[j] = blk[j];

        lc[j] = lens[j];

    }

    mean = sche_metric_average_runs(pp, lc, (size_t) good,


                                  &mlen);

    free(pp);

    free(lc);

    for (j = 0; j < good; j++) {

        sche_metric_row_free_slice(blk[j], lens[j]);


    }

    free(blk);

    free(lens);

    if (!mean || mlen == 0)

        return -1;

    *out_mean = mean;



    *out_n = mlen;

    return 0;



}

int main(int argc, char **argv)
{
    struct Opts O;
    struct option lo[] = {

        {"seq-bin", required_argument, 0, 's'},
        {"omp-bin", required_argument, 0, 'p'},
        {"cuda-bin", required_argument, 0, 'u'},
        {"repeat", required_argument, 0, 'r'},
        {"compare-threshold", required_argument, 0, 'T'},


        {"output-dir", required_argument, 0, 'o'},




        {"run-name", required_argument, 0, 'n'},


        {"repo", required_argument, 0, 'R'},




        {"image", required_argument, 0, 'i'},


        {"omp-threads", required_argument, 0, 't'},


        {"cuda-blocks", required_argument, 0, 'b'},




        {"metrics-csv", required_argument, 0, 'm'},




        {0, 0, 0, 0}


    };


    memset(&O, 0, sizeof O);
    strcpy(O.repo, ".");
    snprintf(O.outroot, sizeof O.outroot, "scheduler/runs");
    O.repeat = 1;
    O.thrpix = 3;
    O.threads[0] = 8;
    O.nthr = 1;
    O.blocks[0] = 8;
    O.blocks[1] = 16;
    O.blocks[2] = 32;
    O.nblk = 3;
    for (;;) {
        int c, li = 0;
        c = getopt_long(argc, argv, "s:p:u:r:T:o:n:R:i:t:b:m:", lo, &li);
        if (c == -1)
            break;
        switch (c) {
        case 's':
            snprintf(O.seq, sizeof O.seq, "%s", optarg);
            break;
        case 'p':
            snprintf(O.omp, sizeof O.omp, "%s", optarg);
            O.has_omp = 1;
            break;
        case 'u':
            snprintf(O.cuda, sizeof O.cuda, "%s", optarg);
            O.has_cuda = 1;
            break;
        case 'r':
            O.repeat = atoi(optarg);
            if (O.repeat < 1)
                O.repeat = 1;
            break;
        case 'T':
            O.thrpix = atoi(optarg);
            break;
        case 'o':
            snprintf(O.outroot, sizeof O.outroot, "%s", optarg);
            break;
        case 'n':
            snprintf(O.runname, sizeof O.runname, "%s", optarg);
            break;
        case 'R':
            snprintf(O.repo, sizeof O.repo, "%s", optarg);
            break;
        case 'i': {
            char **ni = realloc(O.imgs, sizeof(char *) * (O.nimg + 1));
            if (!ni)
                return 2;
            O.imgs = ni;
            O.imgs[O.nimg++] = strdup(optarg);
        } break;
        case 't':
            if (parse_int_csv(optarg, O.threads, MAXT, &O.nthr) != 0)
                return 2;
            break;
        case 'b':
            if (parse_int_csv(optarg, O.blocks, MAXT, &O.nblk) != 0)
                return 2;
            break;
        case 'm':
            snprintf(O.metrics, sizeof O.metrics, "%s", optarg);
            break;
        default:
            return 2;
        }
    }
    while (optind < argc) {
        char **ni = realloc(O.imgs, sizeof(char *) * (O.nimg + 1));
        if (!ni)
            return 2;
        O.imgs = ni;
        O.imgs[O.nimg++] = strdup(argv[optind++]);
    }
    if (!O.seq[0]) {
        fputs("run_benchmark: --seq-bin gerekli\n", stderr);
        return 2;
    }
    if (access(O.seq, X_OK) != 0) {
        perror("seq-bin");
        return 2;
    }
    if (O.has_omp && access(O.omp, X_OK) != 0) {
        perror("omp-bin");
        return 2;
    }
    if (O.has_cuda && access(O.cuda, X_OK) != 0) {
        perror("cuda-bin");
        return 2;
    }
    if (O.nimg == 0 && glob_imgs(&O) != 0) {
        fputs("Gorsel yok: --image veya repo/images/*.pgm\n", stderr);
        return 2;
    }
    if (O.nimg == 0) {
        fputs("Gorsel listesi bos\n", stderr);
        return 2;
    }
    if (!O.runname[0]) {
        time_t tt = time(NULL);
        struct tm *g = gmtime(&tt);
        strftime(O.runname, sizeof O.runname, "%Y%m%dT%H%M%SZ", g);
    }
    {
        char rr[MAXL], seqd[MAXL], bref[SCHE_PATH_MAX], eref[SCHE_PATH_MAX];
        char *avseq[4];
        size_t ix;
        int ec = 0, anyb = 0;
        pjoin(rr, sizeof rr, O.outroot, O.runname);
        if (mkdir_p(rr) != 0) {
            perror("run root");
            free_img_list(&O);
            return 2;
        }
        if (!O.metrics[0])
            pjoin(O.metrics, sizeof O.metrics, rr, "metrics.csv");
        remove(O.metrics);
        for (ix = 0; ix < O.nimg; ix++) {
            char stem[256];
            ScheMetricRow *mean = NULL;
            size_t nr = 0;
            int nok = 0, lrc = 0;
            path_stem(O.imgs[ix], stem, sizeof stem);
            pjoin(seqd, sizeof seqd, rr, "sequential");
            pjoin(seqd, sizeof seqd, seqd, stem);
            avseq[0] = O.seq;
            avseq[1] = O.imgs[ix];
            avseq[2] = NULL;
            if (run_variant(seqd, avseq, O.repeat, &mean, &nr, &nok,
                            &lrc) != 0 ||
                !mean) {
                fputs("sequential basarisiz\n", stderr);
                anyb = 1;
                continue;
            }
            sche_metrics_csv_append(O.metrics, O.runname, stem, mean, nr);
            sche_metric_row_free_slice(mean, nr);
            mean = NULL;
            if (sche_fs_find_outputs(seqd, bref, eref) != 0) {
                fputs("sequential PGM bulunamadi\n", stderr);
                anyb = 1;
                continue;
            }
            if (O.has_omp) {
                int ti;
                for (ti = 0; ti < O.nthr; ti++) {
                    char ompd[MAXL], bcmp[SCHE_PATH_MAX], ecmp[SCHE_PATH_MAX];
                    char ths[32];
                    char *avom[4];
                    snprintf(ths, sizeof ths, "%d", O.threads[ti]);
                    pjoin(ompd, sizeof ompd, rr, "openmp");
                    {
                        char tdir[64];
                        snprintf(tdir, sizeof tdir, "t%d", O.threads[ti]);
                        pjoin(ompd, sizeof ompd, ompd, tdir);
                    }
                    pjoin(ompd, sizeof ompd, ompd, stem);
                    avom[0] = O.omp;
                    avom[1] = O.imgs[ix];
                    avom[2] = ths;
                    avom[3] = NULL;
                    if (run_variant(ompd, avom, O.repeat, &mean, &nr, &nok,
                                    &lrc) != 0 ||
                        !mean) {
                        anyb = 1;
                    } else {
                        sche_metrics_csv_append(O.metrics, O.runname, stem,
                                                mean, nr);
                        sche_metric_row_free_slice(mean, nr);
                        mean = NULL;
                        if (nok == O.repeat &&
                            sche_fs_find_outputs(ompd, bcmp, ecmp) == 0) {
                            SchePgmCompare c0, c1;
                            c0 = sche_pgm_compare_files(bref, bcmp, O.thrpix);
                            c1 = sche_pgm_compare_files(eref, ecmp, O.thrpix);
                            if (!c0.ok || !c1.ok)
                                ec = 1;
                        } else
                            ec = 1;
                    }
                }
            }
            if (O.has_cuda) {
                int bi;
                for (bi = 0; bi < O.nblk; bi++) {
                    char cud[MAXL], bcmp[SCHE_PATH_MAX], ecmp[SCHE_PATH_MAX];
                    char bs[32];
                    char *avcu[4];
                    snprintf(bs, sizeof bs, "%d", O.blocks[bi]);
                    pjoin(cud, sizeof cud, rr, "cuda");
                    {
                        char bdir[64];
                        snprintf(bdir, sizeof bdir, "b%d", O.blocks[bi]);
                        pjoin(cud, sizeof cud, cud, bdir);
                    }
                    pjoin(cud, sizeof cud, cud, stem);
                    avcu[0] = O.cuda;
                    avcu[1] = O.imgs[ix];
                    avcu[2] = bs;
                    avcu[3] = NULL;
                    if (run_variant(cud, avcu, O.repeat, &mean, &nr, &nok,
                                    &lrc) != 0 ||
                        !mean) {
                        anyb = 1;
                    } else {
                        sche_metrics_csv_append(O.metrics, O.runname, stem,
                                                mean, nr);
                        sche_metric_row_free_slice(mean, nr);
                        mean = NULL;
                        if (nok == O.repeat &&
                            sche_fs_find_outputs(cud, bcmp, ecmp) == 0) {
                            SchePgmCompare c0, c1;
                            c0 = sche_pgm_compare_files(bref, bcmp, O.thrpix);
                            c1 = sche_pgm_compare_files(eref, ecmp, O.thrpix);
                            if (!c0.ok || !c1.ok)
                                ec = 1;
                        } else
                            ec = 1;
                    }
                }
            }
            (void)lrc;
        }
        printf("METRICS: %s\n", O.metrics);
        free_img_list(&O);
        return anyb ? 2 : ec;
    }
}

#endif
