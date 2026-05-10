#define _DEFAULT_SOURCE

#include "sche_fsutil.h"

#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ends_suffix(const char *s, const char *suf) {
    size_t la = strlen(s), lb = strlen(suf);
    if (la < lb)
        return 0;
    return strcmp(s + la - lb, suf) == 0;
}

static int filt_blurred(const struct dirent *e) {
    return ends_suffix(e->d_name, "_blurred.pgm");
}

static int filt_edges(const struct dirent *e) {
    return ends_suffix(e->d_name, "_edges.pgm");
}

static int pick_first(const char *dir,
                     int (*filt)(const struct dirent *),
                     char out[SCHE_PATH_MAX]) {
    struct dirent **list = NULL;
    int nf, i;

    nf = scandir(dir, &list, filt, alphasort);
    if (nf < 0)
        return -1;

    if (nf == 0) {
        free(list);
        return -1;
    }
    /* nf > 0 */
    snprintf(out, SCHE_PATH_MAX, "%s/%s", dir, list[0]->d_name);

    for (i = 0; i < nf; i++)
        free(list[i]);
    free(list);
    return 0;
}

int sche_fs_find_outputs(const char *dir, char blur[SCHE_PATH_MAX],
                         char edges[SCHE_PATH_MAX]) {

    blur[0] = '\0';
    edges[0] = '\0';
    if (pick_first(dir, filt_blurred, blur) != 0)
        return -1;
    if (pick_first(dir, filt_edges, edges) != 0)
        return -1;
    return 0;
}
