#include "sche_compat.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char *sche_strdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *p = malloc(n);
    if (!p)
        return NULL;
    memcpy(p, s, n);
    return p;
}

char *sche_read_file(const char *path, size_t *out_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp)
        return NULL;
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return NULL;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return NULL;
    }
    rewind(fp);
    char *buf = malloc((size_t)sz + 1);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    size_t r = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);
    buf[r] = '\0';
    if (out_len)
        *out_len = r;
    return buf;
}
