#ifndef SCHE_COMPAT_H
#define SCHE_COMPAT_H

#include <stddef.h>

char *sche_strdup(const char *s);

/* Okunabilir tam dosya; basarisizda NULL. */
char *sche_read_file(const char *path, size_t *out_len);

#endif
