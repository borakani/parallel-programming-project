#ifndef SCHE_FSUTIL_H
#define SCHE_FSUTIL_H

#define SCHE_PATH_MAX 1024

/* dir icinde sıralı ilk blurry/edges PGM yolları yazilir; yoksa -1 */
int sche_fs_find_outputs(const char *dir, char blur[SCHE_PATH_MAX],
                         char edges[SCHE_PATH_MAX]);

#endif
