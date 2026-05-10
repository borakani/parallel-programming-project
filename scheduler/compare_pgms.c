#include "sche_pgm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {

    SchePgmCompare c;
    int thr = 3;
    int ai = 1;

    while (ai < argc && argv[ai][0] == '-') {

        if (strcmp(argv[ai], "-t") == 0 || strcmp(argv[ai], "--threshold") == 0) {

            if (ai + 1 >= argc) {
                fprintf(stderr, "-t yaninda sayi gerekiyor\n");
                return 2;

            }

            thr = atoi(argv[ai + 1]);

            ai += 2;

            continue;

        }

        fprintf(stderr, "Bilinmeyen secenek: %s\n", argv[ai]);
        return 2;

    }

    if (ai + 2 != argc) {
        fprintf(stderr, "Usage: compare_pgms [-t eksik_pixel] dosya_a.pgm dosya_b.pgm\n");
        return 2;

    }

    c = sche_pgm_compare_files(argv[ai], argv[ai + 1], thr);


    printf("A:               %s\n", argv[ai]);



    printf("B:               %s\n", argv[ai + 1]);



    printf("Max |diff|:      %d\n", c.max_abs_diff);



    printf("Mean |diff|:     %.6f\n", c.mean_abs_diff);



    printf("RMSE:            %.6f\n", c.rmse);



    printf("> esik piksel:   %ld (esik=%d)\n",
           (long)c.differing_pixels, thr);

    printf("Sonuc:           %s\n", c.ok ? "OK" : "FAIL");


    return c.ok ? 0 : 1;

}
