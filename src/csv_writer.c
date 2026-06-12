/*
 * csv_writer.c — Minimal CSV Output Utility
 */

#include "csv_writer.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CsvWriter csv_open(const char *path, const char *header) {
    CsvWriter w;
    memset(&w, 0, sizeof(w));
    strncpy(w.path, path, sizeof(w.path) - 1);

    w.fp = fopen(path, "w");
    if (!w.fp) {
        perror(path);
        exit(EXIT_FAILURE);
    }

    /* Write header line */
    fprintf(w.fp, "%s\n", header);
    fflush(w.fp);

    printf("  [csv] Writing %s\n", path);
    return w;
}

void csv_write_row(CsvWriter *w, const char *fmt, ...) {
    if (!w || !w->fp) return;

    va_list ap;
    va_start(ap, fmt);
    vfprintf(w->fp, fmt, ap);
    va_end(ap);

    fputc('\n', w->fp);

    /*
     * Flush after every row. Slightly slower, but ensures partial results
     * are readable if the program is killed mid-benchmark (e.g., on
     * power-constrained devices like Jetson Nano under thermal throttle).
     */
    fflush(w->fp);
}

void csv_close(CsvWriter *w) {
    if (w && w->fp) {
        fclose(w->fp);
        w->fp = NULL;
        printf("  [csv] Closed %s\n", w->path);
    }
}
