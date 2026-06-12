/*
 * csv_writer.h — Minimal CSV Output Utility
 *
 * Wraps a FILE* with a header-on-open, row-write, and close interface.
 * Each row is flushed immediately so partial results are visible even
 * if the program is interrupted.
 *
 * Usage:
 *   CsvWriter w = csv_open("results/foo.csv", "col_a,col_b,col_c");
 *   csv_write_row(&w, "%d,%.4f,%.2f", n, val1, val2);
 *   csv_close(&w);
 */

#ifndef CSV_WRITER_H
#define CSV_WRITER_H

#include <stdio.h>

typedef struct {
    FILE *fp;           /* Underlying file pointer  */
    char  path[256];    /* Path (kept for logging)  */
} CsvWriter;

/*
 * Open (or create) the CSV file at path, write the header line, and
 * return a CsvWriter. Exits the process on failure.
 */
CsvWriter csv_open(const char *path, const char *header);

/*
 * Write a formatted row to the CSV.
 * fmt is a printf-style format string. A newline is appended automatically.
 */
void csv_write_row(CsvWriter *w, const char *fmt, ...);

/*
 * Flush and close the underlying file.
 */
void csv_close(CsvWriter *w);

#endif /* CSV_WRITER_H */
