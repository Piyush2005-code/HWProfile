/*
 * profiler.c — Timer, Affinity, and Calibration Implementations
 */

#include "profiler.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * Thread CPU pinning
 * ───────────────────────────────────────────────────────────────────────── */
int pin_thread_to_core(int core_id) {
#ifdef __linux__
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET((unsigned)core_id, &cs);
    int rc = pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs);
    if (rc != 0)
        fprintf(stderr, "  [warn] pin_thread_to_core(%d): %s\n",
                core_id, strerror(rc));
    return rc;
#else
    /* macOS and others: silently no-op */
    (void)core_id;
    return 0;
#endif
}

/* ─────────────────────────────────────────────────────────────────────────
 * Directory creation
 * ───────────────────────────────────────────────────────────────────────── */
void ensure_dir(const char *path) {
    if (mkdir(path, 0755) != 0 && errno != EEXIST)
        perror(path);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Statistics
 * ───────────────────────────────────────────────────────────────────────── */
double median3(double a, double b, double c) {
    /* Branchless median of three */
    if ((a <= b && b <= c) || (c <= b && b <= a)) return b;
    if ((b <= a && a <= c) || (c <= a && a <= b)) return a;
    return c;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Calibration
 *
 * Strategy:
 *   1. Run fn(arg, PROBE) and measure elapsed time.
 *   2. Extrapolate: iters = PROBE * target_ms / elapsed_ms.
 *   3. Clamp to a minimum of PROBE so we always run at least one meaningful
 *      batch, even on very fast machines.
 * ───────────────────────────────────────────────────────────────────────── */
long long calibrate(bench_fn_t fn, void *arg, double target_ms) {
    const long long PROBE = 10000LL;

    hr_time_t t;
    timer_start(&t);
    fn(arg, PROBE);
    double elapsed_ms = timer_elapsed_ms(&t);

    /* Guard against zero elapsed (faster than timer resolution) */
    if (elapsed_ms < 0.001) elapsed_ms = 0.001;

    long long iters = (long long)((double)PROBE * target_ms / elapsed_ms);
    if (iters < PROBE) iters = PROBE;

    return iters;
}
