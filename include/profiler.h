/*
 * profiler.h — Timing, Affinity, and Utility Functions
 *
 * Provides:
 *   - High-resolution monotonic timer (CLOCK_MONOTONIC)
 *   - Thread-to-core pinning (Linux only; silently skipped on macOS)
 *   - File-system helpers
 *   - Statistics utilities
 *   - Auto-calibration: determine loop count for a target measurement time
 */

#ifndef PROFILER_H
#define PROFILER_H

#include <time.h>
#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────
 * High-resolution monotonic timer
 *
 * Usage:
 *   hr_time_t t;
 *   timer_start(&t);
 *   do_work();
 *   double ms = timer_elapsed_ms(&t);
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct timespec hr_time_t;

/*
 * Record the current time into *t.
 * Uses CLOCK_MONOTONIC — not affected by wall-clock adjustments.
 */
static inline void timer_start(hr_time_t *t) {
    clock_gettime(CLOCK_MONOTONIC, t);
}

/*
 * Return elapsed time in milliseconds since *start.
 */
static inline double timer_elapsed_ms(const hr_time_t *start) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec  - start->tv_sec ) * 1e3
         + (double)(now.tv_nsec - start->tv_nsec) * 1e-6;
}

/*
 * Return elapsed time in seconds since *start.
 */
static inline double timer_elapsed_s(const hr_time_t *start) {
    return timer_elapsed_ms(start) * 1e-3;
}

/* ─────────────────────────────────────────────────────────────────────────
 * CPU core pinning
 *
 * On Linux: calls pthread_setaffinity_np to bind the calling thread
 *           to the specified core_id.
 * On macOS: does nothing (returns 0). macOS does not expose affinity.
 *
 * Returns 0 on success (or on unsupported platform).
 * ───────────────────────────────────────────────────────────────────────── */
int pin_thread_to_core(int core_id);

/* ─────────────────────────────────────────────────────────────────────────
 * File system utilities
 * ───────────────────────────────────────────────────────────────────────── */

/*
 * Create directory at path if it does not already exist.
 * (Non-recursive; equivalent to mkdir -p for a single directory.)
 */
void ensure_dir(const char *path);

/* ─────────────────────────────────────────────────────────────────────────
 * Statistics
 * ───────────────────────────────────────────────────────────────────────── */

/*
 * Return the median of three doubles.
 * Using 3-sample median for benchmark stability avoids outliers from
 * OS preemption without the overhead of a full sort.
 */
double median3(double a, double b, double c);

/* ─────────────────────────────────────────────────────────────────────────
 * Auto-calibration
 *
 * Many benchmarks need a loop count that keeps the measurement window
 * long enough to be accurate (~500ms–1s) regardless of machine speed.
 *
 * calibrate() runs the benchmark for a short probe duration and extrapolates
 * how many iterations are needed to fill target_ms milliseconds.
 *
 * Signature of the probe function:
 *   void fn(void *arg, long long iters)
 *
 * Example:
 *   long long N = calibrate(my_kernel, &arg, 500.0);
 *   // Now run my_kernel(&arg, N) and time it.
 * ───────────────────────────────────────────────────────────────────────── */
typedef void (*bench_fn_t)(void *arg, long long iters);

long long calibrate(bench_fn_t fn, void *arg, double target_ms);

#endif /* PROFILER_H */
