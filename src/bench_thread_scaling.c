/*
 * bench_thread_scaling.c — Phase A: CPU Thread Scaling
 *
 * Answers: "How many parallel threads should my CV/ML pipeline use?"
 *
 * Method:
 *   Run a compute-bound double-precision FMA kernel on N threads
 *   simultaneously (N = 1 .. num_cpus). Measure aggregate throughput
 *   (GFLOP/s) and efficiency (actual speedup / ideal speedup × 100%).
 *
 * Reading the results:
 *   - Efficiency 100% at N=1 by definition.
 *   - Efficiency stays near 100% while threads are fully independent.
 *   - It drops when threads compete for shared resources (cache, memory
 *     bus, power budget, or SMT slots).
 *   - The "saturation point" — first N where efficiency < ~90% — is the
 *     optimal thread count for this workload class.
 *
 * Why 8-chain FMA?
 *   Using 8 independent accumulator chains maximises instruction-level
 *   parallelism (ILP) within each thread, ensuring we saturate the FP
 *   units and measure true per-core peak rather than latency-bound numbers.
 *   (See bench_compute.c for the ILP sweep that explains this in detail.)
 */

#include "benchmarks.h"
#include "profiler.h"
#include "csv_writer.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Worker thread
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    int       thread_id;
    int       num_cpus;
    int       has_affinity;
    long long iters;        /* total FMA iterations to perform         */
    double    sink;         /* dummy result — prevents dead-code elim  */
} WorkerArg;

static void *fma_worker(void *varg) {
    WorkerArg *a = (WorkerArg *)varg;

    /* Pin this thread to a specific core (Linux only) */
    if (a->has_affinity)
        pin_thread_to_core(a->thread_id % a->num_cpus);

    /*
     * 8 independent double-precision accumulators.
     *
     * Why 8?  The FMA has a latency of 4–5 cycles on most modern CPUs but a
     * throughput of 1 per cycle (2 on some x86 with two FP units).  To hide
     * the latency we need at least LATENCY_CYCLES independent chains running
     * simultaneously.  8 is safe for all known architectures in 2024.
     */
    double a0=1.1, a1=1.2, a2=1.3, a3=1.4;
    double a4=1.5, a5=1.6, a6=1.7, a7=1.8;
    const double MUL = 1.000000001;
    const double ADD = 0.000000001;
    const long long n = a->iters / 8;   /* each unrolled body = 8 FMAs */

    for (long long i = 0; i < n; i++) {
        a0 = a0 * MUL + ADD;
        a1 = a1 * MUL + ADD;
        a2 = a2 * MUL + ADD;
        a3 = a3 * MUL + ADD;
        a4 = a4 * MUL + ADD;
        a5 = a5 * MUL + ADD;
        a6 = a6 * MUL + ADD;
        a7 = a7 * MUL + ADD;
    }
    /* Prevent compiler from removing the loop entirely */
    a->sink = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    return NULL;
}

/* ─────────────────────────────────────────────────────────────────────────
 * Calibration shim
 * ───────────────────────────────────────────────────────────────────────── */
static WorkerArg g_calib_arg;
static void calib_shim(void *arg, long long iters) {
    WorkerArg *a = (WorkerArg *)arg;
    a->iters       = iters;
    a->has_affinity = 0;
    fma_worker(a);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Run N threads, return aggregate GFLOP/s
 * ───────────────────────────────────────────────────────────────────────── */
static double run_n_threads(int n, const HWProfile *hw, long long iters) {
    pthread_t  *tids = malloc((size_t)n * sizeof(pthread_t));
    WorkerArg  *args = malloc((size_t)n * sizeof(WorkerArg));

    hr_time_t wall;
    timer_start(&wall);

    for (int i = 0; i < n; i++) {
        args[i].thread_id   = i;
        args[i].num_cpus    = hw->num_cpus;
        args[i].has_affinity= hw->has_affinity;
        args[i].iters       = iters;
        args[i].sink        = 0.0;
        pthread_create(&tids[i], NULL, fma_worker, &args[i]);
    }
    for (int i = 0; i < n; i++)
        pthread_join(tids[i], NULL);

    double elapsed_s = timer_elapsed_s(&wall);

    /* Consume sinks so the compiler cannot remove the threads */
    volatile double dummy = 0.0;
    for (int i = 0; i < n; i++) dummy += args[i].sink;
    (void)dummy;

    free(tids);
    free(args);

    /*
     * Total FLOP count:
     *   iters/8 loop-bodies per thread × 8 FMAs × 2 FLOPs/FMA × n threads
     */
    double flops = (double)(iters / 8) * 8.0 * 2.0 * (double)n;
    return flops / elapsed_s / 1e9;   /* GFLOP/s */
}

/* ─────────────────────────────────────────────────────────────────────────
 * Public entry point
 * ───────────────────────────────────────────────────────────────────────── */
void bench_thread_scaling(const HWProfile *hw, const char *csv_path) {
    printf("=== [Phase A] Thread Scaling ===\n");

    /*
     * Auto-calibrate: find an iteration count that makes each single-thread
     * measurement last roughly 600 ms — long enough for the OS to settle,
     * short enough for the full sweep to finish in a reasonable time.
     */
    memset(&g_calib_arg, 0, sizeof(g_calib_arg));
    long long iters = calibrate(calib_shim, &g_calib_arg, 600.0);
    printf("  calibration: %lld iterations per thread (~600 ms each)\n\n",
           iters);

    CsvWriter w = csv_open(csv_path,
                           "threads,throughput_gflops,speedup,efficiency_pct");

    double baseline = 0.0;  /* single-thread GFLOP/s */

    for (int n = 1; n <= hw->num_cpus; n++) {
        /*
         * Run 3 independent trials and take the median.
         * This guards against OS jitter on any single measurement.
         */
        double r[3];
        for (int k = 0; k < 3; k++)
            r[k] = run_n_threads(n, hw, iters);
        double gflops = median3(r[0], r[1], r[2]);

        if (n == 1) baseline = gflops;

        double speedup    = gflops / baseline;
        double efficiency = (speedup / (double)n) * 100.0;

        printf("  threads = %2d | %7.3f GFLOP/s | speedup %5.2fx"
               " | efficiency %5.1f%%\n",
               n, gflops, speedup, efficiency);

        csv_write_row(&w, "%d,%.6f,%.4f,%.2f",
                      n, gflops, speedup, efficiency);
    }

    csv_close(&w);
    printf("\n[Phase A] Done.\n\n");
}
