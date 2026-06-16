/*
 * bench_memory.c — Phase B: Memory Bandwidth and Latency
 *
 * Answers:
 *   "What is the throughput ceiling for reading/writing image tensors?"
 *   "What is the cost of a scattered (random) memory access at each
 *    cache level?"
 *
 * ── B1: Sequential Bandwidth Sweep ──────────────────────────────────────
 *
 *   Measures read / write / copy / triad bandwidth (GB/s) at buffer sizes
 *   that span L1 → L2 → L3 → DRAM.  The sweep points are derived at
 *   runtime from detected cache sizes — nothing is hardcoded.
 *
 *   The classic STREAM triad: C[i] = A[i] + s * B[i]
 *   This is the most demanding kernel: 2 reads + 1 write per element.
 *   It is the closest model to an image blend or linear combination op.
 *
 * ── B2: Pointer-Chasing Latency ──────────────────────────────────────────
 *
 *   Builds a buffer filled with a random circular permutation of
 *   cache-line-aligned pointers, then traverses the chain.
 *
 *   Because each memory access depends on the value just read (the
 *   "next pointer"), the CPU cannot prefetch ahead.  The measured
 *   nanoseconds per hop directly reflects access latency at each
 *   memory level.
 *
 *   Reading the results:
 *     Small buffer (fits in L1): ~2–5 ns
 *     Medium buffer (fits in L2): ~10–20 ns
 *     Large buffer (DRAM): ~60–120 ns (higher on power-constrained SoCs)
 */

#include "benchmarks.h"
#include "profiler.h"
#include "csv_writer.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdint.h>

/*
 * Global sink — prevents the compiler from eliminating measurement loops.
 * We store accumulated results here so they are "visible" to the linker.
 */
volatile double g_bw_sink = 0.0;

/* ═══════════════════════════════════════════════════════════════════════════
 * B1: Sequential Bandwidth
 * ═════════════════════════════════════════════════════════════════════════ */

/*
 * Generate the list of buffer sizes to sweep.
 *
 * For each detected cache level we test: size/4, size/2, size (the boundary).
 * For DRAM we test: 4×, 8×, 16× the largest detected cache, capped at RAM/8.
 *
 * Returns the number of sizes written into *sizes (at most max_n).
 * Sizes are sorted ascending.
 */
static int make_sweep_sizes(const HWProfile *hw,
                             size_t *sizes, int max_n) {
    int ns = 0;

    size_t cache_levels[3] = { hw->l1d_size, hw->l2_size, hw->l3_size };

    /* Per-level: quarter, half, full */
    for (int ci = 0; ci < 3; ci++) {
        size_t cs = cache_levels[ci];
        if (cs == 0) continue;
        size_t fracs[3] = { cs / 4, cs / 2, cs };
        for (int f = 0; f < 3; f++) {
            size_t sz = fracs[f];
            if (sz < 4096) continue;            /* skip tiny blocks */
            /* Round to 4 KB for alignment */
            sz = (sz / 4096) * 4096;
            /* Dedup */
            int dup = 0;
            for (int j = 0; j < ns; j++) if (sizes[j] == sz) { dup=1; break; }
            if (!dup && ns < max_n) sizes[ns++] = sz;
        }
    }

    /* DRAM region: 4× to 32× largest cache */
    size_t largest = 0;
    for (int ci = 0; ci < 3; ci++)
        if (cache_levels[ci] > largest) largest = cache_levels[ci];
    if (largest == 0) largest = 8 * 1024 * 1024;   /* 8 MB fallback */

    for (int mul = 4; mul <= 32; mul *= 2) {
        size_t sz = largest * (size_t)mul;
        /* Never exceed RAM/8 to avoid OOM and swap */
        if (hw->total_ram > 0 && sz > hw->total_ram / 8)
            sz = hw->total_ram / 8;
        sz = (sz / (1024*1024)) * (1024*1024);      /* align to 1 MB */
        if (sz < 4096) continue;
        int dup = 0;
        for (int j = 0; j < ns; j++) if (sizes[j] == sz) { dup=1; break; }
        if (!dup && ns < max_n) sizes[ns++] = sz;
    }

    /* Sort ascending (simple insertion sort — small array) */
    for (int i = 1; i < ns; i++) {
        size_t key = sizes[i];
        int j = i - 1;
        while (j >= 0 && sizes[j] > key) { sizes[j+1] = sizes[j]; j--; }
        sizes[j+1] = key;
    }

    return ns;
}

/*
 * Measure bandwidth (GB/s) for a given kernel on a pre-allocated buffer.
 * Repeats the kernel until at least MIN_MEASURE_MS have elapsed to get
 * a stable reading for small (cache-resident) buffers.
 */
#define MIN_MEASURE_MS 150.0

typedef enum { BW_READ, BW_WRITE, BW_COPY, BW_TRIAD } BwKernel;

static double measure_bandwidth(BwKernel ktype,
                                 double *A, double *B, double *C,
                                 size_t n_doubles) {
    double best = 0.0;

    /* Run several trials; keep best (highest = least OS noise) */
    for (int trial = 0; trial < 4; trial++) {
        long long reps = 0;
        hr_time_t t;
        timer_start(&t);

        do {
            switch (ktype) {
            case BW_READ: {
                double acc = 0.0;
                for (size_t i = 0; i < n_doubles; i++) acc += A[i];
                g_bw_sink = acc;
                break;
            }
            case BW_WRITE:
                for (size_t i = 0; i < n_doubles; i++) A[i] = 1.23;
                break;
            case BW_COPY:
                for (size_t i = 0; i < n_doubles; i++) C[i] = A[i];
                g_bw_sink = C[0];
                break;
            case BW_TRIAD:
                for (size_t i = 0; i < n_doubles; i++)
                    C[i] = A[i] + 3.0 * B[i];
                g_bw_sink = C[0];
                break;
            }
            reps++;
        } while (timer_elapsed_ms(&t) < MIN_MEASURE_MS);

        double elapsed_s = timer_elapsed_s(&t);

        /* Bytes transferred per kernel per element */
        double bytes_per_elem;
        switch (ktype) {
        case BW_READ:  bytes_per_elem = 1.0 * sizeof(double); break;
        case BW_WRITE: bytes_per_elem = 1.0 * sizeof(double); break;
        case BW_COPY:  bytes_per_elem = 2.0 * sizeof(double); break;
        case BW_TRIAD: bytes_per_elem = 3.0 * sizeof(double); break;
        default:       bytes_per_elem = 1.0 * sizeof(double); break;
        }

        double gbps = ((double)reps * (double)n_doubles * bytes_per_elem)
                      / elapsed_s / 1e9;
        if (gbps > best) best = gbps;
    }
    return best;
}

static void bench_bandwidth(const HWProfile *hw, const char *csv_path) {
    printf("  [B1] Sequential bandwidth sweep...\n");

    size_t sizes[64];
    int ns = make_sweep_sizes(hw, sizes, 64);

    CsvWriter w = csv_open(csv_path,
        "buffer_size_kb,read_gbps,write_gbps,copy_gbps,triad_gbps");

    /* Allocate three buffers at the maximum size we'll test */
    size_t max_bytes = sizes[ns - 1];
    size_t n_max     = max_bytes / sizeof(double);

    double *A = NULL, *B = NULL, *C = NULL;
    posix_memalign((void **)&A, 64, n_max * sizeof(double));
    posix_memalign((void **)&B, 64, n_max * sizeof(double));
    posix_memalign((void **)&C, 64, n_max * sizeof(double));

    /* Touch all pages (force physical allocation, avoid copy-on-write) */
    memset(A, 0, n_max * sizeof(double));
    memset(B, 1, n_max * sizeof(double));
    memset(C, 2, n_max * sizeof(double));

    for (int i = 0; i < ns; i++) {
        size_t n = sizes[i] / sizeof(double);
        size_t kb = sizes[i] / 1024;

        double rd = measure_bandwidth(BW_READ,  A, B, C, n);
        double wr = measure_bandwidth(BW_WRITE, A, B, C, n);
        double cp = measure_bandwidth(BW_COPY,  A, B, C, n);
        double tr = measure_bandwidth(BW_TRIAD, A, B, C, n);

        printf("  %8zu KB | R %6.2f | W %6.2f | Copy %6.2f | Triad %6.2f GB/s\n",
               kb, rd, wr, cp, tr);
        csv_write_row(&w, "%zu,%.4f,%.4f,%.4f,%.4f", kb, rd, wr, cp, tr);
    }

    free(A); free(B); free(C);
    csv_close(&w);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * B2: Pointer-Chasing Latency
 * ═════════════════════════════════════════════════════════════════════════ */

static void bench_latency(const HWProfile *hw, const char *csv_path) {
    printf("  [B2] Pointer-chasing latency sweep...\n");

    size_t sizes[64];
    int ns = make_sweep_sizes(hw, sizes, 64);

    CsvWriter w = csv_open(csv_path, "buffer_size_kb,latency_ns");

    const size_t cl    = hw->cache_line_bytes;
    const long long HOPS = 4 * 1024 * 1024LL;   /* 4M pointer hops per trial */

    for (int i = 0; i < ns; i++) {
        size_t buf_bytes = sizes[i];
        /* Align buffer size to cache line boundary */
        buf_bytes = (buf_bytes / cl) * cl;
        size_t n_lines = buf_bytes / cl;
        if (n_lines < 8) continue;

        /* Allocate buffer aligned to cache line */
        uint8_t *buf = NULL;
        posix_memalign((void **)&buf, cl, buf_bytes);
        memset(buf, 0, buf_bytes);

        /*
         * Build a random circular permutation of cache-line indices.
         *
         * Fisher-Yates shuffle: produces a uniformly random permutation.
         * We seed with buf_bytes so different sizes give different patterns.
         *
         * The chain: line[perm[0]] → line[perm[1]] → ... → line[perm[N-1]]
         *            → line[perm[0]] (wraps around)
         */
        size_t *perm = malloc(n_lines * sizeof(size_t));
        for (size_t j = 0; j < n_lines; j++) perm[j] = j;
        srand((unsigned int)buf_bytes ^ 0xDEADBEEFu);
        for (size_t j = n_lines - 1; j > 0; j--) {
            size_t k = (size_t)rand() % (j + 1);
            size_t tmp = perm[j]; perm[j] = perm[k]; perm[k] = tmp;
        }

        /* Write pointer chain: each cache line's first word is "next pointer" */
        for (size_t j = 0; j < n_lines - 1; j++)
            *(uintptr_t *)(buf + perm[j] * cl) =
                (uintptr_t)(buf + perm[j+1] * cl);
        *(uintptr_t *)(buf + perm[n_lines-1] * cl) =
            (uintptr_t)(buf + perm[0] * cl);

        /* Warm-up: traverse the full chain once to populate TLB + prefetchers */
        uintptr_t ptr = *(uintptr_t *)(buf + perm[0] * cl);
        for (size_t j = 0; j < n_lines; j++)
            ptr = *(uintptr_t *)ptr;
        g_bw_sink = (double)ptr;

        /* Timed chase */
        hr_time_t t;
        timer_start(&t);
        for (long long h = 0; h < HOPS; h++)
            ptr = *(uintptr_t *)ptr;
        double elapsed_s = timer_elapsed_s(&t);
        g_bw_sink = (double)ptr;

        double ns_per_hop = (elapsed_s * 1e9) / (double)HOPS;
        printf("  %8zu KB | latency = %6.1f ns\n", buf_bytes/1024, ns_per_hop);
        csv_write_row(&w, "%zu,%.2f", buf_bytes/1024, ns_per_hop);

        free(perm);
        free(buf);
    }

    csv_close(&w);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public entry point
 * ═════════════════════════════════════════════════════════════════════════ */
void bench_memory(const HWProfile *hw,
                  const char *bw_csv, const char *lat_csv) {
    printf("=== [Phase B] Memory Bandwidth and Latency ===\n");
    bench_bandwidth(hw, bw_csv);
    bench_latency(hw, lat_csv);
    printf("[Phase B] Done.\n\n");
}
