/*
 * bench_compute.c — Phase C: Compute Throughput and ILP Sweep
 *
 * Answers: "How many sequential operations should I chain in a CV/ML kernel
 *           to maximize CPU throughput?"
 *
 * ── The ILP Sweep (most important measurement) ───────────────────────────
 *
 *   Modern CPUs are superscalar: they can execute multiple independent
 *   instructions per clock cycle.  But if each instruction depends on the
 *   RESULT of the previous one, the pipeline must stall until that result
 *   is ready — this is a "dependency chain" or "latency-bound" loop.
 *
 *   We measure throughput as a function of chain depth:
 *
 *     chain_depth = 1:  every FMA reads the previous result.
 *                       Throughput ≤ 1 / FMA_LATENCY_CYCLES (stall-bound).
 *
 *     chain_depth ≥ FMA_LATENCY:  enough independent accumulators to keep
 *                       every execution unit busy.
 *                       Throughput approaches PEAK_THROUGHPUT.
 *
 *   The plateau in the throughput-vs-depth curve is the true peak.
 *   The depth at which it plateaus tells you how many independent
 *   accumulators your loop needs (loop unrolling factor for CV kernels).
 *
 * ── SIMD variants ────────────────────────────────────────────────────────
 *
 *   If the CPU supports SIMD (detected at compile time via -march=native),
 *   we also run a vectorised FMA kernel.  The ratio
 *       SIMD_throughput / scalar_throughput
 *   tells you how much your CV kernel benefits from explicit vectorisation.
 *
 *   Supported at compile time:
 *     __ARM_NEON    → float32x4_t  (AArch64 / Apple Silicon / Jetson)
 *     __AVX2__      → __m256       (x86-64 with AVX2)
 *     __SSE2__      → __m128       (x86-64 baseline)
 *     (scalar only if none of the above)
 */

#include "benchmarks.h"
#include "profiler.h"
#include "csv_writer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Pull in SIMD headers based on what -march=native activated */
#if defined(__ARM_NEON)
#   include <arm_neon.h>
#elif defined(__AVX2__)
#   include <immintrin.h>
#elif defined(__SSE2__)
#   include <xmmintrin.h>   /* SSE */
#   include <emmintrin.h>   /* SSE2 */
#endif

/* Global sink — prevents compiler dead-code elimination */
volatile double g_compute_sink = 0.0;

/* ═══════════════════════════════════════════════════════════════════════════
 * Scalar FMA (double-precision) — varying chain depth
 *
 * chain_depth: number of independent accumulator chains.
 * iters:       total FMA operations to perform.
 *
 * We implement depths 1, 2, 4, 8, 16 via separate unrolled functions.
 * Using separate functions avoids the overhead of a runtime loop counter
 * and lets the compiler see the exact dependency structure.
 * ═════════════════════════════════════════════════════════════════════════ */

#define MUL 1.000000001
#define ADD 0.000000001

/* 1 chain: maximum latency exposure */
static void scalar_f64_chain1(long long iters) {
    double a0 = 1.0;
    for (long long i = 0; i < iters; i++)
        a0 = a0 * MUL + ADD;
    g_compute_sink = a0;
}

/* 2 independent chains */
static void scalar_f64_chain2(long long iters) {
    double a0=1.0, a1=2.0;
    for (long long i = 0; i < iters/2; i++) {
        a0 = a0*MUL + ADD;
        a1 = a1*MUL + ADD;
    }
    g_compute_sink = a0 + a1;
}

/* 4 independent chains */
static void scalar_f64_chain4(long long iters) {
    double a0=1.0, a1=2.0, a2=3.0, a3=4.0;
    for (long long i = 0; i < iters/4; i++) {
        a0 = a0*MUL + ADD;
        a1 = a1*MUL + ADD;
        a2 = a2*MUL + ADD;
        a3 = a3*MUL + ADD;
    }
    g_compute_sink = a0+a1+a2+a3;
}

/* 8 independent chains */
static void scalar_f64_chain8(long long iters) {
    double a0=1.0, a1=2.0, a2=3.0, a3=4.0;
    double a4=5.0, a5=6.0, a6=7.0, a7=8.0;
    for (long long i = 0; i < iters/8; i++) {
        a0=a0*MUL+ADD; a1=a1*MUL+ADD;
        a2=a2*MUL+ADD; a3=a3*MUL+ADD;
        a4=a4*MUL+ADD; a5=a5*MUL+ADD;
        a6=a6*MUL+ADD; a7=a7*MUL+ADD;
    }
    g_compute_sink = a0+a1+a2+a3+a4+a5+a6+a7;
}

/* 16 independent chains */
static void scalar_f64_chain16(long long iters) {
    double a0=1, a1=2, a2=3, a3=4, a4=5, a5=6, a6=7, a7=8;
    double b0=9, b1=10,b2=11,b3=12,b4=13,b5=14,b6=15,b7=16;
    for (long long i = 0; i < iters/16; i++) {
        a0=a0*MUL+ADD; a1=a1*MUL+ADD; a2=a2*MUL+ADD; a3=a3*MUL+ADD;
        a4=a4*MUL+ADD; a5=a5*MUL+ADD; a6=a6*MUL+ADD; a7=a7*MUL+ADD;
        b0=b0*MUL+ADD; b1=b1*MUL+ADD; b2=b2*MUL+ADD; b3=b3*MUL+ADD;
        b4=b4*MUL+ADD; b5=b5*MUL+ADD; b6=b6*MUL+ADD; b7=b7*MUL+ADD;
    }
    g_compute_sink = a0+a1+a2+a3+a4+a5+a6+a7+b0+b1+b2+b3+b4+b5+b6+b7;
}

/* ─── Single-precision scalar FMA (f32) ────────────────────────────────── */
#define MULf 1.000001f
#define ADDf 0.000001f

static void scalar_f32_chain1(long long iters) {
    float a0 = 1.0f;
    for (long long i = 0; i < iters; i++)
        a0 = a0 * MULf + ADDf;
    g_compute_sink = (double)a0;
}

static void scalar_f32_chain8(long long iters) {
    float a0=1,a1=2,a2=3,a3=4,a4=5,a5=6,a6=7,a7=8;
    for (long long i = 0; i < iters/8; i++) {
        a0=a0*MULf+ADDf; a1=a1*MULf+ADDf;
        a2=a2*MULf+ADDf; a3=a3*MULf+ADDf;
        a4=a4*MULf+ADDf; a5=a5*MULf+ADDf;
        a6=a6*MULf+ADDf; a7=a7*MULf+ADDf;
    }
    g_compute_sink = (double)(a0+a1+a2+a3+a4+a5+a6+a7);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * SIMD FMA kernels (compiled only when the ISA extension is available)
 * ═════════════════════════════════════════════════════════════════════════ */

#if defined(__ARM_NEON)
/*
 * NEON: float32x4_t holds 4 × f32 in a 128-bit register.
 * vmlaq_f32(a, b, c) = a + b*c  (fused on hardware)
 * We use 4 independent vector accumulators = 16 independent f32 chains.
 */
static void simd_f32_neon_chain4(long long iters) {
    float32x4_t a0 = {1,2,3,4};
    float32x4_t a1 = {5,6,7,8};
    float32x4_t a2 = {9,10,11,12};
    float32x4_t a3 = {13,14,15,16};
    float32x4_t m  = vdupq_n_f32(MULf);
    float32x4_t c  = vdupq_n_f32(ADDf);

    for (long long i = 0; i < iters/4; i++) {
        a0 = vmlaq_f32(c, a0, m);
        a1 = vmlaq_f32(c, a1, m);
        a2 = vmlaq_f32(c, a2, m);
        a3 = vmlaq_f32(c, a3, m);
    }
    float32x4_t sum = vaddq_f32(vaddq_f32(a0,a1), vaddq_f32(a2,a3));
    g_compute_sink = (double)vaddvq_f32(sum);
}
#endif /* __ARM_NEON */

#if defined(__AVX2__)
/*
 * AVX2: __m256 holds 8 × f32 in a 256-bit register.
 * _mm256_fmadd_ps(a, b, c) = a*b + c
 */
static void simd_f32_avx2_chain4(long long iters) {
    __m256 a0 = _mm256_set1_ps(1.0f);
    __m256 a1 = _mm256_set1_ps(2.0f);
    __m256 a2 = _mm256_set1_ps(3.0f);
    __m256 a3 = _mm256_set1_ps(4.0f);
    __m256 m  = _mm256_set1_ps(MULf);
    __m256 c  = _mm256_set1_ps(ADDf);

    for (long long i = 0; i < iters/4; i++) {
        a0 = _mm256_fmadd_ps(a0, m, c);
        a1 = _mm256_fmadd_ps(a1, m, c);
        a2 = _mm256_fmadd_ps(a2, m, c);
        a3 = _mm256_fmadd_ps(a3, m, c);
    }
    __m256 s = _mm256_add_ps(_mm256_add_ps(a0,a1), _mm256_add_ps(a2,a3));
    float tmp[8]; _mm256_storeu_ps(tmp, s);
    double acc = 0;
    for (int k=0;k<8;k++) acc += tmp[k];
    g_compute_sink = acc;
}
#elif defined(__SSE2__)
/*
 * SSE2 fallback: __m128 holds 4 × f32.
 * No FMA in SSE2, so we use separate mul + add.
 */
static void simd_f32_sse2_chain4(long long iters) {
    __m128 a0 = _mm_set1_ps(1.0f);
    __m128 a1 = _mm_set1_ps(2.0f);
    __m128 a2 = _mm_set1_ps(3.0f);
    __m128 a3 = _mm_set1_ps(4.0f);
    __m128 m  = _mm_set1_ps(MULf);
    __m128 c  = _mm_set1_ps(ADDf);

    for (long long i = 0; i < iters/4; i++) {
        a0 = _mm_add_ps(_mm_mul_ps(a0,m), c);
        a1 = _mm_add_ps(_mm_mul_ps(a1,m), c);
        a2 = _mm_add_ps(_mm_mul_ps(a2,m), c);
        a3 = _mm_add_ps(_mm_mul_ps(a3,m), c);
    }
    __m128 s = _mm_add_ps(_mm_add_ps(a0,a1), _mm_add_ps(a2,a3));
    float tmp[4]; _mm_storeu_ps(tmp, s);
    g_compute_sink = (double)(tmp[0]+tmp[1]+tmp[2]+tmp[3]);
}
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 * Calibration shim and measurement helper
 * ═════════════════════════════════════════════════════════════════════════ */
typedef void (*kernel_fn)(long long iters);

static void shim(void *arg, long long iters) {
    ((kernel_fn)arg)(iters);
}

/*
 * Run kernel, calibrate for ~target_ms, then do 3 timed trials.
 * Returns GFLOP/s (flops_per_iter FLOPs per iteration).
 */
static double measure_kernel(kernel_fn fn, int flops_per_iter,
                              double target_ms) {
    long long iters = calibrate(shim, (void *)fn, target_ms);

    double r[3];
    for (int k = 0; k < 3; k++) {
        hr_time_t t;
        timer_start(&t);
        fn(iters);
        double s = timer_elapsed_s(&t);
        r[k] = ((double)iters * (double)flops_per_iter) / s / 1e9;
    }
    return median3(r[0], r[1], r[2]);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Public entry point
 * ═════════════════════════════════════════════════════════════════════════ */
void bench_compute(const HWProfile *hw, const char *csv_path) {
    printf("=== [Phase C] Compute Throughput + ILP Sweep ===\n");
    (void)hw;   /* hw available for future extensions (e.g. SMT detection) */

    CsvWriter w = csv_open(csv_path,
        "kernel,precision,chain_depth,simd_lanes,gflops_per_core");

    /* ── Scalar f64: ILP sweep ─────────────────────────────────────────── */
    printf("  Scalar f64 (double) ILP sweep...\n");
    struct { int depth; kernel_fn fn; } f64_kernels[] = {
        { 1,  scalar_f64_chain1  },
        { 2,  scalar_f64_chain2  },
        { 4,  scalar_f64_chain4  },
        { 8,  scalar_f64_chain8  },
        { 16, scalar_f64_chain16 },
    };
    int nf64 = (int)(sizeof(f64_kernels)/sizeof(f64_kernels[0]));
    for (int i = 0; i < nf64; i++) {
        double g = measure_kernel(f64_kernels[i].fn, 2, 400.0);
        printf("  f64 chain %2d: %.3f GFLOP/s\n", f64_kernels[i].depth, g);
        csv_write_row(&w, "scalar_fma,f64,%d,1,%.6f",
                      f64_kernels[i].depth, g);
    }

    /* ── Scalar f32: ILP sweep ─────────────────────────────────────────── */
    printf("  Scalar f32 (float) ILP sweep...\n");
    struct { int depth; kernel_fn fn; } f32_kernels[] = {
        { 1, scalar_f32_chain1 },
        { 8, scalar_f32_chain8 },
    };
    int nf32 = (int)(sizeof(f32_kernels)/sizeof(f32_kernels[0]));
    for (int i = 0; i < nf32; i++) {
        double g = measure_kernel(f32_kernels[i].fn, 2, 400.0);
        printf("  f32 chain %2d: %.3f GFLOP/s\n", f32_kernels[i].depth, g);
        csv_write_row(&w, "scalar_fma,f32,%d,1,%.6f",
                      f32_kernels[i].depth, g);
    }

    /* ── SIMD kernels (if available) ──────────────────────────────────── */
#if defined(__ARM_NEON)
    printf("  NEON f32x4 (4-chain)...\n");
    {
        double g = measure_kernel(simd_f32_neon_chain4, 2*4, 400.0);
        printf("  NEON f32x4 chain 4: %.3f GFLOP/s (%.1fx scalar)\n",
               g, g / measure_kernel(scalar_f32_chain8, 2, 400.0));
        csv_write_row(&w, "simd_fma,f32,4,4,%.6f", g);
    }
#elif defined(__AVX2__)
    printf("  AVX2 f32x8 (4-chain)...\n");
    {
        double g = measure_kernel(simd_f32_avx2_chain4, 2*8, 400.0);
        printf("  AVX2 f32x8 chain 4: %.3f GFLOP/s\n", g);
        csv_write_row(&w, "simd_fma,f32,4,8,%.6f", g);
    }
#elif defined(__SSE2__)
    printf("  SSE2 f32x4 (4-chain, no FMA)...\n");
    {
        double g = measure_kernel(simd_f32_sse2_chain4, 2*4, 400.0);
        printf("  SSE2 f32x4 chain 4: %.3f GFLOP/s\n", g);
        csv_write_row(&w, "simd_mul_add,f32,4,4,%.6f", g);
    }
#else
    printf("  No SIMD detected (scalar only).\n");
#endif

    csv_close(&w);
    printf("[Phase C] Done.\n\n");
}
