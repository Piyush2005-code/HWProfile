/*
 * main.c — ArchScope Lite Entry Point
 *
 * Orchestrates the four benchmark phases:
 *   Phase 0: Hardware discovery (always runs first)
 *   Phase A: CPU thread scaling
 *   Phase B: Memory bandwidth + latency
 *   Phase C: Compute throughput + ILP sweep
 *   Phase D: Synchronization overhead
 *
 * Usage:
 *   ./profiler          — run all phases
 *   ./profiler discover — run Phase 0 only (print hardware profile)
 *   ./profiler A        — run Phase A only (thread scaling)
 *   ./profiler B        — run Phase B only (memory)
 *   ./profiler C        — run Phase C only (compute)
 *   ./profiler D        — run Phase D only (sync)
 *
 * Results are written as CSV files into ./results/
 * Generate plots with: python3 scripts/plot_*.py
 */

#include "hw_detect.h"
#include "benchmarks.h"
#include "profiler.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* ─────────────────────────────────────────────────────────────────────────
 * Banner
 * ───────────────────────────────────────────────────────────────────────── */
static void print_banner(void) {
    printf("\n");
    printf("  ╔══════════════════════════════════════════════════════╗\n");
    printf("  ║          ArchScope Lite — Hardware Profiler          ║\n");
    printf("  ║   Detects hardware at runtime. No hardcoded values.  ║\n");
    printf("  ╚══════════════════════════════════════════════════════╝\n");
    printf("\n");
}

/* ─────────────────────────────────────────────────────────────────────────
 * main
 * ───────────────────────────────────────────────────────────────────────── */
int main(int argc, char **argv) {
    print_banner();

    /* ── Phase 0: Hardware discovery ─────────────────────────────────── */
    HWProfile hw;
    hw_detect(&hw);
    hw_print(&hw);

    ensure_dir("results");
    hw_save_csv(&hw, "results/hw_profile.csv");
    printf("\n");

    /* ── Parse optional phase selector ─────────────────────────────────── */
    /*
     * If no argument is given, all phases run.
     * If an argument like "A", "B", "C", "D", or "discover" is given,
     * only that phase runs.
     */
    int run_A = 1, run_B = 1, run_C = 1, run_D = 1;

    if (argc >= 2) {
        const char *sel = argv[1];
        if (strcmp(sel, "discover") == 0) {
            /* Already done above */
            printf("Hardware profile written to results/hw_profile.csv\n\n");
            return 0;
        }
        /* Selective run */
        run_A = (strchr(sel, 'A') != NULL || strchr(sel, 'a') != NULL);
        run_B = (strchr(sel, 'B') != NULL || strchr(sel, 'b') != NULL);
        run_C = (strchr(sel, 'C') != NULL || strchr(sel, 'c') != NULL);
        run_D = (strchr(sel, 'D') != NULL || strchr(sel, 'd') != NULL);
    }

    /* ── Phase A: Thread scaling ─────────────────────────────────────── */
    if (run_A)
        bench_thread_scaling(&hw, "results/thread_scaling.csv");

    /* ── Phase B: Memory bandwidth + latency ─────────────────────────── */
    if (run_B)
        bench_memory(&hw,
                     "results/memory_bandwidth.csv",
                     "results/memory_latency.csv");

    /* ── Phase C: Compute throughput + ILP sweep ─────────────────────── */
    if (run_C)
        bench_compute(&hw, "results/compute_throughput.csv");

    /* ── Phase D: Synchronization overhead ───────────────────────────── */
    if (run_D)
        bench_sync(&hw, "results/sync_overhead.csv");

    /* ── Summary ─────────────────────────────────────────────────────── */
    printf("══════════════════════════════════════════════════════════\n");
    printf("  All selected phases complete.\n");
    printf("  CSV results are in:  results/\n");
    printf("  Generate plots with: make plots\n");
    printf("                   or: python3 scripts/plot_<phase>.py\n");
    printf("══════════════════════════════════════════════════════════\n\n");

    return 0;
}
