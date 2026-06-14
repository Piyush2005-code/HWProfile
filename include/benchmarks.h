/*
 * benchmarks.h — Public API for All Benchmark Phases
 *
 * Each function runs one profiling phase and writes results to CSV file(s).
 * All parameters are derived from the HWProfile — nothing is hardcoded.
 */

#ifndef BENCHMARKS_H
#define BENCHMARKS_H

#include "hw_detect.h"

/*
 * Phase A — CPU Thread Scaling
 *
 * Runs a compute-bound FMA kernel with 1..num_cpus threads.
 * Measures throughput (GFLOP/s) and efficiency (%) per thread count.
 *
 * Key question answered: how many parallel workers should a CV/ML
 * preprocessing pipeline use?
 *
 * Output: csv_path  (e.g., "results/thread_scaling.csv")
 */
void bench_thread_scaling(const HWProfile *hw, const char *csv_path);

/*
 * Phase B — Memory Bandwidth and Latency
 *
 * B1: Sequential read/write/copy/triad bandwidth swept across buffer
 *     sizes from L1 → L2 → L3 → DRAM (derived from detected cache sizes).
 *
 * B2: Random-access latency via pointer chasing, revealing ns/access
 *     at each level of the memory hierarchy.
 *
 * Key question answered: what is the ceiling throughput for copying
 * image tensors, and what does random access cost at each cache level?
 *
 * Output: bw_csv  (e.g., "results/memory_bandwidth.csv")
 *         lat_csv (e.g., "results/memory_latency.csv")
 */
void bench_memory(const HWProfile *hw,
                  const char *bw_csv,
                  const char *lat_csv);

/*
 * Phase C — Compute Throughput and ILP Sweep
 *
 * Measures peak floating-point and integer throughput (GFLOP/s) on a
 * single core, varying the number of independent dependency chains (ILP).
 *
 * The throughput vs. chain-depth curve reveals the CPU's instruction-level
 * parallelism ceiling — i.e., how many sequential ops you can chain before
 * you are limited by latency rather than throughput.
 *
 * Includes SIMD variants if the CPU supports them (detected at compile time).
 *
 * Key question answered: how many sequential FMA ops per chain maximizes
 * throughput in a CV/ML kernel?
 *
 * Output: csv_path  (e.g., "results/compute_throughput.csv")
 */
void bench_compute(const HWProfile *hw, const char *csv_path);

/*
 * Phase D — Synchronization Overhead
 *
 * Measures the latency (ns/operation) of:
 *   - pthread mutex lock/unlock (uncontended + contended at N threads)
 *   - pthread barrier (N threads)
 *   - C11 atomic fetch_add (uncontended + N-thread contention)
 *   - Condition variable producer→consumer hand-off (pipeline stage cost)
 *
 * Key question answered: what is the real cost of connecting parallel
 * pipeline stages with a mutex or condition variable?
 *
 * Output: csv_path  (e.g., "results/sync_overhead.csv")
 */
void bench_sync(const HWProfile *hw, const char *csv_path);

#endif /* BENCHMARKS_H */
