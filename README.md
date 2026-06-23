# Hardware Profiling Engine

A lightweight, hardware-agnostic CPU profiler written in pure C.
Detects hardware configuration at runtime, runs four focused benchmark
phases, and writes CSV results for Python-based visualization.

**Designed to answer two questions for CV/ML pipeline developers:**
1. *How many parallel threads should I use?*  → Phase A
2. *How many sequential ops can I chain before performance degrades?*  → Phase C

---

## Requirements

| Component | Minimum |
|-----------|---------|
| C compiler | GCC 7+ or Clang 5+ |
| C standard | C11 (for `stdatomic.h`) |
| Libraries | pthreads, libm |
| Python (plots) | 3.8+, with `matplotlib pandas numpy` |

Works on:
- **Linux** — x86-64, AArch64 (Jetson Nano, Raspberry Pi, etc.)
- **macOS** — Apple Silicon (M-series) and Intel

---

## Build and Run

```bash
# Compile
make

# Run all 4 phases
make run

# Run a single phase
make run ARG=A        # thread scaling only
make run ARG=B        # memory bandwidth + latency only
make run ARG=C        # compute throughput + ILP sweep only
make run ARG=D        # synchronization overhead only

# Generate plots (requires Python)
pip3 install matplotlib pandas numpy
make plots
```

Results (CSV files) are written to `results/`.
Plots (PNG files) are also written to `results/`.

---

## Output Files

| File | Phase | Description |
|------|-------|-------------|
| `results/hw_profile.csv` | 0 | Detected hardware summary |
| `results/thread_scaling.csv` | A | GFLOP/s + efficiency vs thread count |
| `results/memory_bandwidth.csv` | B | Read/write/copy/triad GB/s vs buffer size |
| `results/memory_latency.csv` | B | Pointer-chasing latency (ns) vs buffer size |
| `results/compute_throughput.csv` | C | GFLOP/s vs ILP chain depth + SIMD |
| `results/sync_overhead.csv` | D | Latency (ns) per synchronization primitive |

---

## Phase Descriptions

### Phase A — Thread Scaling
Runs a compute-bound double-precision FMA kernel with 1 to N threads
(N = detected logical CPU count). Measures:
- Aggregate throughput (GFLOP/s)
- Speedup relative to single-thread
- Efficiency (%) = actual speedup / ideal speedup × 100

**Reading the plot:** The thread count where efficiency first drops below
~90% is your optimal parallel worker count for compute-heavy preprocessing.

### Phase B — Memory Bandwidth + Latency
**B1 (Bandwidth):** Sequential read/write/copy/triad kernels at buffer
sizes from L1 → L2 → L3 → DRAM. Buffer sweep points are generated
automatically from detected cache sizes.

**B2 (Latency):** Pointer-chasing benchmark that prevents hardware prefetch.
Reveals the true memory access latency at each cache level and DRAM.

**Reading the plot:** The bandwidth drop between cache levels tells you the
cost of processing image tensors that don't fit in a given cache.

### Phase C — Compute Throughput + ILP Sweep
Measures peak FMA throughput for varying numbers of independent
accumulator chains (dependency chain depth 1, 2, 4, 8, 16).

**Reading the plot:**
- At chain depth 1: throughput is limited by FMA latency.
- At the plateau: throughput reaches the hardware peak.
- The depth at which it plateaus = the loop unrolling factor your kernel
  needs to hide FMA latency and maximize throughput.

Includes SIMD variants (NEON on ARM, AVX2 on x86) when available.

### Phase D — Synchronization Overhead
Measures latency (ns) of:
- Mutex lock/unlock (uncontended and contended)
- Atomic `fetch_add` (uncontended and contended)
- `pthread_barrier_wait`
- Condition variable producer→consumer hand-off (pipeline stage latency)

**Reading the plot:** The condvar hand-off latency is the irreducible cost
of passing a frame between two pipeline stages. Budget for this when
deciding how to partition your CV pipeline.

---

## How Hardware is Detected

| Property | Linux | macOS |
|----------|-------|-------|
| CPU count | `sysconf(_SC_NPROCESSORS_ONLN)` | same |
| L1d size | `/sys/devices/system/cpu/cpu0/cache/index*/` | `sysctlbyname("hw.l1dcachesize")` |
| L2/L3 size | sysfs (same) | `sysctlbyname("hw.l2/l3cachesize")` |
| Cache line | sysfs `coherency_line_size` | `sysctlbyname("hw.cachelinesize")` |
| Total RAM | `sysconf(_SC_PHYS_PAGES * _SC_PAGESIZE)` | same |
| SIMD width | compile-time macros (`-march=native`) | same |
| Arch/OS | `uname()` | same |

---

## Project Structure

```
hardware-profiling-engine/
├── Makefile
├── README.md
├── include/
│   ├── hw_detect.h          # HWProfile struct + discovery API
│   ├── profiler.h           # Timer, affinity, calibration
│   ├── csv_writer.h         # CSV output utility
│   └── benchmarks.h        # Forward declarations for all phases
├── src/
│   ├── main.c               # Entry point + phase orchestration
│   ├── hw_detect.c          # Runtime hardware detection
│   ├── profiler.c           # Timer, affinity, calibration impl
│   ├── csv_writer.c         # CSV writer impl
│   ├── bench_thread_scaling.c
│   ├── bench_memory.c
│   ├── bench_compute.c
│   └── bench_sync.c
├── scripts/
│   ├── plot_thread_scaling.py
│   ├── plot_memory_hierarchy.py
│   ├── plot_compute_throughput.py
│   └── plot_sync_overhead.py
└── results/                 # Auto-created by 'make run'
```
