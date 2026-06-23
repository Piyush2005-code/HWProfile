# Hardware Profiling Engine
# Makefile
#
# Targets:
#   make            — compile the profiler binary
#   make run        — compile + run all benchmark phases
#   make run ARG=A  — compile + run only Phase A
#   make plots      — generate all plots (requires Python + matplotlib)
#   make clean      — remove compiled objects and binary
#   make distclean  — also remove results/

# ── Compiler selection ────────────────────────────────────────────────────
#
# Prefer GCC for best -march=native auto-vectorization on ARM.
# Fall back to clang (default on macOS).
#
CC := $(shell command -v gcc 2>/dev/null || echo clang)

# ── Compiler flags ────────────────────────────────────────────────────────
#
# -O3            : Full optimization (enables auto-vectorization)
# -march=native  : Use all CPU features of the build machine.
#                  This activates NEON on AArch64, AVX2 on compatible x86, etc.
#                  The hw_detect.c code reads the resulting __ARM_NEON /
#                  __AVX2__ macros at compile time to know what SIMD is
#                  available.
# -std=c11       : Required for C11 atomics (stdatomic.h) in bench_sync.c
# -Wall -Wextra  : Enable warnings; treat your code as a learning resource.
# -Iinclude      : Find our headers in include/

CFLAGS  := -O3 -march=native -std=c11 -D_GNU_SOURCE -Wall -Wextra -Iinclude

# ── Linker flags ──────────────────────────────────────────────────────────
#
# -lpthread : pthreads (thread scaling, synchronization benchmarks)
# -lm       : math library (sqrt, fabs, etc. used in some kernels)

LDFLAGS := -lpthread -lm

# ── Source files ──────────────────────────────────────────────────────────

SRCS := src/main.c            \
        src/hw_detect.c       \
        src/profiler.c        \
        src/csv_writer.c      \
        src/bench_thread_scaling.c \
        src/bench_memory.c    \
        src/bench_compute.c   \
        src/bench_sync.c

OBJS    := $(SRCS:.c=.o)
TARGET  := profiler

# ── Rules ─────────────────────────────────────────────────────────────────

.PHONY: all run plots clean distclean help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo ""
	@echo "  Build successful: ./$(TARGET)"
	@echo "  Run all phases:   make run"
	@echo "  Run single phase: make run ARG=A"
	@echo ""

# Pattern rule: compile each .c to a .o
src/%.o: src/%.c
	$(CC) $(CFLAGS) -c -o $@ $<

# ── Run ───────────────────────────────────────────────────────────────────

run: $(TARGET)
	@mkdir -p results logs
	./$(TARGET) $(ARG) 2>&1 | tee logs/profiler_run.log

# ── Plots ─────────────────────────────────────────────────────────────────
#
# Requires: pip install matplotlib pandas numpy
# Each script reads its CSV and saves plots to results/

plots:
	@echo "Generating plots..."
	python3 scripts/plot_thread_scaling.py
	python3 scripts/plot_memory_hierarchy.py
	python3 scripts/plot_compute_throughput.py
	python3 scripts/plot_sync_overhead.py
	@echo "Plots saved in results/"

# ── Cleanup ───────────────────────────────────────────────────────────────

clean:
	rm -f $(OBJS) $(TARGET)

distclean: clean
	rm -rf results/

# ── Help ──────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  make           — build the profiler"
	@echo "  make run       — build + run all 4 phases"
	@echo "  make run ARG=A — run only Phase A (thread scaling)"
	@echo "  make run ARG=B — run only Phase B (memory)"
	@echo "  make run ARG=C — run only Phase C (compute / ILP)"
	@echo "  make run ARG=D — run only Phase D (synchronization)"
	@echo "  make plots     — generate PNG plots from CSV results"
	@echo "  make clean     — remove object files and binary"
	@echo "  make distclean — also remove results/"
	@echo ""
