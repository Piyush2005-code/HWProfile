/*
 * hw_detect.c — Runtime Hardware Discovery
 *
 * Detects:
 *   - Number of online logical CPUs      (POSIX sysconf)
 *   - Cache sizes L1d / L2 / L3         (sysfs on Linux, sysctlbyname on macOS)
 *   - Cache line size                    (same sources)
 *   - Total physical RAM                 (POSIX sysconf)
 *   - SIMD width                         (compile-time preprocessor macros)
 *   - Thread affinity support            (compile-time __linux__ guard)
 *   - Architecture / OS strings         (uname)
 */

#include "hw_detect.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>         /* sysconf, _SC_* */
#include <sys/utsname.h>    /* uname           */
#include <errno.h>

/* ═══════════════════════════════════════════════════════════════════════════
 * macOS-specific detection via sysctl(3)
 * ═════════════════════════════════════════════════════════════════════════ */
#ifdef __APPLE__
#include <sys/sysctl.h>

/*
 * Query a single size_t value from the sysctl MIB by name.
 * Returns 0 if the key is not found (e.g. hw.l3cachesize on M-series chips).
 */
static size_t macos_sysctl_size(const char *name) {
    size_t val = 0;
    size_t len = sizeof(val);
    sysctlbyname(name, &val, &len, NULL, 0);
    return val;
}
#endif /* __APPLE__ */

/* ═══════════════════════════════════════════════════════════════════════════
 * Linux-specific detection via /sys/devices/system/cpu/
 * ═════════════════════════════════════════════════════════════════════════ */
#ifdef __linux__

/*
 * sysfs cache size values look like "32K" or "512K" or "8192K".
 * Parse the integer and multiply by 1024 to get bytes.
 */
static size_t sysfs_read_size_kb(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;

    char buf[32] = {0};
    if (!fgets(buf, sizeof(buf), f)) { fclose(f); return 0; }
    fclose(f);

    /* strtoul stops at 'K', giving us the numeric part */
    unsigned long kb = strtoul(buf, NULL, 10);
    return (size_t)kb * 1024UL;
}

/*
 * Walk /sys/devices/system/cpu/cpu0/cache/index{0..7}/ looking for a
 * cache entry at the requested hierarchy level that is not instruction-only.
 *
 * Returns the cache size in bytes, or 0 if not found.
 */
static size_t linux_find_cache_size(int want_level) {
    char path[256];

    for (int idx = 0; idx <= 7; idx++) {
        /* ── Read level number ── */
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/level", idx);
        FILE *f = fopen(path, "r");
        if (!f) break;                  /* no more cache entries */
        int level = 0;
        fscanf(f, "%d", &level);
        fclose(f);
        if (level != want_level) continue;

        /* ── Skip instruction-only caches (e.g. index0 on some x86 CPUs) ── */
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/type", idx);
        f = fopen(path, "r");
        char type[32] = "Unified";
        if (f) { fscanf(f, "%31s", type); fclose(f); }
        if (strcmp(type, "Instruction") == 0) continue;

        /* ── Read size ── */
        snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu0/cache/index%d/size", idx);
        return sysfs_read_size_kb(path);
    }
    return 0;
}

/*
 * Read the cache line (coherency) size for index0 (L1d or L1).
 */
static size_t linux_cache_line_size(void) {
    const char *path =
        "/sys/devices/system/cpu/cpu0/cache/index0/coherency_line_size";
    FILE *f = fopen(path, "r");
    if (!f) return 64;          /* universal safe default */
    size_t val = 64;
    fscanf(f, "%zu", &val);
    fclose(f);
    return val;
}

#endif /* __linux__ */

/* ═══════════════════════════════════════════════════════════════════════════
 * Main API
 * ═════════════════════════════════════════════════════════════════════════ */

void hw_detect(HWProfile *p) {
    memset(p, 0, sizeof(*p));

    /* ── Logical CPU count ────────────────────────────────────────────── */
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    p->num_cpus = (n > 0) ? (int)n : 1;

    /* ── Physical RAM ─────────────────────────────────────────────────── */
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    p->total_ram = (pages > 0 && page_size > 0)
                 ? (size_t)pages * (size_t)page_size
                 : 0;

    /* ── Cache hierarchy ──────────────────────────────────────────────── */
#ifdef __APPLE__
    p->l1d_size         = macos_sysctl_size("hw.l1dcachesize");
    p->l2_size          = macos_sysctl_size("hw.l2cachesize");
    p->l3_size          = macos_sysctl_size("hw.l3cachesize");
    p->cache_line_bytes = macos_sysctl_size("hw.cachelinesize");
#elif defined(__linux__)
    p->l1d_size         = linux_find_cache_size(1);
    p->l2_size          = linux_find_cache_size(2);
    p->l3_size          = linux_find_cache_size(3);
    p->cache_line_bytes = linux_cache_line_size();
#endif

    /* ── Fallbacks if detection failed ───────────────────────────────── */
    if (p->l1d_size == 0)         p->l1d_size         = 32768;   /* 32 KB */
    if (p->cache_line_bytes == 0) p->cache_line_bytes  = 64;

    /* ── SIMD width ───────────────────────────────────────────────────── */
    /*
     * This is a compile-time decision: -march=native causes the compiler
     * to define the appropriate macros for the build machine.
     * On cross-compilation the macros reflect the target flags passed.
     */
#if   defined(__AVX512F__)
    p->simd_width_bits = 512;
#elif defined(__AVX2__)
    p->simd_width_bits = 256;
#elif defined(__AVX__)
    p->simd_width_bits = 128;
#elif defined(__SSE2__)
    p->simd_width_bits = 128;
#elif defined(__ARM_NEON)
    p->simd_width_bits = 128;
#else
    p->simd_width_bits = 0;
#endif

    /* ── Thread affinity ─────────────────────────────────────────────── */
#ifdef __linux__
    p->has_affinity = 1;
#else
    p->has_affinity = 0;
#endif

    /* ── Architecture and OS strings ────────────────────────────────── */
    struct utsname un;
    if (uname(&un) == 0) {
        snprintf(p->arch, sizeof(p->arch), "%s", un.machine);
        snprintf(p->os,   sizeof(p->os),   "%s %s", un.sysname, un.release);
    } else {
        snprintf(p->arch, sizeof(p->arch), "unknown");
        snprintf(p->os,   sizeof(p->os),   "unknown");
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * hw_print — Human-readable summary
 * ───────────────────────────────────────────────────────────────────────── */
void hw_print(const HWProfile *p) {
    printf("\n");
    printf("  +-----------------------------------------------+\n");
    printf("  |         Detected Hardware Profile             |\n");
    printf("  +-----------------------------------------------+\n");
    printf("  | OS              : %-27s |\n", p->os);
    printf("  | Architecture    : %-27s |\n", p->arch);
    printf("  | Logical CPUs    : %-27d |\n", p->num_cpus);

    if (p->l1d_size)
        printf("  | L1d cache       : %-24zu KB |\n", p->l1d_size / 1024);
    else
        printf("  | L1d cache       : %-27s |\n", "(fallback 32 KB)");

    if (p->l2_size)
        printf("  | L2 cache        : %-24zu KB |\n", p->l2_size / 1024);
    else
        printf("  | L2 cache        : %-27s |\n", "not detected");

    if (p->l3_size)
        printf("  | L3 cache        : %-24zu KB |\n", p->l3_size / 1024);
    else
        printf("  | L3 cache        : %-27s |\n", "absent");

    printf("  | Cache line      : %-24zu B  |\n", p->cache_line_bytes);

    if (p->total_ram)
        printf("  | Total RAM       : %-24zu MB |\n", p->total_ram/(1024*1024));
    else
        printf("  | Total RAM       : %-27s |\n", "unknown");

    if (p->simd_width_bits)
        printf("  | SIMD width      : %-24d b  |\n", p->simd_width_bits);
    else
        printf("  | SIMD width      : %-27s |\n", "scalar only");

    printf("  | CPU pinning     : %-27s |\n",
           p->has_affinity ? "supported (Linux)" : "not supported");
    printf("  +-----------------------------------------------+\n\n");
}

/* ─────────────────────────────────────────────────────────────────────────
 * hw_save_csv — key,value CSV
 * ───────────────────────────────────────────────────────────────────────── */
void hw_save_csv(const HWProfile *p, const char *path) {
    FILE *f = fopen(path, "w");
    if (!f) { perror(path); return; }

    fprintf(f, "key,value\n");
    fprintf(f, "os,%s\n",               p->os);
    fprintf(f, "arch,%s\n",             p->arch);
    fprintf(f, "num_cpus,%d\n",         p->num_cpus);
    fprintf(f, "l1d_size_kb,%zu\n",     p->l1d_size  / 1024);
    fprintf(f, "l2_size_kb,%zu\n",      p->l2_size   / 1024);
    fprintf(f, "l3_size_kb,%zu\n",      p->l3_size   / 1024);
    fprintf(f, "cache_line_bytes,%zu\n",p->cache_line_bytes);
    fprintf(f, "total_ram_mb,%zu\n",    p->total_ram / (1024*1024));
    fprintf(f, "simd_width_bits,%d\n",  p->simd_width_bits);
    fprintf(f, "has_affinity,%d\n",     p->has_affinity);

    fclose(f);
    printf("  [hw_detect] Profile saved -> %s\n", path);
}
