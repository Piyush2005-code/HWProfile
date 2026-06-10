/*
 * hw_detect.h — Runtime Hardware Discovery
 *
 * Populates a HWProfile struct by querying the OS at runtime.
 * No values are hardcoded; every benchmark derives its parameters
 * from this struct.
 *
 * Supported platforms:
 *   Linux  : sysfs + sysconf + uname
 *   macOS  : sysctlbyname + sysconf + uname
 */

#ifndef HW_DETECT_H
#define HW_DETECT_H

#include <stddef.h>

/* ─────────────────────────────────────────────────────────────────────────
 * HWProfile
 *
 * Central description of the machine. Built once by hw_detect() and
 * passed (by const pointer) to every benchmark function.
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    /* CPU */
    int    num_cpus;            /* Logical (online) CPU count            */

    /* Cache hierarchy — sizes in bytes; 0 means "not detected / absent" */
    size_t l1d_size;            /* L1 data cache per core                */
    size_t l2_size;             /* L2 cache (may be shared)              */
    size_t l3_size;             /* L3 cache (often absent on small SoCs) */
    size_t cache_line_bytes;    /* Cache line size (typically 64 B)      */

    /* Memory */
    size_t total_ram;           /* Physical RAM in bytes                 */

    /* SIMD — detected at compile time from -march=native flags          */
    int    simd_width_bits;     /* 0 = scalar, 128 = SSE2/NEON,
                                   256 = AVX2, 512 = AVX-512            */

    /* OS capabilities */
    int    has_affinity;        /* 1 if pthread_setaffinity_np available */

    /* Strings */
    char   arch[64];            /* e.g. "aarch64", "x86_64", "arm64"    */
    char   os[128];             /* e.g. "Linux 6.1.57", "Darwin 24.4.0" */
} HWProfile;

/*
 * hw_detect() — Fill *p from OS APIs.
 *
 * Call once at program start. All fields are set; unknown cache levels
 * are left as 0 so callers can check "if (hw.l3_size) { ... }".
 */
void hw_detect(HWProfile *p);

/*
 * hw_print() — Pretty-print the profile to stdout.
 */
void hw_print(const HWProfile *p);

/*
 * hw_save_csv() — Write profile as key,value CSV to the given path.
 */
void hw_save_csv(const HWProfile *p, const char *path);

#endif /* HW_DETECT_H */
