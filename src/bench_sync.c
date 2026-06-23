/*
 * bench_sync.c — Phase D: Synchronization Overhead
 *
 * Answers: "What is the real cost of the boundaries between pipeline stages
 *           in my CV/ML pipeline?"
 *
 * In a typical CV pipeline you have something like:
 *
 *   [Decode] → [Preprocess] → [Inference] → [Postprocess]
 *
 * Each arrow is a synchronization point.  If you parallelize with threads,
 * each arrow costs at least one mutex lock/unlock or condition variable
 * signal/wait.  These benchmarks quantify that cost on your actual hardware.
 *
 * Measurements:
 *
 *   D1  Mutex — uncontended:
 *       Single thread locks and unlocks the same mutex in a tight loop.
 *       This is the minimum overhead added by wrapping a queue push/pop.
 *
 *   D2  Mutex — contended (N threads):
 *       N threads compete for the same mutex.  Latency grows as threads
 *       spend time spinning or sleeping waiting for the lock.
 *
 *   D3  C11 Atomic fetch_add — uncontended + N-thread contention:
 *       Useful if you use a shared frame counter or a lock-free queue index.
 *
 *   D4  pthread_barrier_wait (N threads):
 *       Cost of synchronizing all workers at the end of a batch.
 *
 *   D5  Condition variable ping-pong (producer → consumer):
 *       The most realistic model for a pipeline hand-off.
 *       Two threads alternate signalling each other:
 *         Thread A signals → Thread B wakes → Thread B signals → Thread A wakes → ...
 *       Measured time / (2 × ROUNDS) = one-way latency for handing a frame
 *       from one pipeline stage to the next.
 */

#include "benchmarks.h"
#include "profiler.h"
#include "csv_writer.h"

#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdatomic.h>   /* C11 atomics */

/*
 * pthread_barrier_t is a Linux extension (POSIX option _POSIX_BARRIERS).
 * macOS does not implement it. We guard the barrier benchmark and provide
 * a portable mutex+condvar simulation on non-Linux platforms.
 */
#ifdef __linux__
#include <sched.h>
#endif

/* ─────────────────────────────────────────────────────────────────────────
 * D1: Mutex round-trip, uncontended
 *
 * Single thread: lock → unlock × ITERS
 * Measures the cost of the system call and cache line ping-pong when
 * nobody else is competing.
 * ───────────────────────────────────────────────────────────────────────── */
static double bench_mutex_uncontended(void) {
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    const long long ITERS = 1000000LL;

    /* Warm up */
    for (int i = 0; i < 1000; i++) {
        pthread_mutex_lock(&mu);
        pthread_mutex_unlock(&mu);
    }

    hr_time_t t;
    timer_start(&t);
    for (long long i = 0; i < ITERS; i++) {
        pthread_mutex_lock(&mu);
        pthread_mutex_unlock(&mu);
    }
    double elapsed_s = timer_elapsed_s(&t);
    pthread_mutex_destroy(&mu);

    return (elapsed_s * 1e9) / (double)ITERS;   /* ns per lock+unlock pair */
}

/* ─────────────────────────────────────────────────────────────────────────
 * D2: Mutex contention (N threads)
 *
 * All threads compete for a single shared counter via a mutex.
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    pthread_mutex_t *mu;
    long long        iters;
    volatile long long counter;
} ContendArg;

static void *contend_worker(void *varg) {
    ContendArg *a = (ContendArg *)varg;
    for (long long i = 0; i < a->iters; i++) {
        pthread_mutex_lock(a->mu);
        a->counter++;
        pthread_mutex_unlock(a->mu);
    }
    return NULL;
}

static double bench_mutex_contended(int n_threads) {
    pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
    const long long ITERS_TOTAL = 200000LL;
    long long iters_each = ITERS_TOTAL / n_threads;
    if (iters_each < 1) iters_each = 1;

    ContendArg arg = { &mu, iters_each, 0 };
    pthread_t *tids = malloc((size_t)n_threads * sizeof(pthread_t));

    hr_time_t t;
    timer_start(&t);
    for (int i = 0; i < n_threads; i++)
        pthread_create(&tids[i], NULL, contend_worker, &arg);
    for (int i = 0; i < n_threads; i++)
        pthread_join(tids[i], NULL);
    double elapsed_s = timer_elapsed_s(&t);

    free(tids);
    pthread_mutex_destroy(&mu);

    /* ns per operation = elapsed / total_ops */
    long long total_ops = iters_each * n_threads;
    return (elapsed_s * 1e9) / (double)total_ops;
}

/* ─────────────────────────────────────────────────────────────────────────
 * D3: C11 Atomic fetch_add
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    atomic_long *counter;
    long long    iters;
} AtomicArg;

static void *atomic_worker(void *varg) {
    AtomicArg *a = (AtomicArg *)varg;
    for (long long i = 0; i < a->iters; i++)
        atomic_fetch_add_explicit(a->counter, 1, memory_order_relaxed);
    return NULL;
}

static double bench_atomic(int n_threads) {
    atomic_long counter = ATOMIC_VAR_INIT(0);
    const long long ITERS_TOTAL = 2000000LL;
    long long iters_each = ITERS_TOTAL / n_threads;

    AtomicArg  arg  = { &counter, iters_each };
    pthread_t *tids = malloc((size_t)n_threads * sizeof(pthread_t));

    hr_time_t t;
    timer_start(&t);
    for (int i = 0; i < n_threads; i++)
        pthread_create(&tids[i], NULL, atomic_worker, &arg);
    for (int i = 0; i < n_threads; i++)
        pthread_join(tids[i], NULL);
    double elapsed_s = timer_elapsed_s(&t);

    free(tids);
    long long total_ops = iters_each * n_threads;
    return (elapsed_s * 1e9) / (double)total_ops;
}

/* ─────────────────────────────────────────────────────────────────────────
 * D4: pthread barrier
 *
 * pthread_barrier is a Linux extension. On macOS we simulate it with a
 * mutex + condition variable "rendezvous" pattern, which is functionally
 * equivalent and measures the same scheduling round-trip.
 * ───────────────────────────────────────────────────────────────────────── */

#ifdef __linux__
/* ── Linux: native pthread_barrier ────────────────────────────────────── */
typedef struct {
    pthread_barrier_t *barrier;
    long long          rounds;
} BarrierArg;

static void *barrier_worker(void *varg) {
    BarrierArg *a = (BarrierArg *)varg;
    for (long long i = 0; i < a->rounds; i++)
        pthread_barrier_wait(a->barrier);
    return NULL;
}

static double bench_barrier(int n_threads) {
    const long long ROUNDS = 5000LL;
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)n_threads);

    BarrierArg  arg  = { &bar, ROUNDS };
    pthread_t  *tids = malloc((size_t)n_threads * sizeof(pthread_t));

    hr_time_t t;
    timer_start(&t);
    for (int i = 0; i < n_threads; i++)
        pthread_create(&tids[i], NULL, barrier_worker, &arg);
    for (int i = 0; i < n_threads; i++)
        pthread_join(tids[i], NULL);
    double elapsed_s = timer_elapsed_s(&t);

    free(tids);
    pthread_barrier_destroy(&bar);

    return (elapsed_s * 1e9) / (double)ROUNDS;
}

#else /* !__linux__ */
/* ── Portable fallback: mutex + condvar rendezvous ─────────────────────── */
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    int             arrived;   /* count of threads that reached barrier */
    int             n;         /* total thread count for this barrier   */
    int             generation;/* prevents spurious wakeups across rounds */
} PortableBarrier;

static void portable_barrier_wait(PortableBarrier *b) {
    pthread_mutex_lock(&b->mu);
    int gen = b->generation;
    b->arrived++;
    if (b->arrived == b->n) {
        b->arrived = 0;
        b->generation++;
        pthread_cond_broadcast(&b->cv);
    } else {
        while (b->generation == gen)
            pthread_cond_wait(&b->cv, &b->mu);
    }
    pthread_mutex_unlock(&b->mu);
}

typedef struct {
    PortableBarrier *bar;
    long long        rounds;
} BarrierArg;

static void *barrier_worker(void *varg) {
    BarrierArg *a = (BarrierArg *)varg;
    for (long long i = 0; i < a->rounds; i++)
        portable_barrier_wait(a->bar);
    return NULL;
}

static double bench_barrier(int n_threads) {
    const long long ROUNDS = 5000LL;
    PortableBarrier bar = {
        PTHREAD_MUTEX_INITIALIZER, PTHREAD_COND_INITIALIZER,
        0, n_threads, 0
    };

    BarrierArg  arg  = { &bar, ROUNDS };
    pthread_t  *tids = malloc((size_t)n_threads * sizeof(pthread_t));

    hr_time_t t;
    timer_start(&t);
    for (int i = 0; i < n_threads; i++)
        pthread_create(&tids[i], NULL, barrier_worker, &arg);
    for (int i = 0; i < n_threads; i++)
        pthread_join(tids[i], NULL);
    double elapsed_s = timer_elapsed_s(&t);

    free(tids);
    pthread_mutex_destroy(&bar.mu);
    pthread_cond_destroy(&bar.cv);

    return (elapsed_s * 1e9) / (double)ROUNDS;
}
#endif /* __linux__ */

/* ─────────────────────────────────────────────────────────────────────────
 * D5: Condition variable ping-pong (pipeline stage hand-off)
 *
 * Two threads alternate:
 *   Thread A: sends "frame ready" → Thread B
 *   Thread B: processes, sends "done" → Thread A
 *
 * Total wall time / (2 × ROUNDS) = one-way latency.
 * ───────────────────────────────────────────────────────────────────────── */
typedef struct {
    pthread_mutex_t mu;
    pthread_cond_t  cond_a;   /* signalled by B when it is done */
    pthread_cond_t  cond_b;   /* signalled by A when data is ready */
    int             turn;     /* 0 = A's turn to send, 1 = B's turn */
    long long       rounds;
    volatile int    stop;
} PingPongState;

static void *pingpong_thread_b(void *varg) {
    PingPongState *s = (PingPongState *)varg;
    
    pthread_mutex_lock(&s->mu);
    while (!s->stop) {
        while (s->turn != 1 && !s->stop) {
            pthread_cond_wait(&s->cond_b, &s->mu);
        }
        if (s->stop) break;
        
        s->turn = 0;
        pthread_cond_signal(&s->cond_a);
    }
    pthread_mutex_unlock(&s->mu);
    return NULL;
}

static double bench_condvar_pingpong(void) {
    const long long ROUNDS = 50000LL;

    PingPongState s;
    pthread_mutex_init(&s.mu, NULL);
    pthread_cond_init(&s.cond_a, NULL);
    pthread_cond_init(&s.cond_b, NULL);
    s.turn   = 0;
    s.rounds = ROUNDS;
    s.stop   = 0;

    pthread_t tb;
    pthread_create(&tb, NULL, pingpong_thread_b, &s);

    /* Warm up */
    pthread_mutex_lock(&s.mu);
    for (int w = 0; w < 100; w++) {
        s.turn = 1;
        pthread_cond_signal(&s.cond_b);
        while (s.turn != 0) {
            pthread_cond_wait(&s.cond_a, &s.mu);
        }
    }
    pthread_mutex_unlock(&s.mu);

    hr_time_t t;
    timer_start(&t);

    pthread_mutex_lock(&s.mu);
    for (long long i = 0; i < ROUNDS; i++) {
        s.turn = 1;
        pthread_cond_signal(&s.cond_b);
        while (s.turn != 0) {
            pthread_cond_wait(&s.cond_a, &s.mu);
        }
    }
    
    s.stop = 1;
    pthread_cond_signal(&s.cond_b);
    pthread_mutex_unlock(&s.mu);

    double elapsed_s = timer_elapsed_s(&t);
    pthread_join(tb, NULL);

    pthread_mutex_destroy(&s.mu);
    pthread_cond_destroy(&s.cond_a);
    pthread_cond_destroy(&s.cond_b);

    /* Each round = A→B signal + B→A signal = 2 one-way hops */
    return (elapsed_s * 1e9) / (double)(ROUNDS * 2LL);
}

/* ─────────────────────────────────────────────────────────────────────────
 * Public entry point
 * ───────────────────────────────────────────────────────────────────────── */
void bench_sync(const HWProfile *hw, const char *csv_path) {
    printf("=== [Phase D] Synchronization Overhead ===\n");

    CsvWriter w = csv_open(csv_path, "primitive,threads,latency_ns");

    /* D1: Mutex uncontended */
    printf("  Mutex (uncontended)...\n");
    double mu_unc = bench_mutex_uncontended();
    printf("  mutex uncontended: %.1f ns\n", mu_unc);
    csv_write_row(&w, "mutex_uncontended,1,%.2f", mu_unc);

    /* D2: Mutex contended at 2..N threads */
    for (int n = 2; n <= hw->num_cpus; n++) {
        printf("  Mutex contended (%d threads)...\n", n);
        double v = bench_mutex_contended(n);
        printf("  mutex contended(%d): %.1f ns\n", n, v);
        csv_write_row(&w, "mutex_contended,%d,%.2f", n, v);
    }

    /* D3: Atomic fetch_add */
    printf("  Atomic fetch_add (1 thread)...\n");
    double at1 = bench_atomic(1);
    printf("  atomic fetch_add(1): %.1f ns\n", at1);
    csv_write_row(&w, "atomic_fetch_add,1,%.2f", at1);

    for (int n = 2; n <= hw->num_cpus; n++) {
        printf("  Atomic fetch_add (%d threads)...\n", n);
        double v = bench_atomic(n);
        printf("  atomic fetch_add(%d): %.1f ns\n", n, v);
        csv_write_row(&w, "atomic_fetch_add,%d,%.2f", n, v);
    }

    /* D4: Barrier */
    for (int n = 2; n <= hw->num_cpus; n++) {
        printf("  pthread_barrier (%d threads)...\n", n);
        double v = bench_barrier(n);
        printf("  barrier(%d): %.1f ns\n", n, v);
        csv_write_row(&w, "pthread_barrier,%d,%.2f", n, v);
    }

    /* D5: Condvar ping-pong */
    printf("  Condvar ping-pong (pipeline hand-off, 2 threads)...\n");
    double pp = bench_condvar_pingpong();
    printf("  condvar hand-off: %.1f ns one-way\n", pp);
    csv_write_row(&w, "condvar_handoff,2,%.2f", pp);

    csv_close(&w);
    printf("[Phase D] Done.\n\n");
}
