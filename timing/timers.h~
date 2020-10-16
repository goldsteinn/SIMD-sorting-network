#ifndef _TIMING_H_
#define _TIMING_H_

#include <misc/cpp_attributes.h>

#include <stdint.h>
#include <time.h>

// TODO:
// replace clock_gettime with direct vdso syscall

namespace timers {

static constexpr clockid_t REAL   = CLOCK_REALTIME;
static constexpr clockid_t ELAPSE = CLOCK_MONOTONIC;
static constexpr clockid_t PCPU   = CLOCK_PROCESS_CPUTIME_ID;
static constexpr clockid_t TCPU   = CLOCK_THREAD_CPUTIME_ID;

enum time_units { CYCLES = 0, NSEC = 1, USEC = 2, MSEC = 3, SEC = 4 };
static constexpr char units_to_str[5][8] = { "cycles",
                                             "nsec",
                                             "usec",
                                             "msec",
                                             "sec" };
static const char *
unit_to_str(time_units unit) {
    return units_to_str[(uint32_t)unit];
}


static uint64_t ALWAYS_INLINE
get_cycles() {
    uint32_t hi, lo;
    __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
    return (((uint64_t)lo) | (((uint64_t)hi) << 32));
}

static void ALWAYS_INLINE
gettime(clockid_t id, struct timespec * ts) {
    clock_gettime(id, ts);
}



static uint64_t
ts_to_ns(struct timespec * ts) {
    return ts->tv_sec * 1000 * 1000 * 1000 + ts->tv_nsec;
}


static uint64_t
ts_to_us(struct timespec * ts) {
    return ts->tv_sec * 1000 * 1000 + ts->tv_nsec / 1000;
}


static uint64_t
ts_to_ms(struct timespec * ts) {
    return ts->tv_sec * 1000 + ts->tv_nsec / (1000 * 1000);
}

static uint64_t
ts_to_s(struct timespec * ts) {
    return ts->tv_sec + ts->tv_nsec / (1000 * 1000 * 1000);
}


uint64_t ALWAYS_INLINE
get_ns(clockid_t id = ELAPSE) {
    struct timespec t;
    gettime(id, &t);
    return ts_to_ns(&t);
}




}  // namespace clock

#endif
