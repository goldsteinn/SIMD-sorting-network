#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <x86intrin.h>
#include <algorithm>

#include <timing/stats.h>
#include <vec_sort/vec_sort.h>

template<typename T, uint32_t n>
struct sarr {
    typedef uint32_t aliasing_u32 __attribute__((aligned(1), may_alias));


    T arr[n] ALIGN_ATTR(64);

    void
    finit() {
        for (uint32_t i = 0; i < n; ++i) {
            arr[i] = i;
        }
    }

    void
    binit() {
        for (uint32_t i = 0; i < n; ++i) {
            arr[i] = (n - 1) - i;
        }
    }

    void
    show() {
        for (uint32_t i = 0; i < n; ++i) {
            fprintf(stderr, "%d: %d\n", i, (uint32_t)arr[i]);
        }
    }

    void
    verify() {
        for (uint32_t i = 1; i < n; ++i) {
            assert(arr[i] >= arr[i - 1]);
        }
    }

    void
    randomize() {
        aliasing_u32 * _arr = (aliasing_u32 *)arr;
        for (uint32_t i = 0;
             i < ((sizeof(T) * n + (sizeof(uint32_t) - 1))) / sizeof(uint32_t);
             ++i) {
            _arr[i] = rand();
        }
    }
};


enum SORT { SSORT = 0, VSORT = 1, USORT = 2 };
template<SORT s,
         typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network_algorithm,
         vsort::simd_instructions simd_set,
         vsort::builtin_usage     builtin_perm>
void NEVER_INLINE
ALIGN_ATTR(64) do_sort(T * arr) {
    if constexpr (s == SSORT) {
        std::sort(arr, arr + n);
    }
    else if constexpr (s == VSORT) {
        vsort::sorta<T, n, network_algorithm, simd_set, builtin_perm>(arr);
    }
    else {
        vsort::sortu<T, n, network_algorithm, simd_set, builtin_perm>(arr);
    }
}


template<typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network_algorithm,
         vsort::simd_instructions simd_set,
         vsort::builtin_usage     builtin_perm>
void
corr_test() {
    static constexpr uint32_t tsize = ((1u) << 20);
    sarr<T, n>                s1;
    sarr<T, n>                s2;
    sarr<T, n>                s3;

    uint32_t i = 0;
    for (i = 0; i < tsize; ++i) {
        s1.randomize();
        memcpy(s2.arr, s1.arr, n * sizeof(T));
        memcpy(s3.arr, s1.arr, n * sizeof(T));
        do_sort<SSORT, T, n, network_algorithm, simd_set, builtin_perm>(s1.arr);
        do_sort<VSORT, T, n, network_algorithm, simd_set, builtin_perm>(s2.arr);
        do_sort<USORT, T, n, network_algorithm, simd_set, builtin_perm>(s3.arr);
        if (!(!memcmp(s1.arr, s2.arr, n * sizeof(T)))) {
            fprintf(stderr,
                    "FAILED[A] : [%zu][%d][%d][%d]\n",
                    sizeof(T),
                    n,
                    (uint32_t)simd_set,
                    (uint32_t)builtin_perm);
            break;
        }
        if (!(!memcmp(s1.arr, s3.arr, n * sizeof(T)))) {
            fprintf(stderr,
                    "FAILED[U] : [%zu][%d][%d][%d]\n",
                    sizeof(T),
                    n,
                    (uint32_t)simd_set,
                    (uint32_t)builtin_perm);
            break;
        }
    }
    if (i == tsize) {
        fprintf(stderr,
                "SUCCESS: [%zu][%d][%d][%d]\n",
                sizeof(T),
                n,
                (uint32_t)simd_set,
                (uint32_t)builtin_perm);
    }
}

template<typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network_algorithm,
         vsort::simd_instructions simd_set,
         vsort::builtin_usage     builtin_perm>
double
perf_test() {
    static constexpr uint32_t tsize = ((1u) << 20);
    sarr<T, n>                s;
    uint64_t                  total_cycles = 0;
    for (uint32_t i = 0; i < tsize; ++i) {
        s.randomize();
        uint64_t start = _rdtsc();
        do_sort<VSORT, T, n, network_algorithm, simd_set, builtin_perm>(s.arr);
        uint64_t end = _rdtsc();


        total_cycles += end - start;
    }
    double ret = total_cycles;
    return ret / tsize;
}

enum OPERATION { CORRECT = 0, PERFORMANCE = 1 };
template<OPERATION op,
         typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network_algorithm,
         vsort::simd_instructions simd_set,
         vsort::builtin_usage     builtin_perm>
void
test() {
    if constexpr (op == CORRECT) {
        corr_test<T, n, network_algorithm, simd_set, builtin_perm>();
    }
    else {
        double r = perf_test<T, n, network_algorithm, simd_set, builtin_perm>();
        fprintf(stderr,
                "[%zu][%d][%d][%d]: %.3lf\n",
                sizeof(T),
                n,
                (uint32_t)simd_set,
                (uint32_t)builtin_perm,
                r);
    }
}

template<OPERATION op,
         typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network_algorithm>
void
test_all_kernel() {
    if constexpr (sizeof(T) * n <= 64) {
        if constexpr (n >= 2) {
            test<op,
                 T,
                 n,
                 network_algorithm,
                 vsort::simd_instructions::AVX2,
                 vsort::builtin_usage::BUILTIN_FIRST>();
            test<op,
                 T,
                 n,
                 network_algorithm,
                 vsort::simd_instructions::AVX2,
                 vsort::builtin_usage::BUILTIN_FALLBACK>();
            test<op,
                 T,
                 n,
                 network_algorithm,
                 vsort::simd_instructions::AVX2,
                 vsort::builtin_usage::BUILTIN_NONE>();

            test<op,
                 T,
                 n,
                 network_algorithm,
                 vsort::simd_instructions::AVX512,
                 vsort::builtin_usage::BUILTIN_FIRST>();
            test<op,
                 T,
                 n,
                 network_algorithm,
                 vsort::simd_instructions::AVX512,
                 vsort::builtin_usage::BUILTIN_FALLBACK>();
            test<op,
                 T,
                 n,
                 network_algorithm,
                 vsort::simd_instructions::AVX512,
                 vsort::builtin_usage::BUILTIN_NONE>();
        }
        test_all_kernel<op, T, n + 1, network_algorithm>();
    }
}

template<OPERATION op, template<uint32_t _n> typename network_algorithm>
void
test_all() {
    test_all_kernel<op, uint8_t, 2, network_algorithm>();
    test_all_kernel<op, uint16_t, 2, network_algorithm>();
    test_all_kernel<op, uint32_t, 2, network_algorithm>();
    test_all_kernel<op, uint64_t, 2, network_algorithm>();

    test_all_kernel<op, int8_t, 2, network_algorithm>();
    test_all_kernel<op, int16_t, 2, network_algorithm>();
    test_all_kernel<op, int32_t, 2, network_algorithm>();
    test_all_kernel<op, int64_t, 2, network_algorithm>();
}

#define v_to_string(X)  _v_to_string(X)
#define _v_to_string(X) #X

#ifndef TEST_ALGORITHM
#define TEST_ALGORITHM bitonic
#endif

#ifndef NRUNS
#define NRUNS 256
#endif

#ifndef WARMUP
#define WARMUP 2
#endif

#ifndef TEST_TYPE
#define TEST_TYPE uint32_t
#endif

#ifndef TEST_N
#define TEST_N 4
#endif

#ifndef TEST_SIMD
#define TEST_SIMD 2
#endif

#ifndef TEST_BUILTIN
#define TEST_BUILTIN 2
#endif


int
main(int argc, char ** argv) {

    const char *     hdr = "type,test_n,algorithm,simd,builtin";
    stats::stats_out so;
    if (argc > 1) {
        so.export_hdr(stderr, hdr);
    }
    else {
        test_all<CORRECT, vsort::best>();
        exit(0);
        char test_fields[128] = "";
        sprintf(test_fields,
                "%s,%d,%s,%d,%d",
                v_to_string(TEST_TYPE),
                TEST_N,
                v_to_string(TEST_ALGORITHM),
                TEST_SIMD,
                TEST_BUILTIN);

        double * results = (double *)calloc(NRUNS, sizeof(double));
        for (uint32_t i = 0; i < WARMUP; ++i) {
            perf_test<TEST_TYPE,
                      TEST_N,
                      vsort::TEST_ALGORITHM,
                      (vsort::simd_instructions)TEST_SIMD,
                      (vsort::builtin_usage)TEST_BUILTIN>();
        }
        for (uint32_t i = 0; i < NRUNS; ++i) {
            results[i] = perf_test<TEST_TYPE,
                                   TEST_N,
                                   vsort::TEST_ALGORITHM,
                                   (vsort::simd_instructions)TEST_SIMD,
                                   (vsort::builtin_usage)TEST_BUILTIN>();
        }


        so.get_stats(results, NRUNS, timers::time_units::CYCLES);
        so.print_csv(stderr, stats::DONT_PRINT_HEADER, test_fields);
        free(results);
    }
}
