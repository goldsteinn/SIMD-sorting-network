#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

#include <util/constexpr_util.h>
#include <util/cpp_attributes.h>
#include "x86intrin.h"

#include <algorithms/networks.h>
#include <algorithms/vec_sort.h>


template<typename T, uint32_t n>
struct sarr {
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
};


enum SORT { SSORT = 0, VSORT = 1 };
template<typename T, uint32_t n, SORT s, uint32_t USE_AVX512 = 0>
void NEVER_INLINE
do_sort(T * arr) {
    if constexpr (s == SSORT) {
        std::sort(arr, arr + n);
    }
    else {
        if constexpr (USE_AVX512) {
            vsort::vec_sort<T, n, typename vsort::bitonic<n>::network>::sort(
                arr);
        }
        else {
            vsort::vec_sort<T,
                            n,
                            typename vsort::bitonic<n>::network,
                            vop::instruction_set::AVX2>::sort(arr);
        }
    }
}


static constexpr uint32_t tsize = ((1u) << 16);
template<typename T, uint32_t n, uint32_t USE_AVX512 = 0>
void
corr_test() {
    sarr<T, n> s1;
    sarr<T, n> s2;

    uint32_t i = 0;
    for (i = 0; i < tsize; ++i) {
        for (uint32_t _i = 0; _i < n; ++_i) {
            s1.arr[_i] = (T)rand();
        }
        memcpy(s2.arr, s1.arr, n * sizeof(T));
        do_sort<T, n, SSORT>(s1.arr);
        do_sort<T, n, VSORT, USE_AVX512>(s2.arr);
        if (!(!memcmp(s1.arr, s2.arr, n * sizeof(T)))) {
            fprintf(stderr,
                    "FAILED : [%zu][%d][%d]\n",
                    sizeof(T),
                    n,
                    USE_AVX512);
            break;
        }
    }
    if (i == tsize) {
        fprintf(stderr, "SUCCESS: [%zu][%d][%d]\n", sizeof(T), n, USE_AVX512);
    }
}


template<typename T, uint32_t n, SORT s, uint32_t USE_AVX512 = 0>
double
inner_perf_test() {
    sarr<T, n> s1;

    uint64_t running_total = 0;
    for (uint32_t i = 0; i < tsize; ++i) {
        for (uint32_t _i = 0; _i < n; ++_i) {
            s1.arr[_i] = (T)rand();
        }

        uint64_t start = _rdtsc();
        COMPILER_BARRIER()
        do_sort<T, n, s>(s1.arr);
        COMPILER_DO_NOT_OPTIMIZE_OUT(s1.arr);
        COMPILER_BARRIER()
        uint64_t end = _rdtsc();

        running_total += (end - start);
    }
    double r = running_total;
    return r / tsize;
}


void
perf_print(const char * const h, double cycles) {
    fprintf(stderr, "%-20s: %.3lf \"Cycles\"\n", h, cycles);
}
template<typename T, uint32_t n, uint32_t USE_AVX512 = 0>
void
perf_test() {
    double vsort_cycles = inner_perf_test<T, n, VSORT, USE_AVX512>();
    double ssort_cycles = inner_perf_test<T, n, SSORT>();


    perf_print("AVX Sort Network", vsort_cycles);
    perf_print("std::sort", ssort_cycles);
}

template<typename T, uint32_t n, uint32_t USE_AVX512 = 0>
void
test() {
    corr_test<T, n, USE_AVX512>();
    //    perf_test<T, n, USE_AVX512>();
}
template<typename T, uint32_t n>
void
test_all_kernel() {
    if constexpr (sizeof(T) * n <= 64) {
        if constexpr (n >= 4) {
            test<T, n, 1>();
            test<T, n, 0>();
        }
        test_all_kernel<T, n + 3>();
    }
}

void
test_all() {
    test_all_kernel<uint8_t, 9>();
    test_all_kernel<uint16_t, 5>();
    test_all_kernel<uint32_t, 4>();
    test_all_kernel<uint64_t, 4>();
}
int
main() {
    test_all();
}
