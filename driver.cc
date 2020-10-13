#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>

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


enum SORT { SSORT = 0, VSORT = 1 };
template<typename T, uint32_t n, SORT s, uint32_t USE_AVX512>
void NEVER_INLINE
do_sort(T * arr) {
    if constexpr (s == SSORT) {
        std::sort(arr, arr + n);
    }
    else {
        if constexpr (USE_AVX512) {
            vsort::sort<T, n, vsort::bitonic, vsort::instruction_set::AVX512>(
                arr);
        }
        else {
            vsort::sort<T, n, vsort::bitonic, vsort::instruction_set::AVX2>(
                arr);
        }
    }
}


template<typename T, uint32_t n, uint32_t USE_AVX512>
void
corr_test() {
    static constexpr uint32_t tsize = ((1u) << 20);
    sarr<T, n>                s1;
    sarr<T, n>                s2;

    uint32_t i = 0;
    for (i = 0; i < tsize; ++i) {
        s1.randomize();
        memcpy(s2.arr, s1.arr, n * sizeof(T));
        do_sort<T, n, SSORT, USE_AVX512>(s1.arr);
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

template<typename T, uint32_t n, uint32_t USE_AVX512>
void
test() {
    corr_test<T, n, USE_AVX512>();
}

template<typename T, uint32_t n>
void
test_all_kernel() {
    if constexpr (sizeof(T) * n <= 64) {
        if constexpr (n >= 4) {
            test<T, n, 1>();
            test<T, n, 0>();
        }
        test_all_kernel<T, n + 1>();
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
