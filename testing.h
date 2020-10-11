#ifndef _TESTING_H_
#define _TESTING_H_


#include <stdio.h>
#include <type_traits>
#include <utility>

template<typename T>
void
show_t(uint32_t n, T * arr) {
    fprintf(stderr, "[%u] -> [%u", n, arr[0]);
    for (uint32_t i = 1; i < n; ++i) {
        fprintf(stderr, ", %u", (uint32_t)(arr[i]));
    }
    fprintf(stderr, "]\n");
}

#include <immintrin.h>
template<typename T>
void
show_vec(__m128i v) {
    T arr[16 / sizeof(T)];
    _mm_store_si128((__m128i *)arr, v);
    show_t<T>(16 / sizeof(T), arr);
}

template<typename T>
void
show_vec(__m256i v) {
    T arr[32 / sizeof(T)];
    _mm256_store_si256((__m256i *)arr, v);
    show_t<T>(32 / sizeof(T), arr);
}

template<typename T>
void
show_vec(__m512i v) {
    T arr[32 / sizeof(T)];
    _mm512_store_si512((__m512i *)arr, v);
    show_t<T>(32 / sizeof(T), arr);
}

#endif
