#ifndef _DISPLAY_H_
#define _DISPLAY_H_

#include <stdio.h>
#include <immintrin.h>
#include <util/cpp_attributes.h>

template<typename T>
void
show_v(const char * const hdr, __m128i v) {
    const uint32_t s = sizeof(__m128i) / sizeof(T);
    T arr[s] ALIGN_ATTR(sizeof(__m128i));
    _mm_store_si128((__m128i *)arr, v);
    fprintf(stderr, "%s -> [%d", hdr, (uint32_t)arr[0]);
    for(uint32_t i = 1; i < s; ++i) {
        fprintf(stderr, ", %d", (uint32_t)arr[i]);
    }
    fprintf(stderr, "]\n");
}

template<typename T>
void
show_v(const char * const hdr, __m256i v) {
    const uint32_t s = sizeof(__m256i) / sizeof(T);
    T arr[s];
    memcpy(arr, &v, sizeof(__m256i));
    fprintf(stderr, "%s -> [%d", hdr, arr[0]);
    for(uint32_t i = 1; i < s; ++i) {
        fprintf(stderr, ", %d", arr[i]);
    }
    fprintf(stderr, "]\n");
}

template<typename T>
void
show_v(const char * const hdr, __m512i v) {
    const uint32_t s = sizeof(__m512i) / sizeof(T);
    T arr[s];
    memcpy(arr, &v, sizeof(__m512i));
    fprintf(stderr, "%s -> [%d", hdr, arr[0]);
    for(uint32_t i = 1; i < s; ++i) {
        fprintf(stderr, ", %d", arr[i]);
    }
    fprintf(stderr, "]\n");
}

#endif
