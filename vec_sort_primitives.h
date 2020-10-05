#ifndef _VEC_SORT_PRIMITIVES_H_
#define _VEC_SORT_PRIMITIVES_H_

#include <immintrin.h>
#include <stdint.h>
#include "cpp_attributes.h"

struct __m128_wrapper {
    typedef __m128i type;
} ALIGN_ATTR(sizeof(__m128i));

struct __m256_wrapper {
    typedef __m256i type;
} ALIGN_ATTR(sizeof(__m256i));

struct __m512_wrapper {
    typedef __m512i type;
} ALIGN_ATTR(sizeof(__m512i));


template<typename T, uint32_t n>
using get_vec =
    typename std::conditional_t<n * sizeof(T) <= 32,
                                typename std::conditional_t<n * sizeof(T) <= 16,
                                                            __m128_wrapper,
                                                            __m256_wrapper>,
                                __m512_wrapper>;

template<typename T, uint32_t n>
using vec_t = typename get_vec<T, n>::type;


template<typename T, uint32_t n>
void ALWAYS_INLINE
vec_store(T * arr, vec_t<T, n> v) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        _mm_store_si128((__m128i *)arr, v);
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        _mm256_store_si256((__m256i *)arr, v);
    }
    else {
        _mm512_store_si512((__m512i *)arr, v);
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE
vec_load(T * arr) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        return _mm_load_si128((__m128i *)arr);
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        return _mm256_load_si256((__m256i *)arr);
    }
    else {
        return _mm512_load_si512((__m512i *)arr);
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_min(vec_t<T, n> v1, vec_t<T, n> v2) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm_min_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm_min_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm_min_epi32(v1, v2);
            }
            else {  // 64
                return _mm_min_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm_min_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm_min_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm_min_epu32(v1, v2);
            }
            else {  // 64
                return _mm_min_epu64(v1, v2);
            }
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm256_min_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm256_min_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm256_min_epi32(v1, v2);
            }
            else {  // 64
                return _mm256_min_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm256_min_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm256_min_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm256_min_epu32(v1, v2);
            }
            else {  // 64
                return _mm256_min_epu64(v1, v2);
            }
        }
    }
    else {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm512_min_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm512_min_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm512_min_epi32(v1, v2);
            }
            else {  // 64
                return _mm512_min_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm512_min_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm512_min_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm512_min_epu32(v1, v2);
            }
            else {  // 64
                return _mm512_min_epu64(v1, v2);
            }
        }
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_max(vec_t<T, n> v1, vec_t<T, n> v2) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm_max_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm_max_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm_max_epi32(v1, v2);
            }
            else {  // 64
                return _mm_max_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm_max_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm_max_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm_max_epu32(v1, v2);
            }
            else {  // 64
                return _mm_max_epu64(v1, v2);
            }
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm256_max_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm256_max_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm256_max_epi32(v1, v2);
            }
            else {  // 64
                return _mm256_max_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm256_max_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm256_max_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm256_max_epu32(v1, v2);
            }
            else {  // 64
                return _mm256_max_epu64(v1, v2);
            }
        }
    }
    else {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm512_max_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm512_max_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm512_max_epi32(v1, v2);
            }
            else {  // 64
                return _mm512_max_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm512_max_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm512_max_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm512_max_epu32(v1, v2);
            }
            else {  // 64
                return _mm512_max_epu64(v1, v2);
            }
        }
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_blend(vec_t<T, n> v_max, vec_t<T, n> v_min, const uint64_t mask) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // needs work
            return _mm_blendv_epi8(v_max, v_min, mask);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return _mm_blend_epi16(v_max, v_min, mask);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return _mm_blend_epi32(v_max, v_min, mask);
        }
        else {  // 64
            return _mm_mask_mov_epi64(v_max, mask, v_min);
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // needs work
            return _mm256_blendv_epi8(v_max, v_min, mask);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return _mm256_mask_mov_epi16(v_max, mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return _mm256_blend_epi32(v_max, v_min, mask);
        }
        else {  // 64
            return _mm256_mask_mov_epi64(v_max, mask, v_min);
        }
    }
    else {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return _mm512_mask_mov_epi8(v_max, mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return _mm512_mask_mov_epi16(v_max, mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return _mm512_mask_mov_epi32(v_max, mask, v_min);
        }
        else {  // 64
            return _mm512_mask_mov_epi64(v_max, mask, v_min);
        }
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_perm(vec_t<T, n> v1, vec_t<T, n> v2) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return _mm_permutexvar_epi8(v1, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return _mm_permutexvar_epi16(v1, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return _mm_permutexvar_epi32(v1, v2);
        }
        else {  // 64
            return _mm_permutexvar_epi64(v1, v2);
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return _mm256_permutexvar_epi8(v1, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return _mm256_permutexvar_epi16(v1, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return _mm256_permutexvar_epi32(v1, v2);
        }
        else {  // 64
            return _mm256_permutexvar_epi64(v1, v2);
        }
    }
    else {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return _mm512_permutexvar_epi8(v1, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return _mm512_permutexvar_epi16(v1, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return _mm512_permutexvar_epi32(v1, v2);
        }
        else {  // 64
            return _mm512_permutexvar_epi64(v1, v2);
        }
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_set(T e0, T e1) {
    return _mm_set_epi64x(e0, e1);
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_set(T e0, T e1, T e2, T e3) {
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        return _mm_set_epi32(e0, e1, e2, e3);
    }
    else {
        return _mm256_set_epi64x(e0, e1, e2, e3);
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_set(T e0, T e1, T e2, T e3, T e4, T e5, T e6, T e7) {
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        return _mm_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7);
    }
    else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        return _mm256_set_epi32(e0, e1, e2, e3, e4, e5, e6, e7);
    }
    else {
        return _mm512_set_epi64x(e0, e1, e2, e3, e4, e5, e6, e7);
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
// clang-format off
vec_set(T e0, T e1, T e2, T e3, T e4, T e5, T e6, T e7,
        T e8, T e9, T e10, T e11, T e12, T e13, T e14, T e15)
// clang-format on
{

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        // clang-format off
            return _mm_set_epi8(e0, e1, e2, e3, e4, e5, e6, e7, e8,
                                e9, e10, e11, e12, e13, e14, e15);
        // clang-format on
    }
    else if constexpr (sizeof(T) == sizeof(uint16_t)) {
        // clang-format off
        return _mm256_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7, e8,
                             e9, e10, e11, e12, e13, e14, e15);
        // clang-format on
    }
    else {
        // clang-format off
        return _mm512_set_epi32(e0, e1, e2, e3, e4, e5, e6, e7, e8,
                                e9, e10, e11, e12, e13, e14, e15);
        // clang-format on
    }
}


template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
// clang-format off
vec_set(T e0, T e1, T e2, T e3, T e4, T e5, T e6, T e7,
        T e8, T e9, T e10, T e11, T e12, T e13, T e14, T e15,
        T e16, T e17, T e18, T e19, T e20, T e21, T e22, T e23,
        T e24, T e25, T e26, T e27, T e28, T e29, T e30, T e31)
// clang-format on
{

    if constexpr (sizeof(T) == sizeof(uint8_t)) {
        // clang-format off
            return _mm256_set_epi8(e0, e1, e2, e3, e4, e5, e6, e7,
                                e8, e9, e10, e11, e12, e13, e14, e15,
                                e16, e17, e18, e19, e20, e21, e22, e23,
                                e24, e25, e26, e27, e28, e29, e30, e31);
        // clang-format on
    }
    else {
        // clang-format off
            return _mm512_set_epi16(e0, e1, e2, e3, e4, e5, e6, e7,
                                e8, e9, e10, e11, e12, e13, e14, e15,
                                e16, e17, e18, e19, e20, e21, e22, e23,
                                e24, e25, e26, e27, e28, e29, e30, e31);
        // clang-format on
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
// clang-format off
vec_set(T e0, T e1, T e2, T e3, T e4, T e5, T e6, T e7,
        T e8, T e9, T e10, T e11, T e12, T e13, T e14, T e15,
        T e16, T e17, T e18, T e19, T e20, T e21, T e22, T e23,
        T e24, T e25, T e26, T e27, T e28, T e29, T e30, T e31,
        T e32, T e33, T e34, T e35, T e36, T e37, T e38, T e39,
        T e40, T e41, T e42, T e43, T e44, T e45, T e46, T e47,
        T e48, T e49, T e50, T e51, T e52, T e53, T e54, T e55,
        T e56, T e57, T e58, T e59, T e60, T e61, T e62, T e63)
// clang-format on
{
    // clang-format off
    return _mm512_set_epi8(e0, e1, e2, e3, e4, e5, e6, e7,
                           e8, e9, e10, e11, e12, e13, e14, e15,
                           e16, e17, e18, e19, e20, e21, e22, e23,
                           e24, e25, e26, e27, e28, e29, e30, e31,
                           e32, e33, e34, e35, e36, e37, e38, e39,
                           e40, e41, e42, e43, e44, e45, e46, e47,
                           e48, e49, e50, e51, e52, e53, e54, e55,
                           e56, e57, e58, e59, e60, e61, e62, e63);
    // clang-format on
}
#endif
