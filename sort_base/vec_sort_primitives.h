#ifndef _VEC_SORT_PRIMITIVES_H_
#define _VEC_SORT_PRIMITIVES_H_

#include <immintrin.h>
#include <stdint.h>
#include <util/constexpr_util.h>
#include <util/cpp_attributes.h>

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

template<typename T>
static constexpr uint32_t ele_per_lane = sizeof(__m128i) / sizeof(T);

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

template<typename T, uint32_t s>
constexpr uint32_t
inbounds(uint32_t v) {
    return (v >= s) && (v < (s + ele_per_lane<T>));
}

template<typename T,
         uint32_t i,
         uint32_t s,
         uint32_t n,
         uint32_t mask,
         uint32_t... e>
constexpr uint32_t
group_inbounds() {
    if constexpr (i >= ele_per_lane<T>) {
        return mask;
    }
    else {
        constexpr uint32_t perm[n] = { static_cast<uint32_t>(e)... };
        constexpr uint32_t pidx    = perm[n - (s + i + 1)];
        if constexpr (inbounds<T, s>(pidx)) {
            constexpr uint32_t new_mask =
                mask | (pidx - s) << (i * ulog2(ele_per_lane<T>));
            return group_inbounds<T, i + 1, s, n, new_mask, e...>();
        }
        else {
            return 0;
        }
    }
}

template<typename T, uint32_t s, uint32_t n, uint32_t mask, T... e>
constexpr uint32_t
test_proximity() {
    if constexpr (s >= n) {
        return mask;
    }
    else {
        constexpr uint32_t new_mask = group_inbounds<T, 0, s, n, 0, e...>();
        if constexpr (new_mask != 0) {
            if constexpr (mask == 0) {
                return test_proximity<T,
                                      s + ele_per_lane<T>,
                                      n,
                                      new_mask,
                                      e...>();
            }
            else if constexpr (new_mask != mask) {
                return 0;
            }
            else {
                return test_proximity<T,
                                      s + ele_per_lane<T>,
                                      n,
                                      new_mask,
                                      e...>();
            }
        }
        else {
            return 0;
        }
    }
}

template<typename T, uint32_t n, T... e>
constexpr uint32_t
test_shuffle() {
    if constexpr (sizeof(T) == sizeof(uint32_t)) {
        return test_proximity<T, 0, n, 0, e...>();
    }
    else {
        return 0;
    }
}

template<typename T, uint32_t n, T... e>
struct shuffle {
    static constexpr uint32_t should_shuffle = test_shuffle<T, n, e...>();
};

template<typename T, uint32_t n, T... e>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_set_perm(vec_t<T, n> v1) {

    constexpr uint32_t do_shuffle = shuffle<T, n, e...>::should_shuffle;
    if constexpr (do_shuffle) {
        if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
            return _mm256_shuffle_epi32(v1, do_shuffle);
        }
        else {
            return _mm512_shuffle_epi32(v1, (_MM_PERM_ENUM)do_shuffle);
        }
    }
    else {
        if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm_permutexvar_epi8(_mm_set_epi8(e...), v1);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm_permutexvar_epi16(_mm_set_epi16(e...), v1);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm_permutexvar_epi32(_mm_set_epi32(e...), v1);
            }
            else {  // 64
                return _mm_permutexvar_epi64(_mm_set_epi64x(e...), v1);
            }
        }
        else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm256_permutexvar_epi8(_mm256_set_epi8(e...), v1);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm256_permutexvar_epi16(_mm256_set_epi16(e...), v1);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm256_permutexvar_epi32(_mm256_set_epi32(e...), v1);
            }
            else {  // 64
                return _mm256_permutexvar_epi64(_mm256_set_epi64x(e...), v1);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                return _mm512_permutexvar_epi8(_mm512_set_epi8(e...), v1);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                return _mm512_permutexvar_epi16(_mm512_set_epi16(e...), v1);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                return _mm512_permutexvar_epi32(_mm512_set_epi32(e...), v1);
            }
            else {  // 64
                return _mm512_permutexvar_epi64(_mm512_set_epi64x(e...), v1);
            }
        }
    }
}


template<typename T, uint32_t n, T... e>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
compare_exchange(vec_t<T, n> v, const uint32_t v_mask) {
    vec_t<T, n> cmp   = vec_set_perm<T, n, e...>(v);
    vec_t<T, n> s_min = vec_min<T, n>(v, cmp);
    vec_t<T, n> s_max = vec_max<T, n>(v, cmp);
    return vec_blend<T, n>(s_max, s_min, v_mask);
}

#endif
