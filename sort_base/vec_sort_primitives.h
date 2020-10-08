#ifndef _VEC_SORT_PRIMITIVES_H_
#define _VEC_SORT_PRIMITIVES_H_

#include <immintrin.h>
#include <stdint.h>
#include <util/constexpr_util.h>
#include <util/cpp_attributes.h>


#include <instructions/instruction_optimizer.h>
#include <instructions/instruction_sets.h>
#include <instructions/instruction_types.h>


template<typename T, uint32_t n>
void ALWAYS_INLINE
vec_store(T * arr, vec_t<T, n> v) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        // AVX
        _mm_store_si128((__m128i *)arr, v);
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        // AVX2
        _mm256_store_si256((__m256i *)arr, v);
    }
    else {
        // AVX512F
        _mm512_store_si512((__m512i *)arr, v);
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE
vec_load(T * arr) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        // AVX
        return _mm_load_si128((__m128i *)arr);
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        // AVX2
        return _mm256_load_si256((__m256i *)arr);
    }
    else {
        // AVX512_F
        return _mm512_load_si512((__m512i *)arr);
    }
}

template<typename T, uint32_t n>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_min(vec_t<T, n> v1, vec_t<T, n> v2) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // SSE4.1
                return _mm_min_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // SSE2
                return _mm_min_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // SSE4.1
                return _mm_min_epi32(v1, v2);
            }
            else {  // 64
                // AVX512F
                // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm_min_epi64(v1, v2);
                }
                else {
                    vec_t<T, n> cmp_mask = _mm_cmpgt_epi64(v1, v2);
                    return _mm_blendv_epi8(v2, v1, cmp_mask);
                }
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // SSE4.1
                return _mm_min_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // SSE2
                return _mm_min_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // SSE4.1
                return _mm_min_epu32(v1, v2);
            }
            else {  // 64
                // AVX512F
                // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm_min_epu64(v1, v2);
                }
                else {
                    vec_t<T, n> sign_bit = _mm_set1_epi64x((1UL) << 63);
                    vec_t<T, n> cmp_mask =
                        _mm_cmpgt_epi64(_mm_xor_si128(v1, sign_bit),
                                        _mm_xor_si128(v2, sign_bit));
                    return _mm_blendv_epi8(v2, v1, cmp_mask);
                }
            }
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX2
                return _mm256_min_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX2
                return _mm256_min_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX2
                return _mm256_min_epi32(v1, v2);
            }
            else {  // 64
                // AVX512F
                // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm256_min_epi64(v1, v2);
                }
                else {
                    vec_t<T, n> cmp_mask = _mm256_cmpgt_epi64(v1, v2);
                    return _mm256_blendv_epi8(v2, v1, cmp_mask);
                }
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX2
                return _mm256_min_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX2
                return _mm256_min_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX2
                return _mm256_min_epu32(v1, v2);
            }
            else {  // 64
                // AVX512F
                // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm256_min_epu64(v1, v2);
                }
                else {
                    vec_t<T, n> sign_bit = _mm256_set1_epi64x((1UL) << 63);
                    vec_t<T, n> cmp_mask =
                        _mm256_cmpgt_epi64(_mm256_xor_si256(v1, sign_bit),
                                           _mm256_xor_si256(v2, sign_bit));
                    return _mm256_blendv_epi8(v2, v1, cmp_mask);
                }
            }
        }
    }
    else {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX512BW
                return _mm512_min_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX512BW
                return _mm512_min_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX512F
                return _mm512_min_epi32(v1, v2);
            }
            else {  // 64
                // AVX512F
                return _mm512_min_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX512BW
                return _mm512_min_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX512BW
                return _mm512_min_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX512F
                return _mm512_min_epu32(v1, v2);
            }
            else {  // 64
                // AVX512F
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
                // SSE4.1
                return _mm_max_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // SSE2
                return _mm_max_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // SSE4.1
                return _mm_max_epi32(v1, v2);
            }
            else {  // 64
                    // AVX512F
                    // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm_max_epi64(v1, v2);
                }
                else {
                    vec_t<T, n> cmp_mask = _mm_cmpgt_epi64(v1, v2);
                    return _mm_blendv_epi8(v1, v2, cmp_mask);
                }
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // SSE4.1
                return _mm_max_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // SSE2
                return _mm_max_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // SSE4.1
                return _mm_max_epu32(v1, v2);
            }
            else {  // 64
                    // AVX512F
                    // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm_max_epu64(v1, v2);
                }
                else {
                    vec_t<T, n> sign_bit = _mm_set1_epi64x((1UL) << 63);
                    vec_t<T, n> cmp_mask =
                        _mm_cmpgt_epi64(_mm_xor_si128(v1, sign_bit),
                                        _mm_xor_si128(v2, sign_bit));
                    return _mm_blendv_epi8(v1, v2, cmp_mask);
                }
            }
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX2
                return _mm256_max_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX2
                return _mm256_max_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX2
                return _mm256_max_epi32(v1, v2);
            }
            else {  // 64
                    // AVX512F
                    // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm256_max_epi64(v1, v2);
                }
                else {
                    vec_t<T, n> cmp_mask = _mm256_cmpgt_epi64(v1, v2);
                    return _mm256_blendv_epi8(v1, v2, cmp_mask);
                }
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX2
                return _mm256_max_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX2
                return _mm256_max_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX2
                return _mm256_max_epu32(v1, v2);
            }
            else {  // 64
                    // AVX512F
                    // AVX512VL
                if constexpr (AVX_512_F && AVX_512_VL) {
                    return _mm256_max_epu64(v1, v2);
                }
                else {
                    vec_t<T, n> sign_bit = _mm256_set1_epi64x((1UL) << 63);
                    vec_t<T, n> cmp_mask =
                        _mm256_cmpgt_epi64(_mm256_xor_si256(v1, sign_bit),
                                           _mm256_xor_si256(v2, sign_bit));
                    return _mm256_blendv_epi8(v1, v2, cmp_mask);
                }
            }
        }
    }
    else {
        if constexpr (std::is_signed<T>::value) {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX512BW
                return _mm512_max_epi8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX512BW
                return _mm512_max_epi16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX512F
                return _mm512_max_epi32(v1, v2);
            }
            else {  // 64
                // AVX512F
                return _mm512_max_epi64(v1, v2);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint8_t)) {
                // AVX512BW
                return _mm512_max_epu8(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint16_t)) {
                // AVX512BW
                return _mm512_max_epu16(v1, v2);
            }
            else if constexpr (sizeof(T) == sizeof(uint32_t)) {
                // AVX512F
                return _mm512_max_epu32(v1, v2);
            }
            else {  // 64
                    // AVX512F
                return _mm512_max_epu64(v1, v2);
            }
        }
    }
}

template<typename T, uint32_t n, T... e>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_blend(vec_t<T, n> v_max, vec_t<T, n> v_min) {
    static constexpr uint64_t blend_mask = get_blend_mask<T, n, e...>();

    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // SSE4.1
            return _mm_mask_mov_epi8(v_max, blend_mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (AVX_512_VL && AVX_512_F) {
                return _mm_mask_mov_epi16(v_max, blend_mask, v_min);
            }
            else {
                // SSE4.1
                return _mm_blend_epi16(v_max, v_min, blend_mask);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // AVX2
            return _mm_blend_epi32(v_max, v_min, blend_mask);
        }
        else {  // 64
            // AVX512F
            // AVX51VL
            return _mm_mask_mov_epi64(v_max, blend_mask, v_min);
        }
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // AVX2
            return _mm256_mask_mov_epi8(v_max, blend_mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (AVX_512_VL && AVX_512_F) {
                return _mm256_mask_mov_epi16(v_max, blend_mask, v_min);
            }
            else {
                // AVX2
                return _mm256_blend_epi16(v_max, v_min, blend_mask);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // AVX2
            return _mm256_blend_epi32(v_max, v_min, blend_mask);
        }
        else {  // 64
                // AVX512F
            // AVX51VL
            return _mm256_mask_mov_epi64(v_max, blend_mask, v_min);
        }
    }
    else {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // AVX512BW
            return _mm512_mask_mov_epi8(v_max, blend_mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            // AVX512BW
            return _mm512_mask_mov_epi16(v_max, blend_mask, v_min);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // AVX512F
            return _mm512_mask_mov_epi32(v_max, blend_mask, v_min);
        }
        else {  // 64
            // AVX512F
            return _mm512_mask_mov_epi64(v_max, blend_mask, v_min);
        }
    }
}


template<typename T, uint32_t n, T... e>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
vec_set_perm(vec_t<T, n> v1) {
    static constexpr uint64_t shuffle_mask = get_shuffle_mask<T, n, e...>();
    if constexpr (shuffle_mask) {
        if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
            if constexpr (sizeof(T) == sizeof(uint16_t)) {
                constexpr uint32_t shuffle_lo = shuffle_mask;
                constexpr uint32_t shuffle_hi = (shuffle_mask >> 32);

                return _mm_shufflehi_epi16(_mm_shufflelo_epi16(v1, shuffle_lo),
                                           shuffle_hi);
            }
            else {
                return _mm_shuffle_epi32(v1, shuffle_mask);
            }
        }
        else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
            if constexpr (sizeof(T) == sizeof(uint16_t)) {
                constexpr uint32_t shuffle_lo = shuffle_mask;
                constexpr uint32_t shuffle_hi = (shuffle_mask >> 32);

                return _mm256_shufflehi_epi16(
                    _mm256_shufflelo_epi16(v1, shuffle_lo),
                    shuffle_hi);
            }
            else {
                return _mm256_shuffle_epi32(v1, shuffle_mask);
            }
        }
        else {
            if constexpr (sizeof(T) == sizeof(uint16_t)) {
                constexpr uint32_t shuffle_lo = shuffle_mask;
                constexpr uint32_t shuffle_hi = (shuffle_mask >> 32);

                return _mm512_shufflehi_epi16(
                    _mm512_shufflelo_epi16(v1, shuffle_lo),
                    shuffle_hi);
            }
            else {
                return _mm512_shuffle_epi32(v1, (_MM_PERM_ENUM)shuffle_mask);
            }
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
                return _mm_permutexvar_epi64(_mm_set_epi64(e...), v1);
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
                return _mm512_permutexvar_epi64(_mm512_set_epi64(e...), v1);
            }
        }
    }
}


template<typename T, uint32_t n, T... e>
vec_t<T, n> ALWAYS_INLINE CONST_ATTR
compare_exchange(vec_t<T, n> v) {
    vec_t<T, n> cmp   = vec_set_perm<T, n, e...>(v);
    vec_t<T, n> s_min = vec_min<T, n>(v, cmp);
    vec_t<T, n> s_max = vec_max<T, n>(v, cmp);
    return vec_blend<T, n, e...>(s_max, s_min);
}

#endif
