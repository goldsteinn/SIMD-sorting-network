#ifndef _INSTRUCTIONS_H_
#define _INSTRUCTIONS_H_

#include <immintrin.h>
#include <stdint.h>
#include <utility>

namespace vop {
enum INSTRUCTION_SET { AVX = 0, AVX2 = 1, AVX512 = 2 };

namespace internal {
struct avail_instructions {
#if defined __AVX512F__
    static constexpr uint32_t AVX512F = 1;
#else
    static constexpr uint32_t AVX512F    = 0;
#endif

#if defined __AVX512VL__
    static constexpr uint32_t AVX512VL = 1;
#else
    static constexpr uint32_t AVX512VL   = 0;
#endif

#if defined __AVX512VBMI__
    static constexpr uint32_t AVX512VBMI = 1;
#else
    static constexpr uint32_t AVX512VBMI = 0;
#endif

#if defined __AVX512BW__
    static constexpr uint32_t AVX512BW = 1;
#else
    static constexpr uint32_t AVX512BW   = 0;
#endif


#if defined __AVX2__
    static constexpr uint32_t AVX2 = 1;
#else
    static constexpr uint32_t AVX2       = 0;
#endif

#if defined __AVX__
    static constexpr uint32_t AVX = 1;
#else
    static constexpr uint32_t AVX        = 0;
#endif

#if defined __SSE2__
    static constexpr uint32_t SSE2 = 1;
#else
    static constexpr uint32_t SSE2       = 0;
#endif

#if defined __SSE3__
    static constexpr uint32_t SSE3 = 1;
#else
    static constexpr uint32_t SSE3       = 0;
#endif

#if defined __SSE4_1__
    static constexpr uint32_t SSE4_1 = 1;
#else
    static constexpr uint32_t SSE4_1     = 0;
#endif

#if defined __SSE4_2__
    static constexpr uint32_t SSE4_2 = 1;
#else
    static constexpr uint32_t SSE4_2     = 0;
#endif
};

template<typename T, uint32_t n, uint32_t... e>
struct vector_ops_support_impl {

    // this checks whether we can promote to next level (i.e epi16 -> epi32)
    template<uint64_t mask, uint32_t level>
    static constexpr uint32_t
    check_adjacency() {
        // make generic later
        if constexpr (level == 4) {
            constexpr uint64_t compressed_mask = (mask >> 2) & mask;
            constexpr uint64_t reduced_mask    = 0x3333333333333333 & mask;
            return compressed_mask == reduced_mask;
        }
        else /* level == 2 */ {
            constexpr uint64_t compressed_mask = (mask >> 1) & mask;
            constexpr uint64_t reduced_mask    = 0x5555555555555555 & mask;
            return compressed_mask == reduced_mask;
        }
    }


    static constexpr uint64_t
    build_blend_mask() {
        constexpr uint32_t perm[n] = { static_cast<uint32_t>(e)... };

        uint64_t blend_mask = 0;

        for (uint32_t i = 0; i < n; ++i) {
            if (perm[i] < (n - (i + 1))) {
                if constexpr (sizeof(T) < sizeof(uint64_t)) {
                    blend_mask |= ((1UL) << i);
                }
                else /* sizeof(T) == sizeof(uint64_t) */ {
                    blend_mask |= ((3UL) << (2 * i));
                }
            }
        }
        return blend_mask;
    }

    template<uint64_t blend_mask, uint32_t... seq>
    static constexpr decltype(auto)
    build_blend_vec_initializer_kernel(
        std::integer_sequence<uint32_t, seq...> _seq) {
        return std::integer_sequence<uint32_t,
                                     !!(blend_mask & ((1UL) << seq))...>{};
    }

    // this is for epi8 blend
    template<uint64_t blend_mask>
    static constexpr decltype(auto)
    build_blend_vec_initializer() {
        return build_blend_vec_initializer_kernel<blend_mask>(
            std::make_integer_sequence<uint32_t, n>{});
    }

    template<uint32_t offset, uint32_t ele_per_lane, uint32_t lane_size>
    static constexpr uint64_t
    build_shuffle_mask_impl() {
        constexpr uint32_t perms[n] = { static_cast<uint32_t>(e)... };
        uint64_t           mask     = 0;
        for (uint32_t i = 0; i < n; i += lane_size) {
            uint64_t lane_mask   = 0;
            uint32_t lower_bound = n - (i + ele_per_lane + offset);
            uint32_t upper_bound = n - (i + offset);
            for (uint32_t j = 0; j < ele_per_lane; ++j) {
                uint32_t p = perms[i + j + offset];
                if (p >= lower_bound && p < upper_bound) {
                    uint64_t idx = p - lower_bound;
                    uint32_t slot =
                        (ele_per_lane - (j + 1)) * ulog2(ele_per_lane);
                    lane_mask |= (idx << slot);
                }
                else {
                    // moving elements outside of lane so cant use shuffle
                    return 0;
                }
            }
            if (i && (mask != lane_mask)) {
                return 0;
            }
            mask = lane_mask;
        }
        return mask;
    }

    static constexpr decltype(auto)
    build_shuffle_vec_initializer() {
        // 32 will be truncated off for index calculation but will ensure never
        // hit indexes[i + 7] == 1 case
        return std::integer_sequence<uint32_t, (32 | (e))...>{};
    }


    static constexpr decltype(auto)
    build_shuffle_mask() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return build_shuffle_mask_impl<0, 16, 16>();
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            constexpr uint64_t shuffle_mask_lo =
                build_shuffle_mask_impl<0, 4, 8>();
            constexpr uint64_t shuffle_mask_hi =
                build_shuffle_mask_impl<4, 4, 8>();
            return shuffle_mask_lo | (shuffle_mask_hi << 32);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return build_shuffle_mask_impl<0, 4, 4>();
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            constexpr uint64_t shuffle_mask =
                build_shuffle_mask_impl<0, 2, 2>();

            // epi64 can only be to swap adjacent indices with shuffle_epi32
            return shuffle_mask ? 0x4e : 0;
        }
    }
};


template<typename T, uint32_t n, uint32_t... e>
struct vector_ops_support {
    static constexpr uint64_t shuffle_mask =
        vector_ops_support_impl<T, n, e...>::build_shuffle_mask();

    static constexpr uint64_t blend_mask =
        vector_ops_support_impl<T, n, e...>::build_blend_mask();


    using shuffle_vec_initialize = decltype(
        vector_ops_support_impl<T, n, e...>::build_shuffle_vec_initializer());

    using blend_vec_initialize =
        decltype(vector_ops_support_impl<T, n, e...>::
                     template build_blend_vec_initializer<blend_mask>());
};


template<typename T, INSTRUCTION_SET operations, uint32_t vec_size>
struct vector_ops;

template<typename T, INSTRUCTION_SET operations>
struct vector_ops<T, operations, sizeof(__m128i)> {


    template<uint32_t... e>
    static constexpr __m128i ALWAYS_INLINE CONST_ATTR
    build_set_vec() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // SSE2
            return _mm_set_epi8(e...);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            // SSE2
            return _mm_set_epi16(e...);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // SSE2
            return _mm_set_epi32(e...);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // SSE2
            return _mm_set_epi64(e...);
        }
    }

    template<uint32_t... e>
    static constexpr __m128i ALWAYS_INLINE CONST_ATTR
    build_set_vec_wrapper(std::integer_sequence<uint32_t, e...> _e) {
        return build_set_vec<e...>();
    }


    static __m128i ALWAYS_INLINE CONST_ATTR
    vec_min(__m128i v1, __m128i v2) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // SSE4.1
                return _mm_min_epi8(v1, v2);
            }
            else {
                // SSE4.1
                return _mm_min_epu8(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (std::is_signed<T>::value) {
                // SSE2
                return _mm_min_epi16(v1, v2);
            }
            else {
                // SSE2
                return _mm_min_epu16(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (std::is_signed<T>::value) {
                // SSE4.1
                return _mm_min_epi32(v1, v2);
            }
            else {
                // SSE4.1
                return _mm_min_epu32(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {

            if constexpr (operations >= INSTRUCTION_SET::AVX512 &&
                          avail_instructions::AVX512F &&
                          avail_instructions::AVX512VL) {
                if constexpr (std::is_signed<T>::value) {
                    // AVX512F & AVX512VL
                    return _mm_min_epi64(v1, v2);
                }
                else {
                    // AVX512F & AVX512VL
                    return _mm_min_epu64(v1, v2);
                }
            }
            else {
                if constexpr (std::is_signed<T>::value) {
                    // SSE4.2
                    __m128i cmp_mask = _mm_cmpgt_epi64(v1, v2);
                    // SSE4.1
                    return _mm_blendv_epi8(v2, v1, cmp_mask);
                }
                else {
                    // SSE2
                    __m128i sign_bits = _mm_set1_epi64x((1UL) << 63);
                    // SSE4.2 & SSE2
                    __m128i cmp_mask =
                        _mm_cmpgt_epi64(_mm_xor_si128(v1, sign_bits),
                                        _mm_xor_si128(v2, sign_bits));
                    // SSE4.1
                    return _mm_blendv_epi8(v2, v1, cmp_mask);
                }
            }
        }
    }


    static __m128i ALWAYS_INLINE CONST_ATTR
    vec_max(__m128i v1, __m128i v2) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // SSE4.1
                return _mm_max_epi8(v1, v2);
            }
            else {
                // SSE4.1
                return _mm_max_epu8(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (std::is_signed<T>::value) {
                // SSE2
                return _mm_max_epi16(v1, v2);
            }
            else {
                // SSE2
                return _mm_max_epu16(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (std::is_signed<T>::value) {
                // SSE4.1
                return _mm_max_epi32(v1, v2);
            }
            else {
                // SSE4.1
                return _mm_max_epu32(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {

            if constexpr (operations >= INSTRUCTION_SET::AVX512 &&
                          avail_instructions::AVX512F &&
                          avail_instructions::AVX512VL) {
                if constexpr (std::is_signed<T>::value) {
                    // AVX512F & AVX512VL
                    return _mm_max_epi64(v1, v2);
                }
                else {
                    // AVX512F & AVX512VL
                    return _mm_max_epu64(v1, v2);
                }
            }
            else {
                if constexpr (std::is_signed<T>::value) {
                    // SSE4.2
                    __m128i cmp_mask = _mm_cmpgt_epi64(v1, v2);
                    // SSE4.1
                    return _mm_blendv_epi8(v1, v2, cmp_mask);
                }
                else {
                    // SSE2
                    __m128i sign_bits = _mm_set1_epi64x((1UL) << 63);
                    // SSE4.2 & SSE2
                    __m128i cmp_mask =
                        _mm_cmpgt_epi64(_mm_xor_si128(v1, sign_bits),
                                        _mm_xor_si128(v2, sign_bits));
                    // SSE4.1
                    return _mm_blendv_epi8(v1, v2, cmp_mask);
                }
            }
        }
    }

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    vec_blend(__m128i v1, __m128i v2) {
        constexpr uint64_t blend_mask =
            vector_ops_support<T, 16 / sizeof(T), e...>::blend_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (operations >= INSTRUCTION_SET::AVX512 &&
                          avail_instructions::AVX512VL &&
                          avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm_mask_mov_epi8(v1, blend_mask, v2);
            }
            else {
                return _mm_blendv_epi8(
                    v1,
                    v2,
                    build_set_vec_wrapper(
                        typename vector_ops_support<T, 16 / sizeof(T), e...>::
                            blend_vec_initialize{}));
                // TODO
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (operations >= INSTRUCTION_SET::AVX512 &&
                          avail_instructions::AVX512F &&
                          avail_instructions::AVX512VL) {
                // AVX512F & AVX512VL
                return _mm_mask_mov_epi16(v1, blend_mask, v2);
            }
            else {
                // SSE4.1
                return _mm_blend_epi16(v1, v2, blend_mask);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // AVX2
            return _mm_blend_epi32(v1, v2, blend_mask);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            if constexpr (operations >= INSTRUCTION_SET::AVX512 &&
                          avail_instructions::AVX512F &&
                          avail_instructions::AVX512VL) {
                // AVX512F & AVX512VL
                return _mm_mask_mov_epi64(v1, blend_mask, v2);
            }
            else {
                // build_blend_mask will create proper mask for epi64 if AVX512F
                // and AVX512VL are not available

                // AVX2
                return _mm_blend_epi32(v1, v2, blend_mask);
            }
        }
    }

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    vec_permutate(__m128i v) {
        constexpr uint64_t shuffle_mask =
            vector_ops_support<T, 16 / sizeof(T), e...>::shuffle_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // SSE3
            return _mm_shuffle_epi8(
                v,
                build_set_vec_wrapper(
                    typename vector_ops_support<T, 16 / sizeof(T), e...>::
                        shuffle_vec_initialize{}));
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (shuffle_mask) {
                constexpr uint32_t shuffle_mask_lo = shuffle_mask;
                constexpr uint32_t shuffle_mask_hi = (shuffle_mask >> 32);

                // SSE 2
                return _mm_shufflehi_epi16(
                    _mm_shufflelo_epi16(v, shuffle_mask_lo),
                    shuffle_mask_hi);
            }
            else if constexpr (operations >= INSTRUCTION_SET::AVX512 &&
                               avail_instructions::AVX512VL &&
                               avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm_permutexvar_epi16(_mm_set_epi16(e...), v);
            }
            else {
                // TODO
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // SSE2
            return _mm_shuffle_epi32(v, shuffle_mask);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // SSE2
            return _mm_shuffle_epi32(v, shuffle_mask);
        }
    }
};


}  // namespace internal

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
using get_vec_t =
    typename std::conditional_t<n * sizeof(T) <= 32,
                                typename std::conditional_t<n * sizeof(T) <= 16,
                                                            __m128_wrapper,
                                                            __m256_wrapper>,
                                __m512_wrapper>;

template<typename T, uint32_t n>
using vec_t = typename get_vec_t<T, n>::type;

template<typename T, uint32_t n>
constexpr vec_t<T, n> ALWAYS_INLINE
vec_load(T * const arr) {
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
constexpr void ALWAYS_INLINE
vec_store(T * const arr, vec_t<T, n> v) {
    if constexpr (n * sizeof(T) <= sizeof(__m128i)) {
        return _mm_store_si128((__m128i *)arr, v);
    }
    else if constexpr (n * sizeof(T) <= sizeof(__m256i)) {
        return _mm256_store_si256((__m256i *)arr, v);
    }
    else {
        return _mm512_store_si512((__m512i *)arr, v);
    }
}

template<typename T, uint32_t n, uint32_t... e>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
compare_exchange(vec_t<T, n> v) {
    using vec_ops = typename internal::
        vector_ops<T, INSTRUCTION_SET::AVX512, sizeof(T) * n>;
    vec_t<T, n> cmp   = vec_ops::template vec_permutate<e...>(v);
    vec_t<T, n> s_min = vec_ops::vec_min(v, cmp);
    vec_t<T, n> s_max = vec_ops::vec_max(v, cmp);
    return vec_ops::template vec_blend<e...>(s_max, s_min);
}


}  // namespace vop


#endif
