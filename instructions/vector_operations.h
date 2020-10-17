#ifndef _INSTRUCTIONS_H_
#define _INSTRUCTIONS_H_

#include <immintrin.h>
#include <mmintrin.h>
#include <stdint.h>
#include <xmmintrin.h>
#include <utility>

#include <instructions/vector_operation_support.h>
#include <util/cpp_attributes.h>

namespace vsort {

enum simd_instructions { AVX = 0, AVX2 = 1, AVX512 = 2 };
enum builtin_usage {
    BUILTIN_FIRST    = 0,
    BUILTIN_FALLBACK = 1,
    BUILTIN_NONE     = 2
};
namespace vop {

static constexpr simd_instructions simd_instructions_default =
    (internal::avail_instructions::AVX512F |
     internal::avail_instructions::AVX512VL |
     internal::avail_instructions::AVX512VBMI |
     internal::avail_instructions::AVX512BW)
        ? simd_instructions::AVX512
        : simd_instructions::AVX2;

static constexpr builtin_usage builtin_perm_default =
    (internal::avail_instructions::CLANG_BUILTIN
         ? builtin_usage::BUILTIN_FIRST
         : (internal::avail_instructions::GCC_BUILTIN
                ? builtin_usage::BUILTIN_FALLBACK
                : builtin_usage::BUILTIN_NONE));


namespace internal {
template<typename T,
         simd_instructions simd_set,
         builtin_usage     builtin_perm,
         uint32_t          vec_size>
struct vector_ops;

template<typename T, simd_instructions simd_set, builtin_usage builtin_perm>
struct vector_ops<T, simd_set, builtin_perm, sizeof(__m64)> {
    static constexpr uint32_t vec_size = sizeof(__m64);
    static constexpr uint32_t n        = vec_size / sizeof(T);

    template<uint32_t... e>
    static constexpr __m64 ALWAYS_INLINE CONST_ATTR
    build_set_vec() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return _mm_set_pi8(e...);
        }
        else /* sizeof(T) == sizeof(uint16_t) */ {
            return _mm_set_pi16(e...);
        }
    }

    static __m64 ALWAYS_INLINE CONST_ATTR
    vec_min(__m64 v1, __m64 v2) {

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                __m64 cmp_mask = _mm_cmpgt_pi8(v2, v1);
                return _mm_or_si64(_mm_and_si64(cmp_mask, v1),
                                   _mm_andnot_si64(cmp_mask, v2));
            }
            else {
                // SSE
                return _mm_min_pu8(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint16_t) */ {
        }
        if constexpr (std::is_signed<T>::value) {
            // SSE
            return _mm_min_pi16(v1, v2);
        }
        else {
            __m64 sign_bits = _mm_set1_pi16(1 << 15);
            __m64 cmp_mask  = _mm_cmpgt_pi16(_mm_xor_si64(v1, sign_bits),
                                            _mm_xor_si64(v2, sign_bits));
            return _mm_or_si64(_mm_and_si64(cmp_mask, v2),
                               _mm_andnot_si64(cmp_mask, v1));
        }
    }

    static __m64 ALWAYS_INLINE CONST_ATTR
    vec_max(__m64 v1, __m64 v2) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // todo
                __m64 cmp_mask = _mm_cmpgt_pi8(v1, v2);
                return _mm_or_si64(_mm_and_si64(cmp_mask, v1),
                                   _mm_andnot_si64(cmp_mask, v2));
            }
            else {
                // SSE
                return _mm_max_pu8(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint16_t) */ {
        }
        if constexpr (std::is_signed<T>::value) {
            // SSE
            return _mm_max_pi16(v1, v2);
        }
        else {
            __m64 sign_bits = _mm_set1_pi16(1 << 15);
            __m64 cmp_mask  = _mm_cmpgt_pi16(_mm_xor_si64(v1, sign_bits),
                                            _mm_xor_si64(v2, sign_bits));
            return _mm_or_si64(_mm_and_si64(cmp_mask, v1),
                               _mm_andnot_si64(cmp_mask, v2));
        }
    }


    template<uint32_t... e>
    static __m64 ALWAYS_INLINE CONST_ATTR
    vec_blend(__m64 v1, __m64 v2) {
        using vop_support = internal::blend_support<T, n, e...>;

        constexpr __m64 blend_mask = (__m64)vop_support::blend_mask;

        return _mm_or_si64(_mm_and_si64(blend_mask, v2),
                           _mm_andnot_si64(blend_mask, v1));
    }

    template<uint32_t... e>
    static __m64 ALWAYS_INLINE CONST_ATTR
    vec_permutate(__m64 v) {

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return _mm_shuffle_pi8(v, build_set_vec<e...>());
        }
        else /* sizeof(T) == sizeof(uint16_t) */ {
            constexpr uint32_t shuffle_mask =
                shuffle_support<T, n, e...>::shuffle_mask;
            return _mm_shuffle_pi16(v, shuffle_mask);
        }
    }

};  // namespace internal

template<typename T, simd_instructions simd_set, builtin_usage builtin_perm>
struct vector_ops<T, simd_set, builtin_perm, sizeof(__m128i)> {
    static constexpr uint32_t vec_size = sizeof(__m128i);
    static constexpr uint32_t n        = vec_size / sizeof(T);

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle_impl(__m128i v) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector(
                (internal::vec_types::vec16x1)v,
                (internal::vec_types::vec16x1)v,
                e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (internal::vec_types::vec16x1)v,
                (internal::vec_types::vec16x1)_mm_set_epi8(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector(
                (internal::vec_types::vec8x2)v,
                (internal::vec_types::vec8x2)v,
                e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (internal::vec_types::vec8x2)v,
                (internal::vec_types::vec8x2)_mm_set_epi16(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector(
                (internal::vec_types::vec4x4)v,
                (internal::vec_types::vec4x4)v,
                e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (internal::vec_types::vec4x4)v,
                (internal::vec_types::vec4x4)_mm_set_epi32(e...));
#else
            return v;
#endif
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector(
                (internal::vec_types::vec2x8)v,
                (internal::vec_types::vec2x8)v,
                e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (internal::vec_types::vec2x8)v,
                (internal::vec_types::vec2x8)_mm_set_epi64x(e...));
#else
            return v;
#endif
        }
    }

    template<uint32_t... e, uint32_t... seq>
    static __m128i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle_reverse(__m128i                                 v,
                            std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr uint32_t arr[n] = { static_cast<uint32_t>(e)... };
        return builtin_shuffle_impl<arr[(n - 1) - seq]...>(v);
    }

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle(__m128i v) {
#if defined(__clang__)
        return builtin_shuffle_reverse<e...>(
            v,
            std::make_integer_sequence<uint32_t, n>{});
#elif defined(__GNUC__)
        return builtin_shuffle_impl<e...>(v);
#else
        return v;
#endif
    }

    template<uint32_t size, uint32_t... e>
    static constexpr __m128i ALWAYS_INLINE CONST_ATTR
    build_set_vec() {
        if constexpr ((size == sizeof(uint8_t)) ||
                      ((size == 0) && (sizeof(T) == sizeof(uint8_t)))) {
            // SSE2
            return _mm_set_epi8(e...);
        }
        else if constexpr ((size == sizeof(uint16_t)) ||
                           ((size == 0) && (sizeof(T) == sizeof(uint16_t)))) {
            // SSE2
            return _mm_set_epi16(e...);
        }
        else if constexpr ((size == sizeof(uint32_t)) ||
                           ((size == 0) && (sizeof(T) == sizeof(uint32_t)))) {
            // SSE2
            return _mm_set_epi32(e...);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // SSE2
            return _mm_set_epi64x(e...);
        }
    }

    template<uint32_t size, uint32_t... e>
    static constexpr __m128i ALWAYS_INLINE CONST_ATTR
    build_set_vec_wrapper(std::integer_sequence<uint32_t, e...> _e) {
        return build_set_vec<size, e...>();
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

            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512F &&
                          internal::avail_instructions::AVX512VL) {
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

            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512F &&
                          internal::avail_instructions::AVX512VL) {
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

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    vec_blend(__m128i v1, __m128i v2) {
        using vop_support = internal::blend_support<T, n, e...>;

        constexpr uint64_t blend_mask = vop_support::blend_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512VL &&
                          internal::avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm_mask_mov_epi8(v1, blend_mask, v2);
            }
            else {
                // SSE4.1
                return _mm_blendv_epi8(
                    v1,
                    v2,
                    build_set_vec_wrapper<0>(
                        typename vop_support::blend_vec_initialize{}));
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512VL &&
                          internal::avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
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
            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512F &&
                          internal::avail_instructions::AVX512VL) {
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
    vec_permutate_manual(__m128i v) {
        using vop_support = internal::shuffle_support<T, n, e...>;

        constexpr uint64_t shuffle_mask = vop_support::shuffle_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // SSE3
            return _mm_shuffle_epi8(
                v,
                build_set_vec_wrapper<0>(
                    typename vop_support::shuffle_vec_initialize{}));
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (shuffle_mask) {
                constexpr uint32_t shuffle_mask_lo = shuffle_mask;
                constexpr uint32_t shuffle_mask_hi = (shuffle_mask >> 32);

                // SSE2
                return _mm_shufflehi_epi16(
                    _mm_shufflelo_epi16(v, shuffle_mask_lo),
                    shuffle_mask_hi);
            }
            else {
                return _mm_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename vop_support::shuffle_vec_initialize{}));
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

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    vec_permutate(__m128i v) {
        if constexpr (avail_instructions::BUILTIN_SHUFFLE &&
                      builtin_perm == builtin_usage::BUILTIN_FIRST) {
            return builtin_shuffle<e...>(v);
        }
        else {
            return vec_permutate_manual<e...>(v);
        }
    }
};


template<typename T, simd_instructions simd_set, builtin_usage builtin_perm>
struct vector_ops<T, simd_set, builtin_perm, sizeof(__m256i)> {

    static constexpr uint32_t vec_size = sizeof(__m256i);
    static constexpr uint32_t n        = vec_size / sizeof(T);


    template<uint32_t... e>
    static __m256i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle_impl(__m256i v) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector(
                (internal::vec_types::vec32x1)v,
                (internal::vec_types::vec32x1)v,
                e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (internal::vec_types::vec32x1)v,
                (internal::vec_types::vec32x1)_mm256_set_epi8(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector(
                (internal::vec_types::vec16x2)v,
                (internal::vec_types::vec16x2)v,
                e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (internal::vec_types::vec16x2)v,
                (internal::vec_types::vec16x2)_mm256_set_epi16(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector(
                (internal::vec_types::vec8x4)v,
                (internal::vec_types::vec8x4)v,
                e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (internal::vec_types::vec8x4)v,
                (internal::vec_types::vec8x4)_mm256_set_epi32(e...));
#else
            return v;
#endif
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector(
                (internal::vec_types::vec4x8)v,
                (internal::vec_types::vec4x8)v,
                e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (internal::vec_types::vec4x8)v,
                (internal::vec_types::vec4x8)_mm256_set_epi64x(e...));
#else
            return v;
#endif
        }
    }

    template<uint32_t... e, uint32_t... seq>
    static __m256i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle_reverse(__m256i                                 v,
                            std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr uint32_t arr[n] = { static_cast<uint32_t>(e)... };
        return builtin_shuffle_impl<arr[(n - 1) - seq]...>(v);
    }

    template<uint32_t... e>
    static __m256i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle(__m256i v) {
#if defined(__clang__)
        return builtin_shuffle_reverse<e...>(
            v,
            std::make_integer_sequence<uint32_t, n>{});
#elif defined(__GNUC__)
        return builtin_shuffle_impl<e...>(v);
#else
        return v;
#endif
    }

    template<uint32_t size, uint32_t... e>
    static constexpr __m256i ALWAYS_INLINE CONST_ATTR
    build_set_vec() {
        if constexpr ((size == sizeof(uint8_t)) ||
                      ((size == 0) && (sizeof(T) == sizeof(uint8_t)))) {
            // SSE2
            return _mm256_set_epi8(e...);
        }
        else if constexpr ((size == sizeof(uint16_t)) ||
                           ((size == 0) && (sizeof(T) == sizeof(uint16_t)))) {
            // SSE2
            return _mm256_set_epi16(e...);
        }
        else if constexpr ((size == sizeof(uint32_t)) ||
                           ((size == 0) && (sizeof(T) == sizeof(uint32_t)))) {
            // SSE2
            return _mm256_set_epi32(e...);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // SSE2
            return _mm256_set_epi64x(e...);
        }
    }

    template<uint32_t size, uint32_t... e>
    static constexpr __m256i ALWAYS_INLINE CONST_ATTR
    build_set_vec_wrapper(std::integer_sequence<uint32_t, e...> _e) {
        return build_set_vec<size, e...>();
    }


    static __m256i ALWAYS_INLINE CONST_ATTR
    vec_min(__m256i v1, __m256i v2) {

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX2
                return _mm256_min_epi8(v1, v2);
            }
            else {
                // AVX2
                return _mm256_min_epu8(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX2
                return _mm256_min_epi16(v1, v2);
            }
            else {
                // AVX2
                return _mm256_min_epu16(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX2
                return _mm256_min_epi32(v1, v2);
            }
            else {
                // AVX2
                return _mm256_min_epu32(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {

            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512F &&
                          internal::avail_instructions::AVX512VL) {
                if constexpr (std::is_signed<T>::value) {
                    // AVX512F & AVX512VL
                    return _mm256_min_epi64(v1, v2);
                }
                else {
                    // AVX512F & AVX512VL
                    return _mm256_min_epu64(v1, v2);
                }
            }
            else {
                if constexpr (std::is_signed<T>::value) {
                    // AVX2
                    __m256i cmp_mask = _mm256_cmpgt_epi64(v1, v2);
                    // AVX2
                    return _mm256_blendv_epi8(v1, v2, cmp_mask);
                }
                else {
                    // AVX
                    __m256i sign_bits = _mm256_set1_epi64x((1UL) << 63);
                    // AVX2
                    __m256i cmp_mask =
                        _mm256_cmpgt_epi64(_mm256_xor_si256(v1, sign_bits),
                                           _mm256_xor_si256(v2, sign_bits));
                    // AVX2
                    return _mm256_blendv_epi8(v1, v2, cmp_mask);
                }
            }
        }
    }


    static __m256i ALWAYS_INLINE CONST_ATTR
    vec_max(__m256i v1, __m256i v2) {

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX2
                return _mm256_max_epi8(v1, v2);
            }
            else {
                // AVX2
                return _mm256_max_epu8(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX2
                return _mm256_max_epi16(v1, v2);
            }
            else {
                // AVX2
                return _mm256_max_epu16(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX2
                return _mm256_max_epi32(v1, v2);
            }
            else {
                // AVX2
                return _mm256_max_epu32(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {

            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512F &&
                          internal::avail_instructions::AVX512VL) {
                if constexpr (std::is_signed<T>::value) {
                    // AVX512F & AVX512VL
                    return _mm256_max_epi64(v1, v2);
                }
                else {
                    // AVX512F & AVX512VL
                    return _mm256_max_epu64(v1, v2);
                }
            }
            else {
                if constexpr (std::is_signed<T>::value) {
                    // AVX2
                    __m256i cmp_mask = _mm256_cmpgt_epi64(v1, v2);
                    // AVX2
                    return _mm256_blendv_epi8(v2, v1, cmp_mask);
                }
                else {
                    // AVX
                    __m256i sign_bits = _mm256_set1_epi64x((1UL) << 63);
                    // AVX2
                    __m256i cmp_mask =
                        _mm256_cmpgt_epi64(_mm256_xor_si256(v1, sign_bits),
                                           _mm256_xor_si256(v2, sign_bits));


                    // AVX2
                    return _mm256_blendv_epi8(v2, v1, cmp_mask);
                }
            }
        }
    }

    template<uint32_t... e>
    static __m256i ALWAYS_INLINE CONST_ATTR
    vec_blend(__m256i v1, __m256i v2) {
        using vop_support = internal::blend_support<T, n, e...>;

        constexpr uint64_t blend_mask = vop_support::blend_mask;
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (simd_set >= simd_instructions::AVX512 &&
                          internal::avail_instructions::AVX512VL &&
                          internal::avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm256_mask_mov_epi8(v1, blend_mask, v2);
            }
            else {
                // AVX2
                return _mm256_blendv_epi8(
                    v1,
                    v2,
                    build_set_vec_wrapper<0>(
                        typename vop_support::blend_vec_initialize{}));
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr ((blend_mask & 0xff) == ((blend_mask >> 8) & 0xff)) {
                // epi16 blend uses mask_index % 8 so needs to be mirror for
                // this to work
                // AVX2
                return _mm256_blend_epi16(v1, v2, blend_mask & 0xff);
            }
            else if constexpr (simd_set >= simd_instructions::AVX512 &&
                               internal::avail_instructions::AVX512VL &&
                               internal::avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm256_mask_mov_epi16(v1, blend_mask, v2);
            }

            else {
                // AVX2
                return _mm256_blendv_epi8(
                    v1,
                    v2,
                    build_set_vec_wrapper<1>(
                        typename vop_support::blend_vec_initialize{}));
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // AVX2
            return _mm256_blend_epi32(v1, v2, blend_mask);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // AVX2
            return _mm256_blend_epi32(v1, v2, blend_mask);
        }
    }


    template<uint32_t... e>
    static __m256i ALWAYS_INLINE CONST_ATTR
    vec_permutate_manual(__m256i v) {
        using vop_support = internal::shuffle_support<T, n, e...>;

        constexpr uint64_t shuffle_mask = vop_support::shuffle_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // gcc misses some optimizations
            if constexpr (shuffle_mask) {
                // AVX2
                return _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<0>(
                        typename vop_support::shuffle_vec_initialize{}));
            }
            else if constexpr (simd_set >= simd_instructions::AVX512 &&
                               internal::avail_instructions::AVX512VL &&
                               internal::avail_instructions::AVX512VBMI) {
                // AVX512VL & AVX512VBMI
                return _mm256_permutexvar_epi8(_mm256_set_epi8(e...), v);
            }
            // this is true if all movement is within lane

            else if constexpr (internal::avail_instructions::BUILTIN_SHUFFLE &&
                               builtin_perm ==
                                   builtin_usage::BUILTIN_FALLBACK) {
                return builtin_shuffle<e...>(v);
            }
            else {

                // neither GCC, clang, or AVX512 so need to implement perm
                // manually

                // AVX2
                __m256i lo_hi_swap = _mm256_permute4x64_epi64(v, 0x4e);
                // AVX2
                __m256i same_lane = _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<0>(
                        typename shuffle_across_lane_support<T, n, e...>::
                            across_lanes_same_vec_initialize{}));
                __m256i other_lane = _mm256_shuffle_epi8(
                    lo_hi_swap,
                    build_set_vec_wrapper<0>(
                        typename shuffle_across_lane_support<T, n, e...>::
                            across_lanes_other_vec_initialize{}));
                // AVX2
                return _mm256_or_si256(same_lane, other_lane);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (shuffle_mask) {
                constexpr uint32_t shuffle_mask_lo = shuffle_mask;
                constexpr uint32_t shuffle_mask_hi = (shuffle_mask >> 32);

                // AVX2
                return _mm256_shufflehi_epi16(
                    _mm256_shufflelo_epi16(v, shuffle_mask_lo),
                    shuffle_mask_hi);
            }
            else if constexpr (shuffle_inlane_support<T, n, e...>::
                                   in_same_lanes) {
                // AVX2
                return _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename vop_support::shuffle_vec_initialize{}));
            }
            else if constexpr (simd_set >= simd_instructions::AVX512 &&
                               internal::avail_instructions::AVX512VL &&
                               internal::avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm256_permutexvar_epi16(_mm256_set_epi16(e...), v);
            }


            else if constexpr (internal::avail_instructions::BUILTIN_SHUFFLE &&
                               builtin_perm ==
                                   builtin_usage::BUILTIN_FALLBACK) {
                return builtin_shuffle<e...>(v);
            }
            else {
                // AVX2
                __m256i lo_hi_swap = _mm256_permute4x64_epi64(v, 0x4e);

                // AVX2
                __m256i same_lane = _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename shuffle_across_lane_support<T, n, e...>::
                            across_lanes_same_vec_initialize{}));
                __m256i other_lane = _mm256_shuffle_epi8(
                    lo_hi_swap,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename shuffle_across_lane_support<T, n, e...>::
                            across_lanes_other_vec_initialize{}));


                // AVX2
                return _mm256_or_si256(same_lane, other_lane);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (shuffle_mask) {
                // AVX2
                return _mm256_shuffle_epi32(v, shuffle_mask);
            }
            else {
                // AVX2
                return _mm256_permutevar8x32_epi32(v, _mm256_set_epi32(e...));
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // permute4x64 can fully rearrange epi64
            static_assert(shuffle_mask);

            // AVX2
            return _mm256_permute4x64_epi64(v, shuffle_mask);
        }
    }

    template<uint32_t... e>
    static __m256i ALWAYS_INLINE CONST_ATTR
    vec_permutate(__m256i v) {
        if constexpr (avail_instructions::BUILTIN_SHUFFLE &&
                      builtin_perm == builtin_usage::BUILTIN_FIRST) {
            return builtin_shuffle<e...>(v);
        }
        else {
            return vec_permutate_manual<e...>(v);
        }
    }
};

template<typename T, simd_instructions simd_set, builtin_usage builtin_perm>
struct vector_ops<T, simd_set, builtin_perm, sizeof(__m512i)> {

    static constexpr uint32_t vec_size = sizeof(__m512i);
    static constexpr uint32_t n        = vec_size / sizeof(T);

    template<uint32_t... e>
    static __m512i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle_impl(__m512i v) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector(
                (internal::vec_types::vec64x1)v,
                (internal::vec_types::vec64x1)v,
                e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (internal::vec_types::vec64x1)v,
                (internal::vec_types::vec64x1)_mm512_set_epi8(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector(
                (internal::vec_types::vec32x2)v,
                (internal::vec_types::vec32x2)v,
                e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (internal::vec_types::vec32x2)v,
                (internal::vec_types::vec32x2)_mm512_set_epi16(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector(
                (internal::vec_types::vec16x4)v,
                (internal::vec_types::vec16x4)v,
                e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (internal::vec_types::vec16x4)v,
                (internal::vec_types::vec16x4)_mm512_set_epi32(e...));
#else
            return v;
#endif
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector(
                (internal::vec_types::vec8x8)v,
                (internal::vec_types::vec8x8)v,
                e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (internal::vec_types::vec8x8)v,
                (internal::vec_types::vec8x8)_mm512_set_epi64(e...));
#else
            return v;
#endif
        }
    }

    template<uint32_t... e, uint32_t... seq>
    static __m512i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle_reverse(__m512i                                 v,
                            std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr uint32_t arr[n] = { static_cast<uint32_t>(e)... };
        return builtin_shuffle_impl<arr[(n - 1) - seq]...>(v);
    }

    template<uint32_t... e>
    static __m512i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle(__m512i v) {
#if defined(__clang__)
        return builtin_shuffle_reverse<e...>(
            v,
            std::make_integer_sequence<uint32_t, n>{});
#elif defined(__GNUC__)
        return builtin_shuffle_impl<e...>(v);
#else
        return v;
#endif
    }

    template<uint32_t size, uint32_t... e>
    static constexpr __m512i ALWAYS_INLINE CONST_ATTR
    build_set_vec() {
        if constexpr ((size == sizeof(uint8_t)) ||
                      ((size == 0) && (sizeof(T) == sizeof(uint8_t)))) {
            // SSE2
            return _mm512_set_epi8(e...);
        }
        else if constexpr ((size == sizeof(uint16_t)) ||
                           ((size == 0) && (sizeof(T) == sizeof(uint16_t)))) {
            // SSE2
            return _mm512_set_epi16(e...);
        }
        else if constexpr ((size == sizeof(uint32_t)) ||
                           ((size == 0) && (sizeof(T) == sizeof(uint32_t)))) {
            // SSE2
            return _mm512_set_epi32(e...);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // SSE2
            return _mm512_set_epi64(e...);
        }
    }

    template<uint32_t size, uint32_t... e>
    static constexpr __m512i ALWAYS_INLINE CONST_ATTR
    build_set_vec_wrapper(std::integer_sequence<uint32_t, e...> _e) {
        return build_set_vec<size, e...>();
    }


    static __m512i ALWAYS_INLINE CONST_ATTR
    vec_min(__m512i v1, __m512i v2) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX512BW
                return _mm512_min_epi8(v1, v2);
            }
            else {
                // AVX512BW
                return _mm512_min_epu8(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX512BW
                return _mm512_min_epi16(v1, v2);
            }
            else {
                // AVX512BW
                return _mm512_min_epu16(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX512F
                return _mm512_min_epi32(v1, v2);
            }
            else {
                // AVX512F
                return _mm512_min_epu32(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            if constexpr (std::is_signed<T>::value) {
                // AVX512F
                return _mm512_min_epi64(v1, v2);
            }
            else {
                // AVX512F
                return _mm512_min_epu64(v1, v2);
            }
        }
    }


    static __m512i ALWAYS_INLINE CONST_ATTR
    vec_max(__m512i v1, __m512i v2) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX512BW
                return _mm512_max_epi8(v1, v2);
            }
            else {
                // AVX512BW
                return _mm512_max_epu8(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX512BW
                return _mm512_max_epi16(v1, v2);
            }
            else {
                // AVX512BW
                return _mm512_max_epu16(v1, v2);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (std::is_signed<T>::value) {
                // AVX512F
                return _mm512_max_epi32(v1, v2);
            }
            else {
                // AVX512F
                return _mm512_max_epu32(v1, v2);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {

            if constexpr (std::is_signed<T>::value) {
                // AVX512F
                return _mm512_max_epi64(v1, v2);
            }
            else {
                // AVX512F
                return _mm512_max_epu64(v1, v2);
            }
        }
    }

    template<uint32_t... e>
    static __m512i ALWAYS_INLINE CONST_ATTR
    vec_blend(__m512i v1, __m512i v2) {
        using vop_support = internal::blend_support<T, n, e...>;

        constexpr uint64_t blend_mask = vop_support::blend_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            //  AVX512BW
            return _mm512_mask_mov_epi8(v1, blend_mask, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            //  AVX512BW
            return _mm512_mask_mov_epi16(v1, blend_mask, v2);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            // AVX512F
            return _mm512_mask_mov_epi32(v1, blend_mask, v2);
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            // AVX512F
            return _mm512_mask_mov_epi64(v1, blend_mask, v2);
        }
    }


    template<uint32_t... e>
    static __m512i ALWAYS_INLINE CONST_ATTR
    vec_permutate_manual(__m512i v) {
        using vop_support = internal::shuffle_support<T, n, e...>;

        constexpr uint64_t shuffle_mask = vop_support::shuffle_mask;
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // AVX512VBMI
            return _mm512_permutexvar_epi8(_mm512_set_epi8(e...), v);
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            // AVX512BW
            return _mm512_permutexvar_epi16(_mm512_set_epi16(e...), v);
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            if constexpr (shuffle_mask) {
                // AVX512F
                return _mm512_shuffle_epi32(v, (_MM_PERM_ENUM)shuffle_mask);
            }
            else {
                // AVX512F
                return _mm512_permutexvar_epi32(_mm512_set_epi32(e...), v);
            }
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            if constexpr (shuffle_mask) {
                // AVX512F
                return _mm512_permutex_epi64(v, shuffle_mask);
            }
            else {
                // AVX512F
                return _mm512_permutexvar_epi64(_mm512_set_epi64(e...), v);
            }
        }
    }

    template<uint32_t... e>
    static __m512i ALWAYS_INLINE CONST_ATTR
    vec_permutate(__m512i v) {
        if constexpr (avail_instructions::BUILTIN_SHUFFLE &&
                      builtin_perm == builtin_usage::BUILTIN_FIRST) {
            return builtin_shuffle<e...>(v);
        }
        else {
            return vec_permutate_manual<e...>(v);
        }
    }
};
}  // namespace internal

template<typename T, uint32_t n>
using vec_t = typename internal::vec_types::get_vec_t<T, n>::type;

template<typename T, uint32_t n>
constexpr vec_t<T, n> ALWAYS_INLINE
vec_load(T * const arr) {
    // compiler will optimize to loadu if n * sizeof(T) == sizeof(vec_t<T, n>)
    vec_t<T, n> r;
    memcpy(&r, arr, n * sizeof(T));
    return r;
}

template<typename T, uint32_t n>
constexpr void ALWAYS_INLINE
vec_store(T * const arr, vec_t<T, n> v) {
    // compiler will optimize to storeu if n * sizeof(T) == sizeof(vec_t<T, n>)
    memcpy(arr, &v, n * sizeof(T));
}

template<typename T,
         uint32_t          n,
         simd_instructions simd_set,
         builtin_usage     builtin_perm,
         uint32_t... e>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
compare_exchange(vec_t<T, n> v) {
    if constexpr (sizeof(T) * n < sizeof(__m64)) {
        using vec_ops =
            typename internal::vector_ops<T, simd_set, builtin_perm, 8>;
        vec_t<T, n> cmp = vec_ops::template vec_permutate<7, 6, 5, 4, e...>(v);
        vec_t<T, n> s_min = vec_ops::vec_min(v, cmp);
        vec_t<T, n> s_max = vec_ops::vec_max(v, cmp);
        vec_t<T, n> ret =
            vec_ops::template vec_blend<7, 6, 5, 4, e...>(s_max, s_min);
        return ret;
    }
    else {
        using vec_ops = typename internal::
            vector_ops<T, simd_set, builtin_perm, sizeof(T) * n>;
        vec_t<T, n> cmp   = vec_ops::template vec_permutate<e...>(v);
        vec_t<T, n> s_min = vec_ops::vec_min(v, cmp);
        vec_t<T, n> s_max = vec_ops::vec_max(v, cmp);
        vec_t<T, n> ret   = vec_ops::template vec_blend<e...>(s_max, s_min);
        return ret;
    }
}


}  // namespace vop

}  // namespace vsort
#endif
