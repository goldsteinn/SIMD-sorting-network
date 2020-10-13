#ifndef _INSTRUCTIONS_H_
#define _INSTRUCTIONS_H_

#include <immintrin.h>
#include <stdint.h>
#include <utility>

namespace vop {

enum instruction_set { AVX = 0, AVX2 = 1, AVX512 = 2 };

struct avail_instructions {
#if defined __AVX512F__
    static constexpr uint32_t AVX512F = 1;
#else
    static constexpr uint32_t AVX512F       = 0;
#endif

#if defined __AVX512VL__
    static constexpr uint32_t AVX512VL = 1;
#else
    static constexpr uint32_t AVX512VL      = 0;
#endif

#if defined __AVX512VBMI__
    static constexpr uint32_t AVX512VBMI = 1;
#else
    static constexpr uint32_t AVX512VBMI    = 0;
#endif

#if defined __AVX512BW__
    static constexpr uint32_t AVX512BW = 1;
#else
    static constexpr uint32_t AVX512BW      = 0;
#endif


#if defined __AVX2__
    static constexpr uint32_t AVX2 = 1;
#else
    static constexpr uint32_t AVX2          = 0;
#endif

#if defined __AVX__
    static constexpr uint32_t AVX = 1;
#else
    static constexpr uint32_t AVX           = 0;
#endif

#if defined __SSE2__
    static constexpr uint32_t SSE2 = 1;
#else
    static constexpr uint32_t SSE2          = 0;
#endif

#if defined __SSE3__
    static constexpr uint32_t SSE3 = 1;
#else
    static constexpr uint32_t SSE3          = 0;
#endif

#if defined __SSE4_1__
    static constexpr uint32_t SSE4_1 = 1;
#else
    static constexpr uint32_t SSE4_1        = 0;
#endif

#if defined __SSE4_2__
    static constexpr uint32_t SSE4_2 = 1;
#else
    static constexpr uint32_t SSE4_2        = 0;
#endif

    static constexpr instruction_set instruction_set_default =
        (AVX512F | AVX512VL | AVX512VBMI | AVX512BW) ? instruction_set::AVX512
                                                     : instruction_set::AVX2;


#if defined(__clang__)
    static constexpr uint32_t CLANG_BUILTIN = 1;
    static constexpr uint32_t GCC_BUILTIN   = 0;
#elif defined(__GNUC__)
    static constexpr uint32_t CLANG_BUILTIN = 0;
    static constexpr uint32_t GCC_BUILTIN   = 1;
#else
    static constexpr uint32_t CLANG_BUILTIN = 0;
    static constexpr uint32_t GCC_BUILTIN   = 0;
#endif
};

namespace internal {

struct vec_types {
#if defined(__clang__)
    typedef uint8_t vec16x1 __attribute__((ext_vector_type(16)));
    typedef uint8_t vec32x1 __attribute__((ext_vector_type(32)));
    typedef uint8_t vec64x1 __attribute__((ext_vector_type(64)));

    typedef uint16_t vec8x2 __attribute__((ext_vector_type(8)));
    typedef uint16_t vec16x2 __attribute__((ext_vector_type(16)));
    typedef uint16_t vec32x2 __attribute__((ext_vector_type(32)));

    typedef uint32_t vec4x4 __attribute__((ext_vector_type(4)));
    typedef uint32_t vec8x4 __attribute__((ext_vector_type(8)));
    typedef uint32_t vec16x4 __attribute__((ext_vector_type(16)));

    typedef uint64_t vec2x8 __attribute__((ext_vector_type(2)));
    typedef uint64_t vec4x8 __attribute__((ext_vector_type(4)));
    typedef uint64_t vec8x8 __attribute__((ext_vector_type(8)));

#elif defined(__GNUC__)
    typedef uint8_t           vec16x1 __attribute__((vector_size(16)));
    typedef uint8_t           vec32x1 __attribute__((vector_size(32)));
    typedef uint8_t           vec64x1 __attribute__((vector_size(64)));

    typedef uint16_t vec8x2 __attribute__((vector_size(16)));
    typedef uint16_t vec16x2 __attribute__((vector_size(32)));
    typedef uint16_t vec32x2 __attribute__((vector_size(64)));

    typedef uint32_t vec4x4 __attribute__((vector_size(16)));
    typedef uint32_t vec8x4 __attribute__((vector_size(32)));
    typedef uint32_t vec16x4 __attribute__((vector_size(64)));

    typedef uint64_t vec2x8 __attribute__((vector_size(16)));
    typedef uint64_t vec4x8 __attribute__((vector_size(32)));
    typedef uint64_t vec8x8 __attribute__((vector_size(64)));
#endif

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
    using get_vec_t = typename std::conditional_t<
        n * sizeof(T) <= 32,
        typename std::
            conditional_t<n * sizeof(T) <= 16, __m128_wrapper, __m256_wrapper>,
        __m512_wrapper>;
};


template<typename T, uint32_t n, instruction_set operations, uint32_t... e>
struct vector_ops_support_impl {

    template<uint32_t... seq>
    static constexpr decltype(auto)
    expand_seq_kernel(std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr auto _e = std::integer_sequence<uint32_t, e...>{};
        return std::integer_sequence<
            uint32_t,
            sizeof(T) * get_pos<uint32_t, seq / sizeof(T)>(_e) +
                ((sizeof(T) - 1) - (seq % sizeof(T)))...>{};
    }

    static constexpr decltype(auto)
    expand_seq() {
        return expand_seq_kernel(
            std::make_integer_sequence<uint32_t, n * sizeof(T)>{});
    }

    static constexpr uint64_t
    build_blend_mask() {
        constexpr uint32_t perm[n] = { static_cast<uint32_t>(e)... };

        uint64_t blend_mask = 0;

        for (uint32_t i = 0; i < n; ++i) {
            // if (perm[i] < (n - (i + 1))) {
            if (perm[(n - 1) - i] > i) {
                if constexpr (sizeof(T) < sizeof(uint64_t) ||
                              // for __m512i use normal mask for epi64
                              (n * sizeof(T) > sizeof(__m256i))) {
                    blend_mask |= ((1UL) << i);
                }
                else /* sizeof(T) == sizeof(uint64_t) */ {
                    // blend_epi32 for epi64
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
        return std::integer_sequence<
            uint32_t,
            ((!!(blend_mask & ((1UL) << (n - ((seq / sizeof(T)) + 1)))))
             << 7)...>{};
    }

    // this is for epi8 blend
    template<uint64_t blend_mask>
    static constexpr decltype(auto)
    build_blend_vec_initializer() {
        // will use blend_mask directly unless using epi8 blend in which case we
        // need to scale by sizeof(T)
        return build_blend_vec_initializer_kernel<blend_mask>(
            std::make_integer_sequence<uint32_t, sizeof(T) * n>{});
    }

    template<uint32_t offset, uint32_t ele_per_lane, uint32_t lane_size>
    static constexpr uint64_t
    build_shuffle_mask_impl() {
        constexpr uint32_t perms[n]   = { static_cast<uint32_t>(e)... };
        constexpr uint32_t ele_offset = offset == sizeof(T) * n ? 0 : offset;
        constexpr uint32_t in_lanes_check = offset == sizeof(T) * n;
        uint64_t           mask           = 0;
        for (uint32_t i = 0; i < n; i += lane_size) {
            uint64_t lane_mask   = 0;
            uint32_t lower_bound = n - (i + ele_per_lane + ele_offset);
            uint32_t upper_bound = n - (i + ele_offset);
            for (uint32_t j = 0; j < ele_per_lane; ++j) {
                uint32_t p = perms[i + j + ele_offset];
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
            if ((!in_lanes_check) && i && (mask != lane_mask)) {
                return 0;
            }
            mask = lane_mask;
        }
        return mask;
    }


    static constexpr decltype(auto)
    build_shuffle_vec_initializer() {
        return expand_seq();
    }


    static constexpr uint64_t
    build_shuffle_mask() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return build_shuffle_mask_impl<0, 16, 16>();
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            constexpr uint64_t shuffle_mask_lo =
                build_shuffle_mask_impl<4, 4, 8>();
            constexpr uint64_t shuffle_mask_hi =
                build_shuffle_mask_impl<0, 4, 8>();
            if constexpr (shuffle_mask_lo == 0 || shuffle_mask_hi == 0) {
                return 0;
            }
            else {
                return shuffle_mask_lo | (shuffle_mask_hi << 32);
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return build_shuffle_mask_impl<0, 4, 4>();
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            return build_shuffle_mask_impl<0, 4, 4>();
        }
    }

    static constexpr uint64_t
    in_same_lanes() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return build_shuffle_mask_impl<n * sizeof(T), 16, 16>();
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return build_shuffle_mask_impl<n * sizeof(T), 8, 8>();
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return build_shuffle_mask_impl<n * sizeof(T), 4, 4>();
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            return build_shuffle_mask_impl<n * sizeof(T), 2, 2>();
        }
    }

    template<uint64_t build_cross_lane,
             uint64_t across_lanes_mask,
             uint32_t ele_per_lane,
             uint32_t... seq>
    static constexpr decltype(auto)
    build_across_lanes_vec_initializer_kernel(
        std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr uint32_t perms[n] = { static_cast<uint32_t>(e)... };

        return std::integer_sequence < uint32_t,
               ((across_lanes_mask & ((1UL) << (seq / sizeof(T)))) ==
                (build_cross_lane << (seq / sizeof(T))))
                   ? ((sizeof(T) * (perms[seq / sizeof(T)] % ele_per_lane)) +
                      ((sizeof(T) - 1) - ((seq % sizeof(T)))))

                   : ((1u << 7))... > {};
    }

    template<uint64_t build_cross_lane, uint64_t across_lanes_mask>
    static constexpr decltype(auto)
    build_across_lanes_vec_initializer() {
        return build_across_lanes_vec_initializer_kernel<build_cross_lane,
                                                         across_lanes_mask,
                                                         sizeof(__m128i) /
                                                             sizeof(T)>(
            std::make_integer_sequence<uint32_t, n * sizeof(T)>{});
    }


    template<uint32_t lane_size, uint32_t ele_per_lane>
    static constexpr uint64_t
    across_lanes_mask_impl() {
        constexpr uint32_t perms[n] = { static_cast<uint32_t>(e)... };

        uint64_t across_lanes_mask = 0;
        for (uint32_t i = 0; i < n; i += lane_size) {

            uint32_t lower_bound = n - (i + ele_per_lane);
            uint32_t upper_bound = n - (i);

            for (uint32_t j = 0; j < ele_per_lane; ++j) {
                uint32_t p = perms[i + j];

                // not in lane
                if (!(p >= lower_bound && p < upper_bound)) {
                    across_lanes_mask |= (1UL) << (i + j);
                }
            }
        }
        return across_lanes_mask;
    }

    static constexpr uint64_t
    across_lanes_mask() {
        if constexpr (sizeof(T) == sizeof(uint8_t) ||
                      sizeof(T) == sizeof(uint16_t)) {
            return across_lanes_mask_impl<sizeof(__m128i) / sizeof(T),
                                          sizeof(__m128i) / sizeof(T)>();
        }
        else /* sizeof(T) == sizeof(uint64_t) ||
                sizeof(T) == sizeof(uin32_t) */
        {
            return 0;
        }
    }
};  // namespace internal


template<typename T, uint32_t n, instruction_set operations, uint32_t... e>
struct vector_ops_support {
    using vop_support_impl = vector_ops_support_impl<T, n, operations, e...>;

    static constexpr uint64_t shuffle_mask =
        vop_support_impl::build_shuffle_mask();

    static constexpr uint64_t in_same_lanes = vop_support_impl::in_same_lanes();
    static constexpr uint64_t across_lanes_mask =
        vop_support_impl::across_lanes_mask();

    static constexpr uint64_t blend_mask = vop_support_impl::build_blend_mask();

    using across_lanes_other_vec_initialize =
        decltype(vop_support_impl::template build_across_lanes_vec_initializer<
                 1,
                 across_lanes_mask>());
    using across_lanes_same_vec_initialize =
        decltype(vop_support_impl::template build_across_lanes_vec_initializer<
                 0,
                 across_lanes_mask>());

    using shuffle_vec_initialize =
        decltype(vop_support_impl::build_shuffle_vec_initializer());

    using blend_vec_initialize = decltype(
        vop_support_impl::template build_blend_vec_initializer<blend_mask>());
};


template<typename T, instruction_set operations, uint32_t vec_size>
struct vector_ops;

template<typename T, instruction_set operations>
struct vector_ops<T, operations, sizeof(__m128i)> {
    static constexpr uint32_t vec_size = sizeof(__m128i);
    static constexpr uint32_t n        = vec_size / sizeof(T);

    template<uint32_t... e>
    static __m128i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle(__m128i v) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector((vec_types::vec16x1)v,
                                                    (vec_types::vec16x1)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (vec_types::vec16x1)v,
                (vec_types::vec16x1)_mm_set_epi8(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector((vec_types::vec8x2)v,
                                                    (vec_types::vec8x2)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (vec_types::vec8x2)v,
                (vec_types::vec8x2)_mm_set_epi16(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector((vec_types::vec4x4)v,
                                                    (vec_types::vec4x4)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (vec_types::vec4x4)v,
                (vec_types::vec4x4)_mm_set_epi32(e...));
#else
            return v;
#endif
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
#if defined(__clang__)
            return (__m128i)__builtin_shufflevector((vec_types::vec2x8)v,
                                                    (vec_types::vec2x8)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m128i)__builtin_shuffle(
                (vec_types::vec2x8)v,
                (vec_types::vec2x8)_mm_set_epi64x(e...));
#else
            return v;
#endif
        }
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

            if constexpr (operations >= instruction_set::AVX512 &&
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

            if constexpr (operations >= instruction_set::AVX512 &&
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
        using vop_support = vector_ops_support<T, n, operations, e...>;

        constexpr uint64_t blend_mask = vop_support::blend_mask;

        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (operations >= instruction_set::AVX512 &&
                          avail_instructions::AVX512VL &&
                          avail_instructions::AVX512BW) {
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
            if constexpr (operations >= instruction_set::AVX512 &&
                          avail_instructions::AVX512VL &&
                          avail_instructions::AVX512BW) {
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
            if constexpr (operations >= instruction_set::AVX512 &&
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
        using vop_support = vector_ops_support<T, n, operations, e...>;

        constexpr uint64_t shuffle_mask = vop_support::shuffle_mask;

        if constexpr (avail_instructions::CLANG_BUILTIN) {
            return builtin_shuffle<e...>(v);
        }
        else if constexpr (sizeof(T) == sizeof(uint8_t)) {
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
};


template<typename T, instruction_set operations>
struct vector_ops<T, operations, sizeof(__m256i)> {

    static constexpr uint32_t vec_size = sizeof(__m256i);
    static constexpr uint32_t n        = vec_size / sizeof(T);


    template<uint32_t... e>
    static __m256i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle(__m256i v) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector((vec_types::vec32x1)v,
                                                    (vec_types::vec32x1)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (vec_types::vec32x1)v,
                (vec_types::vec32x1)_mm256_set_epi8(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector((vec_types::vec16x2)v,
                                                    (vec_types::vec16x2)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (vec_types::vec16x2)v,
                (vec_types::vec16x2)_mm256_set_epi16(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector((vec_types::vec8x4)v,
                                                    (vec_types::vec8x4)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (vec_types::vec8x4)v,
                (vec_types::vec8x4)_mm256_set_epi32(e...));
#else
            return v;
#endif
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
#if defined(__clang__)
            return (__m256i)__builtin_shufflevector((vec_types::vec4x8)v,
                                                    (vec_types::vec4x8)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m256i)__builtin_shuffle(
                (vec_types::vec4x8)v,
                (vec_types::vec4x8)_mm256_set_epi64x(e...));
#else
            return v;
#endif
        }
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

            if constexpr (operations >= instruction_set::AVX512 &&
                          avail_instructions::AVX512F &&
                          avail_instructions::AVX512VL) {
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

            if constexpr (operations >= instruction_set::AVX512 &&
                          avail_instructions::AVX512F &&
                          avail_instructions::AVX512VL) {
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
        using vop_support = vector_ops_support<T, n, operations, e...>;

        constexpr uint64_t blend_mask = vop_support::blend_mask;
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            if constexpr (operations >= instruction_set::AVX512 &&
                          avail_instructions::AVX512VL &&
                          avail_instructions::AVX512BW) {
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
            else if constexpr (operations >= instruction_set::AVX512 &&
                               avail_instructions::AVX512VL &&
                               avail_instructions::AVX512BW) {
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
    vec_permutate(__m256i v) {
        using vop_support = vector_ops_support<T, n, operations, e...>;

        constexpr uint64_t shuffle_mask = vop_support::shuffle_mask;

        if constexpr (avail_instructions::CLANG_BUILTIN) {
            return builtin_shuffle<e...>(v);
        }
        else if constexpr (sizeof(T) == sizeof(uint8_t)) {
            // gcc misses some optimizations
            if constexpr (shuffle_mask) {
                // AVX2
                return _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<0>(
                        typename vop_support::shuffle_vec_initialize{}));
            }
            else if constexpr (operations >= instruction_set::AVX512 &&
                               avail_instructions::AVX512VL &&
                               avail_instructions::AVX512VBMI) {
                // AVX512VL & AVX512VBMI
                return _mm256_permutexvar_epi8(_mm256_set_epi8(e...), v);
            }
            // this is true if all movement is within lane

            else if constexpr (avail_instructions::GCC_BUILTIN) {
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
                        typename vop_support::
                            across_lanes_same_vec_initialize{}));
                __m256i other_lane = _mm256_shuffle_epi8(
                    lo_hi_swap,
                    build_set_vec_wrapper<0>(
                        typename vop_support::
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
            else if constexpr (vop_support::in_same_lanes) {
                // AVX2
                return _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename vop_support::shuffle_vec_initialize{}));
            }
            else if constexpr (operations >= instruction_set::AVX512 &&
                               avail_instructions::AVX512VL &&
                               avail_instructions::AVX512BW) {
                // AVX512VL & AVX512BW
                return _mm256_permutexvar_epi16(_mm256_set_epi16(e...), v);
            }


            else if constexpr (avail_instructions::GCC_BUILTIN) {
                return builtin_shuffle<e...>(v);
            }
            else {
                // AVX2
                __m256i lo_hi_swap = _mm256_permute4x64_epi64(v, 0x4e);

                // AVX2
                __m256i same_lane = _mm256_shuffle_epi8(
                    v,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename vop_support::
                            across_lanes_same_vec_initialize{}));
                __m256i other_lane = _mm256_shuffle_epi8(
                    lo_hi_swap,
                    build_set_vec_wrapper<sizeof(uint8_t)>(
                        typename vop_support::
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
};

template<typename T, instruction_set operations>
struct vector_ops<T, operations, sizeof(__m512i)> {

    static constexpr uint32_t vec_size = sizeof(__m512i);
    static constexpr uint32_t n        = vec_size / sizeof(T);

    template<uint32_t... e>
    static __m512i ALWAYS_INLINE CONST_ATTR
    builtin_shuffle(__m512i v) {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector((vec_types::vec64x1)v,
                                                    (vec_types::vec64x1)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (vec_types::vec64x1)v,
                (vec_types::vec64x1)_mm512_set_epi8(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector((vec_types::vec32x2)v,
                                                    (vec_types::vec32x2)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (vec_types::vec32x2)v,
                (vec_types::vec32x2)_mm512_set_epi16(e...));
#else
            return v;
#endif
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector((vec_types::vec16x4)v,
                                                    (vec_types::vec16x4)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (vec_types::vec16x4)v,
                (vec_types::vec16x4)_mm512_set_epi32(e...));
#else
            return v;
#endif
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
#if defined(__clang__)
            return (__m512i)__builtin_shufflevector((vec_types::vec8x8)v,
                                                    (vec_types::vec8x8)v,
                                                    e...);
#elif defined(__GNUC__)
            return (__m512i)__builtin_shuffle(
                (vec_types::vec8x8)v,
                (vec_types::vec8x8)_mm512_set_epi64x(e...));
#else
            return v;
#endif
        }
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
        using vop_support = vector_ops_support<T, n, operations, e...>;

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
    vec_permutate(__m512i v) {
        using vop_support = vector_ops_support<T, n, operations, e...>;

        constexpr uint64_t shuffle_mask = vop_support::shuffle_mask;
        if constexpr (avail_instructions::CLANG_BUILTIN) {
            return builtin_shuffle<e...>(v);
        }
        else if constexpr (sizeof(T) == sizeof(uint8_t)) {
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
};

}  // namespace internal


template<typename T, uint32_t n>
using vec_t = typename internal::vec_types::get_vec_t<T, n>::type;

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

template<typename T, uint32_t n, instruction_set operations, uint32_t... e>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
compare_exchange(vec_t<T, n> v) {
    using vec_ops = typename internal::vector_ops<T, operations, sizeof(T) * n>;
    vec_t<T, n> cmp   = vec_ops::template vec_permutate<e...>(v);
    vec_t<T, n> s_min = vec_ops::vec_min(v, cmp);
    vec_t<T, n> s_max = vec_ops::vec_max(v, cmp);
    vec_t<T, n> ret   = vec_ops::template vec_blend<e...>(s_max, s_min);
    return ret;
}


}  // namespace vop

#endif
