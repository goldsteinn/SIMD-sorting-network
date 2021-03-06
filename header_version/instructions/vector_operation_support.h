#ifndef _VECTOR_OPERATION_SUPPORT_H_
#define _VECTOR_OPERATION_SUPPORT_H_

#include <immintrin.h>
#include <stdint.h>
#include <utility>

#include <util/constexpr_util.h>
#include <util/cpp_attributes.h>
#include <util/integer_range.h>

namespace vsort {
namespace vop {
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
    typedef uint8_t vec16x1 __attribute__((vector_size(16)));
    typedef uint8_t vec32x1 __attribute__((vector_size(32)));
    typedef uint8_t vec64x1 __attribute__((vector_size(64)));

    typedef uint16_t vec8x2 __attribute__((vector_size(16)));
    typedef uint16_t vec16x2 __attribute__((vector_size(32)));
    typedef uint16_t vec32x2 __attribute__((vector_size(64)));

    typedef uint32_t vec4x4 __attribute__((vector_size(16)));
    typedef uint32_t vec8x4 __attribute__((vector_size(32)));
    typedef uint32_t vec16x4 __attribute__((vector_size(64)));

    typedef uint64_t          vec2x8 __attribute__((vector_size(16)));
    typedef uint64_t          vec4x8 __attribute__((vector_size(32)));
    typedef uint64_t          vec8x8 __attribute__((vector_size(64)));
#endif
    struct __m64_wrapper {
        typedef __m64 type;
    } ALIGN_ATTR(sizeof(__m64));

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
        typename std::conditional_t<
            n * sizeof(T) <= 16,
            typename std::conditional_t<n * sizeof(T) <= 8,
                                        __m64_wrapper,
                                        __m128_wrapper>,
            __m256_wrapper>,
        __m512_wrapper>;
};


template<typename T, uint32_t n, uint32_t... e>
struct vector_ops_support_impl {
    static constexpr uint64_t shuffle_as_epi32_flag = (1UL) << 63;

    template<uint32_t factor, uint32_t... seq>
    static constexpr decltype(auto)
    expand_seq_kernel(std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr uint32_t _e[sizeof...(e)] = { static_cast<uint32_t>(e)... };
        return std::integer_sequence<uint32_t,
                                     factor * _e[seq / factor] +
                                         ((factor - 1) - (seq % factor))...>{};
    }

    template<uint32_t factor>
    static constexpr decltype(auto)
    expand_seq() {
        return expand_seq_kernel<factor>(
            std::make_integer_sequence<uint32_t, n * factor>{});
    }

    template<uint32_t factor, uint32_t... seq>
    static constexpr decltype(auto)
    shrink_seq_kernel(std::integer_sequence<uint32_t, seq...> _seq) {
        constexpr uint32_t _e[sizeof...(e)] = { static_cast<uint32_t>(e)... };
        return std::integer_sequence<uint32_t,
                                     (_e[seq * factor] / factor)...>{};
    }

    template<uint32_t factor>
    static constexpr decltype(auto)
    shrink_seq() {
        return shrink_seq_kernel<factor>(
            std::make_integer_sequence<uint32_t, n / factor>{});
    }

    template<typename fromT, typename toT>
    static constexpr decltype(auto)
    scale_seq() {
        if constexpr (sizeof(fromT) > sizeof(toT)) {
            return expand_seq<sizeof(fromT) / sizeof(toT)>();
        }
        else if constexpr (sizeof(fromT) < sizeof(toT)) {
            return shrink_seq<sizeof(toT) / sizeof(fromT)>();
        }
        else {
            return std::integer_sequence<uint32_t, e...>{};
        }
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
                    if constexpr (n * sizeof(T) <= sizeof(__m64)) {
                        blend_mask |= ((1UL << (8 * sizeof(T))) - 1UL)
                                      << (i * 8 * sizeof(T));
                    }
                    else {
                        // blend_mask for x/y/zmm register
                        blend_mask |= ((1UL) << i);
                    }
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
        if constexpr (sizeof(T) < sizeof(uint32_t)) {
            return build_blend_vec_initializer_kernel<blend_mask>(
                std::make_integer_sequence<uint32_t, sizeof(T) * n>{});
        }
        else {
            return 0;
        }
    }

    template<uint32_t offset,
             uint32_t ele_per_lane,
             uint32_t lane_size,
             uint32_t... _e>
    static constexpr uint64_t
    build_shuffle_mask_impl() {
        constexpr uint32_t _n         = sizeof...(_e);
        constexpr uint32_t perms[_n]  = { static_cast<uint32_t>(_e)... };
        constexpr uint32_t ele_offset = offset == sizeof(T) * _n ? 0 : offset;
        constexpr uint32_t in_lanes_check = offset == sizeof(T) * _n;
        uint64_t           mask           = 0;
        for (uint32_t i = 0; i < _n; i += lane_size) {
            uint64_t lane_mask   = 0;
            uint32_t lower_bound = _n - (i + ele_per_lane + ele_offset);
            uint32_t upper_bound = _n - (i + ele_offset);
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

    template<uint32_t... shrunk_e>
    static constexpr uint64_t
    build_shuffle_mask_as_epi32_impl_wrapper(
        std::integer_sequence<uint32_t, shrunk_e...> _shrunk_e) {
        return build_shuffle_mask_impl<0, 4, 4, shrunk_e...>();
    }

    static constexpr uint64_t
    build_shuffle_mask_as_epi32() {
        constexpr uint32_t perms[n] = { static_cast<uint32_t>(e)... };
        constexpr uint32_t T_per_int =
            (sizeof(T) > sizeof(uint32_t)) ? 1 : (sizeof(uint32_t) / sizeof(T));
        for (uint32_t i = 0; i < n; i += T_per_int) {
            uint32_t base_ele = perms[i];
            if ((base_ele % T_per_int) != (T_per_int - 1)) {
                return 0;
            }
            for (uint32_t j = 1; j < (T_per_int); j++) {
                if (perms[i + j] != (base_ele - j)) {
                    return 0;
                }
            }
        }
        return build_shuffle_mask_as_epi32_impl_wrapper(
            scale_seq<T, uint32_t>());
    }


    static constexpr decltype(auto)
    build_shuffle_vec_initializer() {
        return scale_seq<T, uint8_t>();
    }


    static constexpr uint64_t
    build_shuffle_mask() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            constexpr uint64_t shuffle_mask_epi32 =
                build_shuffle_mask_as_epi32();
            if constexpr (shuffle_mask_epi32) {
                return shuffle_mask_epi32 | shuffle_as_epi32_flag;
            }
            else {
                return !!build_shuffle_mask_impl<0, 16, 16, e...>();
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            if constexpr (n <= 4) {
                return build_shuffle_mask_impl<0, 4, 4, e...>();
            }
            else {
                constexpr uint64_t shuffle_mask_epi32 =
                    build_shuffle_mask_as_epi32();
                if constexpr (shuffle_mask_epi32) {
                    return shuffle_mask_epi32 | shuffle_as_epi32_flag;
                }
                else {
                    constexpr uint64_t shuffle_mask_lo =
                        build_shuffle_mask_impl<4, 4, 8, e...>();
                    constexpr uint64_t shuffle_mask_hi =
                        build_shuffle_mask_impl<0, 4, 8, e...>();
                    if constexpr (shuffle_mask_lo == 0 ||
                                  shuffle_mask_hi == 0) {
                        return 0;
                    }
                    else {
                        return shuffle_mask_lo | (shuffle_mask_hi << 32);
                    }
                }
            }
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return build_shuffle_mask_impl<0, 4, 4, e...>();
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            constexpr uint64_t shuffle_mask_epi32 =
                build_shuffle_mask_as_epi32();
            if constexpr (shuffle_mask_epi32) {
                return shuffle_mask_epi32 | shuffle_as_epi32_flag;
            }
            else {
                return build_shuffle_mask_impl<0, 4, 4, e...>();
            }
        }
    }

    static constexpr uint64_t
    in_same_lanes() {
        if constexpr (sizeof(T) == sizeof(uint8_t)) {
            return build_shuffle_mask_impl<n * sizeof(T), 16, 16, e...>();
        }
        else if constexpr (sizeof(T) == sizeof(uint16_t)) {
            return build_shuffle_mask_impl<n * sizeof(T), 8, 8, e...>();
        }
        else if constexpr (sizeof(T) == sizeof(uint32_t)) {
            return build_shuffle_mask_impl<n * sizeof(T), 4, 4, e...>();
        }
        else /* sizeof(T) == sizeof(uint64_t) */ {
            return build_shuffle_mask_impl<n * sizeof(T), 2, 2, e...>();
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


template<typename T, uint32_t n, uint32_t... e>
struct blend_support {
    using vop_support_impl = vector_ops_support_impl<T, n, e...>;

    static constexpr uint64_t blend_mask = vop_support_impl::build_blend_mask();

    using blend_vec_initialize = decltype(
        vop_support_impl::template build_blend_vec_initializer<blend_mask>());
};

template<typename T, uint32_t n, uint32_t... e>
struct shuffle_support {
    using vop_support_impl = vector_ops_support_impl<T, n, e...>;
    static constexpr uint64_t shuffle_as_epi32_flag =
        vop_support_impl::shuffle_as_epi32_flag;

    static constexpr uint64_t shuffle_mask =
        vop_support_impl::build_shuffle_mask();

    static constexpr uint64_t in_same_lanes = vop_support_impl::in_same_lanes();

    using shuffle_vec_initialize =
        decltype(vop_support_impl::build_shuffle_vec_initializer());
};


template<typename T, uint32_t n, uint32_t... e>
struct shuffle_across_lane_support {
    using vop_support_impl = vector_ops_support_impl<T, n, e...>;
    static constexpr uint64_t across_lanes_mask =
        vop_support_impl::across_lanes_mask();


    using across_lanes_other_vec_initialize =
        decltype(vop_support_impl::template build_across_lanes_vec_initializer<
                 1,
                 across_lanes_mask>());
    using across_lanes_same_vec_initialize =
        decltype(vop_support_impl::template build_across_lanes_vec_initializer<
                 0,
                 across_lanes_mask>());
};


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
    static constexpr uint32_t BUILTIN_SHUFFLE = CLANG_BUILTIN | GCC_BUILTIN;
};

}  // namespace internal
}  // namespace vop
}  // namespace vsort
#endif
