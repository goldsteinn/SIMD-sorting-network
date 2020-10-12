#ifndef _NETWORKS_H_
#define _NETWORKS_H_

#include <algorithms/transformations.h>
#include <stdint.h>
#include <util/integer_range.h>

namespace vsort {
namespace internal {
template<uint32_t n>
struct bitonic_network {


    template<uint32_t lo, uint32_t s, uint32_t offset, uint32_t order>
    static constexpr decltype(auto)
    bitonic_create_pairs() {
        if constexpr (s != lo) {
            if constexpr (order) {
                return merge<uint32_t>(
                    std::integer_sequence<uint32_t, lo, lo + offset>{},
                    bitonic_create_pairs<lo + 1, s, offset, order>());
            }
            else {
                return merge<uint32_t>(
                    std::integer_sequence<uint32_t, lo + offset, lo>{},
                    bitonic_create_pairs<lo + 1, s, offset, order>());
            }
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t lo, uint32_t s, uint32_t order>
    static constexpr decltype(auto)
    bitonic_merge() {
        if constexpr (s > 1) {
            constexpr uint32_t m = next_p2(s) >> 1;
            constexpr auto     _new_pairs =
                bitonic_create_pairs<lo, (lo + s) - (m), m, order>();
            constexpr auto _pairs_lo = bitonic_merge<lo, m, order>();
            constexpr auto _pairs_hi = bitonic_merge<lo + m, s - m, order>();
            return merge<uint32_t>(_new_pairs,
                                   merge<uint32_t>(_pairs_lo, _pairs_hi));
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t lo, uint32_t s, uint32_t order>
    static constexpr decltype(auto)
    bitonic_sort_kernel() {
        if constexpr (s > 1) {
            constexpr uint32_t m         = s >> 1;
            constexpr auto     _pairs_lo = bitonic_sort_kernel<lo, m, !order>();
            constexpr auto     _pairs_hi =
                bitonic_sort_kernel<lo + m, s - m, order>();
            return merge<uint32_t>(merge<uint32_t>(_pairs_lo, _pairs_hi),
                                   bitonic_merge<lo, s, order>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }


    static constexpr decltype(auto)
    bitonic_sort() {
        return bitonic_sort_kernel<0, n, 1>();
    }

    using network =
        typename transform::unidirectional<n, decltype(bitonic_sort())>::type;
};


}  // namespace internal


template<uint32_t n>
struct bitonic {
    using network = typename transform::
        build<n, typename internal::bitonic_network<n>::network>::type;
};

static_assert(equal_sequences<>(
    std::integer_sequence<uint32_t, 0, 1, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3>{},
    internal::bitonic_network<4>::network{}));


static_assert(
    std::is_same<
        std::integer_sequence<uint32_t, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3, 0, 1>,
        typename transform::build<
            4,
            typename internal::bitonic_network<4>::network>::type>::value);

static_assert(equal_sequences<>(std::integer_sequence<uint32_t,
                                                      // clang-format off
                                    0, 1, 2, 3, 0, 3, 1, 2,
                                    2, 3, 0, 1, 4, 5, 6, 7,
                                    5, 6, 4, 7, 4, 5, 6, 7,
                                    3, 4, 2, 5, 1, 6, 0, 7,
                                    1, 3, 0, 2, 0, 1, 2, 3,
                                    4, 6, 5, 7, 4, 5, 6, 7
                                                      // clang-format on
                                                      >{},
                                internal::bitonic_network<8>::network{}));


static_assert(equal_sequences<>(
    std::integer_sequence<uint32_t,
                          // clang-format off
                      0, 1, 2, 3, 4, 5, 6, 7,
                      0, 3, 1, 2, 5, 6, 4, 7,
                      2, 3, 0, 1, 4, 5, 6, 7,
                      3, 4, 2, 5, 1, 6, 0, 7,
                      1, 3, 0, 2, 4, 6, 5, 7,
                      0, 1, 2, 3, 4, 5, 6, 7
                          // clang-format on
                          >{},
    typename transform::
        group<8, typename internal::bitonic_network<8>::network>::type{}));

static_assert(equal_sequences<>(
    std::integer_sequence<uint32_t,
                          // clang-format off
                      6, 7, 4, 5, 2, 3, 0, 1,
                      4, 5, 6, 7, 0, 1, 2, 3,
                      6, 7, 4, 5, 2, 3, 0, 1,
                      0, 1, 2, 3, 4, 5, 6, 7,
                      5, 4, 7, 6, 1, 0, 3, 2,
                      6, 7, 4, 5, 2, 3, 0, 1
                          // clang-format on
                          >{},
    typename transform::
        build<8, typename internal::bitonic_network<8>::network>::type{}));


static_assert(
    std::is_same<std::integer_sequence<uint32_t,
                                       // clang-format off
        0, 1, 3, 4,
        2, 4, 2, 3,
        1, 4, 1, 2,
        0, 3, 0, 1,
        2, 3
                                       // clang-format on
                                       >,
                 typename internal::bitonic_network<5>::network>::value);

static_assert(std::is_same<
              std::integer_sequence<uint32_t,
                                    // clang-format off
                  1, 0, 3, 4,
                  2, 4, 2, 3,
                  0, 4, 0, 2,
                  1, 3, 0, 1,
                  2, 3
                                    // clang-format on
                                    >,
              decltype(internal::bitonic_network<5>::bitonic_sort())>::value);


static_assert(
    std::is_same<std::integer_sequence<uint32_t,
                                       // clang-format off
        1, 2, 0, 1,
        1, 2, 4, 5,
        3, 5, 3, 4,
        2, 4, 1, 5,
        0, 2, 1, 3,
        0, 1, 2, 3,
        4, 5
                                       // clang-format on
                                       >,
                 typename internal::bitonic_network<6>::network>::value);

static_assert(
    std::is_same<std::integer_sequence<uint32_t,
                                       // clang-format off
        1, 2, 0, 1,
        1, 2, 3, 4,
        5, 6, 4, 5,
        3, 6, 3, 4,
        5, 6, 2, 4,
        1, 5, 0, 6,
        0, 2, 1, 3,
        0, 1, 2, 3,
        4, 6, 4, 5
                                       // clang-format on
                                       >,
                 typename internal::bitonic_network<7>::network>::value);

static_assert(
    std::is_same<std::integer_sequence<uint32_t,
                                       // clang-format off
        1, 2, 0, 2,
        0, 1, 4, 5,
        3, 4, 4, 5,
        0, 4, 1, 3,
        2, 4, 3, 5,
        4, 5, 2, 3,
        0, 1, 7, 8,
        6, 7, 7, 8,
        9, 10, 11, 12,
        10, 11, 9, 12,
        9, 10, 11, 12,
        8, 10, 7, 11,
        6, 12, 6, 8,
        7, 9, 6, 7,
        8, 9, 10, 12,
        10, 11, 5, 8,
        4, 9, 3, 10,
        2, 11, 1, 12,
        1, 5, 0, 4,
        3, 6, 2, 7,
        1, 3, 0, 2,
        0, 1, 2, 3,
        5, 6, 4, 7,
        4, 5, 6, 7,
        8, 12, 8, 10,
        9, 11, 8, 9,
        10, 11
                                       // clang-format on
                                       >,
                 typename internal::bitonic_network<13>::network>::value);

static_assert(
    std::is_same<std::integer_sequence<uint32_t,
                                       // clang-format off
    1, 2, 4, 5,
    7, 8, 9, 10,
    11, 12, 0, 2,
    3, 4, 6, 7,
    10, 11, 9, 12,
    0, 1, 4, 5,
    7, 8, 9, 10,
    11, 12, 0, 4,
    1, 3, 8, 10,
    7, 11, 6, 12,
    2, 4, 3, 5,
    0, 1, 6, 8,
    7, 9, 10, 12,
    4, 5, 2, 3,
    6, 7, 8, 9,
    10, 11, 1, 12,
    5, 8, 4, 9,
    3, 10, 2, 11,
    1, 5, 0, 4,
    3, 6, 2, 7,
    8, 12, 9, 11,
    1, 3, 0, 2,
    5, 6, 4, 7,
    8, 10, 0, 1,
    2, 3, 4, 5,
    6, 7, 8, 9,
    10, 11
                                       // clang-format on
                                       >,
                 typename transform::group<13,
                                           typename internal::bitonic_network<
                                               13>::network>::type>::value);


}  // namespace vsort

#endif
