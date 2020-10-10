#ifndef _NETWORKS_H_
#define _NETWORKS_H_

#include <algorithms/transformations.h>
#include <stdint.h>
#include <util/integer_range.h>

namespace vsort {
namespace internal {
template<uint32_t n>
struct bitonic_network {


    template<uint32_t begin, uint32_t end, uint32_t offset, uint32_t order>
    static constexpr decltype(auto)
    bitonic_create_pairs() {
        if constexpr (end != begin) {
            if constexpr (order) {
                return merge<uint32_t>(
                    std::integer_sequence<uint32_t, begin, begin + offset>{},
                    bitonic_create_pairs<begin + 1, end, offset, order>());
            }
            else {
                return merge<uint32_t>(
                    std::integer_sequence<uint32_t, begin + offset, begin>{},
                    bitonic_create_pairs<begin + 1, end, offset, order>());
            }
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t begin, uint32_t end, uint32_t order>
    static constexpr decltype(auto)
    bitonic_merge() {
        if constexpr (end - begin > 1) {
            constexpr auto _new_pairs =
                bitonic_create_pairs<begin,
                                     begin + ((end - begin) / 2),
                                     (end - begin) / 2,
                                     order>();
            constexpr auto _pairs_lo =
                bitonic_merge<begin, begin + ((end - begin) / 2), order>();
            constexpr auto _pairs_hi =
                bitonic_merge<begin + ((end - begin) / 2), end, order>();
            return merge<uint32_t>(_new_pairs,
                                   merge<uint32_t>(_pairs_lo, _pairs_hi));
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t begin, uint32_t end, uint32_t order>
    static constexpr decltype(auto)
    bitonic_sort_kernel() {
        if constexpr (end - begin > 1) {
            constexpr auto _pairs_lo =
                bitonic_sort_kernel<begin,
                                    begin + ((end - begin) / 2),
                                    !order>();
            constexpr auto _pairs_hi =
                bitonic_sort_kernel<begin + ((end - begin) / 2), end, order>();
            return merge<uint32_t>(merge<uint32_t>(_pairs_lo, _pairs_hi),
                                   bitonic_merge<begin, end, order>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }


    static constexpr decltype(auto)
    bitonic_sort() {
        return bitonic_sort_kernel<0, n, 1>();
    }

    using raw_network = decltype(bitonic_sort());
    using network = typename transform::unidirectional<n, raw_network>::type;
};


}  // namespace internal


template<uint32_t n>
struct bitonic {
    using network = typename internal::bitonic_network<n>::network;
};

static_assert(equal_sequences<>(
    std::integer_sequence<uint32_t, 0, 1, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3>{},
    bitonic<4>::network{}));

static_assert(
    std::is_same<
        std::integer_sequence<uint32_t, 0, 1, 2, 3, 1, 2, 0, 3, 0, 1, 2, 3>,
        typename bitonic<4>::network>::value);

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
                                bitonic<8>::network{}));


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
    typename transform::group<8, typename bitonic<8>::network>::type{}));

}  // namespace vsort

#endif
