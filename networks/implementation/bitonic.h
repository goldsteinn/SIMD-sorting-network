#ifndef _BITONIC_NETWORK_H_
#define _BITONIC_NETWORK_H_

#include <stdint.h>

#include <networks/transformations.h>
#include <util/integer_range.h>

namespace vsort {
namespace network {
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
}  // namespace network
}  // namespace vsort
#endif
