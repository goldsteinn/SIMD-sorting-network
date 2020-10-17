#ifndef _ODDEVEN_NETWORK_H_
#define _ODDEVEN_NETWORK_H_

#include <stdint.h>

#include <networks/transformations.h>
#include <util/integer_range.h>

namespace vsort {
namespace network {
namespace internal {

template<uint32_t n>
struct oddeven_network {

    template<uint32_t i, uint32_t j>
    static constexpr decltype(auto)
    oddeven_create_pair() {
        if constexpr (i < n && j < n) {
            return std::integer_sequence<uint32_t, i, j>{};
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t i, uint32_t lo, uint32_t s, uint32_t r, uint32_t m>
    static constexpr decltype(auto)
    oddeven_create_pairs() {
        if constexpr ((i + r) < (lo + s)) {
            return merge<uint32_t>(oddeven_create_pair<i, i + r>(),
                            oddeven_create_pairs<i + m, lo, s, r, m>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t lo, uint32_t s, uint32_t r>
    static constexpr decltype(auto)
    oddeven_merge() {
        constexpr uint32_t m = 2 * r;
        if constexpr (m < s) {
            auto constexpr _pairs_lo = oddeven_merge<lo, s, m>();
            auto constexpr _pairs_hi = oddeven_merge<lo + r, s, m>();
            return merge<uint32_t>(merge<uint32_t>(_pairs_lo, _pairs_hi),
                                   oddeven_create_pairs<lo + r, lo, s, r, m>());
        }
        else {
            return oddeven_create_pair<lo, lo + r>();
        }
    }

    template<uint32_t lo, uint32_t s>
    static constexpr decltype(auto)
    oddeven_sort_kernel() {
        if constexpr (s > 1) {
            constexpr uint32_t m         = s / 2;
            constexpr auto     _pairs_lo = oddeven_sort_kernel<lo, m>();
            constexpr auto     _pairs_hi = oddeven_sort_kernel<lo + m, m>();
            return merge<uint32_t>(merge<uint32_t>(_pairs_lo, _pairs_hi),
                                   oddeven_merge<lo, s, 1>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    static constexpr decltype(auto)
    oddeven_sort() {
        return oddeven_sort_kernel<0, next_p2(n)>();
    }

    using network = decltype(oddeven_sort());
};

}  // namespace internal
}  // namespace network
}  // namespace vsort

#endif
