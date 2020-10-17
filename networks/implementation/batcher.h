#ifndef _BATCHER_NETWORK_H_
#define _BATCHER_NETWORK_H_

#include <stdint.h>

#include <networks/transformations.h>
#include <util/integer_range.h>

namespace vsort {
namespace network {
namespace internal {
template<uint32_t n>
struct batcher_network {

    template<uint32_t i,
             uint32_t p,
             uint32_t r,
             uint32_t bound,
             uint32_t offset>
    static constexpr decltype(auto)
    batcher_create_pairs() {
        if constexpr (i < bound) {
            if constexpr ((i & p) == r) {
                return merge<uint32_t>(
                    std::integer_sequence<uint32_t, i, i + offset>{},
                    batcher_create_pairs<i + 1, p, r, bound, offset>());
            }
            else {
                return batcher_create_pairs<i + 1, p, r, bound, offset>();
            }
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t q, uint32_t r, uint32_t d, uint32_t p>
    static constexpr decltype(auto)
    batcher_inner() {
        if constexpr (d > 0) {
            constexpr uint32_t _q = q >> 1;
            return merge<uint32_t>(batcher_create_pairs<0, p, r, (n - d), d>(),
                                   batcher_inner<_q, p, q - p, p>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t p>
    static constexpr decltype(auto)
    batcher_outer() {
        if constexpr (p > 0) {
            constexpr uint32_t q  = next_p2(n) >> 1;
            constexpr uint32_t  _p = p >> 1;
            return merge<uint32_t>(batcher_inner<q, 0, p, p>(),
                                   batcher_outer<_p>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    static constexpr decltype(auto)
    batcher_sort() {
        constexpr uint32_t p = next_p2(n) >> 1;
        return batcher_outer<p>();
    }

    using network = decltype(batcher_sort());
};

}  // namespace internal
}  // namespace network
}  // namespace vsort

#endif
