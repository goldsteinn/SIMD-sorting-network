#ifndef _BALANCED_NETWORK_H_
#define _BALANCED_NETWORK_H_

#include <stdint.h>

#include <util/integer_range.h>

namespace vsort {
namespace network {
namespace internal {

template<uint32_t n>
struct balanced_network {

    template<uint32_t i, uint32_t j, uint32_t curr>
    static constexpr decltype(auto)
    balanced_create_pairs() {
        if constexpr (i < next_p2(n)) {
            if constexpr (j < (curr / 2)) {
                constexpr uint32_t wire1 = i + j;
                constexpr uint32_t wire2 = (i + curr) - (j + 1);
                if constexpr (wire1 < n && wire2 < n) {
                    return merge<uint32_t>(
                        std::integer_sequence<uint32_t, wire1, wire2>{},
                        balanced_create_pairs<i, j + 1, curr>());
                }
                else {
                    return balanced_create_pairs<i, j + 1, curr>();
                }
            }
            else {
                return balanced_create_pairs<i + curr, 0, curr>();
            }
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t curr>
    static constexpr decltype(auto)
    balanced_outer() {
        if constexpr (curr > 1) {
            return merge<uint32_t>(balanced_create_pairs<0, 0, curr>(),
                                   balanced_outer<curr / 2>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    template<uint32_t iter>
    static constexpr decltype(auto)
    balanced_sort_kernel() {
        if constexpr (iter > 1) {
            return merge<uint32_t>(balanced_outer<next_p2(n)>(),
                                   balanced_sort_kernel<iter / 2>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    static constexpr decltype(auto)
    balanced_sort() {
        return balanced_sort_kernel<next_p2(n)>();
    }

    using network = decltype(balanced_sort());
};

}  // namespace internal
}  // namespace network
}  // namespace vsort

#endif
