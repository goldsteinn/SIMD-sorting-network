// NOTE: This is NOT optimized! A hand optimized version would be able to cut
// out many of the shuffle / blend instructions

#ifndef _BOSENELSON_NETWORK_H_
#define _BOSENELSON_NETWORK_H_


#include <stdint.h>

#include <networks/transformations.h>
#include <util/integer_range.h>


namespace vsort {
namespace network {
namespace internal {

template<uint32_t n>
struct bosenelson_network {

    template<uint32_t i, uint32_t length_i, uint32_t j, uint32_t length_j>
    static constexpr decltype(auto)
    bosenelson_merge() {
        if constexpr (length_i == 1 && length_j == 1) {
            return std::integer_sequence<uint32_t, i, j>{};
        }
        else if constexpr (length_i == 1 && length_j == 2) {
            return std::integer_sequence<uint32_t, i, j + 1, i, j>{};
        }
        else if constexpr (length_i == 2 && length_j == 1) {
            return std::integer_sequence<uint32_t, i, j, i + 1, j>{};
        }
        else {
            constexpr uint32_t i_mid = length_i / 2;
            constexpr uint32_t j_mid =
                (length_i & 0x1) ? (length_j / 2) : ((length_j + 1) / 2);

            constexpr auto _pairs0 = bosenelson_merge<i, i_mid, j, j_mid>();
            constexpr auto _pairs1 = bosenelson_merge<i + i_mid,
                                                      length_i - i_mid,
                                                      j + j_mid,
                                                      length_j - j_mid>();
            constexpr auto _pairs2 =
                bosenelson_merge<i + i_mid, length_i - i_mid, j, j_mid>();
            return merge<uint32_t>(merge<uint32_t>(_pairs0, _pairs1), _pairs2);
        }
    }

    template<uint32_t i, uint32_t length>
    static constexpr decltype(auto)
    bosenelson_split() {
        if constexpr (length >= 2) {
            constexpr uint32_t mid       = length >> 1;
            constexpr auto     _pairs_lo = bosenelson_split<i, mid>();
            constexpr auto     _pairs_hi =
                bosenelson_split<i + mid, length - mid>();
            return merge<uint32_t>(
                merge<uint32_t>(_pairs_lo, _pairs_hi),
                bosenelson_merge<i, mid, i + mid, length - mid>());
        }
        else {
            return std::make_integer_sequence<uint32_t, 0>{};
        }
    }

    static constexpr decltype(auto)
    bosenelson_sort() {
        return bosenelson_split<0, n>();
    }

    using network = decltype(bosenelson_sort());

};

    

}  // namespace internal
}  // namespace network
}  // namespace vsort


#endif
