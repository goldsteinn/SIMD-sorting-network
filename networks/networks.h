#ifndef _NETWORKS_H_
#define _NETWORKS_H_

#include <networks/implementation/bitonic.h>
#include <networks/implementation/bosenelson.h>
#include <networks/transformations.h>

namespace vsort {

template<uint32_t n>
using bitonic = typename transform::
    build<n, typename network::internal::bitonic_network<n>::network>::type;

    static_assert(std::is_same<typename network::internal::bosenelson_network<9>::network, std::integer_sequence<uint32_t, 0, 1, 2, 3, 0, 2, 1, 3, 1, 2, 4, 5, 7, 8, 6, 8, 6, 7, 4, 7, 4, 6, 5, 8, 5, 7, 5, 6, 0, 5, 0, 4, 1, 6, 1, 5, 1, 4, 2, 7, 3, 8, 3, 7, 2, 5, 2, 4, 3, 6, 3, 5, 3, 4>>::value);
    
template<uint32_t n>
using bosenelson = typename transform::
    build<n, typename network::internal::bosenelson_network<n>::network>::type;



}  // namespace vsort

#endif
