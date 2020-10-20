#ifndef _NETWORKS_H_
#define _NETWORKS_H_

#include <networks/implementation/balanced.h>
#include <networks/implementation/batcher.h>
#include <networks/implementation/bitonic.h>
#include <networks/implementation/bosenelson.h>
#include <networks/implementation/minimum.h>
#include <networks/implementation/oddeven.h>
#include <networks/transformations.h>

namespace vsort {

template<uint32_t n>
using bitonic = typename transform::
    build<n, typename network::internal::bitonic_network<n>::network>::type;


template<uint32_t n>
using bosenelson = typename transform::
    build<n, typename network::internal::bosenelson_network<n>::network>::type;

template<uint32_t n>
using batcher = typename transform::
    build<n, typename network::internal::batcher_network<n>::network>::type;


template<uint32_t n>
using balanced = typename transform::
    build<n, typename network::internal::balanced_network<n>::network>::type;

template<uint32_t n>
using oddeven = typename transform::
    build<n, typename network::internal::oddeven_network<n>::network>::type;

template<uint32_t n>
using minimum = typename transform::
    build<n, typename network::internal::minimum_network<n>::network>::type;


template<uint32_t n>
// the logic for n value on not taken type helps build time a lot
using best =
    std::conditional_t<is_pow2(n) || (n > 32),
                       bitonic<(is_pow2(n) || (n > 32)) ? next_p2(n) : 4>,
                       minimum<(is_pow2(n) || (n > 32)) ? 4 : n>>;


}  // namespace vsort

#endif
