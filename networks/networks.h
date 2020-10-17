#ifndef _NETWORKS_H_
#define _NETWORKS_H_

#include <networks/implementation/balanced.h>
#include <networks/implementation/batcher.h>
#include <networks/implementation/bitonic.h>
#include <networks/implementation/bosenelson.h>
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


}  // namespace vsort

#endif
