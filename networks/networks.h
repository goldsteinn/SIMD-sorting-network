#ifndef _NETWORKS_H_
#define _NETWORKS_H_

#include <networks/implementation/bitonic.h>
#include <networks/transformations.h>

namespace vsort {

template<uint32_t n>
using bitonic = typename transform::
    build<n, typename network::internal::bitonic_network<n>::network>::type;


}  // namespace vsort

#endif
