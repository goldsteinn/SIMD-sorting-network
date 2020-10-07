#ifndef _BITONIC_4_IMPL_H_
#define _BITONIC_4_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 4> {
static constexpr uint32_t n = 4;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
2, 3, 0, 1
// clang-format on
>(v, 0x5);


v = compare_exchange<T, n, 
// clang-format off
0, 1, 2, 3
// clang-format on
>(v, 0x3);


v = compare_exchange<T, n, 
// clang-format off
2, 3, 0, 1
// clang-format on
>(v, 0x5);

vec_store<T, n>(arr, v);
}
};

#endif
