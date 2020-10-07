#ifndef _BITONIC_8_IMPL_H_
#define _BITONIC_8_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 8> {
static constexpr uint32_t n = 8;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55);


v = compare_exchange<T, n, 
// clang-format off
4, 5, 6, 7, 
0, 1, 2, 3
// clang-format on
>(v, 0x33);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55);


v = compare_exchange<T, n, 
// clang-format off
0, 1, 2, 3, 
4, 5, 6, 7
// clang-format on
>(v, 0xf);


v = compare_exchange<T, n, 
// clang-format off
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v, 0x33);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55);

vec_store<T, n>(arr, v);
}
};

#endif
