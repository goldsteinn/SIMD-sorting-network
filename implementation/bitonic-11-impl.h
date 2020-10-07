#ifndef _BITONIC_11_IMPL_H_
#define _BITONIC_11_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 11> {
static constexpr uint32_t n = 11;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
9, 10, 8, 6, 
7, 5, 3, 4, 
2, 0, 1
// clang-format on
>(v, 0x249);


v = compare_exchange<T, n, 
// clang-format off
8, 9, 10, 7, 
5, 6, 4, 2, 
3, 1, 0
// clang-format on
>(v, 0x124);


v = compare_exchange<T, n, 
// clang-format off
10, 8, 9, 6, 
7, 5, 3, 4, 
0, 1, 2
// clang-format on
>(v, 0x149);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 8, 9, 
10, 5, 2, 1, 
4, 3, 0
// clang-format on
>(v, 0xc6);


v = compare_exchange<T, n, 
// clang-format off
9, 10, 6, 5, 
8, 7, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x26a);


v = compare_exchange<T, n, 
// clang-format off
2, 3, 7, 8, 
5, 6, 4, 9, 
10, 1, 0
// clang-format on
>(v, 0xac);


v = compare_exchange<T, n, 
// clang-format off
10, 9, 4, 1, 
2, 3, 8, 5, 
6, 7, 0
// clang-format on
>(v, 0x1e);


v = compare_exchange<T, n, 
// clang-format off
8, 9, 10, 5, 
6, 7, 0, 1, 
2, 3, 4
// clang-format on
>(v, 0x123);


v = compare_exchange<T, n, 
// clang-format off
10, 8, 9, 7, 
4, 5, 6, 3, 
0, 1, 2
// clang-format on
>(v, 0x111);


v = compare_exchange<T, n, 
// clang-format off
10, 9, 8, 6, 
7, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v, 0x55);

vec_store<T, n>(arr, v);
}
};

#endif
