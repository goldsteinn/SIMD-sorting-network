#ifndef _BITONIC_16_IMPL_H_
#define _BITONIC_16_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 16> {
static constexpr uint32_t n = 16;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x5555);


v = compare_exchange<T, n, 
// clang-format off
12, 13, 14, 15, 
8, 9, 10, 11, 
4, 5, 6, 7, 
0, 1, 2, 3
// clang-format on
>(v, 0x3333);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x5555);


v = compare_exchange<T, n, 
// clang-format off
8, 9, 10, 11, 
12, 13, 14, 15, 
0, 1, 2, 3, 
4, 5, 6, 7
// clang-format on
>(v, 0xf0f);


v = compare_exchange<T, n, 
// clang-format off
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v, 0x3333);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x5555);


v = compare_exchange<T, n, 
// clang-format off
0, 1, 2, 3, 
4, 5, 6, 7, 
8, 9, 10, 11, 
12, 13, 14, 15
// clang-format on
>(v, 0xff);


v = compare_exchange<T, n, 
// clang-format off
11, 10, 9, 8, 
15, 14, 13, 12, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v, 0xf0f);


v = compare_exchange<T, n, 
// clang-format off
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v, 0x3333);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x5555);

vec_store<T, n>(arr, v);
}
};

#endif
