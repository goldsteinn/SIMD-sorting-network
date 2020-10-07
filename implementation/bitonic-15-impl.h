#ifndef _BITONIC_15_IMPL_H_
#define _BITONIC_15_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 15> {
static constexpr uint32_t n = 15;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aaa);


v = compare_exchange<T, n, 
// clang-format off
11, 12, 13, 14, 
7, 8, 9, 10, 
3, 4, 5, 6, 
0, 1, 2
// clang-format on
>(v, 0x1999);


v = compare_exchange<T, n, 
// clang-format off
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
2, 0, 1
// clang-format on
>(v, 0x2aa9);


v = compare_exchange<T, n, 
// clang-format off
7, 8, 9, 10, 
11, 12, 13, 14, 
6, 0, 1, 2, 
3, 4, 5
// clang-format on
>(v, 0x787);


v = compare_exchange<T, n, 
// clang-format off
12, 11, 14, 13, 
8, 7, 10, 9, 
4, 3, 6, 5, 
0, 1, 2
// clang-format on
>(v, 0x1999);


v = compare_exchange<T, n, 
// clang-format off
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aaa);


v = compare_exchange<T, n, 
// clang-format off
0, 1, 2, 3, 
4, 5, 6, 7, 
8, 9, 10, 11, 
12, 13, 14
// clang-format on
>(v, 0x7f);


v = compare_exchange<T, n, 
// clang-format off
10, 9, 8, 11, 
14, 13, 12, 3, 
2, 1, 0, 7, 
6, 5, 4
// clang-format on
>(v, 0x70f);


v = compare_exchange<T, n, 
// clang-format off
12, 13, 14, 9, 
8, 11, 10, 5, 
4, 7, 6, 1, 
0, 3, 2
// clang-format on
>(v, 0x1333);


v = compare_exchange<T, n, 
// clang-format off
14, 12, 13, 10, 
11, 8, 9, 6, 
7, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v, 0x1555);

vec_store<T, n>(arr, v);
}
};

#endif
