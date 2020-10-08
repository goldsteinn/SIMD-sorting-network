#ifndef _BITONIC_13_IMPL_H_
#define _BITONIC_13_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 13> {
static constexpr uint32_t n = 13;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
11, 12, 9, 10, 
7, 8, 6, 4, 
5, 3, 1, 2, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
9, 10, 11, 12, 
8, 6, 7, 5, 
3, 4, 0, 1, 
2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
11, 12, 9, 10, 
7, 8, 6, 4, 
5, 3, 2, 0, 
1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 8, 9, 
10, 11, 12, 5, 
0, 1, 2, 3, 
4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
10, 11, 12, 7, 
6, 9, 8, 3, 
2, 5, 4, 0, 
1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
1, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 12, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
12, 2, 3, 4, 
5, 7, 6, 8, 
9, 10, 11, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
8, 9, 10, 11, 
12, 2, 3, 1, 
0, 6, 7, 5, 
4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
12, 11, 8, 9, 
10, 4, 5, 6, 
7, 1, 0, 3, 
2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
12, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
