#ifndef _BITONIC_17_IMPL_H_
#define _BITONIC_17_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 17> {
static constexpr uint32_t n = 17;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
15, 16, 14, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 16, 13, 
12, 8, 9, 10, 
11, 4, 5, 6, 
7, 0, 1, 2, 
3
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
13, 14, 15, 16, 
12, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
11, 12, 13, 14, 
15, 16, 10, 9, 
8, 0, 1, 2, 
3, 4, 5, 6, 
7
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 14, 15, 12, 
13, 11, 10, 9, 
8, 5, 4, 7, 
6, 1, 0, 3, 
2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 8, 9, 10, 
11, 12, 13, 14, 
15, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
7, 13, 12, 15, 
14, 9, 8, 11, 
10, 16, 6, 5, 
4, 3, 2, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 14, 15, 12, 
13, 10, 11, 8, 
9, 7, 6, 5, 
4, 3, 2, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 0, 1, 2, 
3, 4, 5, 6, 
7, 8, 9, 10, 
11, 12, 13, 14, 
15
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 11, 10, 9, 
8, 15, 14, 13, 
12, 3, 2, 1, 
0, 7, 6, 5, 
4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 13, 12, 15, 
14, 9, 8, 11, 
10, 5, 4, 7, 
6, 1, 0, 3, 
2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
