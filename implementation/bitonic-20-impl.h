#ifndef _BITONIC_20_IMPL_H_
#define _BITONIC_20_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 20> {
static constexpr uint32_t n = 20;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
18, 19, 17, 15, 
16, 13, 14, 12, 
10, 11, 8, 9, 
7, 5, 6, 3, 
4, 2, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 18, 19, 16, 
15, 14, 12, 13, 
11, 10, 9, 7, 
8, 6, 5, 2, 
3, 4, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 18, 19, 
15, 13, 14, 10, 
11, 12, 8, 9, 
5, 6, 7, 1, 
2, 3, 4, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 15, 16, 17, 
18, 12, 11, 14, 
13, 10, 7, 6, 
9, 8, 5, 4, 
0, 1, 2, 3
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 17, 18, 15, 
16, 13, 14, 11, 
12, 10, 8, 9, 
6, 7, 5, 4, 
2, 3, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
13, 14, 11, 12, 
15, 18, 19, 16, 
17, 10, 9, 2, 
3, 0, 1, 4, 
7, 8, 5, 6
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 17, 16, 
13, 10, 15, 12, 
11, 14, 5, 8, 
7, 4, 9, 6, 
3, 2, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 15, 14, 
17, 16, 11, 10, 
13, 12, 7, 6, 
9, 8, 3, 2, 
5, 4, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 8, 9, 
2, 3, 4, 5, 
11, 10, 16, 17, 
18, 19, 12, 13, 
14, 15, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 16, 19, 18, 
15, 14, 13, 12, 
6, 7, 1, 0, 
10, 11, 5, 4, 
3, 2, 9, 8
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 16, 17, 
11, 10, 8, 9, 
15, 14, 12, 13, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 17, 16, 
13, 12, 15, 14, 
8, 9, 10, 11, 
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 17, 16, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
