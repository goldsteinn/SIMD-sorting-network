#ifndef _BATCHER_21_IMPL_H_
#define _BATCHER_21_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 21> {
static constexpr uint32_t n = 21;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
4, 3, 2, 1, 
0, 7, 6, 5, 
12, 11, 10, 9, 
8, 15, 14, 13, 
20, 19, 18, 17, 
16
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 18, 17, 
16, 15, 14, 13, 
4, 3, 2, 1, 
0, 7, 6, 5, 
12, 11, 10, 9, 
8
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
12, 11, 10, 9, 
8, 15, 14, 13, 
20, 19, 18, 17, 
16, 3, 2, 1, 
0, 7, 6, 5, 
4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 7, 6, 5, 
20, 11, 10, 9, 
8, 15, 14, 13, 
12, 19, 18, 17, 
4, 1, 0, 3, 
2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
20, 15, 14, 13, 
4, 19, 18, 17, 
12, 7, 6, 5, 
8, 11, 10, 9, 
16, 3, 2, 0, 
1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
20, 17, 18, 19, 
12, 13, 14, 15, 
16, 9, 10, 11, 
4, 5, 6, 7, 
8, 3, 2, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 16, 3, 
18, 15, 12, 7, 
14, 11, 8, 9, 
10, 13, 4, 5, 
6, 17, 2, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
6, 19, 18, 11, 
2, 15, 14, 13, 
12, 17, 10, 3, 
8, 7, 20, 5, 
4, 9, 16, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
14, 19, 18, 15, 
10, 17, 20, 11, 
6, 13, 16, 7, 
2, 9, 12, 3, 
4, 5, 8, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 20, 17, 
14, 15, 16, 13, 
10, 11, 12, 9, 
6, 7, 8, 5, 
2, 3, 4, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
20, 18, 19, 16, 
17, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 1, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
5, 19, 3, 17, 
1, 15, 7, 13, 
12, 11, 10, 9, 
8, 14, 6, 20, 
4, 18, 2, 16, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
13, 19, 11, 17, 
9, 15, 14, 20, 
5, 18, 3, 16, 
1, 7, 6, 12, 
4, 10, 2, 8, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 19, 15, 20, 
13, 18, 11, 16, 
9, 14, 7, 12, 
5, 10, 3, 8, 
1, 6, 2, 4, 
0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
19, 20, 17, 18, 
15, 16, 13, 14, 
11, 12, 9, 10, 
7, 8, 5, 6, 
3, 4, 1, 2, 
0
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
