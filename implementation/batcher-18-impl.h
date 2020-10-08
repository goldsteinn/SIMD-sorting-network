#ifndef _BATCHER_18_IMPL_H_
#define _BATCHER_18_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 18> {
static constexpr uint32_t n = 18;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
1, 0, 7, 6, 
5, 4, 3, 2, 
9, 8, 15, 14, 
13, 12, 11, 10, 
17, 16
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 16, 11, 10, 
13, 12, 15, 14, 
1, 0, 3, 2, 
5, 4, 7, 6, 
9, 8
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
9, 8, 15, 14, 
13, 12, 7, 6, 
17, 16, 11, 10, 
1, 0, 3, 2, 
5, 4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
5, 4, 15, 14, 
9, 8, 11, 10, 
13, 12, 7, 6, 
17, 16, 1, 0, 
3, 2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
13, 12, 15, 14, 
17, 16, 11, 10, 
5, 4, 7, 6, 
9, 8, 3, 2, 
0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
3, 2, 13, 12, 
15, 14, 9, 8, 
11, 10, 5, 4, 
7, 6, 17, 16, 
1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
11, 10, 15, 14, 
7, 6, 17, 16, 
3, 2, 13, 12, 
5, 4, 9, 8, 
1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
15, 14, 17, 16, 
11, 10, 13, 12, 
7, 6, 9, 8, 
3, 2, 5, 4, 
1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 14, 15, 
12, 13, 10, 11, 
8, 9, 6, 7, 
4, 5, 2, 3, 
1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 1, 15, 7, 
13, 5, 11, 3, 
9, 8, 14, 6, 
12, 4, 10, 2, 
16, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 9, 15, 11, 
13, 12, 14, 7, 
16, 1, 10, 3, 
5, 4, 6, 2, 
8, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 13, 15, 14, 
16, 9, 11, 10, 
12, 5, 7, 6, 
8, 1, 3, 2, 
4, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
17, 15, 16, 13, 
14, 11, 12, 9, 
10, 7, 8, 5, 
6, 3, 4, 1, 
2, 0
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
