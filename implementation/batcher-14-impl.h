#ifndef _BATCHER_14_IMPL_H_
#define _BATCHER_14_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 14> {
static constexpr uint32_t n = 14;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
5, 4, 3, 2, 
1, 0, 7, 6, 
13, 12, 11, 10, 
9, 8
// clang-format on
>(v, 0x3f);


v = compare_exchange<T, n, 
// clang-format off
9, 8, 11, 10, 
13, 12, 3, 2, 
1, 0, 7, 6, 
5, 4
// clang-format on
>(v, 0x30f);


v = compare_exchange<T, n, 
// clang-format off
13, 12, 7, 6, 
5, 4, 11, 10, 
9, 8, 1, 0, 
3, 2
// clang-format on
>(v, 0xf3);


v = compare_exchange<T, n, 
// clang-format off
13, 12, 9, 8, 
11, 10, 5, 4, 
7, 6, 3, 2, 
0, 1
// clang-format on
>(v, 0x331);


v = compare_exchange<T, n, 
// clang-format off
7, 6, 11, 10, 
3, 2, 13, 12, 
5, 4, 9, 8, 
1, 0
// clang-format on
>(v, 0xcc);


v = compare_exchange<T, n, 
// clang-format off
11, 10, 13, 12, 
7, 6, 9, 8, 
3, 2, 5, 4, 
1, 0
// clang-format on
>(v, 0xccc);


v = compare_exchange<T, n, 
// clang-format off
12, 13, 10, 11, 
8, 9, 6, 7, 
4, 5, 2, 3, 
1, 0
// clang-format on
>(v, 0x1554);


v = compare_exchange<T, n, 
// clang-format off
13, 5, 11, 3, 
9, 1, 7, 6, 
12, 4, 10, 2, 
8, 0
// clang-format on
>(v, 0x2a);


v = compare_exchange<T, n, 
// clang-format off
13, 9, 11, 7, 
12, 5, 10, 3, 
8, 1, 6, 2, 
4, 0
// clang-format on
>(v, 0x2aa);


v = compare_exchange<T, n, 
// clang-format off
13, 11, 12, 9, 
10, 7, 8, 5, 
6, 3, 4, 1, 
2, 0
// clang-format on
>(v, 0xaaa);

vec_store<T, n>(arr, v);
}
};

#endif
