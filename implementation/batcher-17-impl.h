#ifndef _BATCHER_17_IMPL_H_
#define _BATCHER_17_IMPL_H_

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
0, 7, 6, 5, 
4, 3, 2, 1, 
8, 15, 14, 13, 
12, 11, 10, 9, 
16
// clang-format on
>(v, 0xff);


v = compare_exchange<T, n, 
// clang-format off
16, 11, 10, 9, 
12, 15, 14, 13, 
0, 3, 2, 1, 
4, 7, 6, 5, 
8
// clang-format on
>(v, 0xe0f);


v = compare_exchange<T, n, 
// clang-format off
8, 13, 14, 15, 
12, 7, 6, 5, 
16, 11, 10, 9, 
0, 1, 2, 3, 
4
// clang-format on
>(v, 0x21e3);


v = compare_exchange<T, n, 
// clang-format off
4, 15, 14, 13, 
8, 9, 10, 11, 
12, 5, 6, 7, 
16, 3, 0, 1, 
2
// clang-format on
>(v, 0x331);


v = compare_exchange<T, n, 
// clang-format off
12, 15, 14, 7, 
16, 11, 10, 3, 
4, 13, 6, 5, 
8, 9, 2, 0, 
1
// clang-format on
>(v, 0x1099);


v = compare_exchange<T, n, 
// clang-format off
2, 15, 12, 11, 
14, 13, 8, 7, 
10, 9, 4, 3, 
6, 5, 16, 1, 
0
// clang-format on
>(v, 0x199c);


v = compare_exchange<T, n, 
// clang-format off
10, 15, 14, 13, 
6, 11, 16, 9, 
2, 7, 12, 5, 
4, 3, 8, 1, 
0
// clang-format on
>(v, 0x444);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 16, 13, 
10, 11, 12, 9, 
6, 7, 8, 5, 
2, 3, 4, 1, 
0
// clang-format on
>(v, 0x4444);


v = compare_exchange<T, n, 
// clang-format off
1, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 16, 
0
// clang-format on
>(v, 0x5556);


v = compare_exchange<T, n, 
// clang-format off
9, 15, 7, 13, 
5, 11, 3, 16, 
1, 14, 6, 12, 
4, 10, 2, 8, 
0
// clang-format on
>(v, 0x2aa);


v = compare_exchange<T, n, 
// clang-format off
13, 15, 11, 16, 
9, 14, 7, 12, 
5, 10, 3, 8, 
1, 6, 2, 4, 
0
// clang-format on
>(v, 0x2aaa);


v = compare_exchange<T, n, 
// clang-format off
15, 16, 13, 14, 
11, 12, 9, 10, 
7, 8, 5, 6, 
3, 4, 1, 2, 
0
// clang-format on
>(v, 0xaaaa);

vec_store<T, n>(arr, v);
}
};

#endif
