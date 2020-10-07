#ifndef _BATCHER_19_IMPL_H_
#define _BATCHER_19_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 19> {
static constexpr uint32_t n = 19;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
2, 1, 0, 7, 
6, 5, 4, 3, 
10, 9, 8, 15, 
14, 13, 12, 11, 
18, 17, 16
// clang-format on
>(v, 0xff);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 16, 11, 
14, 13, 12, 15, 
2, 1, 0, 3, 
6, 5, 4, 7, 
10, 9, 8
// clang-format on
>(v, 0x80f);


v = compare_exchange<T, n, 
// clang-format off
10, 9, 8, 15, 
14, 13, 12, 7, 
18, 17, 16, 11, 
2, 1, 0, 3, 
6, 5, 4
// clang-format on
>(v, 0x787);


v = compare_exchange<T, n, 
// clang-format off
6, 5, 4, 15, 
10, 9, 8, 11, 
14, 13, 12, 7, 
18, 17, 16, 1, 
0, 3, 2
// clang-format on
>(v, 0x773);


v = compare_exchange<T, n, 
// clang-format off
14, 13, 12, 15, 
18, 17, 16, 11, 
6, 5, 4, 7, 
10, 9, 8, 3, 
2, 0, 1
// clang-format on
>(v, 0x7071);


v = compare_exchange<T, n, 
// clang-format off
16, 3, 18, 13, 
12, 15, 14, 9, 
8, 11, 10, 5, 
4, 7, 6, 17, 
2, 1, 0
// clang-format on
>(v, 0x13338);


v = compare_exchange<T, n, 
// clang-format off
18, 11, 2, 15, 
14, 7, 6, 17, 
10, 3, 8, 13, 
12, 5, 4, 9, 
16, 1, 0
// clang-format on
>(v, 0x8cc);


v = compare_exchange<T, n, 
// clang-format off
18, 15, 10, 17, 
14, 11, 12, 13, 
16, 7, 2, 9, 
6, 3, 4, 5, 
8, 1, 0
// clang-format on
>(v, 0x8c8c);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 14, 15, 
16, 13, 10, 11, 
12, 9, 6, 7, 
8, 5, 2, 3, 
4, 1, 0
// clang-format on
>(v, 0x4444);


v = compare_exchange<T, n, 
// clang-format off
18, 16, 17, 14, 
15, 12, 13, 10, 
11, 8, 9, 6, 
7, 4, 5, 2, 
3, 1, 0
// clang-format on
>(v, 0x15554);


v = compare_exchange<T, n, 
// clang-format off
3, 17, 1, 15, 
7, 13, 5, 11, 
10, 9, 8, 14, 
6, 12, 4, 18, 
2, 16, 0
// clang-format on
>(v, 0xaa);


v = compare_exchange<T, n, 
// clang-format off
11, 17, 9, 15, 
14, 13, 12, 18, 
3, 16, 1, 7, 
6, 5, 4, 10, 
2, 8, 0
// clang-format on
>(v, 0xa0a);


v = compare_exchange<T, n, 
// clang-format off
15, 17, 13, 18, 
11, 16, 9, 14, 
7, 12, 5, 10, 
3, 8, 1, 6, 
2, 4, 0
// clang-format on
>(v, 0xaaaa);


v = compare_exchange<T, n, 
// clang-format off
17, 18, 15, 16, 
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aaaa);

vec_store<T, n>(arr, v);
}
};

#endif
