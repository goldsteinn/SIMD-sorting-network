#ifndef _BATCHER_25_IMPL_H_
#define _BATCHER_25_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 25> {
static constexpr uint32_t n = 25;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
8, 7, 6, 5, 
4, 3, 2, 1, 
0, 15, 14, 13, 
12, 11, 10, 9, 
24, 23, 22, 21, 
20, 19, 18, 17, 
16
// clang-format on
>(v, 0x1ff);


v = compare_exchange<T, n, 
// clang-format off
16, 23, 22, 21, 
20, 19, 18, 17, 
24, 7, 6, 5, 
4, 3, 2, 1, 
0, 15, 14, 13, 
12, 11, 10, 9, 
8
// clang-format on
>(v, 0x100ff);


v = compare_exchange<T, n, 
// clang-format off
24, 15, 14, 13, 
12, 11, 10, 9, 
8, 23, 22, 21, 
20, 19, 18, 17, 
16, 3, 2, 1, 
0, 7, 6, 5, 
4
// clang-format on
>(v, 0xff0f);


v = compare_exchange<T, n, 
// clang-format off
24, 19, 18, 17, 
16, 23, 22, 21, 
20, 11, 10, 9, 
8, 15, 14, 13, 
12, 7, 6, 5, 
4, 1, 0, 3, 
2
// clang-format on
>(v, 0xf0f03);


v = compare_exchange<T, n, 
// clang-format off
12, 21, 22, 23, 
20, 7, 6, 5, 
4, 15, 14, 13, 
24, 11, 10, 9, 
8, 19, 18, 17, 
16, 3, 2, 0, 
1
// clang-format on
>(v, 0x2010f1);


v = compare_exchange<T, n, 
// clang-format off
20, 23, 22, 21, 
24, 15, 14, 13, 
12, 19, 18, 17, 
16, 7, 6, 5, 
4, 11, 10, 9, 
8, 3, 2, 1, 
0
// clang-format on
>(v, 0x10f0f0);


v = compare_exchange<T, n, 
// clang-format off
24, 23, 20, 21, 
22, 17, 16, 19, 
18, 13, 12, 15, 
14, 9, 8, 11, 
10, 5, 4, 7, 
6, 3, 2, 1, 
0
// clang-format on
>(v, 0x133330);


v = compare_exchange<T, n, 
// clang-format off
10, 23, 22, 7, 
6, 19, 18, 3, 
2, 15, 14, 13, 
12, 11, 24, 9, 
8, 21, 20, 5, 
4, 17, 16, 1, 
0
// clang-format on
>(v, 0x4cc);


v = compare_exchange<T, n, 
// clang-format off
18, 23, 22, 15, 
14, 19, 24, 11, 
10, 21, 20, 7, 
6, 17, 16, 3, 
2, 13, 12, 5, 
4, 9, 8, 1, 
0
// clang-format on
>(v, 0x4cccc);


v = compare_exchange<T, n, 
// clang-format off
22, 23, 24, 19, 
18, 21, 20, 15, 
14, 17, 16, 11, 
10, 13, 12, 7, 
6, 9, 8, 3, 
2, 5, 4, 1, 
0
// clang-format on
>(v, 0x4ccccc);


v = compare_exchange<T, n, 
// clang-format off
24, 22, 23, 20, 
21, 18, 19, 16, 
17, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 1, 
0
// clang-format on
>(v, 0x555554);


v = compare_exchange<T, n, 
// clang-format off
9, 23, 7, 21, 
5, 19, 3, 17, 
1, 15, 14, 13, 
12, 11, 10, 24, 
8, 22, 6, 20, 
4, 18, 2, 16, 
0
// clang-format on
>(v, 0x2aa);


v = compare_exchange<T, n, 
// clang-format off
17, 23, 15, 21, 
13, 19, 11, 24, 
9, 22, 7, 20, 
5, 18, 3, 16, 
1, 14, 6, 12, 
4, 10, 2, 8, 
0
// clang-format on
>(v, 0x2aaaa);


v = compare_exchange<T, n, 
// clang-format off
21, 23, 19, 24, 
17, 22, 15, 20, 
13, 18, 11, 16, 
9, 14, 7, 12, 
5, 10, 3, 8, 
1, 6, 2, 4, 
0
// clang-format on
>(v, 0x2aaaaa);


v = compare_exchange<T, n, 
// clang-format off
23, 24, 21, 22, 
19, 20, 17, 18, 
15, 16, 13, 14, 
11, 12, 9, 10, 
7, 8, 5, 6, 
3, 4, 1, 2, 
0
// clang-format on
>(v, 0xaaaaaa);

vec_store<T, n>(arr, v);
}
};

#endif
