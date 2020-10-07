#ifndef _BITONIC_29_IMPL_H_
#define _BITONIC_29_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 29> {
static constexpr uint32_t n = 29;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
27, 28, 25, 26, 
23, 24, 21, 22, 
19, 20, 17, 18, 
15, 16, 14, 12, 
13, 10, 11, 8, 
9, 7, 5, 6, 
3, 4, 1, 2, 
0
// clang-format on
>(v, 0xaaa952a);


v = compare_exchange<T, n, 
// clang-format off
25, 26, 27, 28, 
21, 22, 23, 24, 
17, 18, 19, 20, 
14, 15, 16, 10, 
11, 12, 13, 7, 
8, 9, 3, 4, 
5, 6, 2, 0, 
1
// clang-format on
>(v, 0x6664c99);


v = compare_exchange<T, n, 
// clang-format off
27, 28, 25, 26, 
23, 24, 21, 22, 
19, 20, 17, 18, 
16, 14, 15, 12, 
13, 10, 11, 9, 
7, 8, 5, 6, 
3, 4, 1, 2, 
0
// clang-format on
>(v, 0xaaa54aa);


v = compare_exchange<T, n, 
// clang-format off
21, 22, 23, 24, 
25, 26, 27, 28, 
20, 14, 15, 16, 
17, 18, 19, 13, 
7, 8, 9, 10, 
11, 12, 0, 1, 
2, 3, 4, 5, 
6
// clang-format on
>(v, 0x1e1c387);


v = compare_exchange<T, n, 
// clang-format off
26, 25, 28, 27, 
22, 21, 24, 23, 
18, 17, 20, 19, 
14, 15, 16, 11, 
10, 13, 12, 7, 
8, 9, 4, 5, 
6, 1, 0, 3, 
2
// clang-format on
>(v, 0x6664c93);


v = compare_exchange<T, n, 
// clang-format off
27, 28, 25, 26, 
23, 24, 21, 22, 
19, 20, 17, 18, 
15, 16, 14, 12, 
13, 10, 11, 8, 
9, 7, 6, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v, 0xaaa9515);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 16, 17, 
18, 19, 20, 21, 
22, 23, 24, 25, 
26, 27, 28, 13, 
0, 1, 2, 3, 
4, 5, 6, 7, 
8, 9, 10, 11, 
12
// clang-format on
>(v, 0x1fc03f);


v = compare_exchange<T, n, 
// clang-format off
24, 23, 22, 25, 
28, 27, 26, 17, 
16, 15, 14, 21, 
20, 19, 18, 9, 
8, 7, 6, 13, 
12, 11, 10, 1, 
0, 3, 2, 5, 
4
// clang-format on
>(v, 0x1c3c3c3);


v = compare_exchange<T, n, 
// clang-format off
26, 27, 28, 23, 
22, 25, 24, 19, 
18, 21, 20, 15, 
14, 17, 16, 11, 
10, 13, 12, 7, 
6, 9, 8, 3, 
2, 5, 4, 0, 
1
// clang-format on
>(v, 0x4cccccd);


v = compare_exchange<T, n, 
// clang-format off
1, 26, 27, 24, 
25, 22, 23, 20, 
21, 18, 19, 16, 
17, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 28, 
0
// clang-format on
>(v, 0x5555556);


v = compare_exchange<T, n, 
// clang-format off
28, 2, 3, 4, 
5, 6, 7, 8, 
9, 10, 11, 12, 
13, 15, 14, 16, 
17, 18, 19, 20, 
21, 22, 23, 24, 
25, 26, 27, 1, 
0
// clang-format on
>(v, 0x3ffc);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 18, 17, 
16, 23, 22, 21, 
28, 27, 26, 25, 
24, 6, 7, 5, 
4, 3, 2, 1, 
0, 14, 15, 13, 
12, 11, 10, 9, 
8
// clang-format on
>(v, 0x1f00ff);


v = compare_exchange<T, n, 
// clang-format off
24, 25, 26, 27, 
28, 19, 18, 17, 
16, 23, 22, 21, 
20, 10, 11, 9, 
8, 14, 15, 13, 
12, 3, 2, 1, 
0, 7, 6, 5, 
4
// clang-format on
>(v, 0x30f0f0f);


v = compare_exchange<T, n, 
// clang-format off
28, 27, 24, 25, 
26, 21, 20, 23, 
22, 17, 16, 19, 
18, 12, 13, 14, 
15, 9, 8, 11, 
10, 5, 4, 7, 
6, 1, 0, 3, 
2
// clang-format on
>(v, 0x1333333);


v = compare_exchange<T, n, 
// clang-format off
28, 26, 27, 24, 
25, 22, 23, 20, 
21, 18, 19, 16, 
17, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v, 0x5555555);

vec_store<T, n>(arr, v);
}
};

#endif
