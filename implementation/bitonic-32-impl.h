#ifndef _BITONIC_32_IMPL_H_
#define _BITONIC_32_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 32> {
static constexpr uint32_t n = 32;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
30, 31, 28, 29, 
26, 27, 24, 25, 
22, 23, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55555555);


v = compare_exchange<T, n, 
// clang-format off
28, 29, 30, 31, 
24, 25, 26, 27, 
20, 21, 22, 23, 
16, 17, 18, 19, 
12, 13, 14, 15, 
8, 9, 10, 11, 
4, 5, 6, 7, 
0, 1, 2, 3
// clang-format on
>(v, 0x33333333);


v = compare_exchange<T, n, 
// clang-format off
30, 31, 28, 29, 
26, 27, 24, 25, 
22, 23, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55555555);


v = compare_exchange<T, n, 
// clang-format off
24, 25, 26, 27, 
28, 29, 30, 31, 
16, 17, 18, 19, 
20, 21, 22, 23, 
8, 9, 10, 11, 
12, 13, 14, 15, 
0, 1, 2, 3, 
4, 5, 6, 7
// clang-format on
>(v, 0xf0f0f0f);


v = compare_exchange<T, n, 
// clang-format off
29, 28, 31, 30, 
25, 24, 27, 26, 
21, 20, 23, 22, 
17, 16, 19, 18, 
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v, 0x33333333);


v = compare_exchange<T, n, 
// clang-format off
30, 31, 28, 29, 
26, 27, 24, 25, 
22, 23, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55555555);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 18, 19, 
20, 21, 22, 23, 
24, 25, 26, 27, 
28, 29, 30, 31, 
0, 1, 2, 3, 
4, 5, 6, 7, 
8, 9, 10, 11, 
12, 13, 14, 15
// clang-format on
>(v, 0xff00ff);


v = compare_exchange<T, n, 
// clang-format off
27, 26, 25, 24, 
31, 30, 29, 28, 
19, 18, 17, 16, 
23, 22, 21, 20, 
11, 10, 9, 8, 
15, 14, 13, 12, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v, 0xf0f0f0f);


v = compare_exchange<T, n, 
// clang-format off
29, 28, 31, 30, 
25, 24, 27, 26, 
21, 20, 23, 22, 
17, 16, 19, 18, 
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v, 0x33333333);


v = compare_exchange<T, n, 
// clang-format off
30, 31, 28, 29, 
26, 27, 24, 25, 
22, 23, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55555555);


v = compare_exchange<T, n, 
// clang-format off
0, 1, 2, 3, 
4, 5, 6, 7, 
8, 9, 10, 11, 
12, 13, 14, 15, 
16, 17, 18, 19, 
20, 21, 22, 23, 
24, 25, 26, 27, 
28, 29, 30, 31
// clang-format on
>(v, 0xffff);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 21, 20, 
19, 18, 17, 16, 
31, 30, 29, 28, 
27, 26, 25, 24, 
7, 6, 5, 4, 
3, 2, 1, 0, 
15, 14, 13, 12, 
11, 10, 9, 8
// clang-format on
>(v, 0xff00ff);


v = compare_exchange<T, n, 
// clang-format off
27, 26, 25, 24, 
31, 30, 29, 28, 
19, 18, 17, 16, 
23, 22, 21, 20, 
11, 10, 9, 8, 
15, 14, 13, 12, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v, 0xf0f0f0f);


v = compare_exchange<T, n, 
// clang-format off
29, 28, 31, 30, 
25, 24, 27, 26, 
21, 20, 23, 22, 
17, 16, 19, 18, 
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v, 0x33333333);


v = compare_exchange<T, n, 
// clang-format off
30, 31, 28, 29, 
26, 27, 24, 25, 
22, 23, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 0, 1
// clang-format on
>(v, 0x55555555);

vec_store<T, n>(arr, v);
}
};

#endif
