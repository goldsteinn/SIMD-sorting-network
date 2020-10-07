#ifndef _BITONIC_31_IMPL_H_
#define _BITONIC_31_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 31> {
static constexpr uint32_t n = 31;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
29, 30, 27, 28, 
25, 26, 23, 24, 
21, 22, 19, 20, 
17, 18, 15, 16, 
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aaaaaaa);


v = compare_exchange<T, n, 
// clang-format off
27, 28, 29, 30, 
23, 24, 25, 26, 
19, 20, 21, 22, 
15, 16, 17, 18, 
11, 12, 13, 14, 
7, 8, 9, 10, 
3, 4, 5, 6, 
2, 0, 1
// clang-format on
>(v, 0x19999999);


v = compare_exchange<T, n, 
// clang-format off
29, 30, 27, 28, 
25, 26, 23, 24, 
21, 22, 19, 20, 
17, 18, 15, 16, 
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aaaaaaa);


v = compare_exchange<T, n, 
// clang-format off
23, 24, 25, 26, 
27, 28, 29, 30, 
15, 16, 17, 18, 
19, 20, 21, 22, 
7, 8, 9, 10, 
11, 12, 13, 14, 
0, 1, 2, 3, 
4, 5, 6
// clang-format on
>(v, 0x7878787);


v = compare_exchange<T, n, 
// clang-format off
28, 27, 30, 29, 
24, 23, 26, 25, 
20, 19, 22, 21, 
16, 15, 18, 17, 
12, 11, 14, 13, 
8, 7, 10, 9, 
4, 5, 6, 1, 
0, 3, 2
// clang-format on
>(v, 0x19999993);


v = compare_exchange<T, n, 
// clang-format off
29, 30, 27, 28, 
25, 26, 23, 24, 
21, 22, 19, 20, 
17, 18, 15, 16, 
13, 14, 11, 12, 
9, 10, 7, 8, 
6, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v, 0x2aaaaa95);


v = compare_exchange<T, n, 
// clang-format off
15, 16, 17, 18, 
19, 20, 21, 22, 
23, 24, 25, 26, 
27, 28, 29, 30, 
14, 0, 1, 2, 
3, 4, 5, 6, 
7, 8, 9, 10, 
11, 12, 13
// clang-format on
>(v, 0x7f807f);


v = compare_exchange<T, n, 
// clang-format off
26, 25, 24, 23, 
30, 29, 28, 27, 
18, 17, 16, 15, 
22, 21, 20, 19, 
10, 9, 8, 7, 
14, 13, 12, 11, 
2, 1, 0, 3, 
6, 5, 4
// clang-format on
>(v, 0x7878787);


v = compare_exchange<T, n, 
// clang-format off
28, 27, 30, 29, 
24, 23, 26, 25, 
20, 19, 22, 21, 
16, 15, 18, 17, 
12, 11, 14, 13, 
8, 7, 10, 9, 
4, 3, 6, 5, 
0, 1, 2
// clang-format on
>(v, 0x19999999);


v = compare_exchange<T, n, 
// clang-format off
29, 30, 27, 28, 
25, 26, 23, 24, 
21, 22, 19, 20, 
17, 18, 15, 16, 
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aaaaaaa);


v = compare_exchange<T, n, 
// clang-format off
0, 1, 2, 3, 
4, 5, 6, 7, 
8, 9, 10, 11, 
12, 13, 14, 15, 
16, 17, 18, 19, 
20, 21, 22, 23, 
24, 25, 26, 27, 
28, 29, 30
// clang-format on
>(v, 0x7fff);


v = compare_exchange<T, n, 
// clang-format off
22, 21, 20, 19, 
18, 17, 16, 23, 
30, 29, 28, 27, 
26, 25, 24, 7, 
6, 5, 4, 3, 
2, 1, 0, 15, 
14, 13, 12, 11, 
10, 9, 8
// clang-format on
>(v, 0x7f00ff);


v = compare_exchange<T, n, 
// clang-format off
26, 25, 24, 27, 
30, 29, 28, 19, 
18, 17, 16, 23, 
22, 21, 20, 11, 
10, 9, 8, 15, 
14, 13, 12, 3, 
2, 1, 0, 7, 
6, 5, 4
// clang-format on
>(v, 0x70f0f0f);


v = compare_exchange<T, n, 
// clang-format off
28, 29, 30, 25, 
24, 27, 26, 21, 
20, 23, 22, 17, 
16, 19, 18, 13, 
12, 15, 14, 9, 
8, 11, 10, 5, 
4, 7, 6, 1, 
0, 3, 2
// clang-format on
>(v, 0x13333333);


v = compare_exchange<T, n, 
// clang-format off
30, 28, 29, 26, 
27, 24, 25, 22, 
23, 20, 21, 18, 
19, 16, 17, 14, 
15, 12, 13, 10, 
11, 8, 9, 6, 
7, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v, 0x15555555);

vec_store<T, n>(arr, v);
}
};

#endif