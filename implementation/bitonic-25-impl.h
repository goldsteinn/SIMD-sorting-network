#ifndef _BITONIC_25_IMPL_H_
#define _BITONIC_25_IMPL_H_

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
23, 24, 21, 22, 
19, 20, 18, 16, 
17, 15, 13, 14, 
12, 10, 11, 9, 
7, 8, 6, 4, 
5, 3, 1, 2, 
0
// clang-format on
>(v, 0xa92492);


v = compare_exchange<T, n, 
// clang-format off
21, 22, 23, 24, 
20, 18, 19, 17, 
15, 16, 12, 13, 
14, 11, 9, 10, 
6, 7, 8, 3, 
4, 5, 2, 0, 
1
// clang-format on
>(v, 0x649249);


v = compare_exchange<T, n, 
// clang-format off
23, 24, 21, 22, 
19, 20, 18, 16, 
17, 15, 14, 12, 
13, 10, 11, 9, 
8, 6, 7, 5, 
3, 4, 1, 2, 
0
// clang-format on
>(v, 0xa9144a);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 20, 21, 
22, 23, 24, 17, 
12, 13, 14, 15, 
16, 11, 6, 7, 
8, 9, 10, 1, 
2, 3, 4, 5, 
0
// clang-format on
>(v, 0x1c30c6);


v = compare_exchange<T, n, 
// clang-format off
22, 23, 24, 19, 
18, 21, 20, 15, 
14, 17, 16, 12, 
13, 9, 8, 11, 
10, 6, 7, 4, 
5, 1, 0, 3, 
2
// clang-format on
>(v, 0x4cd353);


v = compare_exchange<T, n, 
// clang-format off
13, 22, 23, 20, 
21, 18, 19, 16, 
17, 14, 15, 24, 
12, 10, 11, 8, 
9, 7, 6, 5, 
4, 2, 3, 0, 
1
// clang-format on
>(v, 0x556505);


v = compare_exchange<T, n, 
// clang-format off
24, 14, 15, 16, 
17, 19, 18, 20, 
21, 22, 23, 13, 
12, 11, 10, 0, 
1, 2, 3, 5, 
4, 6, 7, 8, 
9
// clang-format on
>(v, 0x3c00f);


v = compare_exchange<T, n, 
// clang-format off
20, 21, 22, 23, 
24, 14, 15, 13, 
12, 18, 19, 17, 
16, 7, 6, 4, 
5, 11, 10, 8, 
9, 1, 0, 3, 
2
// clang-format on
>(v, 0x30f0f3);


v = compare_exchange<T, n, 
// clang-format off
24, 23, 20, 21, 
22, 16, 17, 18, 
19, 13, 12, 15, 
14, 9, 8, 11, 
10, 4, 5, 6, 
7, 2, 3, 0, 
1
// clang-format on
>(v, 0x133335);


v = compare_exchange<T, n, 
// clang-format off
3, 22, 23, 20, 
21, 18, 19, 16, 
17, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 24, 2, 1, 
0
// clang-format on
>(v, 0x555558);


v = compare_exchange<T, n, 
// clang-format off
24, 4, 5, 6, 
7, 8, 9, 10, 
11, 15, 14, 13, 
12, 16, 17, 18, 
19, 20, 21, 22, 
23, 3, 2, 1, 
0
// clang-format on
>(v, 0xff0);


v = compare_exchange<T, n, 
// clang-format off
16, 19, 18, 17, 
20, 23, 22, 21, 
24, 4, 5, 6, 
7, 3, 2, 1, 
0, 12, 13, 14, 
15, 11, 10, 9, 
8
// clang-format on
>(v, 0xf00ff);


v = compare_exchange<T, n, 
// clang-format off
24, 21, 22, 23, 
16, 17, 18, 19, 
20, 8, 9, 10, 
11, 12, 13, 14, 
15, 3, 2, 1, 
0, 7, 6, 5, 
4
// clang-format on
>(v, 0x230f0f);


v = compare_exchange<T, n, 
// clang-format off
24, 23, 20, 21, 
22, 19, 16, 17, 
18, 13, 12, 15, 
14, 9, 8, 11, 
10, 5, 4, 7, 
6, 1, 0, 3, 
2
// clang-format on
>(v, 0x113333);


v = compare_exchange<T, n, 
// clang-format off
24, 22, 23, 20, 
21, 18, 19, 16, 
17, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v, 0x555555);

vec_store<T, n>(arr, v);
}
};

#endif
