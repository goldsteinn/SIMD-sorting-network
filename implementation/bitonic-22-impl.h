#ifndef _BITONIC_22_IMPL_H_
#define _BITONIC_22_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 22> {
static constexpr uint32_t n = 22;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
20, 21, 19, 17, 
18, 16, 14, 15, 
13, 11, 12, 9, 
10, 8, 6, 7, 
5, 3, 4, 2, 
0, 1
// clang-format on
>(v, 0x124a49);


v = compare_exchange<T, n, 
// clang-format off
19, 20, 21, 18, 
16, 17, 15, 13, 
14, 12, 11, 10, 
8, 9, 5, 6, 
7, 2, 3, 4, 
1, 0
// clang-format on
>(v, 0x92124);


v = compare_exchange<T, n, 
// clang-format off
21, 19, 20, 17, 
18, 16, 14, 15, 
11, 12, 13, 9, 
10, 8, 7, 5, 
6, 1, 2, 3, 
4, 0
// clang-format on
>(v, 0xa4a26);


v = compare_exchange<T, n, 
// clang-format off
17, 18, 19, 20, 
21, 16, 13, 12, 
15, 14, 11, 10, 
5, 6, 7, 8, 
9, 4, 0, 1, 
2, 3
// clang-format on
>(v, 0x63063);


v = compare_exchange<T, n, 
// clang-format off
20, 21, 17, 16, 
19, 18, 14, 15, 
12, 13, 11, 8, 
7, 10, 9, 5, 
6, 4, 2, 3, 
0, 1
// clang-format on
>(v, 0x1351a5);


v = compare_exchange<T, n, 
// clang-format off
13, 14, 18, 19, 
16, 17, 15, 20, 
21, 12, 11, 9, 
10, 7, 8, 1, 
2, 4, 3, 5, 
6, 0
// clang-format on
>(v, 0x56286);


v = compare_exchange<T, n, 
// clang-format off
21, 20, 15, 12, 
13, 14, 19, 16, 
17, 18, 11, 6, 
5, 3, 0, 10, 
9, 4, 8, 2, 
1, 7
// clang-format on
>(v, 0xf069);


v = compare_exchange<T, n, 
// clang-format off
19, 20, 21, 16, 
17, 18, 11, 12, 
13, 14, 15, 8, 
9, 10, 4, 3, 
5, 7, 6, 0, 
1, 2
// clang-format on
>(v, 0x91919);


v = compare_exchange<T, n, 
// clang-format off
21, 19, 20, 18, 
15, 16, 17, 14, 
11, 12, 13, 10, 
7, 8, 9, 6, 
4, 5, 3, 1, 
2, 0
// clang-format on
>(v, 0x88892);


v = compare_exchange<T, n, 
// clang-format off
21, 20, 19, 17, 
18, 15, 16, 13, 
14, 11, 12, 9, 
10, 7, 8, 5, 
6, 3, 4, 2, 
1, 0
// clang-format on
>(v, 0x2aaa8);


v = compare_exchange<T, n, 
// clang-format off
5, 6, 7, 8, 
9, 10, 3, 4, 
13, 12, 11, 16, 
17, 18, 19, 20, 
21, 14, 15, 2, 
1, 0
// clang-format on
>(v, 0x7f8);


v = compare_exchange<T, n, 
// clang-format off
17, 16, 19, 18, 
21, 20, 15, 14, 
5, 6, 7, 2, 
1, 0, 11, 12, 
13, 4, 3, 10, 
9, 8
// clang-format on
>(v, 0x300e7);


v = compare_exchange<T, n, 
// clang-format off
20, 21, 17, 16, 
19, 18, 11, 8, 
9, 10, 15, 12, 
13, 14, 3, 2, 
1, 0, 7, 6, 
5, 4
// clang-format on
>(v, 0x130f0f);


v = compare_exchange<T, n, 
// clang-format off
21, 20, 18, 19, 
16, 17, 13, 12, 
15, 14, 9, 8, 
11, 10, 5, 4, 
7, 6, 1, 0, 
3, 2
// clang-format on
>(v, 0x53333);


v = compare_exchange<T, n, 
// clang-format off
21, 20, 19, 18, 
17, 16, 14, 15, 
12, 13, 10, 11, 
8, 9, 6, 7, 
4, 5, 2, 3, 
0, 1
// clang-format on
>(v, 0x5555);

vec_store<T, n>(arr, v);
}
};

#endif
