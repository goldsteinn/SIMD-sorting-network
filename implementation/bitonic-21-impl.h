#ifndef _BITONIC_21_IMPL_H_
#define _BITONIC_21_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 21> {
static constexpr uint32_t n = 21;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
19, 20, 18, 16, 
17, 15, 13, 14, 
12, 10, 11, 8, 
9, 7, 5, 6, 
3, 4, 2, 0, 
1
// clang-format on
>(v, 0x92529);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 20, 17, 
15, 16, 14, 12, 
13, 11, 10, 9, 
7, 8, 6, 5, 
2, 3, 4, 1, 
0
// clang-format on
>(v, 0x49084);


v = compare_exchange<T, n, 
// clang-format off
20, 18, 19, 16, 
17, 15, 13, 14, 
10, 11, 12, 8, 
9, 5, 6, 7, 
1, 2, 3, 4, 
0
// clang-format on
>(v, 0x52526);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 18, 19, 
20, 15, 12, 11, 
14, 13, 10, 7, 
6, 9, 8, 5, 
4, 0, 1, 2, 
3
// clang-format on
>(v, 0x318c3);


v = compare_exchange<T, n, 
// clang-format off
19, 20, 16, 15, 
18, 17, 13, 14, 
11, 12, 10, 8, 
9, 6, 7, 5, 
4, 2, 3, 0, 
1
// clang-format on
>(v, 0x9a945);


v = compare_exchange<T, n, 
// clang-format off
12, 13, 17, 18, 
15, 16, 14, 19, 
20, 11, 10, 9, 
2, 3, 0, 1, 
4, 7, 8, 5, 
6
// clang-format on
>(v, 0x2b00f);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 14, 11, 
12, 13, 18, 15, 
16, 17, 10, 5, 
8, 7, 4, 9, 
6, 3, 2, 0, 
1
// clang-format on
>(v, 0x7831);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 20, 15, 
16, 17, 10, 11, 
12, 13, 14, 7, 
6, 9, 8, 3, 
2, 5, 4, 1, 
0
// clang-format on
>(v, 0x48ccc);


v = compare_exchange<T, n, 
// clang-format off
20, 18, 19, 17, 
14, 15, 16, 13, 
10, 11, 12, 8, 
9, 6, 7, 4, 
5, 2, 3, 1, 
0
// clang-format on
>(v, 0x44554);


v = compare_exchange<T, n, 
// clang-format off
5, 6, 7, 16, 
17, 14, 15, 12, 
13, 10, 11, 9, 
8, 18, 19, 20, 
4, 3, 2, 1, 
0
// clang-format on
>(v, 0x154e0);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 18, 8, 
9, 2, 3, 4, 
5, 6, 7, 16, 
17, 10, 11, 12, 
13, 14, 15, 1, 
0
// clang-format on
>(v, 0x3fc);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 18, 19, 
20, 11, 10, 13, 
12, 15, 14, 1, 
0, 3, 2, 5, 
4, 7, 6, 9, 
8
// clang-format on
>(v, 0x30c0f);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 16, 17, 
18, 15, 14, 8, 
9, 11, 10, 12, 
13, 7, 6, 1, 
0, 3, 2, 5, 
4
// clang-format on
>(v, 0x10303);


v = compare_exchange<T, n, 
// clang-format off
20, 18, 19, 16, 
17, 13, 12, 15, 
14, 8, 9, 10, 
11, 5, 4, 7, 
6, 1, 0, 3, 
2
// clang-format on
>(v, 0x53333);


v = compare_exchange<T, n, 
// clang-format off
20, 19, 18, 17, 
16, 14, 15, 12, 
13, 10, 11, 8, 
9, 6, 7, 4, 
5, 2, 3, 0, 
1
// clang-format on
>(v, 0x5555);

vec_store<T, n>(arr, v);
}
};

#endif
