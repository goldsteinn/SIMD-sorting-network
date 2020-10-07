#ifndef _BITONIC_19_IMPL_H_
#define _BITONIC_19_IMPL_H_

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
17, 18, 16, 14, 
15, 12, 13, 11, 
9, 10, 7, 8, 
6, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v, 0x25295);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 18, 15, 
14, 13, 11, 12, 
10, 9, 8, 6, 
7, 5, 4, 0, 
1, 2, 3
// clang-format on
>(v, 0x10843);


v = compare_exchange<T, n, 
// clang-format off
15, 16, 17, 18, 
14, 12, 13, 9, 
10, 11, 7, 8, 
4, 5, 6, 2, 
3, 0, 1
// clang-format on
>(v, 0x19295);


v = compare_exchange<T, n, 
// clang-format off
18, 14, 15, 16, 
17, 11, 10, 13, 
12, 9, 6, 5, 
8, 7, 0, 3, 
2, 1, 4
// clang-format on
>(v, 0xcc61);


v = compare_exchange<T, n, 
// clang-format off
18, 16, 17, 14, 
15, 12, 13, 10, 
11, 9, 7, 8, 
5, 6, 4, 3, 
2, 1, 0
// clang-format on
>(v, 0x154a0);


v = compare_exchange<T, n, 
// clang-format off
12, 13, 10, 11, 
14, 17, 18, 15, 
16, 9, 4, 1, 
2, 3, 8, 5, 
6, 7, 0
// clang-format on
>(v, 0x3c1e);


v = compare_exchange<T, n, 
// clang-format off
17, 18, 16, 15, 
12, 9, 14, 11, 
10, 13, 6, 5, 
8, 7, 2, 1, 
4, 3, 0
// clang-format on
>(v, 0x21266);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 14, 13, 
16, 15, 10, 9, 
12, 11, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x66aa);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 15, 16, 
13, 14, 11, 12, 
9, 10, 8, 17, 
18, 5, 4, 3, 
2, 1, 0
// clang-format on
>(v, 0xaac0);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 8, 1, 
2, 3, 4, 5, 
6, 7, 16, 9, 
10, 11, 12, 13, 
14, 15, 0
// clang-format on
>(v, 0x1fe);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 18, 11, 
10, 9, 12, 15, 
14, 13, 0, 3, 
2, 1, 4, 7, 
6, 5, 8
// clang-format on
>(v, 0x10e0f);


v = compare_exchange<T, n, 
// clang-format off
18, 16, 17, 13, 
14, 15, 8, 9, 
10, 11, 12, 5, 
6, 7, 0, 1, 
2, 3, 4
// clang-format on
>(v, 0x12323);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 16, 15, 
12, 13, 14, 11, 
8, 9, 10, 7, 
4, 5, 6, 3, 
0, 1, 2
// clang-format on
>(v, 0x1111);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 16, 14, 
15, 12, 13, 10, 
11, 8, 9, 6, 
7, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v, 0x5555);

vec_store<T, n>(arr, v);
}
};

#endif
