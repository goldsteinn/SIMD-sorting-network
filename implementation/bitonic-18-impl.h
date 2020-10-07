#ifndef _BITONIC_18_IMPL_H_
#define _BITONIC_18_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 18> {
static constexpr uint32_t n = 18;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
16, 17, 15, 13, 
14, 11, 12, 9, 
10, 7, 8, 6, 
4, 5, 2, 3, 
0, 1
// clang-format on
>(v, 0x12a95);


v = compare_exchange<T, n, 
// clang-format off
15, 16, 17, 14, 
13, 9, 10, 11, 
12, 8, 6, 7, 
5, 4, 0, 1, 
2, 3
// clang-format on
>(v, 0x8643);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 16, 17, 
13, 11, 12, 9, 
10, 7, 8, 4, 
5, 6, 2, 3, 
0, 1
// clang-format on
>(v, 0xca95);


v = compare_exchange<T, n, 
// clang-format off
12, 13, 14, 15, 
16, 17, 11, 10, 
9, 6, 5, 8, 
7, 0, 3, 2, 
1, 4
// clang-format on
>(v, 0x7061);


v = compare_exchange<T, n, 
// clang-format off
17, 15, 16, 13, 
14, 12, 11, 10, 
9, 7, 8, 5, 
6, 4, 3, 2, 
1, 0
// clang-format on
>(v, 0xa0a0);


v = compare_exchange<T, n, 
// clang-format off
17, 9, 10, 11, 
12, 13, 14, 15, 
16, 4, 1, 2, 
3, 8, 5, 6, 
7, 0
// clang-format on
>(v, 0x1e1e);


v = compare_exchange<T, n, 
// clang-format off
17, 14, 13, 16, 
15, 10, 9, 12, 
11, 6, 5, 8, 
7, 2, 1, 4, 
3, 0
// clang-format on
>(v, 0x6666);


v = compare_exchange<T, n, 
// clang-format off
17, 15, 16, 13, 
14, 11, 12, 9, 
10, 7, 8, 5, 
6, 3, 4, 1, 
2, 0
// clang-format on
>(v, 0xaaaa);


v = compare_exchange<T, n, 
// clang-format off
7, 8, 1, 2, 
3, 4, 5, 6, 
9, 16, 17, 10, 
11, 12, 13, 14, 
15, 0
// clang-format on
>(v, 0x1fe);


v = compare_exchange<T, n, 
// clang-format off
16, 17, 11, 10, 
13, 12, 15, 14, 
7, 0, 9, 2, 
1, 4, 3, 6, 
5, 8
// clang-format on
>(v, 0x10c87);


v = compare_exchange<T, n, 
// clang-format off
17, 16, 15, 14, 
9, 8, 11, 10, 
13, 12, 3, 6, 
5, 0, 7, 2, 
1, 4
// clang-format on
>(v, 0x309);


v = compare_exchange<T, n, 
// clang-format off
17, 16, 13, 12, 
15, 14, 9, 8, 
11, 10, 5, 4, 
7, 6, 1, 0, 
3, 2
// clang-format on
>(v, 0x3333);


v = compare_exchange<T, n, 
// clang-format off
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
