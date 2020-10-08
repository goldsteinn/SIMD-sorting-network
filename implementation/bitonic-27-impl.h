#ifndef _BITONIC_27_IMPL_H_
#define _BITONIC_27_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 27> {
static constexpr uint32_t n = 27;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
25, 26, 23, 24, 
21, 22, 20, 18, 
19, 16, 17, 14, 
15, 13, 11, 12, 
9, 10, 7, 8, 
6, 4, 5, 3, 
1, 2, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
23, 24, 25, 26, 
22, 20, 21, 16, 
17, 18, 19, 13, 
14, 15, 9, 10, 
11, 12, 6, 7, 
8, 3, 4, 5, 
2, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
25, 26, 23, 24, 
21, 22, 20, 18, 
19, 16, 17, 15, 
13, 14, 11, 12, 
9, 10, 8, 6, 
7, 5, 3, 4, 
1, 2, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
20, 21, 22, 23, 
24, 25, 26, 19, 
13, 14, 15, 16, 
17, 18, 12, 6, 
7, 8, 9, 10, 
11, 1, 2, 3, 
4, 5, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
24, 25, 26, 21, 
20, 23, 22, 17, 
16, 19, 18, 13, 
14, 15, 10, 9, 
12, 11, 6, 7, 
8, 4, 5, 1, 
0, 3, 2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
26, 24, 25, 22, 
23, 20, 21, 18, 
19, 16, 17, 14, 
15, 13, 11, 12, 
9, 10, 7, 8, 
4, 5, 6, 2, 
3, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
14, 15, 16, 17, 
18, 19, 20, 21, 
22, 23, 24, 25, 
26, 13, 12, 11, 
0, 1, 2, 3, 
6, 5, 4, 7, 
8, 9, 10
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
22, 21, 24, 23, 
26, 25, 16, 15, 
14, 13, 20, 19, 
18, 17, 8, 7, 
6, 5, 12, 11, 
10, 9, 0, 1, 
2, 3, 4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
25, 26, 22, 21, 
24, 23, 18, 17, 
20, 19, 14, 13, 
16, 15, 10, 9, 
12, 11, 6, 5, 
8, 7, 2, 3, 
4, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
26, 25, 23, 24, 
21, 22, 19, 20, 
17, 18, 15, 16, 
13, 14, 11, 12, 
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
2, 3, 4, 5, 
6, 7, 8, 9, 
10, 11, 12, 15, 
14, 13, 16, 17, 
18, 19, 20, 21, 
22, 23, 24, 25, 
26, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
18, 17, 16, 19, 
22, 21, 20, 23, 
26, 25, 24, 5, 
6, 7, 4, 3, 
2, 1, 0, 13, 
14, 15, 12, 11, 
10, 9, 8
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
24, 25, 26, 23, 
18, 17, 16, 19, 
22, 21, 20, 9, 
10, 11, 8, 13, 
14, 15, 12, 3, 
2, 1, 0, 7, 
6, 5, 4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
26, 24, 25, 21, 
20, 23, 22, 17, 
16, 19, 18, 13, 
12, 15, 14, 9, 
8, 11, 10, 5, 
4, 7, 6, 1, 
0, 3, 2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
26, 25, 24, 22, 
23, 20, 21, 18, 
19, 16, 17, 14, 
15, 12, 13, 10, 
11, 8, 9, 6, 
7, 4, 5, 2, 
3, 0, 1
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
