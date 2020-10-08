#ifndef _BATCHER_32_IMPL_H_
#define _BATCHER_32_IMPL_H_

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
15, 14, 13, 12, 
11, 10, 9, 8, 
7, 6, 5, 4, 
3, 2, 1, 0, 
31, 30, 29, 28, 
27, 26, 25, 24, 
23, 22, 21, 20, 
19, 18, 17, 16
// clang-format on
>(v);


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
>(v);


v = compare_exchange<T, n, 
// clang-format off
27, 26, 25, 24, 
31, 30, 29, 28, 
15, 14, 13, 12, 
11, 10, 9, 8, 
23, 22, 21, 20, 
19, 18, 17, 16, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
29, 28, 31, 30, 
27, 26, 25, 24, 
19, 18, 17, 16, 
23, 22, 21, 20, 
11, 10, 9, 8, 
15, 14, 13, 12, 
7, 6, 5, 4, 
1, 0, 3, 2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
30, 31, 29, 28, 
15, 14, 13, 12, 
23, 22, 21, 20, 
7, 6, 5, 4, 
27, 26, 25, 24, 
11, 10, 9, 8, 
19, 18, 17, 16, 
3, 2, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 30, 29, 28, 
23, 22, 21, 20, 
27, 26, 25, 24, 
15, 14, 13, 12, 
19, 18, 17, 16, 
7, 6, 5, 4, 
11, 10, 9, 8, 
3, 2, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 30, 29, 28, 
25, 24, 27, 26, 
21, 20, 23, 22, 
17, 16, 19, 18, 
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
3, 2, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 30, 15, 14, 
27, 26, 11, 10, 
23, 22, 7, 6, 
19, 18, 3, 2, 
29, 28, 13, 12, 
25, 24, 9, 8, 
21, 20, 5, 4, 
17, 16, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 30, 23, 22, 
27, 26, 19, 18, 
29, 28, 15, 14, 
25, 24, 11, 10, 
21, 20, 7, 6, 
17, 16, 3, 2, 
13, 12, 5, 4, 
9, 8, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 30, 27, 26, 
29, 28, 23, 22, 
25, 24, 19, 18, 
21, 20, 15, 14, 
17, 16, 11, 10, 
13, 12, 7, 6, 
9, 8, 3, 2, 
5, 4, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 30, 28, 29, 
26, 27, 24, 25, 
22, 23, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 15, 29, 13, 
27, 11, 25, 9, 
23, 7, 21, 5, 
19, 3, 17, 1, 
30, 14, 28, 12, 
26, 10, 24, 8, 
22, 6, 20, 4, 
18, 2, 16, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 23, 29, 21, 
27, 19, 25, 17, 
30, 15, 28, 13, 
26, 11, 24, 9, 
22, 7, 20, 5, 
18, 3, 16, 1, 
14, 6, 12, 4, 
10, 2, 8, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 27, 29, 25, 
30, 23, 28, 21, 
26, 19, 24, 17, 
22, 15, 20, 13, 
18, 11, 16, 9, 
14, 7, 12, 5, 
10, 3, 8, 1, 
6, 2, 4, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
31, 29, 30, 27, 
28, 25, 26, 23, 
24, 21, 22, 19, 
20, 17, 18, 15, 
16, 13, 14, 11, 
12, 9, 10, 7, 
8, 5, 6, 3, 
4, 1, 2, 0
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
