#ifndef _BATCHER_24_IMPL_H_
#define _BATCHER_24_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 24> {
static constexpr uint32_t n = 24;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
7, 6, 5, 4, 
3, 2, 1, 0, 
15, 14, 13, 12, 
11, 10, 9, 8, 
23, 22, 21, 20, 
19, 18, 17, 16
// clang-format on
>(v, 0xff);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 21, 20, 
19, 18, 17, 16, 
7, 6, 5, 4, 
3, 2, 1, 0, 
15, 14, 13, 12, 
11, 10, 9, 8
// clang-format on
>(v, 0xff);


v = compare_exchange<T, n, 
// clang-format off
15, 14, 13, 12, 
11, 10, 9, 8, 
23, 22, 21, 20, 
19, 18, 17, 16, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v, 0xff0f);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 17, 16, 
23, 22, 21, 20, 
11, 10, 9, 8, 
15, 14, 13, 12, 
7, 6, 5, 4, 
1, 0, 3, 2
// clang-format on
>(v, 0xf0f03);


v = compare_exchange<T, n, 
// clang-format off
21, 20, 23, 22, 
7, 6, 5, 4, 
15, 14, 13, 12, 
11, 10, 9, 8, 
19, 18, 17, 16, 
3, 2, 0, 1
// clang-format on
>(v, 0x3000f1);


v = compare_exchange<T, n, 
// clang-format off
22, 23, 21, 20, 
15, 14, 13, 12, 
19, 18, 17, 16, 
7, 6, 5, 4, 
11, 10, 9, 8, 
3, 2, 1, 0
// clang-format on
>(v, 0x40f0f0);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 21, 20, 
17, 16, 19, 18, 
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
3, 2, 1, 0
// clang-format on
>(v, 0x33330);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 7, 6, 
19, 18, 3, 2, 
15, 14, 13, 12, 
11, 10, 9, 8, 
21, 20, 5, 4, 
17, 16, 1, 0
// clang-format on
>(v, 0xcc);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 15, 14, 
19, 18, 11, 10, 
21, 20, 7, 6, 
17, 16, 3, 2, 
13, 12, 5, 4, 
9, 8, 1, 0
// clang-format on
>(v, 0xcccc);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 19, 18, 
21, 20, 15, 14, 
17, 16, 11, 10, 
13, 12, 7, 6, 
9, 8, 3, 2, 
5, 4, 1, 0
// clang-format on
>(v, 0xccccc);


v = compare_exchange<T, n, 
// clang-format off
23, 22, 20, 21, 
18, 19, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 1, 0
// clang-format on
>(v, 0x155554);


v = compare_exchange<T, n, 
// clang-format off
23, 7, 21, 5, 
19, 3, 17, 1, 
15, 14, 13, 12, 
11, 10, 9, 8, 
22, 6, 20, 4, 
18, 2, 16, 0
// clang-format on
>(v, 0xaa);


v = compare_exchange<T, n, 
// clang-format off
23, 15, 21, 13, 
19, 11, 17, 9, 
22, 7, 20, 5, 
18, 3, 16, 1, 
14, 6, 12, 4, 
10, 2, 8, 0
// clang-format on
>(v, 0xaaaa);


v = compare_exchange<T, n, 
// clang-format off
23, 19, 21, 17, 
22, 15, 20, 13, 
18, 11, 16, 9, 
14, 7, 12, 5, 
10, 3, 8, 1, 
6, 2, 4, 0
// clang-format on
>(v, 0xaaaaa);


v = compare_exchange<T, n, 
// clang-format off
23, 21, 22, 19, 
20, 17, 18, 15, 
16, 13, 14, 11, 
12, 9, 10, 7, 
8, 5, 6, 3, 
4, 1, 2, 0
// clang-format on
>(v, 0x2aaaaa);

vec_store<T, n>(arr, v);
}
};

#endif
