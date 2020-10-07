#ifndef _BATCHER_20_IMPL_H_
#define _BATCHER_20_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 20> {
static constexpr uint32_t n = 20;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
3, 2, 1, 0, 
7, 6, 5, 4, 
11, 10, 9, 8, 
15, 14, 13, 12, 
19, 18, 17, 16
// clang-format on
>(v, 0xff);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 17, 16, 
15, 14, 13, 12, 
3, 2, 1, 0, 
7, 6, 5, 4, 
11, 10, 9, 8
// clang-format on
>(v, 0xf);


v = compare_exchange<T, n, 
// clang-format off
11, 10, 9, 8, 
15, 14, 13, 12, 
19, 18, 17, 16, 
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v, 0xf0f);


v = compare_exchange<T, n, 
// clang-format off
7, 6, 5, 4, 
11, 10, 9, 8, 
15, 14, 13, 12, 
19, 18, 17, 16, 
1, 0, 3, 2
// clang-format on
>(v, 0xff3);


v = compare_exchange<T, n, 
// clang-format off
15, 14, 13, 12, 
19, 18, 17, 16, 
7, 6, 5, 4, 
11, 10, 9, 8, 
3, 2, 0, 1
// clang-format on
>(v, 0xf0f1);


v = compare_exchange<T, n, 
// clang-format off
17, 16, 19, 18, 
13, 12, 15, 14, 
9, 8, 11, 10, 
5, 4, 7, 6, 
3, 2, 1, 0
// clang-format on
>(v, 0x33330);


v = compare_exchange<T, n, 
// clang-format off
18, 19, 3, 2, 
15, 14, 7, 6, 
11, 10, 9, 8, 
13, 12, 5, 4, 
17, 16, 1, 0
// clang-format on
>(v, 0x400cc);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 11, 10, 
15, 14, 13, 12, 
17, 16, 3, 2, 
7, 6, 5, 4, 
9, 8, 1, 0
// clang-format on
>(v, 0xc0c);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 15, 14, 
17, 16, 11, 10, 
13, 12, 7, 6, 
9, 8, 3, 2, 
5, 4, 1, 0
// clang-format on
>(v, 0xcccc);


v = compare_exchange<T, n, 
// clang-format off
19, 18, 16, 17, 
14, 15, 12, 13, 
10, 11, 8, 9, 
6, 7, 4, 5, 
2, 3, 1, 0
// clang-format on
>(v, 0x15554);


v = compare_exchange<T, n, 
// clang-format off
19, 3, 17, 1, 
15, 7, 13, 5, 
11, 10, 9, 8, 
14, 6, 12, 4, 
18, 2, 16, 0
// clang-format on
>(v, 0xaa);


v = compare_exchange<T, n, 
// clang-format off
19, 11, 17, 9, 
15, 14, 13, 12, 
18, 3, 16, 1, 
7, 6, 5, 4, 
10, 2, 8, 0
// clang-format on
>(v, 0xa0a);


v = compare_exchange<T, n, 
// clang-format off
19, 15, 17, 13, 
18, 11, 16, 9, 
14, 7, 12, 5, 
10, 3, 8, 1, 
6, 2, 4, 0
// clang-format on
>(v, 0xaaaa);


v = compare_exchange<T, n, 
// clang-format off
19, 17, 18, 15, 
16, 13, 14, 11, 
12, 9, 10, 7, 
8, 5, 6, 3, 
4, 1, 2, 0
// clang-format on
>(v, 0x2aaaa);

vec_store<T, n>(arr, v);
}
};

#endif
