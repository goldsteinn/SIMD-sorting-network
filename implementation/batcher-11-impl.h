#ifndef _BATCHER_11_IMPL_H_
#define _BATCHER_11_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 11> {
static constexpr uint32_t n = 11;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
2, 1, 0, 3, 
6, 5, 4, 7, 
10, 9, 8
// clang-format on
>(v, 0xf);


v = compare_exchange<T, n, 
// clang-format off
10, 9, 8, 7, 
2, 1, 0, 3, 
6, 5, 4
// clang-format on
>(v, 0x7);


v = compare_exchange<T, n, 
// clang-format off
6, 5, 4, 7, 
10, 9, 8, 1, 
0, 3, 2
// clang-format on
>(v, 0x73);


v = compare_exchange<T, n, 
// clang-format off
8, 3, 10, 5, 
4, 7, 6, 9, 
2, 0, 1
// clang-format on
>(v, 0x139);


v = compare_exchange<T, n, 
// clang-format off
10, 7, 2, 9, 
6, 3, 4, 5, 
8, 1, 0
// clang-format on
>(v, 0x8c);


v = compare_exchange<T, n, 
// clang-format off
10, 9, 6, 7, 
8, 5, 2, 3, 
4, 1, 0
// clang-format on
>(v, 0x44);


v = compare_exchange<T, n, 
// clang-format off
10, 8, 9, 6, 
7, 4, 5, 2, 
3, 1, 0
// clang-format on
>(v, 0x154);


v = compare_exchange<T, n, 
// clang-format off
3, 9, 1, 7, 
6, 5, 4, 10, 
2, 8, 0
// clang-format on
>(v, 0xa);


v = compare_exchange<T, n, 
// clang-format off
7, 9, 5, 10, 
3, 8, 1, 6, 
2, 4, 0
// clang-format on
>(v, 0xaa);


v = compare_exchange<T, n, 
// clang-format off
9, 10, 7, 8, 
5, 6, 3, 4, 
1, 2, 0
// clang-format on
>(v, 0x2aa);

vec_store<T, n>(arr, v);
}
};

#endif
