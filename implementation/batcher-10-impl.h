#ifndef _BATCHER_10_IMPL_H_
#define _BATCHER_10_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 10> {
static constexpr uint32_t n = 10;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
1, 0, 3, 2, 
5, 4, 7, 6, 
9, 8
// clang-format on
>(v, 0xf);


v = compare_exchange<T, n, 
// clang-format off
9, 8, 7, 6, 
1, 0, 3, 2, 
5, 4
// clang-format on
>(v, 0x3);


v = compare_exchange<T, n, 
// clang-format off
5, 4, 7, 6, 
9, 8, 1, 0, 
3, 2
// clang-format on
>(v, 0x33);


v = compare_exchange<T, n, 
// clang-format off
3, 2, 5, 4, 
7, 6, 9, 8, 
0, 1
// clang-format on
>(v, 0x3d);


v = compare_exchange<T, n, 
// clang-format off
7, 6, 9, 8, 
3, 2, 5, 4, 
1, 0
// clang-format on
>(v, 0xcc);


v = compare_exchange<T, n, 
// clang-format off
8, 9, 6, 7, 
4, 5, 2, 3, 
1, 0
// clang-format on
>(v, 0x154);


v = compare_exchange<T, n, 
// clang-format off
9, 1, 7, 3, 
5, 4, 6, 2, 
8, 0
// clang-format on
>(v, 0xa);


v = compare_exchange<T, n, 
// clang-format off
9, 5, 7, 6, 
8, 1, 3, 2, 
4, 0
// clang-format on
>(v, 0x22);


v = compare_exchange<T, n, 
// clang-format off
9, 7, 8, 5, 
6, 3, 4, 1, 
2, 0
// clang-format on
>(v, 0xaa);

vec_store<T, n>(arr, v);
}
};

#endif
