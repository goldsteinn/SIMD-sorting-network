#ifndef _BITONIC_10_IMPL_H_
#define _BITONIC_10_IMPL_H_

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
8, 9, 7, 5, 
6, 3, 4, 2, 
0, 1
// clang-format on
>(v, 0x129);


v = compare_exchange<T, n, 
// clang-format off
7, 8, 9, 6, 
5, 4, 2, 3, 
1, 0
// clang-format on
>(v, 0x84);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 8, 9, 
5, 3, 4, 0, 
1, 2
// clang-format on
>(v, 0xc9);


v = compare_exchange<T, n, 
// clang-format off
9, 5, 6, 7, 
8, 2, 1, 4, 
3, 0
// clang-format on
>(v, 0x66);


v = compare_exchange<T, n, 
// clang-format off
9, 7, 8, 5, 
6, 3, 4, 1, 
2, 0
// clang-format on
>(v, 0xaa);


v = compare_exchange<T, n, 
// clang-format off
3, 4, 1, 2, 
5, 8, 9, 6, 
7, 0
// clang-format on
>(v, 0x1e);


v = compare_exchange<T, n, 
// clang-format off
8, 9, 7, 6, 
3, 0, 5, 2, 
1, 4
// clang-format on
>(v, 0x109);


v = compare_exchange<T, n, 
// clang-format off
9, 8, 5, 4, 
7, 6, 1, 0, 
3, 2
// clang-format on
>(v, 0x33);


v = compare_exchange<T, n, 
// clang-format off
9, 8, 6, 7, 
4, 5, 2, 3, 
0, 1
// clang-format on
>(v, 0x55);

vec_store<T, n>(arr, v);
}
};

#endif
