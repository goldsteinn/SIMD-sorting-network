#ifndef _BATCHER_9_IMPL_H_
#define _BATCHER_9_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 9> {
static constexpr uint32_t n = 9;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
0, 3, 2, 1, 
4, 7, 6, 5, 
8
// clang-format on
>(v, 0xf);


v = compare_exchange<T, n, 
// clang-format off
8, 5, 6, 7, 
0, 1, 2, 3, 
4
// clang-format on
>(v, 0x23);


v = compare_exchange<T, n, 
// clang-format off
4, 7, 6, 3, 
8, 5, 0, 1, 
2
// clang-format on
>(v, 0x19);


v = compare_exchange<T, n, 
// clang-format off
2, 7, 4, 5, 
6, 3, 8, 0, 
1
// clang-format on
>(v, 0x15);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 8, 5, 
2, 3, 4, 1, 
0
// clang-format on
>(v, 0x44);


v = compare_exchange<T, n, 
// clang-format off
1, 6, 7, 4, 
5, 2, 3, 8, 
0
// clang-format on
>(v, 0x56);


v = compare_exchange<T, n, 
// clang-format off
5, 7, 3, 8, 
1, 6, 2, 4, 
0
// clang-format on
>(v, 0x2a);


v = compare_exchange<T, n, 
// clang-format off
7, 8, 5, 6, 
3, 4, 1, 2, 
0
// clang-format on
>(v, 0xaa);

vec_store<T, n>(arr, v);
}
};

#endif
