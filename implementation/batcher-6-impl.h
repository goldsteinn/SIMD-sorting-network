#ifndef _BATCHER_6_IMPL_H_
#define _BATCHER_6_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 6> {
static constexpr uint32_t n = 6;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
1, 0, 3, 2, 
5, 4
// clang-format on
>(v, 0x3);


v = compare_exchange<T, n, 
// clang-format off
5, 4, 1, 0, 
3, 2
// clang-format on
>(v, 0x3);


v = compare_exchange<T, n, 
// clang-format off
3, 2, 5, 4, 
0, 1
// clang-format on
>(v, 0xd);


v = compare_exchange<T, n, 
// clang-format off
4, 5, 2, 3, 
1, 0
// clang-format on
>(v, 0x14);


v = compare_exchange<T, n, 
// clang-format off
5, 1, 3, 2, 
4, 0
// clang-format on
>(v, 0x2);


v = compare_exchange<T, n, 
// clang-format off
5, 3, 4, 1, 
2, 0
// clang-format on
>(v, 0xa);

vec_store<T, n>(arr, v);
}
};

#endif
