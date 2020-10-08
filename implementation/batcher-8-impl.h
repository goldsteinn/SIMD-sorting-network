#ifndef _BATCHER_8_IMPL_H_
#define _BATCHER_8_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 8> {
static constexpr uint32_t n = 8;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
3, 2, 1, 0, 
7, 6, 5, 4
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
5, 4, 7, 6, 
1, 0, 3, 2
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
6, 7, 3, 2, 
5, 4, 0, 1
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
7, 6, 4, 5, 
2, 3, 1, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
7, 3, 5, 1, 
6, 2, 4, 0
// clang-format on
>(v);


v = compare_exchange<T, n, 
// clang-format off
7, 5, 6, 3, 
4, 1, 2, 0
// clang-format on
>(v);

vec_store<T, n>(arr, v);
}
};

#endif
