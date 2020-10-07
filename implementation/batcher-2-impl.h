#ifndef _BATCHER_2_IMPL_H_
#define _BATCHER_2_IMPL_H_

#include <sort_base/vec_sort_incl.h>

template<typename T>
struct vsort<T, 2> {
static constexpr uint32_t n = 2;
using v_type                = vec_t<T, n>;

static void
sort(T * arr) {
v_type v = vec_load<T, n>(arr);

v = compare_exchange<T, n, 
// clang-format off
0, 1
// clang-format on
>(v, 0x1);

vec_store<T, n>(arr, v);
}
};

#endif
