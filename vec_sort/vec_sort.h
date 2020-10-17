#ifndef _VEC_SORT_H_
#define _VEC_SORT_H_

#include <instructions/vector_operations.h>
#include <networks/networks.h>
#include <util/constexpr_util.h>
#include <vec_sort/sort_builder.h>

namespace vsort {

#define CSWAP(X, Y)                                                            \
    {                                                                          \
        T _tmp = (X);                                                          \
        (X)    = _tmp > (Y) ? (Y) : _tmp;                                      \
        (Y)    = _tmp > (Y) ? _tmp : (Y);                                      \
    }

template<typename T, uint32_t n>
void ALWAYS_INLINE
small_sort(T * const arr) {
    // n == 1 do nothing
    if constexpr (n == 2) {
        CSWAP(arr[0], arr[1]);
    }
    else if constexpr (n == 3) {
        CSWAP(arr[1], arr[2]);
        CSWAP(arr[0], arr[2]);
        CSWAP(arr[0], arr[1]);
    }
}
#undef CSWAP

template<typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
void NEVER_INLINE
sort(T * const arr) {
    if constexpr (n < 4) {
        small_sort<T, n>(arr);
    }
    else {
        vop::vec_t<T, n> v = vop::vec_load<T, n>(arr);
        v = sortgen::
            generate_sort<T, next_p2(n), network<n>, simd_set, builtin_perm>(v);
        vop::vec_store<T, n>(arr, v);
    }
}

}  // namespace vsort

#endif
