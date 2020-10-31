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
         template<uint32_t _n> typename network = vsort::best,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
constexpr vop::vec_t<T, n> ALWAYS_INLINE CONST_ATTR
sortv(vop::vec_t<T, n> v) {
    return sortgen::
        generate_sort<T, next_p2(n), network<n>, simd_set, builtin_perm>(v);
}

template<typename T,
         uint32_t n,
         template<uint32_t _n> typename network = vsort::best,
         uint32_t          aligned,
         uint32_t          extra_memory,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
constexpr void ALWAYS_INLINE
sort(T * const arr) {
    if constexpr (n < 4) {
        small_sort<T, n>(arr);
    }
    else {
        constexpr uint32_t use_extra_memory =
            extra_memory &&
            (!(std::is_same<network<4>, vsort::best<4>>::value && n > 32));
        vop::vec_t<T, n> v =
            vop::vec_load<T, n, aligned, use_extra_memory, simd_set>(arr);
        v = sortv<T, n, network, simd_set, builtin_perm>(v);
        vop::vec_store<T, n, aligned, use_extra_memory, simd_set>(arr, v);
    }
}

template<typename T,
         uint32_t n,
         template<uint32_t _n> typename network = vsort::best,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
void
sortu(T * const arr) {
    sort<T, n, network, 1, 0, simd_set, builtin_perm>(arr);
}

template<typename T,
         uint32_t n,
         template<uint32_t _n> typename network = vsort::best,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
void
sorta(T * const arr) {
    sort<T, n, network, 1, 0, simd_set, builtin_perm>(arr);
}

template<typename T,
         uint32_t n,
         template<uint32_t _n> typename network = vsort::best,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
void
sortue(T * const arr) {
    sort<T, n, network, 1, 1, simd_set, builtin_perm>(arr);
}

template<typename T,
         uint32_t n,
         template<uint32_t _n> typename network = vsort::best,
         simd_instructions simd_set     = vop::simd_instructions_default,
         builtin_usage     builtin_perm = vop::builtin_perm_default>
void
sortae(T * const arr) {
    sort<T, n, network, 1, 1, simd_set, builtin_perm>(arr);
}

}  // namespace vsort

#endif
