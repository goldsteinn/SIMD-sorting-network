#ifndef _VEC_SORT_H_
#define _VEC_SORT_H_

#include <algorithms/networks.h>
#include <sort_base/vec_sort_incl.h>
#include <util/constexpr_util.h>
/*
template<uint32_t       n,
         uint32_t       groups,
         uint32_t       group_idx,
         const uint32_t indexes[n * groups]>
struct permutation_arr {
    constexpr permutation_arr() : arr() {
        for (uint32_t i = n * group_idx; i < n * (group_idx + 1); i += 2) {
            arr[(n - 1) - indexes[i]]     = indexes[i + 1];
            arr[(n - 1) - indexes[i + 1]] = indexes[i];
        }
    }

    uint32_t arr[n];
};

template<typename T, uint32_t n, uint32_t... indexes>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
call_compare_exchange_impl(vec_t<T, n> v) {
    return compare_exchange<T, n, indexes...>(v);
}

template<typename T,
         uint32_t n,
         uint32_t groups,
         //         uint32_t       group_idx,
         const uint32_t indexes[n * groups],
         uint32_t... seq_indexes>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
call_compare_exchange0(vec_t<T, n> v,
                       std::integer_sequence<uint32_t, seq_indexes...>) {
    //    constexpr permutation_arr<n, groups, group_idx, indexes> perm_arr =
      //    permutation_arr<n, groups, group_idx, indexes>();
    return call_compare_exchange_impl<T, n, indexes[seq_indexes]...>(v);
}

template<typename T,
         uint32_t       n,
         uint32_t       groups,
         uint32_t       group_idx,
         const uint32_t indexes[n * groups]>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
call_compare_exchange(vec_t<T, n> v) {
    return call_compare_exchange0<T, n, groups, indexes>(
        v,
        make_integer_range<uint32_t, n * group_idx, n *(group_idx + 1)>{});
}


template<typename T,
         uint32_t       n,
         uint32_t       groups,
         uint32_t       group_idx,
         const uint32_t indexes[n * groups]>
constexpr vec_t<T, n> ALWAYS_INLINE CONST_ATTR
make_group(vec_t<T, n> v) {
    if constexpr (group_idx < (groups - 1)) {
        v = call_compare_exchange<T, n, groups, group_idx, indexes>(v);
        return make_group<T, n, groups, group_idx + 1, indexes>(v);
    }
    else {
        return call_compare_exchange<T, n, groups, group_idx, indexes>(v);
    }
}

template<typename T,
         uint32_t       n,
         uint32_t       groups,
         const uint32_t indexes[n * groups]>
vec_t<T, n>
make_sort(vec_t<T, n> v) {
    return make_group<T, n, groups, 0, indexes>(v);
}

template<typename T, uint32_t n, typename algorithm>
struct vec_sort {

    static void
    sort(T * const arr) {
        vec_t<T, n> v = vec_load<T, n>(arr);
        v = make_sort<T, n, algorithm::groups, algorithm::indexes>(v);
        vec_store<T, n>(arr, v);
    }
};*/


#endif
