#ifndef _VEC_SORT_H_
#define _VEC_SORT_H_

#include <sort_base/vec_sort_incl.h>

#include <util/constexpr_util.h>
#include <util/cpp_attributes.h>
#include <util/integer_range.h>


namespace vsort {
namespace internal {

template<typename T, uint32_t n>
struct network_builder {


    template<uint32_t... perm_indexes_slice>
    static constexpr vop::vec_t<T, n> ALWAYS_INLINE CONST_ATTR
    call_compare_exchange(vop::vec_t<T, n> v,
                          std::integer_sequence<uint32_t, perm_indexes_slice...>
                              _perm_indexes_slice) {
        return vop::compare_exchange<T, n, perm_indexes_slice...>(v);
    }

    template<uint32_t group_idx, uint32_t ngroups, uint32_t... perm_indexes>
    static constexpr vop::vec_t<T, n> ALWAYS_INLINE CONST_ATTR
    build_kernel(
        vop::vec_t<T, n>                                 v,
        std::integer_sequence<uint32_t, perm_indexes...> _perm_indexes) {
        if constexpr (group_idx == ngroups) {
            return v;
        }
        else {
            return build_kernel<group_idx + 1, ngroups>(
                call_compare_exchange<>(
                    v,
                    slice<uint32_t, n * group_idx, n *(group_idx + 1)>(
                        _perm_indexes)),
                _perm_indexes);
        }
    }

    template<uint32_t... perm_indexes>
    static constexpr vop::vec_t<T, n> ALWAYS_INLINE CONST_ATTR
    build(vop::vec_t<T, n>                                 v,
          std::integer_sequence<uint32_t, perm_indexes...> _perm_indexes) {
        constexpr uint32_t ngroups = (sizeof...(perm_indexes)) / n;
        return build_kernel<0, ngroups>(v, _perm_indexes);
    }
};

}  // namespace internal


template<typename T, uint32_t n, typename network>
struct vec_sort {

    static void NEVER_INLINE
    sort(T * const arr) {

        vop::vec_t<T, n> v = vop::vec_load<T, n>(arr);
        v = internal::network_builder<T, n>::build(v, network{});
        vop::vec_store<T, n>(arr, v);
    }
};

}  // namespace vsort

#endif
