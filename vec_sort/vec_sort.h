#ifndef _VEC_SORT_H_
#define _VEC_SORT_H_

#include <instructions/vector_operations.h>
#include <networks/networks.h>
#include <vec_sort/sort_builder.h>


namespace vsort {


template<typename T,
         uint32_t n,
         template<uint32_t _n>
         typename network,
         instruction_set operations = vop::instruction_set_default>
void NEVER_INLINE
sort(T * const arr) {
    vop::vec_t<T, n> v = vop::vec_load<T, n>(arr);
    v = sortgen::generate_sort<T, n, network<n>, operations>(v);
    vop::vec_store<T, n>(arr, v);
}

}  // namespace vsort

#endif
