#ifndef _VEC_SORT_INCL_H_
#define _VEC_SORT_INCL_H_

#include <immintrin.h>
#include <stdint.h>

#include <instructions/instructions.h>
#include <util/cpp_attributes.h>

namespace vsort {
template<typename T,
         uint32_t n,
         typename network,
         vop::instruction_set instructions>
struct vec_sort;

}


#endif
