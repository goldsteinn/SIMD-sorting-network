
/*

Sorting Network Information:
        Sort Size                        : 4
        Underlying Sort Type             : uint64_t
        Network Generation Algorithm     : bitonic
        Network Depth                    : 3
        SIMD Instructions                : 2 / 12
        Optimization Preference          : space
        SIMD Type                        : __m256i
        SIMD Instruction Set(s) Used     : AVX, AVX2, AVX512vl, AVX512f
        SIMD Instruction Set(s) Excluded : None
        Aligned Load & Store             : True
        Integer Aligned Load & Store     : True
        Full Load & Store                : True

Performance Notes:
1) If you are sorting an array where there IS valid memory
   up to the nearest sizeof a SIMD register, you will get an
   improvement enable "EXTRA_MEMORY" (this turns on "Full Load
   & Store". Note that enabling "Full Load & Store" will not
   modify any of the memory not being sorted and will not affect
   the sort in any way. i.e sort(3) [4, 3, 2, 1] with full load
   will still return [2, 3, 4, 1]. Note even if you don't have
   enough memory for a full SIMD register, enabling "INT_ALIGNED"
   will also improve load efficiency and only requires that
   there is valid memory up the next factor of sizeof(int).

2) If your sort size is not a power of 2 you are likely running
   into less efficient instructions. This is especially noticable
   when sorting 8 bit and 16 bit values. If rounding you sort
   size up to the next power of 2 will not cost any additional
   depth it almost definetly worth doing so. The "Best" Network
   Algorithm automatically does this in many cases.

3) There are two optimization settings, "Optimization.SPACE"
   and "Optimization.UOP". The former will essentially break
   ties by picking the instruction that uses less memory (i.e
   doesn't have to store a register's initializing in memory.
   The latter will break ties but simply selecting whatever
   instructions use the least UOPs. Which is best is probably
   application dependent. Note that while "Optimization.SPACE"
   will save .rodata memory it will often cost more in .text
   memory. Generally it is advise to optimize for space if you
   are calling sparingly and uop if you are calling sort in
   a loop.

 */

#include <immintrin.h>
#include <stdint.h>


#ifndef _SIMD_SORT_VEC_bitonic_4_8u_vec_H_
#define _SIMD_SORT_VEC_bitonic_4_8u_vec_H_

/* SIMD Sort */
__m256i __attribute__((const)) bitonic_4_8u_vec(__m256i v) {

    /* Pairs: ([2,3], [0,1]) */
    /* Perm:  ( 2,  3,  0,  1) */
    __m256i perm0 = _mm256_shuffle_epi32(v, uint8_t(0x4e));
    __m256i min0  = _mm256_min_epu64(v, perm0);
    __m256i max0  = _mm256_max_epu64(v, perm0);
    __m256i v0    = _mm256_blend_epi32(max0, min0, 0x33);

    /* Pairs: ([0,3], [1,2]) */
    /* Perm:  ( 0,  1,  2,  3) */
    __m256i perm1 = _mm256_permute4x64_epi64(v0, 0x1b);
    __m256i min1  = _mm256_min_epu64(v0, perm1);
    __m256i max1  = _mm256_max_epu64(v0, perm1);
    __m256i v1    = _mm256_blend_epi32(max1, min1, 0xf);

    /* Pairs: ([2,3], [0,1]) */
    /* Perm:  ( 2,  3,  0,  1) */
    __m256i perm2 = _mm256_shuffle_epi32(v1, uint8_t(0x4e));
    __m256i min2  = _mm256_min_epu64(v1, perm2);
    __m256i max2  = _mm256_max_epu64(v1, perm2);
    __m256i v2    = _mm256_blend_epi32(max2, min2, 0x33);

    return v2;
}
#endif


#ifndef _SIMD_SORT_ARR_bitonic_4_8u_H_
#define _SIMD_SORT_ARR_bitonic_4_8u_H_

/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_4_8u(uint64_t * const arr) {

    __m256i v = _mm256_load_si256((__m256i *)arr);

    v = bitonic_4_8u_vec(v);

    _mm256_store_si256((__m256i *)arr, v);
}
#endif

