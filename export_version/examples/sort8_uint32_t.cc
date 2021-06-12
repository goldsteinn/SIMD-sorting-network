
/*

Sorting Network Information:
        Sort Size                        : 8
        Underlying Sort Type             : uint32_t
        Network Generation Algorithm     : bitonic
        Network Depth                    : 6
        SIMD Instructions                : 2 / 31
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


#ifndef _SIMD_SORT_VEC_bitonic_8_4u_vec_H_
#define _SIMD_SORT_VEC_bitonic_8_4u_vec_H_

/* SIMD Sort */
__m256i __attribute__((const)) bitonic_8_4u_vec(__m256i v) {

    /* Pairs: ([6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  ( 6,  7,  4,  5,  2,  3,  0,  1) */
    __m256i perm0 = _mm256_shuffle_epi32(v, uint8_t(0xb1));
    __m256i min0  = _mm256_min_epu32(v, perm0);
    __m256i max0  = _mm256_max_epu32(v, perm0);
    __m256i v0    = _mm256_blend_epi32(max0, min0, 0x55);

    /* Pairs: ([4,7], [5,6], [0,3], [1,2]) */
    /* Perm:  ( 4,  5,  6,  7,  0,  1,  2,  3) */
    __m256i perm1 = _mm256_shuffle_epi32(v0, uint8_t(0x1b));
    __m256i min1  = _mm256_min_epu32(v0, perm1);
    __m256i max1  = _mm256_max_epu32(v0, perm1);

    /* Pairs: ([6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  ( 6,  7,  4,  5,  2,  3,  0,  1) */
    /* Reordering Permutate and Blend for shorted dependency chain */
    __m256i perm2 = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(min1), _mm256_castsi256_ps(max1), 0xb1));
    __m256i v1    = _mm256_blend_epi32(max1, min1, 0x33);

    __m256i min2 = _mm256_min_epu32(v1, perm2);
    __m256i max2 = _mm256_max_epu32(v1, perm2);

    /* Pairs: ([0,7], [1,6], [2,5], [3,4]) */
    /* Perm:  ( 0,  1,  2,  3,  4,  5,  6,  7) */
    /* Reordering Permutate and Blend for shorted dependency chain */
    __m256i perm3 = _mm256_permutex2var_epi32(
        max2, _mm256_set_epi32(8, 1, 10, 3, 12, 5, 14, 7), min2);
    __m256i v2 = _mm256_blend_epi32(max2, min2, 0x55);

    __m256i min3 = _mm256_min_epu32(v2, perm3);
    __m256i max3 = _mm256_max_epu32(v2, perm3);
    __m256i v3   = _mm256_blend_epi32(max3, min3, 0xf);

    /* Pairs: ([5,7], [4,6], [1,3], [0,2]) */
    /* Perm:  ( 5,  4,  7,  6,  1,  0,  3,  2) */
    __m256i perm4 = _mm256_shuffle_epi32(v3, uint8_t(0x4e));
    __m256i min4  = _mm256_min_epu32(v3, perm4);
    __m256i max4  = _mm256_max_epu32(v3, perm4);

    /* Pairs: ([6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  ( 6,  7,  4,  5,  2,  3,  0,  1) */
    /* Reordering Permutate and Blend for shorted dependency chain */
    __m256i perm5 = _mm256_castps_si256(_mm256_shuffle_ps(
        _mm256_castsi256_ps(min4), _mm256_castsi256_ps(max4), 0xb1));
    __m256i v4    = _mm256_blend_epi32(max4, min4, 0x33);

    __m256i min5 = _mm256_min_epu32(v4, perm5);
    __m256i max5 = _mm256_max_epu32(v4, perm5);
    __m256i v5   = _mm256_blend_epi32(max5, min5, 0x55);

    return v5;
}
#endif


#ifndef _SIMD_SORT_ARR_bitonic_8_4u_H_
#define _SIMD_SORT_ARR_bitonic_8_4u_H_

/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_8_4u(uint32_t * const arr) {

    __m256i v = _mm256_load_si256((__m256i *)arr);

    v = bitonic_8_4u_vec(v);

    _mm256_store_si256((__m256i *)arr, v);
}
#endif

