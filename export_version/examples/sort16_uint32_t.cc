
/*

Sorting Network Information:
        Sort Size                        : 16
        Underlying Sort Type             : uint32_t
        Network Generation Algorithm     : bitonic
        Network Depth                    : 10
        SIMD Instructions                : 2 / 32
        Optimization Preference          : space
        SIMD Type                        : __m512i
        SIMD Instruction Set(s) Used     : AVX512f,
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


#ifndef _SIMD_SORT_VEC_bitonic_16_4u_vec_H_
#define _SIMD_SORT_VEC_bitonic_16_4u_vec_H_

/* SIMD Sort */
__m512i __attribute__((const)) bitonic_16_4u_vec(__m512i v) {

    /* Pairs: ([14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  (14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1)
     */
    __m512i perm0 = _mm512_shuffle_epi32(v, _MM_PERM_ENUM(0xb1));
    __m512i min0  = _mm512_min_epu32(v, perm0);
    __m512i v0    = _mm512_mask_max_epu32(min0, 0xaaaa, v, perm0);

    /* Pairs: ([12,15], [13,14], [8,11], [9,10], [4,7], [5,6], [0,3], [1,2]) */
    /* Perm:  (12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  2,  3)
     */
    __m512i perm1 = _mm512_shuffle_epi32(v0, _MM_PERM_ENUM(0x1b));
    __m512i min1  = _mm512_min_epu32(v0, perm1);
    __m512i v1    = _mm512_mask_max_epu32(min1, 0xcccc, v0, perm1);

    /* Pairs: ([14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  (14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1)
     */
    __m512i perm2 = _mm512_shuffle_epi32(v1, _MM_PERM_ENUM(0xb1));
    __m512i min2  = _mm512_min_epu32(v1, perm2);
    __m512i v2    = _mm512_mask_max_epu32(min2, 0xaaaa, v1, perm2);

    /* Pairs: ([8,15], [9,14], [10,13], [11,12], [0,7], [1,6], [2,5], [3,4]) */
    /* Perm:  ( 8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  6,  7)
     */
    __m512i _tmp0 = _mm512_permutex_epi64(v2, 0x1b);
    __m512i perm3 = _mm512_ror_epi64(_tmp0, 32);
    __m512i min3  = _mm512_min_epu32(v2, perm3);
    __m512i v3    = _mm512_mask_max_epu32(min3, 0xf0f0, v2, perm3);

    /* Pairs: ([13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
    /* Perm:  (13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2)
     */
    __m512i perm4 = _mm512_shuffle_epi32(v3, _MM_PERM_ENUM(0x4e));
    __m512i min4  = _mm512_min_epu32(v3, perm4);
    __m512i v4    = _mm512_mask_max_epu32(min4, 0xcccc, v3, perm4);

    /* Pairs: ([14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  (14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1)
     */
    __m512i perm5 = _mm512_shuffle_epi32(v4, _MM_PERM_ENUM(0xb1));
    __m512i min5  = _mm512_min_epu32(v4, perm5);
    __m512i v5    = _mm512_mask_max_epu32(min5, 0xaaaa, v4, perm5);

    /* Pairs: ([0,15], [1,14], [2,13], [3,12], [4,11], [5,10], [6,9], [7,8]) */
    /* Perm:  ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15)
     */
    __m512i _tmp1 = _mm512_shuffle_i64x2(v5, v5, 0x1b);
    __m512i perm6 = _mm512_shuffle_epi32(_tmp1, _MM_PERM_ENUM(0x1b));
    __m512i min6  = _mm512_min_epu32(v5, perm6);
    __m512i v6    = _mm512_mask_max_epu32(min6, 0xff00, v5, perm6);

    /* Pairs: ([11,15], [10,14], [9,13], [8,12], [3,7], [2,6], [1,5], [0,4]) */
    /* Perm:  (11, 10,  9,  8, 15, 14, 13, 12,  3,  2,  1,  0,  7,  6,  5,  4)
     */
    __m512i perm7 = _mm512_permutex_epi64(v6, 0x4e);
    __m512i min7  = _mm512_min_epu32(v6, perm7);
    __m512i v7    = _mm512_mask_max_epu32(min7, 0xf0f0, v6, perm7);

    /* Pairs: ([13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
    /* Perm:  (13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2)
     */
    __m512i perm8 = _mm512_shuffle_epi32(v7, _MM_PERM_ENUM(0x4e));
    __m512i min8  = _mm512_min_epu32(v7, perm8);
    __m512i v8    = _mm512_mask_max_epu32(min8, 0xcccc, v7, perm8);

    /* Pairs: ([14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
    /* Perm:  (14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1)
     */
    __m512i perm9 = _mm512_shuffle_epi32(v8, _MM_PERM_ENUM(0xb1));
    __m512i min9  = _mm512_min_epu32(v8, perm9);
    __m512i v9    = _mm512_mask_max_epu32(min9, 0xaaaa, v8, perm9);

    return v9;
}
#endif


#ifndef _SIMD_SORT_ARR_bitonic_16_4u_H_
#define _SIMD_SORT_ARR_bitonic_16_4u_H_

/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_16_4u(uint32_t * const arr) {

    __m512i v = _mm512_load_si512((__m512i *)arr);

    v = bitonic_16_4u_vec(v);

    _mm512_store_si512((__m512i *)arr, v);
}
#endif

