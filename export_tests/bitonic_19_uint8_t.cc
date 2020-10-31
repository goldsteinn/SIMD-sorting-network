#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>


template<typename T, uint32_t n>
struct sarr {
    typedef uint32_t aliasing_u32 __attribute__((aligned(1), may_alias));


    T arr[64 / sizeof(T)] __attribute__((aligned(64)));

    void
    finit() {
        for (uint32_t i = 0; i < n; ++i) {
            arr[i] = i;
        }
    }

    void
    binit() {
        for (uint32_t i = 0; i < n; ++i) {
            arr[i] = (n - 1) - i;
        }
    }

    void
    show() {
        for (uint32_t i = 0; i < n; ++i) {
            fprintf(stderr, "%d: %d\n", i, (uint32_t)arr[i]);
        }
    }

    void
    verify() {
        for (uint32_t i = 1; i < n; ++i) {
            assert(arr[i] >= arr[i - 1]);
        }
    }

    void
    randomize() {
        aliasing_u32 * _arr = (aliasing_u32 *)arr;
        for (uint32_t i = 0; i < (64 / sizeof(uint32_t)); ++i) {
            _arr[i] = rand();
        }
    }
};

#define TYPE uint8_t
#define N 19
#define SORT_NAME bitonic_19_uint8_t

#ifndef _SIMD_SORT_bitonic_19_uint8_t_H_
#define _SIMD_SORT_bitonic_19_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 19
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 14
	SIMD Instructions                : 3 / 70
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX2, AVX512vbmi, AVX
	SIMD Instruction Set(s) Excluded : None
	Aligned Load & Store             : True
	Integer Aligned Load & Store     : True
	Full Load & Store                : True
	Scaled Sorting Network           : False

Performance Notes:
1) If you are sorting an array where there IS valid memory up to 
   the nearest sizeof a SIMD register, you will get an improvement enable
   "EXTRA_MEMORY" (this turns on "Full Load & Store". Note that enabling
   "Full Load & Store" will not modify any of the memory not being sorted
   and will not affect the sort in any way. i.e sort(3) [4, 3, 2, 1]
   with full load will still return [2, 3, 4, 1]. Note even if you don't
   have enough memory for a full SIMD register, enabling "INT_ALIGNED"
   will also improve load efficiency and only requires that there is
   valid memory up the next factor of sizeof(int).

2) If your sort size is not a power of 2 you are likely running into 
   less efficient instructions. This is especially noticable when sorting
   8 bit and 16 bit values. If rounding you sort size up to the next
   power of 2 will not cost any additional depth it almost definetly
   worth doing so. The "Best" Network Algorithm automatically does this
   in many cases.

3) There are two optimization settings, "Optimization.SPACE" and 
   "Optimization.UOP". The former will essentially break ties by picking
   the instruction that uses less memory (i.e doesn't have to store
   a register's initializing in memory. The latter will break ties but
   simply selecting whatever instructions use the least UOPs. Which
   is best is probably application dependent. Note that while "Optimization.SPACE"
   will save .rodata memory it will often cost more in .text memory.

 */

#include <immintrin.h>
#include <stdint.h>



 void fill_works(__m256i v) {
      sarr<TYPE, N> t;
      memcpy(t.arr, &v, 32);
      int i = N;for (; i < 32; ++i) {
          assert(t.arr[i] == uint8_t(0xff));
 }
}

/* SIMD Sort */
 __m256i __attribute__((const)) 
bitonic_19_uint8_t_vec(__m256i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [17,18], [16,16], [14,15], [12,13], [11,11], 
                 [9,10], [7,8], [6,6], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 17, 18, 
                 16, 14, 15, 12, 13, 11,  9, 10,  7,  8,  6,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          17, 18, 16, 14, 15, 12, 13, 11, 9, 
                                          10, 7, 8, 6, 4, 5, 2, 3, 0, 1));
      __m256i min0 = _mm256_min_epu8(v, perm0);
      __m256i max0 = _mm256_max_epu8(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi8(max0, 0x25295, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [16,18], [17,17], [15,15], [14,14], [13,13], 
                 [11,12], [10,10], [9,9], [8,8], [6,7], [5,5], [4,4], [0,3], 
                 [1,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 16, 17, 
                 18, 15, 14, 13, 11, 12, 10,  9,  8,  6,  7,  5,  4,  0,  1,  
                 2,  3) */
      __m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          16, 17, 18, 15, 14, 13, 11, 12, 10, 
                                          9, 8, 6, 7, 5, 4, 0, 1, 2, 3));
      __m256i min1 = _mm256_min_epu8(v0, perm1);
      __m256i max1 = _mm256_max_epu8(v0, perm1);
      __m256i v1 = _mm256_mask_mov_epi8(max1, 0x10843, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [15,18], [16,17], [14,14], [12,13], [9,11], 
                 [10,10], [7,8], [4,6], [5,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 15, 16, 
                 17, 18, 14, 12, 13,  9, 10, 11,  7,  8,  4,  5,  6,  2,  3,  
                 0,  1) */
      __m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 15, 16, 17, 18, 14, 12, 13, 
                                              9, 10, 11, 7, 8, 4, 5, 6, 2, 3, 
                                              0, 1), v1);
      __m256i min2 = _mm256_min_epu8(v1, perm2);
      __m256i max2 = _mm256_max_epu8(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi8(max2, 0x19295, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [14,17], [15,16], [11,13], [10,12], [9,9], 
                 [6,8], [5,7], [0,4], [3,3], [2,2], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 14, 
                 15, 16, 17, 11, 10, 13, 12,  9,  6,  5,  8,  7,  0,  3,  2,  
                 1,  4) */
      __m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 14, 15, 16, 17, 11, 10, 
                                              13, 12, 9, 6, 5, 8, 7, 0, 3, 2, 
                                              1, 4), v2);
      __m256i min3 = _mm256_min_epu8(v2, perm3);
      __m256i max3 = _mm256_max_epu8(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi8(max3, 0xcc61, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [16,17], [14,15], [12,13], [10,11], [9,9], 
                 [7,8], [5,6], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 
                 17, 14, 15, 12, 13, 10, 11,  9,  7,  8,  5,  6,  4,  3,  2,  
                 1,  0) */
      __m256i perm4 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 16, 17, 14, 15, 12, 13, 10, 11, 
                                          9, 7, 8, 5, 6, 4, 3, 2, 1, 0));
      __m256i min4 = _mm256_min_epu8(v3, perm4);
      __m256i max4 = _mm256_max_epu8(v3, perm4);
      __m256i v4 = _mm256_mask_mov_epi8(max4, 0x154a0, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [12,18], [13,17], [10,16], [11,15], [14,14], [9,9], 
                 [4,8], [1,7], [2,6], [3,5], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 12, 13, 
                 10, 11, 14, 17, 18, 15, 16,  9,  4,  1,  2,  3,  8,  5,  6,  
                 7,  0) */
      __m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 12, 13, 10, 11, 14, 17, 18, 
                                              15, 16, 9, 4, 1, 2, 3, 8, 5, 6, 
                                              7, 0), v4);
      __m256i min5 = _mm256_min_epu8(v4, perm5);
      __m256i max5 = _mm256_max_epu8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x3c1e, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [17,18], [16,16], [15,15], [12,14], [9,13], 
                 [11,11], [10,10], [6,8], [5,7], [2,4], [1,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 17, 18, 
                 16, 15, 12,  9, 14, 11, 10, 13,  6,  5,  8,  7,  2,  1,  4,  
                 3,  0) */
      __m256i perm6 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          17, 18, 16, 15, 12, 9, 14, 11, 10, 
                                          13, 6, 5, 8, 7, 2, 1, 4, 3, 0));
      __m256i min6 = _mm256_min_epu8(v5, perm6);
      __m256i max6 = _mm256_max_epu8(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi8(max6, 0x21266, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [14,16], [13,15], [10,12], 
                 [9,11], [7,8], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 14, 13, 16, 15, 10,  9, 12, 11,  7,  8,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 14, 13, 16, 15, 10, 
                                              9, 12, 11, 7, 8, 5, 6, 3, 4, 1, 
                                              2, 0), v6);
      __m256i min7 = _mm256_min_epu8(v6, perm7);
      __m256i max7 = _mm256_max_epu8(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi8(max7, 0x66aa, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [6,18], [7,17], [15,16], [13,14], [11,12], [9,10], 
                 [8,8], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,  6,  7, 
                 15, 16, 13, 14, 11, 12,  9, 10,  8, 17, 18,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 6, 7, 15, 16, 13, 14, 11, 
                                              12, 9, 10, 8, 17, 18, 5, 4, 3, 
                                              2, 1, 0), v7);
      __m256i min8 = _mm256_min_epu8(v7, perm8);
      __m256i max8 = _mm256_max_epu8(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi8(max8, 0xaac0, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [8,16], [1,15], [2,14], [3,13], 
                 [4,12], [5,11], [6,10], [7,9], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  
                 8,  1,  2,  3,  4,  5,  6,  7, 16,  9, 10, 11, 12, 13, 14, 
                 15,  0) */
      __m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 8, 1, 2, 3, 4, 5, 
                                              6, 7, 16, 9, 10, 11, 12, 13, 
                                              14, 15, 0), v8);
      __m256i min9 = _mm256_min_epu8(v8, perm9);
      __m256i max9 = _mm256_max_epu8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0x1fe, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [16,18], [17,17], [11,15], [10,14], [9,13], 
                 [12,12], [0,8], [3,7], [2,6], [1,5], [4,4]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 16, 17, 
                 18, 11, 10,  9, 12, 15, 14, 13,  0,  3,  2,  1,  4,  7,  6,  
                 5,  8) */
      __m256i perm10 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 16, 17, 18, 11, 10, 9, 12, 
                                           15, 14, 13, 0, 3, 2, 1, 4, 7, 6, 
                                           5, 8));
      __m256i min10 = _mm256_min_epu8(v9, perm10);
      __m256i max10 = _mm256_max_epu8(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi8(max10, 0x10e0f, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [16,17], [13,15], [14,14], [8,12], [9,11], 
                 [10,10], [5,7], [6,6], [0,4], [1,3], [2,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 
                 17, 13, 14, 15,  8,  9, 10, 11, 12,  5,  6,  7,  0,  1,  2,  
                 3,  4) */
      __m256i perm11 = _mm256_shuffle_epi8(v10, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 16, 17, 13, 14, 15, 8, 
                                           9, 10, 11, 12, 5, 6, 7, 0, 1, 2, 
                                           3, 4));
      __m256i min11 = _mm256_min_epu8(v10, perm11);
      __m256i max11 = _mm256_max_epu8(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi8(max11, 0x12323, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [12,14], 
                 [13,13], [11,11], [8,10], [9,9], [7,7], [4,6], [5,5], [3,3], 
                 [0,2], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 12, 13, 14, 11,  8,  9, 10,  7,  4,  5,  6,  3,  0,  
                 1,  2) */
      __m256i perm12 = _mm256_shuffle_epi8(v11, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 12, 13, 
                                           14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 
                                           0, 1, 2));
      __m256i min12 = _mm256_min_epu8(v11, perm12);
      __m256i max12 = _mm256_max_epu8(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi8(max12, 0x1111, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [14,15], [12,13], 
                 [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm13 = _mm256_shuffle_epi8(v12, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 0, 1));
      __m256i min13 = _mm256_min_epu8(v12, perm13);
      __m256i max13 = _mm256_max_epu8(v12, perm13);
      __m256i v13 = _mm256_mask_mov_epi8(max13, 0x5555, min13);
      
      return v13;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bitonic_19_uint8_t(uint8_t * const arr) 
                             {
      
      __m256i _tmp0 = _mm256_set1_epi8(uint8_t(0xff));
      __m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x7ffff, arr);
      fill_works(v);
      v = bitonic_19_uint8_t_vec(v);
      
      fill_works(v);_mm256_mask_storeu_epi8((void *)arr, 0x7ffff, v);
      
 }


#endif





#define TSIZE 1000
void test() {
    sarr<TYPE, N> s1;
    sarr<TYPE, N> s2;
    
    s1.binit();
    memcpy(s2.arr, s1.arr, 64);
    
    std::sort(s1.arr, s1.arr + N);
    SORT_NAME(s2.arr);
    assert(!memcmp(s1.arr, s2.arr, 64));

    s1.finit();
    memcpy(s2.arr, s1.arr, 64);
    
    std::sort(s1.arr, s1.arr + N);
    SORT_NAME(s2.arr);
    assert(!memcmp(s1.arr, s2.arr, 64));

    for(uint32_t i = 0; i < TSIZE; ++i) {
        s1.randomize();
        memcpy(s2.arr, s1.arr, 64);
    
        std::sort(s1.arr, s1.arr + N);
        SORT_NAME(s2.arr);
        assert(!memcmp(s1.arr, s2.arr, 64));
    }
}

int main() {
    test();
}


