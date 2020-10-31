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
#define N 26
#define SORT_NAME batcher_26_uint8_t

#ifndef _SIMD_SORT_batcher_26_uint8_t_H_
#define _SIMD_SORT_batcher_26_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 26
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 15
	SIMD Instructions                : 2 / 75
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX, AVX512vbmi, AVX512vl, AVX2, AVX512bw
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



/* SIMD Sort */
 __m256i __attribute__((const)) 
batcher_26_uint8_t_vec(__m256i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [9,25], [8,24], [7,23], [6,22], [5,21], [4,20], [3,19], 
                 [2,18], [1,17], [0,16], [15,15], [14,14], [13,13], [12,12], 
                 [11,11], [10,10]) */
      /* Perm:  (31, 30, 29, 28, 27, 26,  9,  8,  7,  6,  5,  4,  3,  2,  1,  
                 0, 15, 14, 13, 12, 11, 10, 25, 24, 23, 22, 21, 20, 19, 18, 
                 17, 16) */
      __m256i perm0 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 9, 8, 7, 6, 5, 4, 3, 2, 
                                              1, 0, 15, 14, 13, 12, 11, 10, 
                                              25, 24, 23, 22, 21, 20, 19, 18, 
                                              17, 16), v);
      __m256i min0 = _mm256_min_epu8(v, perm0);
      __m256i max0 = _mm256_max_epu8(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi8(max0, 0x3ff, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [17,25], [16,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [7,15], [6,14], [5,13], [4,12], [3,11], 
                 [2,10], [1,9], [0,8]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 17, 16, 23, 22, 21, 20, 19, 18, 25, 
                 24,  7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  
                 9,  8) */
      __m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 17, 16, 23, 22, 21, 20, 19, 
                                          18, 25, 24, 7, 6, 5, 4, 3, 2, 1, 0, 
                                          15, 14, 13, 12, 11, 10, 9, 8));
      __m256i min1 = _mm256_min_epu8(v0, perm1);
      __m256i max1 = _mm256_max_epu8(v0, perm1);
      __m256i v1 = _mm256_mask_mov_epi8(max1, 0x300ff, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [15,23], [14,22], [13,21], [12,20], 
                 [11,19], [10,18], [9,17], [8,16], [3,7], [2,6], [1,5], 
                 [0,4]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 10,  9,  
                 8, 23, 22, 21, 20, 19, 18, 17, 16,  3,  2,  1,  0,  7,  6,  
                 5,  4) */
      __m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 3, 2, 1, 0, 7, 
                                              6, 5, 4), v1);
      __m256i min2 = _mm256_min_epu8(v1, perm2);
      __m256i max2 = _mm256_max_epu8(v1, perm2);
      __m256i v2 = _mm256_blend_epi32(max2, min2, 0xd);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [19,23], [18,22], [17,21], [16,20], 
                 [11,15], [10,14], [9,13], [8,12], [7,7], [6,6], [5,5], 
                 [4,4], [1,3], [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 19, 18, 17, 16, 23, 22, 21, 
                 20, 11, 10,  9,  8, 15, 14, 13, 12,  7,  6,  5,  4,  1,  0,  
                 3,  2) */
      __m256i perm3 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 19, 18, 17, 16, 23, 
                                          22, 21, 20, 11, 10, 9, 8, 15, 14, 
                                          13, 12, 7, 6, 5, 4, 1, 0, 3, 2));
      __m256i min3 = _mm256_min_epu8(v2, perm3);
      __m256i max3 = _mm256_max_epu8(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi8(max3, 0xf0f03, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [13,25], [12,24], [23,23], [22,22], [21,21], [20,20], 
                 [7,19], [6,18], [5,17], [4,16], [15,15], [14,14], [11,11], 
                 [10,10], [9,9], [8,8], [3,3], [2,2], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 13, 12, 23, 22, 21, 20,  7,  6,  5,  
                 4, 15, 14, 25, 24, 11, 10,  9,  8, 19, 18, 17, 16,  3,  2,  
                 0,  1) */
      __m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 13, 12, 23, 22, 21, 20, 
                                              7, 6, 5, 4, 15, 14, 25, 24, 11, 
                                              10, 9, 8, 19, 18, 17, 16, 3, 2, 
                                              0, 1), v3);
      __m256i min4 = _mm256_min_epu8(v3, perm4);
      __m256i max4 = _mm256_max_epu8(v3, perm4);
      __m256i v4 = _mm256_mask_mov_epi8(max4, 0x30f1, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [21,25], [20,24], [23,23], [22,22], [15,19], [14,18], 
                 [13,17], [12,16], [7,11], [6,10], [5,9], [4,8], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 21, 20, 23, 22, 25, 24, 15, 14, 13, 
                 12, 19, 18, 17, 16,  7,  6,  5,  4, 11, 10,  9,  8,  3,  2,  
                 1,  0) */
      __m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 21, 20, 23, 22, 25, 24, 
                                              15, 14, 13, 12, 19, 18, 17, 16, 
                                              7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 
                                              1, 0), v4);
      __m256i min5 = _mm256_min_epu8(v4, perm5);
      __m256i max5 = _mm256_max_epu8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x30f0f0, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [21,23], [20,22], [17,19], [16,18], 
                 [13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 21, 20, 23, 22, 17, 16, 19, 
                 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  3,  2,  
                 1,  0) */
      __m256i perm6 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 21, 20, 23, 22, 17, 
                                          16, 19, 18, 13, 12, 15, 14, 9, 8, 
                                          11, 10, 5, 4, 7, 6, 3, 2, 1, 0));
      __m256i min6 = _mm256_min_epu8(v5, perm6);
      __m256i max6 = _mm256_max_epu8(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi8(max6, 0x333330, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [11,25], [10,24], [23,23], [22,22], [7,21], [6,20], [19,19], 
                 [18,18], [3,17], [2,16], [15,15], [14,14], [13,13], [12,12], 
                 [9,9], [8,8], [5,5], [4,4], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 11, 10, 23, 22,  7,  6, 19, 18,  3,  
                 2, 15, 14, 13, 12, 25, 24,  9,  8, 21, 20,  5,  4, 17, 16,  
                 1,  0) */
      __m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 11, 10, 23, 22, 7, 6, 
                                              19, 18, 3, 2, 15, 14, 13, 12, 
                                              25, 24, 9, 8, 21, 20, 5, 4, 17, 
                                              16, 1, 0), v6);
      __m256i min7 = _mm256_min_epu8(v6, perm7);
      __m256i max7 = _mm256_max_epu8(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi8(max7, 0xccc, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [19,25], [18,24], [23,23], [22,22], [15,21], [14,20], 
                 [11,17], [10,16], [7,13], [6,12], [3,9], [2,8], [5,5], 
                 [4,4], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 19, 18, 23, 22, 15, 14, 25, 24, 11, 
                 10, 21, 20,  7,  6, 17, 16,  3,  2, 13, 12,  5,  4,  9,  8,  
                 1,  0) */
      __m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 19, 18, 23, 22, 15, 14, 
                                              25, 24, 11, 10, 21, 20, 7, 6, 
                                              17, 16, 3, 2, 13, 12, 5, 4, 9, 
                                              8, 1, 0), v7);
      __m256i min8 = _mm256_min_epu8(v7, perm8);
      __m256i max8 = _mm256_max_epu8(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi8(max8, 0xccccc, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [23,25], [22,24], [19,21], [18,20], [15,17], [14,16], 
                 [11,13], [10,12], [7,9], [6,8], [3,5], [2,4], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 23, 22, 25, 24, 19, 18, 21, 20, 15, 
                 14, 17, 16, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  5,  4,  
                 1,  0) */
      __m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 23, 22, 25, 24, 19, 18, 
                                              21, 20, 15, 14, 17, 16, 11, 10, 
                                              13, 12, 7, 6, 9, 8, 3, 2, 5, 4, 
                                              1, 0), v8);
      __m256i min9 = _mm256_min_epu8(v8, perm9);
      __m256i max9 = _mm256_max_epu8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0xcccccc, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [24,25], [22,23], [20,21], [18,19], [16,17], [14,15], 
                 [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 1,  0) */
      __m256i perm10 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 24, 25, 22, 23, 20, 
                                           21, 18, 19, 16, 17, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 1, 0));
      __m256i min10 = _mm256_min_epu8(v9, perm10);
      __m256i max10 = _mm256_max_epu8(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi8(max10, 0x1555554, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [9,24], [23,23], [7,22], [21,21], [5,20], [19,19], 
                 [3,18], [17,17], [1,16], [15,15], [14,14], [13,13], [12,12], 
                 [11,11], [10,10], [8,8], [6,6], [4,4], [2,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25,  9, 23,  7, 21,  5, 19,  3, 17,  
                 1, 15, 14, 13, 12, 11, 10, 24,  8, 22,  6, 20,  4, 18,  2, 
                 16,  0) */
      __m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 9, 23, 7, 21, 
                                               5, 19, 3, 17, 1, 15, 14, 13, 
                                               12, 11, 10, 24, 8, 22, 6, 20, 
                                               4, 18, 2, 16, 0), v10);
      __m256i min11 = _mm256_min_epu8(v10, perm11);
      __m256i max11 = _mm256_max_epu8(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi8(max11, 0x2aa, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [17,24], [23,23], [15,22], [21,21], [13,20], 
                 [19,19], [11,18], [9,16], [7,14], [5,12], [3,10], [1,8], 
                 [6,6], [4,4], [2,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 17, 23, 15, 21, 13, 19, 11, 24,  
                 9, 22,  7, 20,  5, 18,  3, 16,  1, 14,  6, 12,  4, 10,  2,  
                 8,  0) */
      __m256i perm12 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 17, 23, 15, 
                                               21, 13, 19, 11, 24, 9, 22, 7, 
                                               20, 5, 18, 3, 16, 1, 14, 6, 
                                               12, 4, 10, 2, 8, 0), v11);
      __m256i min12 = _mm256_min_epu8(v11, perm12);
      __m256i max12 = _mm256_max_epu8(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi8(max12, 0x2aaaa, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [21,24], [23,23], [19,22], [17,20], [15,18], 
                 [13,16], [11,14], [9,12], [7,10], [5,8], [3,6], [1,4], 
                 [2,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 21, 23, 19, 24, 17, 22, 15, 20, 
                 13, 18, 11, 16,  9, 14,  7, 12,  5, 10,  3,  8,  1,  6,  2,  
                 4,  0) */
      __m256i perm13 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 21, 23, 19, 
                                               24, 17, 22, 15, 20, 13, 18, 
                                               11, 16, 9, 14, 7, 12, 5, 10, 
                                               3, 8, 1, 6, 2, 4, 0), v12);
      __m256i min13 = _mm256_min_epu8(v12, perm13);
      __m256i max13 = _mm256_max_epu8(v12, perm13);
      __m256i v13 = _mm256_mask_mov_epi8(max13, 0x2aaaaa, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [23,24], [21,22], [19,20], [17,18], [15,16], 
                 [13,14], [11,12], [9,10], [7,8], [5,6], [3,4], [1,2], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m256i perm14 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 23, 24, 21, 
                                               22, 19, 20, 17, 18, 15, 16, 
                                               13, 14, 11, 12, 9, 10, 7, 8, 
                                               5, 6, 3, 4, 1, 2, 0), v13);
      __m256i min14 = _mm256_min_epu8(v13, perm14);
      __m256i max14 = _mm256_max_epu8(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi8(max14, 0xaaaaaa, min14);
      
      return v14;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
batcher_26_uint8_t(uint8_t * const arr) 
                             {
      
      __m256i v = _mm256_load_si256((__m256i *)arr);
      
      v = batcher_26_uint8_t_vec(v);
      
      _mm256_store_si256((__m256i *)arr, v);
      
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


