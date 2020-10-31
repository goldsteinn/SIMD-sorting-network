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
#define N 24
#define SORT_NAME bosenelson_24_uint8_t

#ifndef _SIMD_SORT_bosenelson_24_uint8_t_H_
#define _SIMD_SORT_bosenelson_24_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 24
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : bosenelson
	Network Depth                    : 24
	SIMD Instructions                : 2 / 120
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX, AVX2, AVX512vl, AVX512bw, AVX512vbmi
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
bosenelson_24_uint8_t_vec(__m256i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [22,23], [21,21], [19,20], [18,18], 
                 [16,17], [15,15], [13,14], [12,12], [10,11], [9,9], [7,8], 
                 [6,6], [4,5], [3,3], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 21, 19, 20, 18, 16, 
                 17, 15, 13, 14, 12, 10, 11,  9,  7,  8,  6,  4,  5,  3,  1,  
                 2,  0) */
      __m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 22, 23, 21, 19, 20, 
                                          18, 16, 17, 15, 13, 14, 12, 10, 11, 
                                          9, 7, 8, 6, 4, 5, 3, 1, 2, 0));
      __m256i min0 = _mm256_min_epu8(v, perm0);
      __m256i max0 = _mm256_max_epu8(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi8(max0, 0x492492, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [21,23], [22,22], [18,20], [19,19], 
                 [15,17], [16,16], [12,14], [13,13], [9,11], [10,10], [6,8], 
                 [7,7], [3,5], [4,4], [0,2], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 21, 22, 23, 18, 19, 20, 15, 
                 16, 17, 12, 13, 14,  9, 10, 11,  6,  7,  8,  3,  4,  5,  0,  
                 1,  2) */
      __m256i perm1 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 21, 22, 23, 18, 
                                              19, 20, 15, 16, 17, 12, 13, 14, 
                                              9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 
                                              1, 2), v0);
      __m256i min1 = _mm256_min_epu8(v0, perm1);
      __m256i max1 = _mm256_max_epu8(v0, perm1);
      __m256i v1 = _mm256_mask_mov_epi8(max1, 0x249249, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [20,23], [21,22], [18,19], [14,17], 
                 [15,16], [12,13], [8,11], [9,10], [6,7], [2,5], [3,4], 
                 [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 20, 21, 22, 23, 18, 19, 14, 
                 15, 16, 17, 12, 13,  8,  9, 10, 11,  6,  7,  2,  3,  4,  5,  
                 0,  1) */
      __m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 20, 21, 22, 23, 
                                              18, 19, 14, 15, 16, 17, 12, 13, 
                                              8, 9, 10, 11, 6, 7, 2, 3, 4, 5, 
                                              0, 1), v1);
      __m256i min2 = _mm256_min_epu8(v1, perm2);
      __m256i max2 = _mm256_max_epu8(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi8(max2, 0x34d34d, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [17,23], [19,22], [18,21], [20,20], 
                 [13,16], [12,15], [14,14], [5,11], [7,10], [6,9], [8,8], 
                 [1,4], [0,3], [2,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 17, 19, 18, 20, 22, 21, 23, 
                 13, 12, 14, 16, 15,  5,  7,  6,  8, 10,  9, 11,  1,  0,  2,  
                 4,  3) */
      __m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 17, 19, 18, 20, 
                                              22, 21, 23, 13, 12, 14, 16, 15, 
                                              5, 7, 6, 8, 10, 9, 11, 1, 0, 2, 
                                              4, 3), v2);
      __m256i min3 = _mm256_min_epu8(v2, perm3);
      __m256i max3 = _mm256_max_epu8(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi8(max3, 0xe30e3, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [11,23], [20,22], [19,21], [12,18], 
                 [17,17], [14,16], [13,15], [8,10], [7,9], [0,6], [5,5], 
                 [2,4], [1,3]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 11, 20, 19, 22, 21, 12, 17, 
                 14, 13, 16, 15, 18, 23,  8,  7, 10,  9,  0,  5,  2,  1,  4,  
                 3,  6) */
      __m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 11, 20, 19, 22, 
                                              21, 12, 17, 14, 13, 16, 15, 18, 
                                              23, 8, 7, 10, 9, 0, 5, 2, 1, 4, 
                                              3, 6), v3);
      __m256i min4 = _mm256_min_epu8(v3, perm4);
      __m256i max4 = _mm256_max_epu8(v3, perm4);
      __m256i v4 = _mm256_mask_mov_epi8(max4, 0x187987, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [16,22], [20,21], [13,19], 
                 [18,18], [17,17], [14,15], [0,12], [11,11], [4,10], [8,9], 
                 [1,7], [6,6], [5,5], [2,3]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 16, 20, 21, 13, 18, 17, 
                 22, 14, 15, 19,  0, 11,  4,  8,  9,  1,  6,  5, 10,  2,  3,  
                 7, 12) */
      __m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 16, 20, 21, 
                                              13, 18, 17, 22, 14, 15, 19, 0, 
                                              11, 4, 8, 9, 1, 6, 5, 10, 2, 3, 
                                              7, 12), v4);
      __m256i min5 = _mm256_min_epu8(v4, perm5);
      __m256i max5 = _mm256_max_epu8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x116117, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [17,22], [15,21], [14,20], 
                 [19,19], [13,18], [16,16], [12,12], [11,11], [5,10], [3,9], 
                 [2,8], [7,7], [1,6], [4,4], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 17, 15, 14, 19, 13, 22, 
                 16, 21, 20, 18, 12, 11,  5,  3,  2,  7,  1, 10,  4,  9,  8,  
                 6,  0) */
      __m256i perm6 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 17, 15, 14, 
                                              19, 13, 22, 16, 21, 20, 18, 12, 
                                              11, 5, 3, 2, 7, 1, 10, 4, 9, 8, 
                                              6, 0), v5);
      __m256i min6 = _mm256_min_epu8(v5, perm6);
      __m256i max6 = _mm256_max_epu8(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi8(max6, 0x2e02e, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [10,22], [16,21], [20,20], 
                 [14,19], [18,18], [17,17], [15,15], [1,13], [12,12], 
                 [11,11], [4,9], [8,8], [2,7], [6,6], [5,5], [3,3], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 10, 16, 20, 14, 18, 17, 
                 21, 15, 19,  1, 12, 11, 22,  4,  8,  2,  6,  5,  9,  3,  7, 
                 13,  0) */
      __m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 10, 16, 20, 
                                              14, 18, 17, 21, 15, 19, 1, 12, 
                                              11, 22, 4, 8, 2, 6, 5, 9, 3, 7, 
                                              13, 0), v6);
      __m256i min7 = _mm256_min_epu8(v6, perm7);
      __m256i max7 = _mm256_max_epu8(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi8(max7, 0x14416, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [11,22], [17,21], [20,20], 
                 [16,19], [14,18], [15,15], [13,13], [1,12], [10,10], [5,9], 
                 [8,8], [4,7], [2,6], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 11, 17, 20, 16, 14, 21, 
                 19, 15, 18, 13,  1, 22, 10,  5,  8,  4,  2,  9,  7,  3,  6, 
                 12,  0) */
      __m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 11, 17, 20, 
                                              16, 14, 21, 19, 15, 18, 13, 1, 
                                              22, 10, 5, 8, 4, 2, 9, 7, 3, 6, 
                                              12, 0), v7);
      __m256i min8 = _mm256_min_epu8(v7, perm8);
      __m256i max8 = _mm256_max_epu8(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi8(max8, 0x34836, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [9,21], [17,20], 
                 [19,19], [15,18], [16,16], [2,14], [13,13], [12,12], 
                 [11,11], [10,10], [5,8], [7,7], [3,6], [4,4], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22,  9, 17, 19, 15, 20, 
                 16, 18,  2, 13, 12, 11, 10, 21,  5,  7,  3,  8,  4,  6, 14,  
                 1,  0) */
      __m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 9, 17, 
                                              19, 15, 20, 16, 18, 2, 13, 12, 
                                              11, 10, 21, 5, 7, 3, 8, 4, 6, 
                                              14, 1, 0), v8);
      __m256i min9 = _mm256_min_epu8(v8, perm9);
      __m256i max9 = _mm256_max_epu8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0x2822c, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [10,21], [8,20], 
                 [17,19], [16,18], [3,15], [14,14], [2,13], [12,12], [11,11], 
                 [9,9], [5,7], [4,6], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 10,  8, 17, 16, 19, 
                 18,  3, 14,  2, 12, 11, 21,  9, 20,  5,  4,  7,  6, 15, 13,  
                 1,  0) */
      __m256i perm10 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               10, 8, 17, 16, 19, 18, 3, 14, 
                                               2, 12, 11, 21, 9, 20, 5, 4, 7, 
                                               6, 15, 13, 1, 0), v9);
      __m256i min10 = _mm256_min_epu8(v9, perm10);
      __m256i max10 = _mm256_max_epu8(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi8(max10, 0x3053c, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [11,21], [20,20], 
                 [7,19], [17,18], [4,16], [15,15], [14,14], [13,13], [2,12], 
                 [10,10], [9,9], [8,8], [5,6], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 11, 20,  7, 17, 18,  
                 4, 15, 14, 13,  2, 21, 10,  9,  8, 19,  5,  6, 16,  3, 12,  
                 1,  0) */
      __m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               11, 20, 7, 17, 18, 4, 15, 14, 
                                               13, 2, 21, 10, 9, 8, 19, 5, 6, 
                                               16, 3, 12, 1, 0), v10);
      __m256i min11 = _mm256_min_epu8(v10, perm11);
      __m256i max11 = _mm256_max_epu8(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi8(max11, 0x208b4, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [11,20], 
                 [8,19], [6,18], [5,17], [16,16], [4,15], [14,14], [13,13], 
                 [3,12], [10,10], [9,9], [7,7], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 11,  8,  6,  5, 
                 16,  4, 14, 13,  3, 20, 10,  9, 19,  7, 18, 17, 15, 12,  2,  
                 1,  0) */
      __m256i perm12 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 11, 8, 6, 5, 16, 4, 14, 
                                               13, 3, 20, 10, 9, 19, 7, 18, 
                                               17, 15, 12, 2, 1, 0), v11);
      __m256i min12 = _mm256_min_epu8(v11, perm12);
      __m256i max12 = _mm256_max_epu8(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi8(max12, 0x978, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [10,19], [7,18], [17,17], [5,16], [15,15], [14,14], [4,13], 
                 [12,12], [11,11], [9,9], [8,8], [6,6], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 10,  7, 17,  
                 5, 15, 14,  4, 12, 11, 19,  9,  8, 18,  6, 16, 13,  3,  2,  
                 1,  0) */
      __m256i perm13 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 10, 7, 17, 5, 15, 14, 
                                               4, 12, 11, 19, 9, 8, 18, 6, 
                                               16, 13, 3, 2, 1, 0), v12);
      __m256i min13 = _mm256_min_epu8(v12, perm13);
      __m256i max13 = _mm256_max_epu8(v12, perm13);
      __m256i v13 = _mm256_mask_mov_epi8(max13, 0x4b0, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [11,19], [8,18], [17,17], [16,16], [5,15], [14,14], [13,13], 
                 [4,12], [10,10], [9,9], [7,7], [6,6], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 11,  8, 17, 
                 16,  5, 14, 13,  4, 19, 10,  9, 18,  7,  6, 15, 12,  3,  2,  
                 1,  0) */
      __m256i perm14 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 11, 8, 17, 16, 5, 14, 
                                               13, 4, 19, 10, 9, 18, 7, 6, 
                                               15, 12, 3, 2, 1, 0), v13);
      __m256i min14 = _mm256_min_epu8(v13, perm14);
      __m256i max14 = _mm256_max_epu8(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi8(max14, 0x930, min14);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [9,18], [17,17], [16,16], [15,15], [5,14], [13,13], 
                 [12,12], [11,11], [10,10], [8,8], [7,7], [6,6], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,  9, 17, 
                 16, 15,  5, 13, 12, 11, 10, 18,  8,  7,  6, 14,  4,  3,  2,  
                 1,  0) */
      __m256i perm15 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 9, 17, 16, 15, 5, 
                                               13, 12, 11, 10, 18, 8, 7, 6, 
                                               14, 4, 3, 2, 1, 0), v14);
      __m256i min15 = _mm256_min_epu8(v14, perm15);
      __m256i max15 = _mm256_max_epu8(v14, perm15);
      __m256i v15 = _mm256_mask_mov_epi8(max15, 0x220, min15);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [10,18], [17,17], [16,16], [9,15], [8,14], [5,13], 
                 [12,12], [11,11], [7,7], [6,6], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 10, 17, 
                 16,  9,  8,  5, 12, 11, 18, 15, 14,  7,  6, 13,  4,  3,  2,  
                 1,  0) */
      __m256i perm16 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 10, 17, 16, 9, 8, 
                                               5, 12, 11, 18, 15, 14, 7, 6, 
                                               13, 4, 3, 2, 1, 0), v15);
      __m256i min16 = _mm256_min_epu8(v15, perm16);
      __m256i max16 = _mm256_max_epu8(v15, perm16);
      __m256i v16 = _mm256_mask_mov_epi8(max16, 0x720, min16);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [11,18], [17,17], [10,16], [15,15], [14,14], 
                 [7,13], [5,12], [9,9], [8,8], [6,6], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 11, 17, 
                 10, 15, 14,  7,  5, 18, 16,  9,  8, 13,  6, 12,  4,  3,  2,  
                 1,  0) */
      __m256i perm17 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 11, 17, 10, 15, 
                                               14, 7, 5, 18, 16, 9, 8, 13, 6, 
                                               12, 4, 3, 2, 1, 0), v16);
      __m256i min17 = _mm256_min_epu8(v16, perm17);
      __m256i max17 = _mm256_max_epu8(v16, perm17);
      __m256i v17 = _mm256_mask_mov_epi8(max17, 0xca0, min17);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [11,17], [16,16], [10,15], [14,14], 
                 [8,13], [6,12], [9,9], [7,7], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 11, 
                 16, 10, 14,  8,  6, 17, 15,  9, 13,  7, 12,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm18 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 11, 16, 10, 
                                               14, 8, 6, 17, 15, 9, 13, 7, 
                                               12, 5, 4, 3, 2, 1, 0), v17);
      __m256i min18 = _mm256_min_epu8(v17, perm18);
      __m256i max18 = _mm256_max_epu8(v17, perm18);
      __m256i v18 = _mm256_mask_mov_epi8(max18, 0xd40, min18);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [11,16], [15,15], [14,14], 
                 [10,13], [7,12], [9,9], [8,8], [6,6], [5,5], [4,4], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 11, 15, 14, 10,  7, 16, 13,  9,  8, 12,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm19 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 11, 15, 
                                               14, 10, 7, 16, 13, 9, 8, 12, 
                                               6, 5, 4, 3, 2, 1, 0), v18);
      __m256i min19 = _mm256_min_epu8(v18, perm19);
      __m256i max19 = _mm256_max_epu8(v18, perm19);
      __m256i v19 = _mm256_mask_mov_epi8(max19, 0xc80, min19);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [11,15], [14,14], 
                 [13,13], [8,12], [10,10], [9,9], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 11, 14, 13,  8, 15, 10,  9, 12,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm20 = _mm256_shuffle_epi8(v19, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 11, 14, 13, 8, 
                                           15, 10, 9, 12, 7, 6, 5, 4, 3, 2, 
                                           1, 0));
      __m256i min20 = _mm256_min_epu8(v19, perm20);
      __m256i max20 = _mm256_max_epu8(v19, perm20);
      __m256i v20 = _mm256_mask_mov_epi8(max20, 0x900, min20);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [11,14], 
                 [13,13], [9,12], [10,10], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 11, 13,  9, 14, 10, 12,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm21 = _mm256_shuffle_epi8(v20, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 11, 13, 9, 
                                           14, 10, 12, 8, 7, 6, 5, 4, 3, 2, 
                                           1, 0));
      __m256i min21 = _mm256_min_epu8(v20, perm21);
      __m256i max21 = _mm256_max_epu8(v20, perm21);
      __m256i v21 = _mm256_mask_mov_epi8(max21, 0xa00, min21);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [11,13], [10,12], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 11, 10, 13, 12,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm22 = _mm256_shuffle_epi8(v21, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 11, 
                                           10, 13, 12, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m256i min22 = _mm256_min_epu8(v21, perm22);
      __m256i max22 = _mm256_max_epu8(v21, perm22);
      __m256i v22 = _mm256_mask_mov_epi8(max22, 0xc00, min22);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [11,12], [10,10], [9,9], [8,8], [7,7], [6,6], 
                 [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 11, 12, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm23 = _mm256_shuffle_epi8(v22, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           11, 12, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m256i min23 = _mm256_min_epu8(v22, perm23);
      __m256i max23 = _mm256_max_epu8(v22, perm23);
      __m256i v23 = _mm256_mask_mov_epi8(max23, 0x800, min23);
      
      return v23;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bosenelson_24_uint8_t(uint8_t * const 
                             arr) {
      
      __m256i v = _mm256_load_si256((__m256i *)arr);
      
      v = bosenelson_24_uint8_t_vec(v);
      
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


