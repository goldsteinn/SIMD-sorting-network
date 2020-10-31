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

#define TYPE uint16_t
#define N 22
#define SORT_NAME bosenelson_22_uint16_t

#ifndef _SIMD_SORT_bosenelson_22_uint16_t_H_
#define _SIMD_SORT_bosenelson_22_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 22
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : bosenelson
	Network Depth                    : 23
	SIMD Instructions                : 2 / 115
	Optimization Preference          : space
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512bw
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
 __m512i __attribute__((const)) 
bosenelson_22_uint16_t_vec(__m512i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [20,21], [19,19], 
                 [17,18], [16,16], [14,15], [13,13], [11,12], [9,10], [8,8], 
                 [6,7], [5,5], [3,4], [2,2], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 19, 17, 18, 
                 16, 14, 15, 13, 11, 12,  9, 10,  8,  6,  7,  5,  3,  4,  2,  
                 0,  1) */
      __m512i perm0 = _mm512_shuffle_epi8(v, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 55, 54, 53, 52, 51, 
                                          50, 49, 48, 47, 46, 45, 44, 41, 40, 
                                          43, 42, 39, 38, 35, 34, 37, 36, 33, 
                                          32, 29, 28, 31, 30, 27, 26, 23, 22, 
                                          25, 24, 19, 18, 21, 20, 17, 16, 13, 
                                          12, 15, 14, 11, 10, 7, 6, 9, 8, 5, 
                                          4, 1, 0, 3, 2));
      __m512i min0 = _mm512_min_epu16(v, perm0);
      __m512i max0 = _mm512_max_epu16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x124a49, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [19,21], [20,20], 
                 [16,18], [17,17], [13,15], [14,14], [12,12], [11,11], 
                 [8,10], [9,9], [5,7], [6,6], [2,4], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 19, 20, 21, 16, 17, 
                 18, 13, 14, 15, 12, 11,  8,  9, 10,  5,  6,  7,  2,  3,  4,  
                 1,  0) */
      __m512i perm1 = _mm512_shuffle_epi8(v0, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 55, 54, 53, 52, 51, 
                                          50, 49, 48, 47, 46, 45, 44, 39, 38, 
                                          41, 40, 43, 42, 33, 32, 35, 34, 37, 
                                          36, 27, 26, 29, 28, 31, 30, 25, 24, 
                                          23, 22, 17, 16, 19, 18, 21, 20, 11, 
                                          10, 13, 12, 15, 14, 5, 4, 7, 6, 9, 
                                          8, 3, 2, 1, 0));
      __m512i min1 = _mm512_min_epu16(v0, perm1);
      __m512i max1 = _mm512_max_epu16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x92124, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [18,21], [19,20], 
                 [16,17], [12,15], [13,14], [11,11], [7,10], [8,9], [5,6], 
                 [1,4], [2,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 18, 19, 20, 21, 16, 
                 17, 12, 13, 14, 15, 11,  7,  8,  9, 10,  5,  6,  1,  2,  3,  
                 4,  0) */
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               18, 19, 20, 21, 16, 17, 12, 
                                               13, 14, 15, 11, 7, 8, 9, 10, 
                                               5, 6, 1, 2, 3, 4, 0), v1);
      __m512i min2 = _mm512_min_epu16(v1, perm2);
      __m512i max2 = _mm512_max_epu16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0xd31a6, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [15,21], [17,20], 
                 [16,19], [18,18], [11,14], [13,13], [12,12], [4,10], [6,9], 
                 [5,8], [7,7], [0,3], [2,2], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 15, 17, 16, 18, 20, 
                 19, 21, 11, 13, 12, 14,  4,  6,  5,  7,  9,  8, 10,  0,  2,  
                 1,  3) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               15, 17, 16, 18, 20, 19, 21, 
                                               11, 13, 12, 14, 4, 6, 5, 7, 9, 
                                               8, 10, 0, 2, 1, 3), v2);
      __m512i min3 = _mm512_min_epu16(v2, perm3);
      __m512i max3 = _mm512_max_epu16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x38871, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [10,21], [18,20], 
                 [17,19], [16,16], [15,15], [12,14], [11,13], [7,9], [6,8], 
                 [5,5], [4,4], [1,3], [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 10, 18, 17, 20, 19, 
                 16, 15, 12, 11, 14, 13, 21,  7,  6,  9,  8,  5,  4,  1,  0,  
                 3,  2) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               10, 18, 17, 20, 19, 16, 15, 
                                               12, 11, 14, 13, 21, 7, 6, 9, 
                                               8, 5, 4, 1, 0, 3, 2), v3);
      __m512i min4 = _mm512_min_epu16(v3, perm4);
      __m512i max4 = _mm512_max_epu16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0x61cc3, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [14,20], 
                 [18,19], [11,17], [16,16], [15,15], [12,13], [10,10], [3,9], 
                 [7,8], [0,6], [5,5], [4,4], [1,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 14, 18, 19, 11, 
                 16, 15, 20, 12, 13, 17, 10,  3,  7,  8,  0,  5,  4,  9,  1,  
                 2,  6) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 14, 18, 19, 11, 16, 15, 
                                               20, 12, 13, 17, 10, 3, 7, 8, 
                                               0, 5, 4, 9, 1, 2, 6), v4);
      __m512i min5 = _mm512_min_epu16(v4, perm5);
      __m512i max5 = _mm512_max_epu16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0x4588b, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [15,20], 
                 [13,19], [12,18], [17,17], [11,16], [14,14], [10,10], [4,9], 
                 [2,8], [1,7], [6,6], [0,5], [3,3]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 15, 13, 12, 17, 
                 11, 20, 14, 19, 18, 16, 10,  4,  2,  1,  6,  0,  9,  3,  8,  
                 7,  5) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 15, 13, 12, 17, 11, 20, 
                                               14, 19, 18, 16, 10, 4, 2, 1, 
                                               6, 0, 9, 3, 8, 7, 5), v5);
      __m512i min6 = _mm512_min_epu16(v5, perm6);
      __m512i max6 = _mm512_max_epu16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0xb817, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [9,20], 
                 [14,19], [18,18], [12,17], [16,16], [15,15], [13,13], 
                 [0,11], [10,10], [3,8], [7,7], [1,6], [5,5], [4,4], [2,2]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21,  9, 14, 18, 12, 
                 16, 15, 19, 13, 17,  0, 10, 20,  3,  7,  1,  5,  4,  8,  2,  
                 6, 11) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 9, 14, 18, 12, 16, 15, 19, 
                                               13, 17, 0, 10, 20, 3, 7, 1, 5, 
                                               4, 8, 2, 6, 11), v6);
      __m512i min7 = _mm512_min_epu16(v6, perm7);
      __m512i max7 = _mm512_max_epu16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x520b, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [10,20], 
                 [15,19], [18,18], [14,17], [12,16], [13,13], [11,11], [9,9], 
                 [4,8], [7,7], [3,6], [1,5], [2,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 10, 15, 18, 14, 
                 12, 19, 17, 13, 16, 11, 20,  9,  4,  7,  3,  1,  8,  6,  2,  
                 5,  0) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 10, 15, 18, 14, 12, 19, 
                                               17, 13, 16, 11, 20, 9, 4, 7, 
                                               3, 1, 8, 6, 2, 5, 0), v7);
      __m512i min8 = _mm512_min_epu16(v7, perm8);
      __m512i max8 = _mm512_max_epu16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0xd41a, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [8,19], [15,18], [17,17], [13,16], [14,14], [1,12], [11,11], 
                 [10,10], [9,9], [4,7], [6,6], [2,5], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,  8, 15, 17, 
                 13, 18, 14, 16,  1, 11, 10,  9, 19,  4,  6,  2,  7,  3,  5, 
                 12,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 8, 15, 17, 13, 18, 14, 
                                               16, 1, 11, 10, 9, 19, 4, 6, 2, 
                                               7, 3, 5, 12, 0), v8);
      __m512i min9 = _mm512_min_epu16(v8, perm9);
      __m512i max9 = _mm512_max_epu16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0xa116, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [9,19], [7,18], [15,17], [14,16], [2,13], [12,12], [1,11], 
                 [10,10], [8,8], [4,6], [3,5], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20,  9,  7, 15, 
                 14, 17, 16,  2, 12,  1, 10, 19,  8, 18,  4,  3,  6,  5, 13, 
                 11,  0) */
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 9, 7, 15, 14, 17, 16, 
                                                2, 12, 1, 10, 19, 8, 18, 4, 
                                                3, 6, 5, 13, 11, 0), v9);
      __m512i min10 = _mm512_min_epu16(v9, perm10);
      __m512i max10 = _mm512_max_epu16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0xc29e, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [10,19], [18,18], [6,17], [15,16], [3,14], [13,13], [12,12], 
                 [2,11], [9,9], [8,8], [7,7], [4,5], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 10, 18,  6, 
                 15, 16,  3, 13, 12,  2, 19,  9,  8,  7, 17,  4,  5, 14, 11,  
                 1,  0) */
      __m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 10, 18, 6, 15, 16, 3, 
                                                13, 12, 2, 19, 9, 8, 7, 17, 
                                                4, 5, 14, 11, 1, 0), v10);
      __m512i min11 = _mm512_min_epu16(v10, perm11);
      __m512i max11 = _mm512_max_epu16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0x845c, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [10,18], [7,17], [5,16], [4,15], [14,14], [3,13], 
                 [12,12], [11,11], [9,9], [8,8], [6,6], [2,2], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 10,  7,  
                 5,  4, 14,  3, 12, 11, 18,  9,  8, 17,  6, 16, 15, 13,  2,  
                 1,  0) */
      __m512i perm12 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 10, 7, 5, 4, 14, 
                                                3, 12, 11, 18, 9, 8, 17, 6, 
                                                16, 15, 13, 2, 1, 0), v11);
      __m512i min12 = _mm512_min_epu16(v11, perm12);
      __m512i max12 = _mm512_max_epu16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x4b8, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [9,17], [6,16], [15,15], [4,14], [13,13], 
                 [3,12], [11,11], [10,10], [8,8], [7,7], [5,5], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,  9,  
                 6, 15,  4, 13,  3, 11, 10, 17,  8,  7, 16,  5, 14, 12,  2,  
                 1,  0) */
      __m512i perm13 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 9, 6, 15, 4, 
                                                13, 3, 11, 10, 17, 8, 7, 16, 
                                                5, 14, 12, 2, 1, 0), v12);
      __m512i min13 = _mm512_min_epu16(v12, perm13);
      __m512i max13 = _mm512_max_epu16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0x258, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [10,17], [7,16], [15,15], [14,14], [4,13], 
                 [12,12], [3,11], [9,9], [8,8], [6,6], [5,5], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 10,  
                 7, 15, 14,  4, 12,  3, 17,  9,  8, 16,  6,  5, 13, 11,  2,  
                 1,  0) */
      __m512i perm14 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 10, 7, 15, 
                                                14, 4, 12, 3, 17, 9, 8, 16, 
                                                6, 5, 13, 11, 2, 1, 0), 
                                                v13);
      __m512i min14 = _mm512_min_epu16(v13, perm14);
      __m512i max14 = _mm512_max_epu16(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi16(max14, 0x498, min14);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [8,16], [15,15], [14,14], [7,13], 
                 [4,12], [11,11], [10,10], [9,9], [6,6], [5,5], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  
                 8, 15, 14,  7,  4, 11, 10,  9, 16, 13,  6,  5, 12,  3,  2,  
                 1,  0) */
      __m512i perm15 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 17, 8, 15, 
                                                14, 7, 4, 11, 10, 9, 16, 13, 
                                                6, 5, 12, 3, 2, 1, 0), v14);
      __m512i min15 = _mm512_min_epu16(v14, perm15);
      __m512i max15 = _mm512_max_epu16(v14, perm15);
      __m512i v15 = _mm512_mask_mov_epi16(max15, 0x190, min15);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [9,16], [15,15], [8,14], [13,13], 
                 [6,12], [4,11], [10,10], [7,7], [5,5], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  
                 9, 15,  8, 13,  6,  4, 10, 16, 14,  7, 12,  5, 11,  3,  2,  
                 1,  0) */
      __m512i perm16 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 17, 9, 15, 8, 
                                                13, 6, 4, 10, 16, 14, 7, 12, 
                                                5, 11, 3, 2, 1, 0), v15);
      __m512i min16 = _mm512_min_epu16(v15, perm16);
      __m512i max16 = _mm512_max_epu16(v15, perm16);
      __m512i v16 = _mm512_mask_mov_epi16(max16, 0x350, min16);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [10,16], [9,15], [14,14], 
                 [13,13], [7,12], [5,11], [8,8], [6,6], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 10,  9, 14, 13,  7,  5, 16, 15,  8, 12,  6, 11,  4,  3,  2,  
                 1,  0) */
      __m512i perm17 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 17, 10, 9, 
                                                14, 13, 7, 5, 16, 15, 8, 12, 
                                                6, 11, 4, 3, 2, 1, 0), v16);
      __m512i min17 = _mm512_min_epu16(v16, perm17);
      __m512i max17 = _mm512_max_epu16(v16, perm17);
      __m512i v17 = _mm512_mask_mov_epi16(max17, 0x6a0, min17);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [10,15], [9,14], 
                 [13,13], [12,12], [6,11], [8,8], [7,7], [5,5], [4,4], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 10,  9, 13, 12,  6, 15, 14,  8,  7, 11,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm18 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 17, 16, 10, 
                                                9, 13, 12, 6, 15, 14, 8, 7, 
                                                11, 5, 4, 3, 2, 1, 0), v17);
      __m512i min18 = _mm512_min_epu16(v17, perm18);
      __m512i max18 = _mm512_max_epu16(v17, perm18);
      __m512i v18 = _mm512_mask_mov_epi16(max18, 0x640, min18);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [10,14], 
                 [13,13], [9,12], [7,11], [8,8], [6,6], [5,5], [4,4], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 10, 13,  9,  7, 14, 12,  8, 11,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm19 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 17, 16, 15, 
                                                10, 13, 9, 7, 14, 12, 8, 11, 
                                                6, 5, 4, 3, 2, 1, 0), v18);
      __m512i min19 = _mm512_min_epu16(v18, perm19);
      __m512i max19 = _mm512_max_epu16(v18, perm19);
      __m512i v19 = _mm512_mask_mov_epi16(max19, 0x680, min19);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [10,13], [12,12], [8,11], [9,9], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 10, 12,  8, 13,  9, 11,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm20 = _mm512_shuffle_epi8(v19, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 31, 30, 29, 
                                           28, 21, 20, 25, 24, 17, 16, 27, 
                                           26, 19, 18, 23, 22, 15, 14, 13, 
                                           12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min20 = _mm512_min_epu16(v19, perm20);
      __m512i max20 = _mm512_max_epu16(v19, perm20);
      __m512i v20 = _mm512_mask_mov_epi16(max20, 0x500, min20);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [10,12], [9,11], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 10,  9, 12, 11,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm21 = _mm512_shuffle_epi8(v20, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 31, 30, 29, 
                                           28, 27, 26, 21, 20, 19, 18, 25, 
                                           24, 23, 22, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min21 = _mm512_min_epu16(v20, perm21);
      __m512i max21 = _mm512_max_epu16(v20, perm21);
      __m512i v21 = _mm512_mask_mov_epi16(max21, 0x600, min21);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [10,11], [9,9], [8,8], [7,7], [6,6], 
                 [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 10, 11,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm22 = _mm512_shuffle_epi8(v21, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 31, 30, 29, 
                                           28, 27, 26, 25, 24, 21, 20, 23, 
                                           22, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min22 = _mm512_min_epu16(v21, perm22);
      __m512i max22 = _mm512_max_epu16(v21, perm22);
      __m512i v22 = _mm512_mask_mov_epi16(max22, 0x400, min22);
      
      return v22;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bosenelson_22_uint16_t(uint16_t * const 
                             arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = bosenelson_22_uint16_t_vec(v);
      
      _mm512_store_si512((__m512i *)arr, v);
      
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


