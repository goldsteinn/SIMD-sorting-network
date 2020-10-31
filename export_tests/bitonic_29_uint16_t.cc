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
#define N 29
#define SORT_NAME bitonic_29_uint16_t

#ifndef _SIMD_SORT_bitonic_29_uint16_t_H_
#define _SIMD_SORT_bitonic_29_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 29
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 2 / 75
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
bitonic_29_uint16_t_vec(__m512i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [27,28], [25,26], [23,24], 
                 [21,22], [19,20], [17,18], [15,16], [14,14], [12,13], 
                 [10,11], [8,9], [7,7], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 14, 12, 13, 10, 11,  8,  9,  7,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m512i perm0 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               27, 28, 25, 26, 23, 24, 21, 
                                               22, 19, 20, 17, 18, 15, 16, 
                                               14, 12, 13, 10, 11, 8, 9, 7, 
                                               5, 6, 3, 4, 1, 2, 0), v);
      __m512i min0 = _mm512_min_epu16(v, perm0);
      __m512i max0 = _mm512_max_epu16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0xaaa952a, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [25,28], [26,27], [21,24], 
                 [22,23], [17,20], [18,19], [14,16], [15,15], [10,13], 
                 [11,12], [7,9], [8,8], [3,6], [4,5], [2,2], [0,1]) */
      /* Perm:  (31, 30, 29, 25, 26, 27, 28, 21, 22, 23, 24, 17, 18, 19, 20, 
                 14, 15, 16, 10, 11, 12, 13,  7,  8,  9,  3,  4,  5,  6,  2,  
                 0,  1) */
      __m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               25, 26, 27, 28, 21, 22, 23, 
                                               24, 17, 18, 19, 20, 14, 15, 
                                               16, 10, 11, 12, 13, 7, 8, 9, 
                                               3, 4, 5, 6, 2, 0, 1), v0);
      __m512i min1 = _mm512_min_epu16(v0, perm1);
      __m512i max1 = _mm512_max_epu16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x6664c99, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [27,28], [25,26], [23,24], 
                 [21,22], [19,20], [17,18], [16,16], [14,15], [12,13], 
                 [10,11], [9,9], [7,8], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 
                 16, 14, 15, 12, 13, 10, 11,  9,  7,  8,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               27, 28, 25, 26, 23, 24, 21, 
                                               22, 19, 20, 17, 18, 16, 14, 
                                               15, 12, 13, 10, 11, 9, 7, 8, 
                                               5, 6, 3, 4, 1, 2, 0), v1);
      __m512i min2 = _mm512_min_epu16(v1, perm2);
      __m512i max2 = _mm512_max_epu16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0xaaa54aa, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [21,28], [22,27], [23,26], 
                 [24,25], [20,20], [14,19], [15,18], [16,17], [13,13], 
                 [7,12], [8,11], [9,10], [0,6], [1,5], [2,4], [3,3]) */
      /* Perm:  (31, 30, 29, 21, 22, 23, 24, 25, 26, 27, 28, 20, 14, 15, 16, 
                 17, 18, 19, 13,  7,  8,  9, 10, 11, 12,  0,  1,  2,  3,  4,  
                 5,  6) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               21, 22, 23, 24, 25, 26, 27, 
                                               28, 20, 14, 15, 16, 17, 18, 
                                               19, 13, 7, 8, 9, 10, 11, 12, 
                                               0, 1, 2, 3, 4, 5, 6), v2);
      __m512i min3 = _mm512_min_epu16(v2, perm3);
      __m512i max3 = _mm512_max_epu16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x1e1c387, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [26,28], [25,27], [22,24], 
                 [21,23], [18,20], [17,19], [14,16], [15,15], [11,13], 
                 [10,12], [7,9], [8,8], [4,6], [5,5], [1,3], [0,2]) */
      /* Perm:  (31, 30, 29, 26, 25, 28, 27, 22, 21, 24, 23, 18, 17, 20, 19, 
                 14, 15, 16, 11, 10, 13, 12,  7,  8,  9,  4,  5,  6,  1,  0,  
                 3,  2) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               26, 25, 28, 27, 22, 21, 24, 
                                               23, 18, 17, 20, 19, 14, 15, 
                                               16, 11, 10, 13, 12, 7, 8, 9, 
                                               4, 5, 6, 1, 0, 3, 2), v3);
      __m512i min4 = _mm512_min_epu16(v3, perm4);
      __m512i max4 = _mm512_max_epu16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0x6664c93, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [27,28], [25,26], [23,24], 
                 [21,22], [19,20], [17,18], [15,16], [14,14], [12,13], 
                 [10,11], [8,9], [7,7], [6,6], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 14, 12, 13, 10, 11,  8,  9,  7,  6,  4,  5,  2,  3,  
                 0,  1) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               27, 28, 25, 26, 23, 24, 21, 
                                               22, 19, 20, 17, 18, 15, 16, 
                                               14, 12, 13, 10, 11, 8, 9, 7, 
                                               6, 4, 5, 2, 3, 0, 1), v4);
      __m512i min5 = _mm512_min_epu16(v4, perm5);
      __m512i max5 = _mm512_max_epu16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0xaaa9515, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [14,28], [15,27], [16,26], 
                 [17,25], [18,24], [19,23], [20,22], [21,21], [13,13], 
                 [0,12], [1,11], [2,10], [3,9], [4,8], [5,7], [6,6]) */
      /* Perm:  (31, 30, 29, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 
                 26, 27, 28, 13,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 
                 11, 12) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               14, 15, 16, 17, 18, 19, 20, 
                                               21, 22, 23, 24, 25, 26, 27, 
                                               28, 13, 0, 1, 2, 3, 4, 5, 6, 
                                               7, 8, 9, 10, 11, 12), v5);
      __m512i min6 = _mm512_min_epu16(v5, perm6);
      __m512i max6 = _mm512_max_epu16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0x1fc03f, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [24,28], [23,27], [22,26], 
                 [25,25], [17,21], [16,20], [15,19], [14,18], [9,13], [8,12], 
                 [7,11], [6,10], [1,5], [0,4], [3,3], [2,2]) */
      /* Perm:  (31, 30, 29, 24, 23, 22, 25, 28, 27, 26, 17, 16, 15, 14, 21, 
                 20, 19, 18,  9,  8,  7,  6, 13, 12, 11, 10,  1,  0,  3,  2,  
                 5,  4) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               24, 23, 22, 25, 28, 27, 26, 
                                               17, 16, 15, 14, 21, 20, 19, 
                                               18, 9, 8, 7, 6, 13, 12, 11, 
                                               10, 1, 0, 3, 2, 5, 4), v6);
      __m512i min7 = _mm512_min_epu16(v6, perm7);
      __m512i max7 = _mm512_max_epu16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x1c3c3c3, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [26,28], [27,27], [23,25], 
                 [22,24], [19,21], [18,20], [15,17], [14,16], [11,13], 
                 [10,12], [7,9], [6,8], [3,5], [2,4], [0,1]) */
      /* Perm:  (31, 30, 29, 26, 27, 28, 23, 22, 25, 24, 19, 18, 21, 20, 15, 
                 14, 17, 16, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  5,  4,  
                 0,  1) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               26, 27, 28, 23, 22, 25, 24, 
                                               19, 18, 21, 20, 15, 14, 17, 
                                               16, 11, 10, 13, 12, 7, 6, 9, 
                                               8, 3, 2, 5, 4, 0, 1), v7);
      __m512i min8 = _mm512_min_epu16(v7, perm8);
      __m512i max8 = _mm512_max_epu16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x4cccccd, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [1,28], [26,27], [24,25], 
                 [22,23], [20,21], [18,19], [16,17], [14,15], [12,13], 
                 [10,11], [8,9], [6,7], [4,5], [2,3], [0,0]) */
      /* Perm:  (31, 30, 29,  1, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3, 
                 28,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               1, 26, 27, 24, 25, 22, 23, 20, 
                                               21, 18, 19, 16, 17, 14, 15, 
                                               12, 13, 10, 11, 8, 9, 6, 7, 4, 
                                               5, 2, 3, 28, 0), v8);
      __m512i min9 = _mm512_min_epu16(v8, perm9);
      __m512i max9 = _mm512_max_epu16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0x5555556, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [2,27], [3,26], [4,25], 
                 [5,24], [6,23], [7,22], [8,21], [9,20], [10,19], [11,18], 
                 [12,17], [13,16], [15,15], [14,14], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 
                 13, 15, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,  
                 1,  0) */
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 2, 3, 4, 5, 6, 7, 8, 9, 
                                                10, 11, 12, 13, 15, 14, 16, 
                                                17, 18, 19, 20, 21, 22, 23, 
                                                24, 25, 26, 27, 1, 0), v9);
      __m512i min10 = _mm512_min_epu16(v9, perm10);
      __m512i max10 = _mm512_max_epu16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x3ffc, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [20,28], [19,27], [18,26], 
                 [17,25], [16,24], [23,23], [22,22], [21,21], [6,15], [7,14], 
                 [5,13], [4,12], [3,11], [2,10], [1,9], [0,8]) */
      /* Perm:  (31, 30, 29, 20, 19, 18, 17, 16, 23, 22, 21, 28, 27, 26, 25, 
                 24,  6,  7,  5,  4,  3,  2,  1,  0, 14, 15, 13, 12, 11, 10,  
                 9,  8) */
      __m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                20, 19, 18, 17, 16, 23, 22, 
                                                21, 28, 27, 26, 25, 24, 6, 7, 
                                                5, 4, 3, 2, 1, 0, 14, 15, 13, 
                                                12, 11, 10, 9, 8), v10);
      __m512i min11 = _mm512_min_epu16(v10, perm11);
      __m512i max11 = _mm512_max_epu16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0x1f00ff, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [24,28], [25,27], [26,26], 
                 [19,23], [18,22], [17,21], [16,20], [10,15], [11,14], 
                 [9,13], [8,12], [3,7], [2,6], [1,5], [0,4]) */
      /* Perm:  (31, 30, 29, 24, 25, 26, 27, 28, 19, 18, 17, 16, 23, 22, 21, 
                 20, 10, 11,  9,  8, 14, 15, 13, 12,  3,  2,  1,  0,  7,  6,  
                 5,  4) */
      __m512i perm12 = _mm512_shuffle_epi8(v11, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 49, 48, 51, 50, 53, 
                                           52, 55, 54, 57, 56, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 21, 20, 23, 
                                           22, 19, 18, 17, 16, 29, 28, 31, 
                                           30, 27, 26, 25, 24, 7, 6, 5, 4, 3, 
                                           2, 1, 0, 15, 14, 13, 12, 11, 10, 
                                           9, 8));
      __m512i min12 = _mm512_min_epu16(v11, perm12);
      __m512i max12 = _mm512_max_epu16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x30f0f0f, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [24,26], 
                 [25,25], [21,23], [20,22], [17,19], [16,18], [12,15], 
                 [13,14], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 24, 25, 26, 21, 20, 23, 22, 17, 16, 19, 
                 18, 12, 13, 14, 15,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m512i perm13 = _mm512_shuffle_epi8(v12, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 49, 
                                           48, 51, 50, 53, 52, 43, 42, 41, 
                                           40, 47, 46, 45, 44, 35, 34, 33, 
                                           32, 39, 38, 37, 36, 25, 24, 27, 
                                           26, 29, 28, 31, 30, 19, 18, 17, 
                                           16, 23, 22, 21, 20, 11, 10, 9, 8, 
                                           15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 
                                           5, 4));
      __m512i min13 = _mm512_min_epu16(v12, perm13);
      __m512i max13 = _mm512_max_epu16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0x1333333, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [26,27], [24,25], 
                 [22,23], [20,21], [18,19], [16,17], [14,15], [12,13], 
                 [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m512i perm14 = _mm512_shuffle_epi8(v13, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 53, 52, 55, 
                                           54, 49, 48, 51, 50, 45, 44, 47, 
                                           46, 41, 40, 43, 42, 37, 36, 39, 
                                           38, 33, 32, 35, 34, 29, 28, 31, 
                                           30, 25, 24, 27, 26, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 
                                           0, 3, 2));
      __m512i min14 = _mm512_min_epu16(v13, perm14);
      __m512i max14 = _mm512_max_epu16(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi16(max14, 0x5555555, min14);
      
      return v14;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bitonic_29_uint16_t(uint16_t * const 
                             arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = bitonic_29_uint16_t_vec(v);
      
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


