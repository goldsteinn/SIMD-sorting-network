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
#define N 28
#define SORT_NAME minimum_28_uint16_t

#ifndef _SIMD_SORT_minimum_28_uint16_t_H_
#define _SIMD_SORT_minimum_28_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 28
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 14
	SIMD Instructions                : 2 / 70
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
minimum_28_uint16_t_vec(__m512i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [26,27], [24,25], 
                 [22,23], [20,21], [18,19], [16,17], [14,15], [12,13], 
                 [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m512i perm0 = _mm512_shuffle_epi8(v, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 53, 52, 55, 54, 49, 
                                          48, 51, 50, 45, 44, 47, 46, 41, 40, 
                                          43, 42, 37, 36, 39, 38, 33, 32, 35, 
                                          34, 29, 28, 31, 30, 25, 24, 27, 26, 
                                          21, 20, 23, 22, 17, 16, 19, 18, 13, 
                                          12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 
                                          6, 1, 0, 3, 2));
      __m512i min0 = _mm512_min_epu16(v, perm0);
      __m512i max0 = _mm512_max_epu16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x5555555, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [1,27], [0,26], [3,25], 
                 [2,24], [5,23], [4,22], [7,21], [6,20], [9,19], [8,18], 
                 [11,17], [10,16], [13,15], [12,14]) */
      /* Perm:  (31, 30, 29, 28,  1,  0,  3,  2,  5,  4,  7,  6,  9,  8, 11, 
                 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20, 23, 22, 25, 24, 
                 27, 26) */
      __m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 1, 0, 3, 2, 5, 4, 7, 6, 9, 
                                               8, 11, 10, 13, 12, 15, 14, 17, 
                                               16, 19, 18, 21, 20, 23, 22, 
                                               25, 24, 27, 26), v0);
      __m512i min1 = _mm512_min_epu16(v0, perm1);
      __m512i max1 = _mm512_max_epu16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x3fff, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [21,27], [7,26], 
                 [23,25], [22,24], [1,20], [19,19], [18,18], [15,17], 
                 [13,16], [11,14], [10,12], [9,9], [8,8], [0,6], [3,5], 
                 [2,4]) */
      /* Perm:  (31, 30, 29, 28, 21,  7, 23, 22, 25, 24, 27,  1, 19, 18, 15, 
                 13, 17, 11, 16, 10, 14, 12,  9,  8, 26,  0,  3,  2,  5,  4, 
                 20,  6) */
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 21, 7, 23, 22, 25, 24, 27, 
                                               1, 19, 18, 15, 13, 17, 11, 16, 
                                               10, 14, 12, 9, 8, 26, 0, 3, 2, 
                                               5, 4, 20, 6), v1);
      __m512i min2 = _mm512_min_epu16(v1, perm2);
      __m512i max2 = _mm512_max_epu16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0xe0ac8f, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [9,26], 
                 [17,25], [24,24], [5,23], [4,22], [19,21], [20,20], [1,18], 
                 [16,16], [14,15], [12,13], [11,11], [2,10], [6,8], [7,7], 
                 [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27,  9, 17, 24,  5,  4, 19, 20, 21,  1, 25, 
                 16, 14, 15, 12, 13, 11,  2, 26,  6,  7,  8, 23, 22,  3, 10, 
                 18,  0) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 9, 17, 24, 5, 4, 19, 
                                               20, 21, 1, 25, 16, 14, 15, 12, 
                                               13, 11, 2, 26, 6, 7, 8, 23, 
                                               22, 3, 10, 18, 0), v2);
      __m512i min3 = _mm512_min_epu16(v2, perm3);
      __m512i max3 = _mm512_max_epu16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0xa5276, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [21,27], [26,26], 
                 [25,25], [5,24], [15,23], [3,22], [18,20], [19,19], [17,17], 
                 [14,16], [11,13], [4,12], [10,10], [7,9], [8,8], [0,6], 
                 [2,2], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 21, 26, 25,  5, 15,  3, 27, 18, 19, 20, 17, 
                 14, 23, 16, 11,  4, 13, 10,  7,  8,  9,  0, 24, 12, 22,  2,  
                 1,  6) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 21, 26, 25, 5, 15, 3, 27, 
                                               18, 19, 20, 17, 14, 23, 16, 
                                               11, 4, 13, 10, 7, 8, 9, 0, 24, 
                                               12, 22, 2, 1, 6), v3);
      __m512i min4 = _mm512_min_epu16(v3, perm4);
      __m512i max4 = _mm512_max_epu16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0x24c8b9, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [25,27], [23,26], 
                 [16,24], [14,22], [17,21], [19,20], [18,18], [15,15], 
                 [5,13], [12,12], [3,11], [6,10], [9,9], [7,8], [1,4], [0,2]) 
                 */
      /* Perm:  (31, 30, 29, 28, 25, 23, 27, 16, 26, 14, 17, 19, 20, 18, 21, 
                 24, 15, 22,  5, 12,  3,  6,  9,  7,  8, 10, 13,  1, 11,  0,  
                 4,  2) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 25, 23, 27, 16, 26, 14, 
                                               17, 19, 20, 18, 21, 24, 15, 
                                               22, 5, 12, 3, 6, 9, 7, 8, 10, 
                                               13, 1, 11, 0, 4, 2), v4);
      __m512i min5 = _mm512_min_epu16(v4, perm5);
      __m512i max5 = _mm512_max_epu16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0x28b40eb, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [21,26], 
                 [25,25], [20,24], [16,23], [17,22], [12,19], [13,18], 
                 [8,15], [9,14], [4,11], [5,10], [3,7], [1,6], [2,2], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 21, 25, 20, 16, 17, 26, 24, 12, 13, 22, 
                 23,  8,  9, 18, 19,  4,  5, 14, 15,  3,  1, 10, 11,  7,  2,  
                 6,  0) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 21, 25, 20, 16, 17, 
                                               26, 24, 12, 13, 22, 23, 8, 9, 
                                               18, 19, 4, 5, 14, 15, 3, 1, 
                                               10, 11, 7, 2, 6, 0), v5);
      __m512i min6 = _mm512_min_epu16(v5, perm6);
      __m512i max6 = _mm512_max_epu16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0x33333a, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [18,25], [24,24], [19,23], [14,22], [21,21], [10,20], 
                 [7,17], [12,16], [11,15], [5,13], [2,9], [4,8], [6,6], 
                 [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 18, 24, 19, 14, 21, 10, 23, 25,  7, 
                 12, 11, 22,  5, 16, 15, 20,  2,  4, 17,  6, 13,  8,  3,  9,  
                 1,  0) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 18, 24, 19, 14, 
                                               21, 10, 23, 25, 7, 12, 11, 22, 
                                               5, 16, 15, 20, 2, 4, 17, 6, 
                                               13, 8, 3, 9, 1, 0), v6);
      __m512i min7 = _mm512_min_epu16(v6, perm7);
      __m512i max7 = _mm512_max_epu16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0xc5cb4, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [24,25], [22,23], [21,21], [15,20], [16,19], [18,18], 
                 [14,17], [10,13], [7,12], [8,11], [9,9], [6,6], [4,5], 
                 [2,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 24, 25, 22, 23, 21, 15, 16, 18, 14, 
                 19, 20, 17, 10,  7,  8, 13,  9, 11, 12,  6,  4,  5,  2,  3,  
                 1,  0) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 24, 25, 22, 23, 
                                               21, 15, 16, 18, 14, 19, 20, 
                                               17, 10, 7, 8, 13, 9, 11, 12, 
                                               6, 4, 5, 2, 3, 1, 0), v7);
      __m512i min8 = _mm512_min_epu16(v7, perm8);
      __m512i max8 = _mm512_max_epu16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x141c594, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [25,26], 
                 [24,24], [23,23], [21,22], [20,20], [17,19], [15,18], 
                 [13,16], [11,14], [9,12], [8,10], [7,7], [5,6], [4,4], 
                 [3,3], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 25, 26, 24, 23, 21, 22, 20, 17, 15, 19, 
                 13, 18, 11, 16,  9, 14,  8, 12, 10,  7,  5,  6,  4,  3,  1,  
                 2,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 25, 26, 24, 23, 21, 
                                               22, 20, 17, 15, 19, 13, 18, 
                                               11, 16, 9, 14, 8, 12, 10, 7, 
                                               5, 6, 4, 3, 1, 2, 0), v8);
      __m512i min9 = _mm512_min_epu16(v8, perm9);
      __m512i max9 = _mm512_max_epu16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0x222ab22, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [22,24], [23,23], [18,21], [19,20], [15,17], 
                 [14,16], [11,13], [10,12], [6,9], [7,8], [3,5], [4,4], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 22, 23, 24, 18, 19, 20, 21, 15, 
                 14, 17, 16, 11, 10, 13, 12,  6,  7,  8,  9,  3,  4,  5,  2,  
                 1,  0) */
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 22, 23, 24, 
                                                18, 19, 20, 21, 15, 14, 17, 
                                                16, 11, 10, 13, 12, 6, 7, 8, 
                                                9, 3, 4, 5, 2, 1, 0), v9);
      __m512i min10 = _mm512_min_epu16(v9, perm10);
      __m512i max10 = _mm512_max_epu16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x4cccc8, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [24,25], [20,23], [21,22], [18,19], [16,17], [14,15], 
                 [12,13], [10,11], [8,9], [4,7], [5,6], [2,3], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 24, 25, 20, 21, 22, 23, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  4,  5,  6,  7,  2,  3,  
                 1,  0) */
      __m512i perm11 = _mm512_shuffle_epi8(v10, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 49, 48, 51, 50, 41, 40, 43, 
                                           42, 45, 44, 47, 46, 37, 36, 39, 
                                           38, 33, 32, 35, 34, 29, 28, 31, 
                                           30, 25, 24, 27, 26, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 9, 8, 11, 10, 
                                           13, 12, 15, 14, 5, 4, 7, 6, 3, 2, 
                                           1, 0));
      __m512i min11 = _mm512_min_epu16(v10, perm11);
      __m512i max11 = _mm512_max_epu16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0x1355534, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [22,23], [20,21], [17,19], [16,18], 
                 [13,15], [12,14], [9,11], [8,10], [6,7], [4,5], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 21, 17, 16, 19, 
                 18, 13, 12, 15, 14,  9,  8, 11, 10,  6,  7,  4,  5,  3,  2,  
                 1,  0) */
      __m512i perm12 = _mm512_shuffle_epi8(v11, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 45, 44, 47, 
                                           46, 41, 40, 43, 42, 35, 34, 33, 
                                           32, 39, 38, 37, 36, 27, 26, 25, 
                                           24, 31, 30, 29, 28, 19, 18, 17, 
                                           16, 23, 22, 21, 20, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min12 = _mm512_min_epu16(v11, perm12);
      __m512i max12 = _mm512_max_epu16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x533350, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [23,24], [21,22], [19,20], [17,18], [15,16], 
                 [13,14], [11,12], [9,10], [7,8], [5,6], [3,4], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  2,  
                 1,  0) */
      __m512i perm13 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 23, 24, 21, 
                                                22, 19, 20, 17, 18, 15, 16, 
                                                13, 14, 11, 12, 9, 10, 7, 8, 
                                                5, 6, 3, 4, 2, 1, 0), v12);
      __m512i min13 = _mm512_min_epu16(v12, perm13);
      __m512i max13 = _mm512_max_epu16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0xaaaaa8, min13);
      
      return v13;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
minimum_28_uint16_t(uint16_t * const 
                             arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = minimum_28_uint16_t_vec(v);
      
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


