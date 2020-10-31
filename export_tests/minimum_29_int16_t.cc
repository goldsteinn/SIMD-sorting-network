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

#define TYPE int16_t
#define N 29
#define SORT_NAME minimum_29_int16_t

#ifndef _SIMD_SORT_minimum_29_int16_t_H_
#define _SIMD_SORT_minimum_29_int16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 29
	Underlying Sort Type             : int16_t
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
minimum_29_int16_t_vec(__m512i v) {
      
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
      __m512i min0 = _mm512_min_epi16(v, perm0);
      __m512i max0 = _mm512_max_epi16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x5555555, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [25,27], [24,26], 
                 [21,23], [20,22], [17,19], [16,18], [13,15], [12,14], 
                 [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
      /* Perm:  (31, 30, 29, 28, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 
                 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m512i perm1 = _mm512_shuffle_epi8(v0, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 51, 50, 49, 48, 55, 
                                          54, 53, 52, 43, 42, 41, 40, 47, 46, 
                                          45, 44, 35, 34, 33, 32, 39, 38, 37, 
                                          36, 27, 26, 25, 24, 31, 30, 29, 28, 
                                          19, 18, 17, 16, 23, 22, 21, 20, 11, 
                                          10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 
                                          0, 7, 6, 5, 4));
      __m512i min1 = _mm512_min_epi16(v0, perm1);
      __m512i max1 = _mm512_max_epi16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x3333333, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [24,28], [27,27], [26,26], 
                 [25,25], [19,23], [18,22], [17,21], [16,20], [11,15], 
                 [10,14], [9,13], [8,12], [3,7], [2,6], [1,5], [0,4]) */
      /* Perm:  (31, 30, 29, 24, 27, 26, 25, 28, 19, 18, 17, 16, 23, 22, 21, 
                 20, 11, 10,  9,  8, 15, 14, 13, 12,  3,  2,  1,  0,  7,  6,  
                 5,  4) */
      __m512i perm2 = _mm512_shuffle_epi8(v1, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 49, 48, 55, 54, 53, 52, 51, 
                                          50, 57, 56, 39, 38, 37, 36, 35, 34, 
                                          33, 32, 47, 46, 45, 44, 43, 42, 41, 
                                          40, 23, 22, 21, 20, 19, 18, 17, 16, 
                                          31, 30, 29, 28, 27, 26, 25, 24, 7, 
                                          6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 
                                          12, 11, 10, 9, 8));
      __m512i min2 = _mm512_min_epi16(v1, perm2);
      __m512i max2 = _mm512_max_epi16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0x10f0f0f, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [20,28], [19,27], [18,26], 
                 [17,25], [16,24], [23,23], [22,22], [21,21], [7,15], [6,14], 
                 [5,13], [4,12], [3,11], [2,10], [1,9], [0,8]) */
      /* Perm:  (31, 30, 29, 20, 19, 18, 17, 16, 23, 22, 21, 28, 27, 26, 25, 
                 24,  7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  
                 9,  8) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               20, 19, 18, 17, 16, 23, 22, 
                                               21, 28, 27, 26, 25, 24, 7, 6, 
                                               5, 4, 3, 2, 1, 0, 15, 14, 13, 
                                               12, 11, 10, 9, 8), v2);
      __m512i min3 = _mm512_min_epi16(v2, perm3);
      __m512i max3 = _mm512_max_epi16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x1f00ff, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [19,28], [23,27], [21,26], 
                 [22,25], [17,24], [18,20], [0,16], [15,15], [7,14], [11,13], 
                 [3,12], [5,10], [6,9], [1,8], [2,4]) */
      /* Perm:  (31, 30, 29, 19, 23, 21, 22, 17, 27, 25, 26, 18, 28, 20, 24,  
                 0, 15,  7, 11,  3, 13,  5,  6,  1, 14,  9, 10,  2, 12,  4,  
                 8, 16) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               19, 23, 21, 22, 17, 27, 25, 
                                               26, 18, 28, 20, 24, 0, 15, 7, 
                                               11, 3, 13, 5, 6, 1, 14, 9, 10, 
                                               2, 12, 4, 8, 16), v3);
      __m512i min4 = _mm512_min_epi16(v3, perm4);
      __m512i max4 = _mm512_max_epi16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0xee08ef, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [26,28], [27,27], [9,25], 
                 [20,24], [23,23], [6,22], [19,21], [17,18], [16,16], 
                 [15,15], [13,14], [10,12], [7,11], [4,8], [3,5], [1,2], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 26, 27, 28,  9, 20, 23,  6, 19, 24, 21, 17, 18, 
                 16, 15, 13, 14, 10,  7, 12, 25,  4, 11, 22,  3,  8,  5,  1,  
                 2,  0) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               26, 27, 28, 9, 20, 23, 6, 19, 
                                               24, 21, 17, 18, 16, 15, 13, 
                                               14, 10, 7, 12, 25, 4, 11, 22, 
                                               3, 8, 5, 1, 2, 0), v4);
      __m512i min5 = _mm512_min_epi16(v4, perm5);
      __m512i max5 = _mm512_max_epi16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0x41a26da, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [12,28], [11,27], [21,26], 
                 [13,25], [8,24], [7,23], [22,22], [4,20], [3,19], [2,18], 
                 [1,17], [16,16], [15,15], [14,14], [5,10], [9,9], [6,6], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 12, 11, 21, 13,  8,  7, 22, 26,  4,  3,  2,  1, 
                 16, 15, 14, 25, 28, 27,  5,  9, 24, 23,  6, 10, 20, 19, 18, 
                 17,  0) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               12, 11, 21, 13, 8, 7, 22, 26, 
                                               4, 3, 2, 1, 16, 15, 14, 25, 
                                               28, 27, 5, 9, 24, 23, 6, 10, 
                                               20, 19, 18, 17, 0), v5);
      __m512i min6 = _mm512_min_epi16(v5, perm6);
      __m512i max6 = _mm512_max_epi16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0x2039be, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [14,28], [15,27], [10,26], 
                 [25,25], [22,24], [11,23], [5,21], [8,20], [19,19], [6,18], 
                 [3,17], [4,16], [13,13], [12,12], [7,9], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 14, 15, 10, 25, 22, 11, 24,  5,  8, 19,  6,  3,  
                 4, 27, 28, 13, 12, 23, 26,  7, 20,  9, 18, 21, 16, 17,  2,  
                 1,  0) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               14, 15, 10, 25, 22, 11, 24, 5, 
                                               8, 19, 6, 3, 4, 27, 28, 13, 
                                               12, 23, 26, 7, 20, 9, 18, 21, 
                                               16, 17, 2, 1, 0), v6);
      __m512i min7 = _mm512_min_epi16(v6, perm7);
      __m512i max7 = _mm512_max_epi16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x40cdf8, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [23,28], [27,27], [15,26], 
                 [25,25], [14,24], [10,22], [9,21], [12,20], [11,19], 
                 [18,18], [7,17], [5,16], [13,13], [3,8], [6,6], [1,4], 
                 [2,2], [0,0]) */
      /* Perm:  (31, 30, 29, 23, 27, 15, 25, 14, 28, 10,  9, 12, 11, 18,  7,  
                 5, 26, 24, 13, 20, 19, 22, 21,  3, 17,  6, 16,  1,  8,  2,  
                 4,  0) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               23, 27, 15, 25, 14, 28, 10, 9, 
                                               12, 11, 18, 7, 5, 26, 24, 13, 
                                               20, 19, 22, 21, 3, 17, 6, 16, 
                                               1, 8, 2, 4, 0), v7);
      __m512i min8 = _mm512_min_epi16(v7, perm8);
      __m512i max8 = _mm512_max_epi16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x80deaa, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [23,24], [13,22], [21,21], [14,20], [15,19], 
                 [9,18], [11,17], [12,16], [10,10], [7,8], [6,6], [2,5], 
                 [4,4], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 23, 24, 13, 21, 14, 15,  9, 11, 
                 12, 19, 20, 22, 16, 17, 10, 18,  7,  8,  6,  2,  4,  3,  5,  
                 1,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 23, 24, 13, 
                                               21, 14, 15, 9, 11, 12, 19, 20, 
                                               22, 16, 17, 10, 18, 7, 8, 6, 
                                               2, 4, 3, 5, 1, 0), v8);
      __m512i min9 = _mm512_min_epi16(v8, perm9);
      __m512i max9 = _mm512_max_epi16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0x80fa84, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [19,25], [24,24], [23,23], [15,22], [20,21], [14,18], 
                 [13,17], [9,16], [6,12], [10,11], [8,8], [7,7], [5,5], 
                 [2,4], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 19, 24, 23, 15, 20, 21, 25, 14, 13,  
                 9, 22, 18, 17,  6, 10, 11, 16,  8,  7, 12,  5,  2,  3,  4,  
                 1,  0) */
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 19, 24, 23, 15, 
                                                20, 21, 25, 14, 13, 9, 22, 
                                                18, 17, 6, 10, 11, 16, 8, 7, 
                                                12, 5, 2, 3, 4, 1, 0), v9);
      __m512i min10 = _mm512_min_epi16(v9, perm10);
      __m512i max10 = _mm512_max_epi16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x18e644, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [25,26], 
                 [24,24], [19,23], [21,22], [18,20], [15,17], [14,16], 
                 [11,13], [8,12], [9,10], [7,7], [5,6], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 25, 26, 24, 19, 21, 22, 18, 23, 20, 15, 
                 14, 17, 16, 11,  8, 13,  9, 10, 12,  7,  5,  6,  4,  3,  2,  
                 1,  0) */
      __m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 25, 26, 24, 19, 21, 
                                                22, 18, 23, 20, 15, 14, 17, 
                                                16, 11, 8, 13, 9, 10, 12, 7, 
                                                5, 6, 4, 3, 2, 1, 0), v10);
      __m512i min11 = _mm512_min_epi16(v10, perm11);
      __m512i max11 = _mm512_max_epi16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0x22ccb20, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [26,28], [27,27], [24,25], 
                 [22,23], [19,21], [17,20], [15,18], [13,16], [11,14], 
                 [10,12], [8,9], [6,7], [3,5], [4,4], [2,2], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 26, 27, 28, 24, 25, 22, 23, 19, 17, 21, 15, 20, 
                 13, 18, 11, 16, 10, 14, 12,  8,  9,  6,  7,  3,  4,  5,  2,  
                 1,  0) */
      __m512i perm12 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                26, 27, 28, 24, 25, 22, 23, 
                                                19, 17, 21, 15, 20, 13, 18, 
                                                11, 16, 10, 14, 12, 8, 9, 6, 
                                                7, 3, 4, 5, 2, 1, 0), v11);
      __m512i min12 = _mm512_min_epi16(v11, perm12);
      __m512i max12 = _mm512_max_epi16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x54aad48, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [27,28], [25,26], [23,24], 
                 [21,22], [19,20], [17,18], [15,16], [13,14], [11,12], 
                 [9,10], [7,8], [5,6], [3,4], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  2,  
                 1,  0) */
      __m512i perm13 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                27, 28, 25, 26, 23, 24, 21, 
                                                22, 19, 20, 17, 18, 15, 16, 
                                                13, 14, 11, 12, 9, 10, 7, 8, 
                                                5, 6, 3, 4, 2, 1, 0), v12);
      __m512i min13 = _mm512_min_epi16(v12, perm13);
      __m512i max13 = _mm512_max_epi16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0xaaaaaa8, min13);
      
      return v13;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
minimum_29_int16_t(int16_t * const arr) 
                             {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = minimum_29_int16_t_vec(v);
      
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


