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
#define N 17
#define SORT_NAME bosenelson_17_uint8_t

#ifndef _SIMD_SORT_bosenelson_17_uint8_t_H_
#define _SIMD_SORT_bosenelson_17_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 17
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : bosenelson
	Network Depth                    : 20
	SIMD Instructions                : 3 / 100
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX512vbmi, AVX, AVX2
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
bosenelson_17_uint8_t_vec(__m256i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [15,16], [14,14], [12,13], 
                 [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 15, 16, 14, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm0 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 15, 16, 14, 12, 13, 
                                              10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 
                                              0, 1), v);
      __m256i min0 = _mm256_min_epu8(v, perm0);
      __m256i max0 = _mm256_max_epu8(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi8(max0, 0x9555, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [14,16], [15,15], [13,13], 
                 [12,12], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 14, 15, 16, 13, 12,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m256i perm1 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 14, 15, 16, 13, 12, 
                                              9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 
                                              3, 2), v0);
      __m256i min1 = _mm256_min_epu8(v0, perm1);
      __m256i max1 = _mm256_max_epu8(v0, perm1);
      __m256i v1 = _mm256_mask_mov_epi8(max1, 0x4333, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [13,16], [14,15], [12,12], 
                 [11,11], [9,10], [8,8], [3,7], [5,6], [0,4], [1,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 13, 14, 15, 16, 12, 11,  9, 10,  8,  3,  5,  6,  0,  7,  1,  
                 2,  4) */
      __m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 13, 14, 15, 16, 12, 
                                              11, 9, 10, 8, 3, 5, 6, 0, 7, 1, 
                                              2, 4), v1);
      __m256i min2 = _mm256_min_epu8(v1, perm2);
      __m256i max2 = _mm256_max_epu8(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi8(max2, 0x622b, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [11,16], [12,15], [14,14], 
                 [13,13], [10,10], [9,9], [8,8], [7,7], [2,6], [1,5], [4,4], 
                 [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 11, 12, 14, 13, 15, 16, 10,  9,  8,  7,  2,  1,  4,  3,  6,  
                 5,  0) */
      __m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 11, 12, 14, 13, 15, 
                                              16, 10, 9, 8, 7, 2, 1, 4, 3, 6, 
                                              5, 0), v2);
      __m256i min3 = _mm256_min_epu8(v2, perm3);
      __m256i max3 = _mm256_max_epu8(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi8(max3, 0x1806, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [7,16], [13,15], [12,14], 
                 [11,11], [10,10], [9,9], [8,8], [3,6], [5,5], [1,4], [2,2], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  
                 7, 13, 12, 15, 14, 11, 10,  9,  8, 16,  3,  5,  1,  6,  2,  
                 4,  0) */
      __m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 7, 13, 12, 15, 14, 
                                              11, 10, 9, 8, 16, 3, 5, 1, 6, 
                                              2, 4, 0), v3);
      __m256i min4 = _mm256_min_epu8(v3, perm4);
      __m256i max4 = _mm256_max_epu8(v3, perm4);
      __m256i v4 = _mm256_mask_mov_epi8(max4, 0x308a, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [10,15], [13,14], 
                 [12,12], [11,11], [9,9], [8,8], [7,7], [6,6], [3,5], [2,4], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 10, 13, 14, 12, 11, 15,  9,  8,  7,  6,  3,  2,  5,  4,  
                 1,  0) */
      __m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 10, 13, 14, 12, 11, 15, 
                                          9, 8, 7, 6, 3, 2, 5, 4, 1, 0));
      __m256i min5 = _mm256_min_epu8(v4, perm5);
      __m256i max5 = _mm256_max_epu8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x240c, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [11,15], [9,14], [8,13], 
                 [12,12], [10,10], [7,7], [6,6], [5,5], [3,4], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 11,  9,  8, 12, 15, 10, 14, 13,  7,  6,  5,  3,  4,  2,  
                 1,  0) */
      __m256i perm6 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 11, 9, 8, 12, 15, 10, 
                                          14, 13, 7, 6, 5, 3, 4, 2, 1, 0));
      __m256i min6 = _mm256_min_epu8(v5, perm6);
      __m256i max6 = _mm256_max_epu8(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi8(max6, 0xb08, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [6,15], [11,14], [9,13], 
                 [8,12], [10,10], [7,7], [5,5], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16,  6, 11,  9,  8, 14, 10, 13, 12,  7, 15,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm7 = _mm256_shuffle_epi8(v6, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 6, 11, 9, 8, 14, 10, 
                                          13, 12, 7, 15, 5, 4, 3, 2, 1, 0));
      __m256i min7 = _mm256_min_epu8(v6, perm7);
      __m256i max7 = _mm256_max_epu8(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi8(max7, 0xb40, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [7,15], [5,14], [10,13], 
                 [9,12], [11,11], [8,8], [6,6], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16,  7,  5, 10,  9, 11, 13, 12,  8, 15,  6, 14,  4,  3,  2,  
                 1,  0) */
      __m256i perm8 = _mm256_shuffle_epi8(v7, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 7, 5, 10, 9, 11, 13, 
                                          12, 8, 15, 6, 14, 4, 3, 2, 1, 0));
      __m256i min8 = _mm256_min_epu8(v7, perm8);
      __m256i max8 = _mm256_max_epu8(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi8(max8, 0x6a0, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [7,14], 
                 [11,13], [10,12], [0,9], [8,8], [6,6], [5,5], [4,4], [3,3], 
                 [2,2], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15,  7, 11, 10, 13, 12,  0,  8, 14,  6,  5,  4,  3,  2,  
                 1,  9) */
      __m256i perm9 = _mm256_shuffle_epi8(v8, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 15, 7, 11, 10, 13, 12, 
                                          0, 8, 14, 6, 5, 4, 3, 2, 1, 9));
      __m256i min9 = _mm256_min_epu8(v8, perm9);
      __m256i max9 = _mm256_max_epu8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0xc81, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [4,13], [11,12], [1,10], [9,9], [0,8], [7,7], [6,6], [5,5], 
                 [3,3], [2,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14,  4, 11, 12,  1,  9,  0,  7,  6,  5, 13,  3,  2, 
                 10,  8) */
      __m256i perm10 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 4, 11, 
                                           12, 1, 9, 0, 7, 6, 5, 13, 3, 2, 
                                           10, 8));
      __m256i min10 = _mm256_min_epu8(v9, perm10);
      __m256i max10 = _mm256_max_epu8(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi8(max10, 0x813, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [5,13], [3,12], [2,11], [10,10], [1,9], [8,8], [7,7], [6,6], 
                 [4,4], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14,  5,  3,  2, 10,  1,  8,  7,  6, 13,  4, 12, 11,  
                 9,  0) */
      __m256i perm11 = _mm256_shuffle_epi8(v10, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 5, 3, 
                                           2, 10, 1, 8, 7, 6, 13, 4, 12, 11, 
                                           9, 0));
      __m256i min11 = _mm256_min_epu8(v10, perm11);
      __m256i max11 = _mm256_max_epu8(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi8(max11, 0x2e, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [6,13], [12,12], [3,11], [10,10], [2,9], [1,8], [7,7], 
                 [5,5], [4,4], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14,  6, 12,  3, 10,  2,  1,  7, 13,  5,  4, 11,  9,  
                 8,  0) */
      __m256i perm12 = _mm256_shuffle_epi8(v11, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 6, 12, 
                                           3, 10, 2, 1, 7, 13, 5, 4, 11, 9, 
                                           8, 0));
      __m256i min12 = _mm256_min_epu8(v11, perm12);
      __m256i max12 = _mm256_max_epu8(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi8(max12, 0x4e, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [7,13], [12,12], [6,11], [3,10], [9,9], [2,8], [5,5], [4,4], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14,  7, 12,  6,  3,  9,  2, 13, 11,  5,  4, 10,  8,  
                 1,  0) */
      __m256i perm13 = _mm256_shuffle_epi8(v12, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 7, 12, 
                                           6, 3, 9, 2, 13, 11, 5, 4, 10, 8, 
                                           1, 0));
      __m256i min13 = _mm256_min_epu8(v12, perm13);
      __m256i max13 = _mm256_max_epu8(v12, perm13);
      __m256i v13 = _mm256_mask_mov_epi8(max13, 0xcc, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [7,12], [11,11], [5,10], [3,9], [8,8], [6,6], 
                 [4,4], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13,  7, 11,  5,  3,  8, 12,  6, 10,  4,  9,  2,  
                 1,  0) */
      __m256i perm14 = _mm256_shuffle_epi8(v13, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 7, 
                                           11, 5, 3, 8, 12, 6, 10, 4, 9, 2, 
                                           1, 0));
      __m256i min14 = _mm256_min_epu8(v13, perm14);
      __m256i max14 = _mm256_max_epu8(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi8(max14, 0xa8, min14);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [7,11], [10,10], [4,9], [3,8], [6,6], 
                 [5,5], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12,  7, 10,  4,  3, 11,  6,  5,  9,  8,  2,  
                 1,  0) */
      __m256i perm15 = _mm256_shuffle_epi8(v14, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 7, 10, 4, 3, 11, 6, 5, 9, 8, 
                                           2, 1, 0));
      __m256i min15 = _mm256_min_epu8(v14, perm15);
      __m256i max15 = _mm256_max_epu8(v14, perm15);
      __m256i v15 = _mm256_mask_mov_epi8(max15, 0x98, min15);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [11,11], [7,10], [5,9], [4,8], [6,6], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 11,  7,  5,  4, 10,  6,  9,  8,  3,  2,  
                 1,  0) */
      __m256i perm16 = _mm256_shuffle_epi8(v15, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 7, 5, 4, 10, 6, 9, 8, 3, 
                                           2, 1, 0));
      __m256i min16 = _mm256_min_epu8(v15, perm16);
      __m256i max16 = _mm256_max_epu8(v15, perm16);
      __m256i v16 = _mm256_mask_mov_epi8(max16, 0xb0, min16);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [11,11], [10,10], [6,9], [5,8], [7,7], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 11, 10,  6,  5,  7,  9,  8,  4,  3,  2,  
                 1,  0) */
      __m256i perm17 = _mm256_shuffle_epi8(v16, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 6, 5, 7, 9, 8, 4, 3, 
                                           2, 1, 0));
      __m256i min17 = _mm256_min_epu8(v16, perm17);
      __m256i max17 = _mm256_max_epu8(v16, perm17);
      __m256i v17 = _mm256_mask_mov_epi8(max17, 0x60, min17);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [11,11], [10,10], [7,9], [6,8], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 11, 10,  7,  6,  9,  8,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm18 = _mm256_shuffle_epi8(v17, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 7, 6, 9, 8, 5, 4, 3, 
                                           2, 1, 0));
      __m256i min18 = _mm256_min_epu8(v17, perm18);
      __m256i max18 = _mm256_max_epu8(v17, perm18);
      __m256i v18 = _mm256_mask_mov_epi8(max18, 0xc0, min18);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [11,11], [10,10], [9,9], [7,8], [6,6], 
                 [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 11, 10,  9,  7,  8,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm19 = _mm256_shuffle_epi8(v18, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 9, 7, 8, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m256i min19 = _mm256_min_epu8(v18, perm19);
      __m256i max19 = _mm256_max_epu8(v18, perm19);
      __m256i v19 = _mm256_mask_mov_epi8(max19, 0x80, min19);
      
      return v19;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bosenelson_17_uint8_t(uint8_t * const 
                             arr) {
      
      __m256i _tmp0 = _mm256_set1_epi8(uint8_t(0xff));
      __m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x1ffff, arr);
      fill_works(v);
      v = bosenelson_17_uint8_t_vec(v);
      
      fill_works(v);_mm256_mask_storeu_epi8((void *)arr, 0x1ffff, v);
      
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


