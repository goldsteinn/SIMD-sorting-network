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

#define TYPE int8_t
#define N 31
#define SORT_NAME bosenelson_31_int8_t

#ifndef _SIMD_SORT_bosenelson_31_int8_t_H_
#define _SIMD_SORT_bosenelson_31_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 31
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : bosenelson
	Network Depth                    : 31
	SIMD Instructions                : 3 / 155
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
          assert(t.arr[i] == int8_t(0x7f));
 }
}

/* SIMD Sort */
 __m256i __attribute__((const)) 
bosenelson_31_int8_t_vec(__m256i v) {
      
      /* Pairs: ([31,31], [29,30], [27,28], [25,26], [23,24], [21,22], 
                 [19,20], [17,18], [15,16], [13,14], [11,12], [9,10], [7,8], 
                 [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m256i perm0 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 29, 30, 27, 
                                              28, 25, 26, 23, 24, 21, 22, 19, 
                                              20, 17, 18, 15, 16, 13, 14, 11, 
                                              12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 
                                              2, 0), v);
      __m256i min0 = _mm256_min_epi8(v, perm0);
      __m256i max0 = _mm256_max_epi8(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi8(max0, 0x2aaaaaaa, min0);
      
      /* Pairs: ([31,31], [28,30], [27,29], [24,26], [23,25], [20,22], 
                 [19,21], [16,18], [15,17], [12,14], [11,13], [8,10], [7,9], 
                 [4,6], [3,5], [0,2], [1,1]) */
      /* Perm:  (31, 28, 27, 30, 29, 24, 23, 26, 25, 20, 19, 22, 21, 16, 15, 
                 18, 17, 12, 11, 14, 13,  8,  7, 10,  9,  4,  3,  6,  5,  0,  
                 1,  2) */
      __m256i perm1 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 28, 27, 30, 
                                              29, 24, 23, 26, 25, 20, 19, 22, 
                                              21, 16, 15, 18, 17, 12, 11, 14, 
                                              13, 8, 7, 10, 9, 4, 3, 6, 5, 0, 
                                              1, 2), v0);
      __m256i min1 = _mm256_min_epi8(v0, perm1);
      __m256i max1 = _mm256_max_epi8(v0, perm1);
      __m256i v1 = _mm256_mask_mov_epi8(max1, 0x19999999, min1);
      
      /* Pairs: ([31,31], [26,30], [28,29], [23,27], [24,25], [18,22], 
                 [20,21], [15,19], [16,17], [10,14], [12,13], [7,11], [8,9], 
                 [2,6], [4,5], [3,3], [0,1]) */
      /* Perm:  (31, 26, 28, 29, 23, 30, 24, 25, 27, 18, 20, 21, 15, 22, 16, 
                 17, 19, 10, 12, 13,  7, 14,  8,  9, 11,  2,  4,  5,  3,  6,  
                 0,  1) */
      __m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 26, 28, 29, 
                                              23, 30, 24, 25, 27, 18, 20, 21, 
                                              15, 22, 16, 17, 19, 10, 12, 13, 
                                              7, 14, 8, 9, 11, 2, 4, 5, 3, 6, 
                                              0, 1), v1);
      __m256i min2 = _mm256_min_epi8(v1, perm2);
      __m256i max2 = _mm256_max_epi8(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi8(max2, 0x15959595, min2);
      
      /* Pairs: ([31,31], [22,30], [25,29], [24,28], [27,27], [26,26], 
                 [15,23], [17,21], [16,20], [19,19], [18,18], [6,14], [9,13], 
                 [8,12], [11,11], [10,10], [7,7], [1,5], [0,4], [3,3], [2,2]) 
                 */
      /* Perm:  (31, 22, 25, 24, 27, 26, 29, 28, 15, 30, 17, 16, 19, 18, 21, 
                 20, 23,  6,  9,  8, 11, 10, 13, 12,  7, 14,  1,  0,  3,  2,  
                 5,  4) */
      __m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 22, 25, 24, 
                                              27, 26, 29, 28, 15, 30, 17, 16, 
                                              19, 18, 21, 20, 23, 6, 9, 8, 
                                              11, 10, 13, 12, 7, 14, 1, 0, 3, 
                                              2, 5, 4), v2);
      __m256i min3 = _mm256_min_epi8(v2, perm3);
      __m256i max3 = _mm256_max_epi8(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi8(max3, 0x3438343, min3);
      
      /* Pairs: ([31,31], [14,30], [26,29], [28,28], [24,27], [25,25], 
                 [23,23], [22,22], [18,21], [20,20], [16,19], [17,17], 
                 [15,15], [10,13], [12,12], [8,11], [9,9], [7,7], [6,6], 
                 [2,5], [4,4], [0,3], [1,1]) */
      /* Perm:  (31, 14, 26, 28, 24, 29, 25, 27, 23, 22, 18, 20, 16, 21, 17, 
                 19, 15, 30, 10, 12,  8, 13,  9, 11,  7,  6,  2,  4,  0,  5,  
                 1,  3) */
      __m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 14, 26, 28, 
                                              24, 29, 25, 27, 23, 22, 18, 20, 
                                              16, 21, 17, 19, 15, 30, 10, 12, 
                                              8, 13, 9, 11, 7, 6, 2, 4, 0, 5, 
                                              1, 3), v3);
      __m256i min4 = _mm256_min_epi8(v3, perm4);
      __m256i max4 = _mm256_max_epi8(v3, perm4);
      __m256i v4 = _mm256_mask_mov_epi8(max4, 0x5054505, min4);
      
      /* Pairs: ([31,31], [30,30], [21,29], [26,28], [25,27], [16,24], 
                 [23,23], [22,22], [18,20], [17,19], [15,15], [14,14], 
                 [5,13], [10,12], [9,11], [0,8], [7,7], [6,6], [2,4], [1,3]) 
                 */
      /* Perm:  (31, 30, 21, 26, 25, 28, 27, 16, 23, 22, 29, 18, 17, 20, 19, 
                 24, 15, 14,  5, 10,  9, 12, 11,  0,  7,  6, 13,  2,  1,  4,  
                 3,  8) */
      __m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 21, 26, 
                                          25, 28, 27, 16, 23, 22, 29, 18, 17, 
                                          20, 19, 24, 15, 14, 5, 10, 9, 12, 
                                          11, 0, 7, 6, 13, 2, 1, 4, 3, 8));
      __m256i min5 = _mm256_min_epi8(v4, perm5);
      __m256i max5 = _mm256_max_epi8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x6270627, min5);
      
      /* Pairs: ([31,31], [30,30], [22,29], [20,28], [26,27], [17,25], 
                 [24,24], [16,23], [21,21], [18,19], [15,15], [14,14], 
                 [6,13], [4,12], [10,11], [1,9], [8,8], [0,7], [5,5], [2,3]) 
                 */
      /* Perm:  (31, 30, 22, 20, 26, 27, 17, 24, 16, 29, 21, 28, 18, 19, 25, 
                 23, 15, 14,  6,  4, 10, 11,  1,  8,  0, 13,  5, 12,  2,  3,  
                 9,  7) */
      __m256i perm6 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(31, 30, 22, 20, 
                                          26, 27, 17, 24, 16, 29, 21, 28, 18, 
                                          19, 25, 23, 15, 14, 6, 4, 10, 11, 
                                          1, 8, 0, 13, 5, 12, 2, 3, 9, 7));
      __m256i min6 = _mm256_min_epi8(v5, perm6);
      __m256i max6 = _mm256_max_epi8(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi8(max6, 0x4570457, min6);
      
      /* Pairs: ([31,31], [30,30], [13,29], [22,28], [19,27], [18,26], 
                 [25,25], [24,24], [17,23], [21,21], [20,20], [0,16], 
                 [15,15], [14,14], [6,12], [3,11], [2,10], [9,9], [8,8], 
                 [1,7], [5,5], [4,4]) */
      /* Perm:  (31, 30, 13, 22, 19, 18, 25, 24, 17, 28, 21, 20, 27, 26, 23,  
                 0, 15, 14, 29,  6,  3,  2,  9,  8,  1, 12,  5,  4, 11, 10,  
                 7, 16) */
      __m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 13, 22, 
                                              19, 18, 25, 24, 17, 28, 21, 20, 
                                              27, 26, 23, 0, 15, 14, 29, 6, 
                                              3, 2, 9, 8, 1, 12, 5, 4, 11, 
                                              10, 7, 16), v6);
      __m256i min7 = _mm256_min_epi8(v6, perm7);
      __m256i max7 = _mm256_max_epi8(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi8(max7, 0x4e204f, min7);
      
      /* Pairs: ([31,31], [30,30], [14,29], [12,28], [20,27], [26,26], 
                 [18,25], [24,24], [23,23], [22,22], [21,21], [19,19], 
                 [1,17], [16,16], [0,15], [13,13], [4,11], [10,10], [2,9], 
                 [8,8], [7,7], [6,6], [5,5], [3,3]) */
      /* Perm:  (31, 30, 14, 12, 20, 26, 18, 24, 23, 22, 21, 27, 19, 25,  1, 
                 16,  0, 29, 13, 28,  4, 10,  2,  8,  7,  6,  5, 11,  3,  9, 
                 17, 15) */
      __m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 14, 12, 
                                              20, 26, 18, 24, 23, 22, 21, 27, 
                                              19, 25, 1, 16, 0, 29, 13, 28, 
                                              4, 10, 2, 8, 7, 6, 5, 11, 3, 9, 
                                              17, 15), v7);
      __m256i min8 = _mm256_min_epi8(v7, perm8);
      __m256i max8 = _mm256_max_epi8(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi8(max8, 0x145017, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [14,28], [21,27], [26,26], 
                 [25,25], [18,24], [23,23], [22,22], [20,20], [19,19], 
                 [17,17], [16,16], [1,15], [13,13], [12,12], [5,11], [10,10], 
                 [9,9], [2,8], [7,7], [6,6], [4,4], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 14, 21, 26, 25, 18, 23, 22, 27, 20, 19, 24, 17, 
                 16,  1, 28, 13, 12,  5, 10,  9,  2,  7,  6, 11,  4,  3,  8, 
                 15,  0) */
      __m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 14, 
                                              21, 26, 25, 18, 23, 22, 27, 20, 
                                              19, 24, 17, 16, 1, 28, 13, 12, 
                                              5, 10, 9, 2, 7, 6, 11, 4, 3, 8, 
                                              15, 0), v8);
      __m256i min9 = _mm256_min_epi8(v8, perm9);
      __m256i max9 = _mm256_max_epi8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0x244026, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [22,27], [26,26], 
                 [21,25], [20,24], [18,23], [19,19], [17,17], [16,16], 
                 [15,15], [14,14], [13,13], [12,12], [6,11], [10,10], [5,9], 
                 [4,8], [2,7], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 22, 26, 21, 20, 18, 27, 25, 24, 19, 23, 17, 
                 16, 15, 14, 13, 12,  6, 10,  5,  4,  2, 11,  9,  8,  3,  7,  
                 1,  0) */
      __m256i perm10 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(31, 30, 29, 
                                           28, 22, 26, 21, 20, 18, 27, 25, 
                                           24, 19, 23, 17, 16, 15, 14, 13, 
                                           12, 6, 10, 5, 4, 2, 11, 9, 8, 3, 
                                           7, 1, 0));
      __m256i min10 = _mm256_min_epi8(v9, perm10);
      __m256i max10 = _mm256_max_epi8(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi8(max10, 0x740074, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [11,27], [22,26], 
                 [25,25], [24,24], [19,23], [21,21], [20,20], [2,18], 
                 [17,17], [16,16], [15,15], [14,14], [13,13], [12,12], 
                 [6,10], [9,9], [8,8], [3,7], [5,5], [4,4], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 11, 22, 25, 24, 19, 26, 21, 20, 23,  2, 17, 
                 16, 15, 14, 13, 12, 27,  6,  9,  8,  3, 10,  5,  4,  7, 18,  
                 1,  0) */
      __m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 11, 22, 25, 24, 19, 26, 
                                               21, 20, 23, 2, 17, 16, 15, 14, 
                                               13, 12, 27, 6, 9, 8, 3, 10, 5, 
                                               4, 7, 18, 1, 0), v10);
      __m256i min11 = _mm256_min_epi8(v10, perm11);
      __m256i max11 = _mm256_max_epi8(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi8(max11, 0x48084c, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [12,27], [10,26], 
                 [22,25], [24,24], [20,23], [21,21], [3,19], [18,18], [2,17], 
                 [16,16], [15,15], [14,14], [13,13], [11,11], [6,9], [8,8], 
                 [4,7], [5,5], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 12, 10, 22, 24, 20, 25, 21, 23,  3, 18,  2, 
                 16, 15, 14, 13, 27, 11, 26,  6,  8,  4,  9,  5,  7, 19, 17,  
                 1,  0) */
      __m256i perm12 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 12, 10, 22, 24, 20, 25, 
                                               21, 23, 3, 18, 2, 16, 15, 14, 
                                               13, 27, 11, 26, 6, 8, 4, 9, 5, 
                                               7, 19, 17, 1, 0), v11);
      __m256i min12 = _mm256_min_epi8(v11, perm12);
      __m256i max12 = _mm256_max_epi8(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi8(max12, 0x50145c, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [13,27], [26,26], 
                 [9,25], [22,24], [21,23], [4,20], [19,19], [18,18], [17,17], 
                 [2,16], [15,15], [14,14], [12,12], [11,11], [10,10], [6,8], 
                 [5,7], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 13, 26,  9, 22, 21, 24, 23,  4, 19, 18, 17,  
                 2, 15, 14, 27, 12, 11, 10, 25,  6,  5,  8,  7, 20,  3, 16,  
                 1,  0) */
      __m256i perm13 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 13, 26, 9, 22, 21, 24, 23, 
                                               4, 19, 18, 17, 2, 15, 14, 27, 
                                               12, 11, 10, 25, 6, 5, 8, 7, 
                                               20, 3, 16, 1, 0), v12);
      __m256i min13 = _mm256_min_epi8(v12, perm13);
      __m256i max13 = _mm256_max_epi8(v12, perm13);
      __m256i v13 = _mm256_mask_mov_epi8(max13, 0x602274, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [14,27], [26,26], 
                 [10,25], [8,24], [22,23], [5,21], [20,20], [4,19], [18,18], 
                 [17,17], [16,16], [2,15], [13,13], [12,12], [11,11], [9,9], 
                 [6,7], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 14, 26, 10,  8, 22, 23,  5, 20,  4, 18, 17, 
                 16,  2, 27, 13, 12, 11, 25,  9, 24,  6,  7, 21, 19,  3, 15,  
                 1,  0) */
      __m256i perm14 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 14, 26, 10, 8, 22, 23, 5, 
                                               20, 4, 18, 17, 16, 2, 27, 13, 
                                               12, 11, 25, 9, 24, 6, 7, 21, 
                                               19, 3, 15, 1, 0), v13);
      __m256i min14 = _mm256_min_epi8(v13, perm14);
      __m256i max14 = _mm256_max_epi8(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi8(max14, 0x404574, min14);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [14,26], 
                 [13,25], [10,24], [7,23], [6,22], [21,21], [20,20], [5,19], 
                 [18,18], [17,17], [4,16], [3,15], [12,12], [11,11], [9,9], 
                 [8,8], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 14, 13, 10,  7,  6, 21, 20,  5, 18, 17,  
                 4,  3, 26, 25, 12, 11, 24,  9,  8, 23, 22, 19, 16, 15,  2,  
                 1,  0) */
      __m256i perm15 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 14, 13, 10, 7, 6, 21, 
                                               20, 5, 18, 17, 4, 3, 26, 25, 
                                               12, 11, 24, 9, 8, 23, 22, 19, 
                                               16, 15, 2, 1, 0), v14);
      __m256i min15 = _mm256_min_epi8(v14, perm15);
      __m256i max15 = _mm256_max_epi8(v14, perm15);
      __m256i v15 = _mm256_mask_mov_epi8(max15, 0x64f8, min15);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [14,25], [12,24], [8,23], [22,22], [6,21], [20,20], [19,19], 
                 [18,18], [5,17], [16,16], [4,15], [13,13], [11,11], [10,10], 
                 [9,9], [7,7], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 14, 12,  8, 22,  6, 20, 19, 18,  5, 
                 16,  4, 25, 13, 24, 11, 10,  9, 23,  7, 21, 17, 15,  3,  2,  
                 1,  0) */
      __m256i perm16 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 14, 12, 8, 22, 6, 
                                               20, 19, 18, 5, 16, 4, 25, 13, 
                                               24, 11, 10, 9, 23, 7, 21, 17, 
                                               15, 3, 2, 1, 0), v15);
      __m256i min16 = _mm256_min_epi8(v15, perm16);
      __m256i max16 = _mm256_max_epi8(v15, perm16);
      __m256i v16 = _mm256_mask_mov_epi8(max16, 0x5170, min16);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [14,24], [9,23], [22,22], [21,21], [6,20], [19,19], 
                 [18,18], [17,17], [16,16], [5,15], [13,13], [12,12], 
                 [11,11], [10,10], [8,8], [7,7], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 14,  9, 22, 21,  6, 19, 18, 17, 
                 16,  5, 24, 13, 12, 11, 10, 23,  8,  7, 20, 15,  4,  3,  2,  
                 1,  0) */
      __m256i perm17 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 14, 9, 22, 21, 
                                               6, 19, 18, 17, 16, 5, 24, 13, 
                                               12, 11, 10, 23, 8, 7, 20, 15, 
                                               4, 3, 2, 1, 0), v16);
      __m256i min17 = _mm256_min_epi8(v16, perm17);
      __m256i max17 = _mm256_max_epi8(v16, perm17);
      __m256i v17 = _mm256_mask_mov_epi8(max17, 0x4260, min17);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [10,23], [22,22], [21,21], [20,20], 
                 [6,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [11,11], [9,9], [8,8], [7,7], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 10, 22, 21, 20,  6, 18, 17, 
                 16, 15, 14, 13, 12, 11, 23,  9,  8,  7, 19,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm18 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 10, 22, 
                                               21, 20, 6, 18, 17, 16, 15, 14, 
                                               13, 12, 11, 23, 9, 8, 7, 19, 
                                               5, 4, 3, 2, 1, 0), v17);
      __m256i min18 = _mm256_min_epi8(v17, perm18);
      __m256i max18 = _mm256_max_epi8(v17, perm18);
      __m256i v18 = _mm256_mask_mov_epi8(max18, 0x440, min18);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [11,23], [22,22], [21,21], [20,20], 
                 [19,19], [6,18], [17,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [10,10], [9,9], [8,8], [7,7], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 11, 22, 21, 20, 19,  6, 17, 
                 16, 15, 14, 13, 12, 23, 10,  9,  8,  7, 18,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm19 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 11, 22, 
                                               21, 20, 19, 6, 17, 16, 15, 14, 
                                               13, 12, 23, 10, 9, 8, 7, 18, 
                                               5, 4, 3, 2, 1, 0), v18);
      __m256i min19 = _mm256_min_epi8(v18, perm19);
      __m256i max19 = _mm256_max_epi8(v18, perm19);
      __m256i v19 = _mm256_mask_mov_epi8(max19, 0x840, min19);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [12,23], [22,22], [21,21], [20,20], 
                 [11,19], [10,18], [6,17], [16,16], [15,15], [14,14], 
                 [13,13], [9,9], [8,8], [7,7], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 12, 22, 21, 20, 11, 10,  6, 
                 16, 15, 14, 13, 23, 19, 18,  9,  8,  7, 17,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm20 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 12, 22, 
                                               21, 20, 11, 10, 6, 16, 15, 14, 
                                               13, 23, 19, 18, 9, 8, 7, 17, 
                                               5, 4, 3, 2, 1, 0), v19);
      __m256i min20 = _mm256_min_epi8(v19, perm20);
      __m256i max20 = _mm256_max_epi8(v19, perm20);
      __m256i v20 = _mm256_mask_mov_epi8(max20, 0x1c40, min20);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [13,23], [22,22], [21,21], [12,20], 
                 [19,19], [18,18], [9,17], [6,16], [15,15], [14,14], [11,11], 
                 [10,10], [8,8], [7,7], [5,5], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 13, 22, 21, 12, 19, 18,  9,  
                 6, 15, 14, 23, 20, 11, 10, 17,  8,  7, 16,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm21 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 13, 22, 
                                               21, 12, 19, 18, 9, 6, 15, 14, 
                                               23, 20, 11, 10, 17, 8, 7, 16, 
                                               5, 4, 3, 2, 1, 0), v20);
      __m256i min21 = _mm256_min_epi8(v20, perm21);
      __m256i max21 = _mm256_max_epi8(v20, perm21);
      __m256i v21 = _mm256_mask_mov_epi8(max21, 0x3240, min21);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [14,23], [22,22], [13,21], [20,20], 
                 [12,19], [18,18], [10,17], [8,16], [6,15], [11,11], [9,9], 
                 [7,7], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 14, 22, 13, 20, 12, 18, 10,  
                 8,  6, 23, 21, 19, 11, 17,  9, 16,  7, 15,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm22 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 14, 22, 
                                               13, 20, 12, 18, 10, 8, 6, 23, 
                                               21, 19, 11, 17, 9, 16, 7, 15, 
                                               5, 4, 3, 2, 1, 0), v21);
      __m256i min22 = _mm256_min_epi8(v21, perm22);
      __m256i max22 = _mm256_max_epi8(v21, perm22);
      __m256i v22 = _mm256_mask_mov_epi8(max22, 0x7540, min22);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [14,22], [21,21], [20,20], 
                 [13,19], [18,18], [17,17], [10,16], [7,15], [12,12], 
                 [11,11], [9,9], [8,8], [6,6], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 14, 21, 20, 13, 18, 17, 
                 10,  7, 22, 19, 12, 11, 16,  9,  8, 15,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm23 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 14, 
                                               21, 20, 13, 18, 17, 10, 7, 22, 
                                               19, 12, 11, 16, 9, 8, 15, 6, 
                                               5, 4, 3, 2, 1, 0), v22);
      __m256i min23 = _mm256_min_epi8(v22, perm23);
      __m256i max23 = _mm256_max_epi8(v22, perm23);
      __m256i v23 = _mm256_mask_mov_epi8(max23, 0x6480, min23);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [14,21], [20,20], 
                 [19,19], [18,18], [13,17], [12,16], [8,15], [11,11], 
                 [10,10], [9,9], [7,7], [6,6], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 14, 20, 19, 18, 13, 
                 12,  8, 21, 17, 16, 11, 10,  9, 15,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm24 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               14, 20, 19, 18, 13, 12, 8, 21, 
                                               17, 16, 11, 10, 9, 15, 7, 6, 
                                               5, 4, 3, 2, 1, 0), v23);
      __m256i min24 = _mm256_min_epi8(v23, perm24);
      __m256i max24 = _mm256_max_epi8(v23, perm24);
      __m256i v24 = _mm256_mask_mov_epi8(max24, 0x7100, min24);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [14,20], 
                 [19,19], [18,18], [17,17], [16,16], [9,15], [13,13], 
                 [12,12], [11,11], [10,10], [8,8], [7,7], [6,6], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 14, 19, 18, 17, 
                 16,  9, 20, 13, 12, 11, 10, 15,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm25 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 14, 19, 18, 17, 16, 9, 20, 
                                               13, 12, 11, 10, 15, 8, 7, 6, 
                                               5, 4, 3, 2, 1, 0), v24);
      __m256i min25 = _mm256_min_epi8(v24, perm25);
      __m256i max25 = _mm256_max_epi8(v24, perm25);
      __m256i v25 = _mm256_mask_mov_epi8(max25, 0x4200, min25);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [14,19], [18,18], [17,17], [16,16], [10,15], [13,13], 
                 [12,12], [11,11], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 14, 18, 17, 
                 16, 10, 19, 13, 12, 11, 15,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm26 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 14, 18, 17, 16, 10, 
                                               19, 13, 12, 11, 15, 9, 8, 7, 
                                               6, 5, 4, 3, 2, 1, 0), v25);
      __m256i min26 = _mm256_min_epi8(v25, perm26);
      __m256i max26 = _mm256_max_epi8(v25, perm26);
      __m256i v26 = _mm256_mask_mov_epi8(max26, 0x4400, min26);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [14,18], [17,17], [16,16], [11,15], [13,13], 
                 [12,12], [10,10], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 14, 17, 
                 16, 11, 18, 13, 12, 15, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm27 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 14, 17, 16, 11, 
                                               18, 13, 12, 15, 10, 9, 8, 7, 
                                               6, 5, 4, 3, 2, 1, 0), v26);
      __m256i min27 = _mm256_min_epi8(v26, perm27);
      __m256i max27 = _mm256_max_epi8(v26, perm27);
      __m256i v27 = _mm256_mask_mov_epi8(max27, 0x4800, min27);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [14,17], [16,16], [12,15], [13,13], 
                 [11,11], [10,10], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 14, 
                 16, 12, 17, 13, 15, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm28 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 14, 16, 12, 
                                               17, 13, 15, 11, 10, 9, 8, 7, 
                                               6, 5, 4, 3, 2, 1, 0), v27);
      __m256i min28 = _mm256_min_epi8(v27, perm28);
      __m256i max28 = _mm256_max_epi8(v27, perm28);
      __m256i v28 = _mm256_mask_mov_epi8(max28, 0x5000, min28);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [14,16], [13,15], [12,12], 
                 [11,11], [10,10], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 14, 13, 16, 15, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm29 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 14, 13, 
                                               16, 15, 12, 11, 10, 9, 8, 7, 
                                               6, 5, 4, 3, 2, 1, 0), v28);
      __m256i min29 = _mm256_min_epi8(v28, perm29);
      __m256i max29 = _mm256_max_epi8(v28, perm29);
      __m256i v29 = _mm256_mask_mov_epi8(max29, 0x6000, min29);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [14,15], [13,13], 
                 [12,12], [11,11], [10,10], [9,9], [8,8], [7,7], [6,6], 
                 [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 14, 15, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m256i perm30 = _mm256_shuffle_epi8(v29, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 14, 15, 13, 
                                           12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m256i min30 = _mm256_min_epi8(v29, perm30);
      __m256i max30 = _mm256_max_epi8(v29, perm30);
      __m256i v30 = _mm256_mask_mov_epi8(max30, 0x4000, min30);
      
      return v30;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bosenelson_31_int8_t(int8_t * const 
                             arr) {
      
      __m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
      __m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x7fffffff, arr);
      fill_works(v);
      v = bosenelson_31_int8_t_vec(v);
      
      fill_works(v);_mm256_mask_storeu_epi8((void *)arr, 0x7fffffff, v);
      
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


