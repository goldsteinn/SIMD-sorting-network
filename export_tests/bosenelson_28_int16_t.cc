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
#define N 28
#define SORT_NAME bosenelson_28_int16_t

#ifndef _SIMD_SORT_bosenelson_28_int16_t_H_
#define _SIMD_SORT_bosenelson_28_int16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 28
	Underlying Sort Type             : int16_t
	Network Generation Algorithm     : bosenelson
	Network Depth                    : 28
	SIMD Instructions                : 2 / 140
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
bosenelson_28_int16_t_vec(__m512i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [26,27], [24,25], 
                 [22,23], [21,21], [19,20], [17,18], [15,16], [14,14], 
                 [12,13], [10,11], [8,9], [7,7], [5,6], [3,4], [1,2], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 26, 27, 24, 25, 22, 23, 21, 19, 20, 17, 18, 
                 15, 16, 14, 12, 13, 10, 11,  8,  9,  7,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m512i perm0 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 26, 27, 24, 25, 22, 23, 
                                               21, 19, 20, 17, 18, 15, 16, 
                                               14, 12, 13, 10, 11, 8, 9, 7, 
                                               5, 6, 3, 4, 1, 2, 0), v);
      __m512i min0 = _mm512_min_epi16(v, perm0);
      __m512i max0 = _mm512_max_epi16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x54a952a, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [25,27], [24,26], 
                 [21,23], [22,22], [18,20], [17,19], [14,16], [15,15], 
                 [11,13], [10,12], [7,9], [8,8], [4,6], [3,5], [0,2], [1,1]) 
                 */
      /* Perm:  (31, 30, 29, 28, 25, 24, 27, 26, 21, 22, 23, 18, 17, 20, 19, 
                 14, 15, 16, 11, 10, 13, 12,  7,  8,  9,  4,  3,  6,  5,  0,  
                 1,  2) */
      __m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 25, 24, 27, 26, 21, 22, 
                                               23, 18, 17, 20, 19, 14, 15, 
                                               16, 11, 10, 13, 12, 7, 8, 9, 
                                               4, 3, 6, 5, 0, 1, 2), v0);
      __m512i min1 = _mm512_min_epi16(v0, perm1);
      __m512i max1 = _mm512_max_epi16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x3264c99, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [23,27], [25,26], 
                 [24,24], [21,22], [16,20], [18,19], [17,17], [14,15], 
                 [9,13], [11,12], [10,10], [7,8], [2,6], [4,5], [3,3], [0,1]) 
                 */
      /* Perm:  (31, 30, 29, 28, 23, 25, 26, 24, 27, 21, 22, 16, 18, 19, 17, 
                 20, 14, 15,  9, 11, 12, 10, 13,  7,  8,  2,  4,  5,  3,  6,  
                 0,  1) */
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 23, 25, 26, 24, 27, 21, 
                                               22, 16, 18, 19, 17, 20, 14, 
                                               15, 9, 11, 12, 10, 13, 7, 8, 
                                               2, 4, 5, 3, 6, 0, 1), v1);
      __m512i min2 = _mm512_min_epi16(v1, perm2);
      __m512i max2 = _mm512_max_epi16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0x2a54a95, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [20,27], [22,26], 
                 [21,25], [24,24], [23,23], [15,19], [14,18], [17,17], 
                 [16,16], [6,13], [8,12], [7,11], [10,10], [9,9], [1,5], 
                 [0,4], [3,3], [2,2]) */
      /* Perm:  (31, 30, 29, 28, 20, 22, 21, 24, 23, 26, 25, 27, 15, 14, 17, 
                 16, 19, 18,  6,  8,  7, 10,  9, 12, 11, 13,  1,  0,  3,  2,  
                 5,  4) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 20, 22, 21, 24, 23, 26, 
                                               25, 27, 15, 14, 17, 16, 19, 
                                               18, 6, 8, 7, 10, 9, 12, 11, 
                                               13, 1, 0, 3, 2, 5, 4), v2);
      __m512i min3 = _mm512_min_epi16(v2, perm3);
      __m512i max3 = _mm512_max_epi16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x70c1c3, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [13,27], [23,26], 
                 [25,25], [21,24], [22,22], [20,20], [16,19], [18,18], 
                 [14,17], [15,15], [9,12], [11,11], [7,10], [8,8], [6,6], 
                 [2,5], [4,4], [0,3], [1,1]) */
      /* Perm:  (31, 30, 29, 28, 13, 23, 25, 21, 26, 22, 24, 20, 16, 18, 14, 
                 19, 15, 17, 27,  9, 11,  7, 12,  8, 10,  6,  2,  4,  0,  5,  
                 1,  3) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 13, 23, 25, 21, 26, 22, 
                                               24, 20, 16, 18, 14, 19, 15, 
                                               17, 27, 9, 11, 7, 12, 8, 10, 
                                               6, 2, 4, 0, 5, 1, 3), v3);
      __m512i min4 = _mm512_min_epi16(v3, perm4);
      __m512i max4 = _mm512_max_epi16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0xa16285, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [19,26], 
                 [23,25], [22,24], [14,21], [20,20], [16,18], [15,17], 
                 [13,13], [5,12], [9,11], [8,10], [0,7], [6,6], [2,4], [1,3]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 19, 23, 22, 25, 24, 14, 20, 26, 16, 15, 
                 18, 17, 21, 13,  5,  9,  8, 11, 10,  0,  6, 12,  2,  1,  4,  
                 3,  7) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 19, 23, 22, 25, 24, 
                                               14, 20, 26, 16, 15, 18, 17, 
                                               21, 13, 5, 9, 8, 11, 10, 0, 6, 
                                               12, 2, 1, 4, 3, 7), v4);
      __m512i min5 = _mm512_min_epi16(v4, perm5);
      __m512i max5 = _mm512_max_epi16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0xc9c327, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [20,26], 
                 [18,25], [23,24], [15,22], [21,21], [19,19], [16,17], 
                 [0,14], [13,13], [6,12], [4,11], [9,10], [1,8], [7,7], 
                 [5,5], [2,3]) */
      /* Perm:  (31, 30, 29, 28, 27, 20, 18, 23, 24, 15, 21, 26, 19, 25, 16, 
                 17, 22,  0, 13,  6,  4,  9, 10,  1,  7, 12,  5, 11,  2,  3,  
                 8, 14) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 20, 18, 23, 24, 15, 
                                               21, 26, 19, 25, 16, 17, 22, 0, 
                                               13, 6, 4, 9, 10, 1, 7, 12, 5, 
                                               11, 2, 3, 8, 14), v5);
      __m512i min6 = _mm512_min_epi16(v5, perm6);
      __m512i max6 = _mm512_max_epi16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0x958257, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [12,26], 
                 [20,25], [17,24], [16,23], [22,22], [15,21], [19,19], 
                 [18,18], [14,14], [13,13], [6,11], [3,10], [2,9], [8,8], 
                 [1,7], [5,5], [4,4], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 12, 20, 17, 16, 22, 15, 25, 19, 18, 24, 
                 23, 21, 14, 13, 26,  6,  3,  2,  8,  1, 11,  5,  4, 10,  9,  
                 7,  0) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 12, 20, 17, 16, 22, 
                                               15, 25, 19, 18, 24, 23, 21, 
                                               14, 13, 26, 6, 3, 2, 8, 1, 11, 
                                               5, 4, 10, 9, 7, 0), v6);
      __m512i min7 = _mm512_min_epi16(v6, perm7);
      __m512i max7 = _mm512_max_epi16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x13904e, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [13,26], 
                 [11,25], [18,24], [23,23], [16,22], [21,21], [20,20], 
                 [19,19], [17,17], [1,15], [14,14], [12,12], [4,10], [9,9], 
                 [2,8], [7,7], [6,6], [5,5], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 13, 11, 18, 23, 16, 21, 20, 19, 24, 17, 
                 22,  1, 14, 26, 12, 25,  4,  9,  2,  7,  6,  5, 10,  3,  8, 
                 15,  0) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 13, 11, 18, 23, 16, 
                                               21, 20, 19, 24, 17, 22, 1, 14, 
                                               26, 12, 25, 4, 9, 2, 7, 6, 5, 
                                               10, 3, 8, 15, 0), v7);
      __m512i min8 = _mm512_min_epi16(v7, perm8);
      __m512i max8 = _mm512_max_epi16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x52816, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [13,25], [19,24], [23,23], [18,22], [16,21], [20,20], 
                 [17,17], [15,15], [1,14], [12,12], [11,11], [5,10], [9,9], 
                 [4,8], [2,7], [6,6], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 13, 19, 23, 18, 16, 20, 24, 22, 17, 
                 21, 15,  1, 25, 12, 11,  5,  9,  4,  2,  6, 10,  8,  3,  7, 
                 14,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 13, 19, 23, 18, 
                                               16, 20, 24, 22, 17, 21, 15, 1, 
                                               25, 12, 11, 5, 9, 4, 2, 6, 10, 
                                               8, 3, 7, 14, 0), v8);
      __m512i min9 = _mm512_min_epi16(v8, perm9);
      __m512i max9 = _mm512_max_epi16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0xd2036, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [20,24], [19,23], [22,22], [17,21], [18,18], 
                 [2,16], [15,15], [14,14], [13,13], [12,12], [11,11], [6,10], 
                 [5,9], [8,8], [3,7], [4,4], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 20, 19, 22, 17, 24, 23, 18, 21,  
                 2, 15, 14, 13, 12, 11,  6,  5,  8,  3, 10,  9,  4,  7, 16,  
                 1,  0) */
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 20, 19, 22, 
                                                17, 24, 23, 18, 21, 2, 15, 
                                                14, 13, 12, 11, 6, 5, 8, 3, 
                                                10, 9, 4, 7, 16, 1, 0), v9);
      __m512i min10 = _mm512_min_epi16(v9, perm10);
      __m512i max10 = _mm512_max_epi16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x1a006c, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [10,24], [20,23], [22,22], [18,21], [19,19], 
                 [3,17], [16,16], [2,15], [14,14], [13,13], [12,12], [11,11], 
                 [6,9], [8,8], [4,7], [5,5], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 10, 20, 22, 18, 23, 19, 21,  3, 
                 16,  2, 14, 13, 12, 11, 24,  6,  8,  4,  9,  5,  7, 17, 15,  
                 1,  0) */
      __m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 10, 20, 22, 
                                                18, 23, 19, 21, 3, 16, 2, 14, 
                                                13, 12, 11, 24, 6, 8, 4, 9, 
                                                5, 7, 17, 15, 1, 0), v10);
      __m512i min11 = _mm512_min_epi16(v10, perm11);
      __m512i max11 = _mm512_max_epi16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0x14045c, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [11,24], [9,23], [20,22], [19,21], [4,18], [17,17], 
                 [16,16], [15,15], [2,14], [13,13], [12,12], [10,10], [6,8], 
                 [5,7], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 11,  9, 20, 19, 22, 21,  4, 17, 
                 16, 15,  2, 13, 12, 24, 10, 23,  6,  5,  8,  7, 18,  3, 14,  
                 1,  0) */
      __m512i perm12 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 11, 9, 20, 
                                                19, 22, 21, 4, 17, 16, 15, 2, 
                                                13, 12, 24, 10, 23, 6, 5, 8, 
                                                7, 18, 3, 14, 1, 0), v11);
      __m512i min12 = _mm512_min_epi16(v11, perm12);
      __m512i max12 = _mm512_max_epi16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x180a74, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [12,24], [23,23], [8,22], [20,21], [5,19], [18,18], 
                 [4,17], [16,16], [15,15], [3,14], [13,13], [11,11], [10,10], 
                 [9,9], [6,7], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 12, 23,  8, 20, 21,  5, 18,  4, 
                 16, 15,  3, 13, 24, 11, 10,  9, 22,  6,  7, 19, 17, 14,  2,  
                 1,  0) */
      __m512i perm13 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 12, 23, 8, 
                                                20, 21, 5, 18, 4, 16, 15, 3, 
                                                13, 24, 11, 10, 9, 22, 6, 7, 
                                                19, 17, 14, 2, 1, 0), v12);
      __m512i min13 = _mm512_min_epi16(v12, perm13);
      __m512i max13 = _mm512_max_epi16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0x101178, min13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [13,24], [12,23], [9,22], [7,21], [6,20], [19,19], 
                 [18,18], [5,17], [16,16], [4,15], [14,14], [11,11], [10,10], 
                 [8,8], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 13, 12,  9,  7,  6, 19, 18,  5, 
                 16,  4, 14, 24, 23, 11, 10, 22,  8, 21, 20, 17, 15,  3,  2,  
                 1,  0) */
      __m512i perm14 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 13, 12, 9, 7, 
                                                6, 19, 18, 5, 16, 4, 14, 24, 
                                                23, 11, 10, 22, 8, 21, 20, 
                                                17, 15, 3, 2, 1, 0), v13);
      __m512i min14 = _mm512_min_epi16(v13, perm14);
      __m512i max14 = _mm512_max_epi16(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi16(max14, 0x32f0, min14);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [13,23], [11,22], [8,21], [20,20], [6,19], 
                 [18,18], [17,17], [5,16], [15,15], [4,14], [12,12], [10,10], 
                 [9,9], [7,7], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 13, 11,  8, 20,  6, 18, 17,  
                 5, 15,  4, 23, 12, 22, 10,  9, 21,  7, 19, 16, 14,  3,  2,  
                 1,  0) */
      __m512i perm15 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 13, 11, 
                                                8, 20, 6, 18, 17, 5, 15, 4, 
                                                23, 12, 22, 10, 9, 21, 7, 19, 
                                                16, 14, 3, 2, 1, 0), v14);
      __m512i min15 = _mm512_min_epi16(v14, perm15);
      __m512i max15 = _mm512_max_epi16(v14, perm15);
      __m512i v15 = _mm512_mask_mov_epi16(max15, 0x2970, min15);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [13,22], [9,21], [20,20], 
                 [19,19], [6,18], [17,17], [16,16], [15,15], [5,14], [12,12], 
                 [11,11], [10,10], [8,8], [7,7], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 13,  9, 20, 19,  6, 17, 
                 16, 15,  5, 22, 12, 11, 10, 21,  8,  7, 18, 14,  4,  3,  2,  
                 1,  0) */
      __m512i perm16 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 13, 
                                                9, 20, 19, 6, 17, 16, 15, 5, 
                                                22, 12, 11, 10, 21, 8, 7, 18, 
                                                14, 4, 3, 2, 1, 0), v15);
      __m512i min16 = _mm512_min_epi16(v15, perm16);
      __m512i max16 = _mm512_max_epi16(v15, perm16);
      __m512i v16 = _mm512_mask_mov_epi16(max16, 0x2260, min16);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [10,21], [20,20], 
                 [19,19], [18,18], [6,17], [16,16], [15,15], [14,14], 
                 [13,13], [12,12], [11,11], [9,9], [8,8], [7,7], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 10, 20, 19, 18,  6, 
                 16, 15, 14, 13, 12, 11, 21,  9,  8,  7, 17,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm17 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                10, 20, 19, 18, 6, 16, 15, 
                                                14, 13, 12, 11, 21, 9, 8, 7, 
                                                17, 5, 4, 3, 2, 1, 0), v16);
      __m512i min17 = _mm512_min_epi16(v16, perm17);
      __m512i max17 = _mm512_max_epi16(v16, perm17);
      __m512i v17 = _mm512_mask_mov_epi16(max17, 0x440, min17);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [11,21], [20,20], 
                 [19,19], [18,18], [10,17], [6,16], [15,15], [14,14], 
                 [13,13], [12,12], [9,9], [8,8], [7,7], [5,5], [4,4], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 11, 20, 19, 18, 10,  
                 6, 15, 14, 13, 12, 21, 17,  9,  8,  7, 16,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm18 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                11, 20, 19, 18, 10, 6, 15, 
                                                14, 13, 12, 21, 17, 9, 8, 7, 
                                                16, 5, 4, 3, 2, 1, 0), v17);
      __m512i min18 = _mm512_min_epi16(v17, perm18);
      __m512i max18 = _mm512_max_epi16(v17, perm18);
      __m512i v18 = _mm512_mask_mov_epi16(max18, 0xc40, min18);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [12,21], [20,20], 
                 [19,19], [11,18], [17,17], [9,16], [6,15], [14,14], [13,13], 
                 [10,10], [8,8], [7,7], [5,5], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 12, 20, 19, 11, 17,  
                 9,  6, 14, 13, 21, 18, 10, 16,  8,  7, 15,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm19 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                12, 20, 19, 11, 17, 9, 6, 14, 
                                                13, 21, 18, 10, 16, 8, 7, 15, 
                                                5, 4, 3, 2, 1, 0), v18);
      __m512i min19 = _mm512_min_epi16(v18, perm19);
      __m512i max19 = _mm512_max_epi16(v18, perm19);
      __m512i v19 = _mm512_mask_mov_epi16(max19, 0x1a40, min19);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [13,21], [20,20], 
                 [12,19], [18,18], [11,17], [16,16], [8,15], [6,14], [10,10], 
                 [9,9], [7,7], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 13, 20, 12, 18, 11, 
                 16,  8,  6, 21, 19, 17, 10,  9, 15,  7, 14,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm20 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                13, 20, 12, 18, 11, 16, 8, 6, 
                                                21, 19, 17, 10, 9, 15, 7, 14, 
                                                5, 4, 3, 2, 1, 0), v19);
      __m512i min20 = _mm512_min_epi16(v19, perm20);
      __m512i max20 = _mm512_max_epi16(v19, perm20);
      __m512i v20 = _mm512_mask_mov_epi16(max20, 0x3940, min20);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [13,20], 
                 [19,19], [18,18], [12,17], [16,16], [9,15], [7,14], [11,11], 
                 [10,10], [8,8], [6,6], [5,5], [4,4], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 13, 19, 18, 12, 
                 16,  9,  7, 20, 17, 11, 10, 15,  8, 14,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm21 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 13, 19, 18, 12, 16, 9, 7, 
                                                20, 17, 11, 10, 15, 8, 14, 6, 
                                                5, 4, 3, 2, 1, 0), v20);
      __m512i min21 = _mm512_min_epi16(v20, perm21);
      __m512i max21 = _mm512_max_epi16(v20, perm21);
      __m512i v21 = _mm512_mask_mov_epi16(max21, 0x3280, min21);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [13,19], [18,18], [17,17], [12,16], [11,15], [8,14], 
                 [10,10], [9,9], [7,7], [6,6], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 13, 18, 17, 
                 12, 11,  8, 19, 16, 15, 10,  9, 14,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm22 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 13, 18, 17, 12, 11, 
                                                8, 19, 16, 15, 10, 9, 14, 7, 
                                                6, 5, 4, 3, 2, 1, 0), v21);
      __m512i min22 = _mm512_min_epi16(v21, perm22);
      __m512i max22 = _mm512_max_epi16(v21, perm22);
      __m512i v22 = _mm512_mask_mov_epi16(max22, 0x3900, min22);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [13,18], [17,17], [16,16], [15,15], [9,14], 
                 [12,12], [11,11], [10,10], [8,8], [7,7], [6,6], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 13, 17, 
                 16, 15,  9, 18, 12, 11, 10, 14,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm23 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 13, 17, 16, 15, 
                                                9, 18, 12, 11, 10, 14, 8, 7, 
                                                6, 5, 4, 3, 2, 1, 0), v22);
      __m512i min23 = _mm512_min_epi16(v22, perm23);
      __m512i max23 = _mm512_max_epi16(v22, perm23);
      __m512i v23 = _mm512_mask_mov_epi16(max23, 0x2200, min23);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [13,17], [16,16], [15,15], [10,14], 
                 [12,12], [11,11], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 13, 
                 16, 15, 10, 17, 12, 11, 14,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm24 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 13, 16, 15, 
                                                10, 17, 12, 11, 14, 9, 8, 7, 
                                                6, 5, 4, 3, 2, 1, 0), v23);
      __m512i min24 = _mm512_min_epi16(v23, perm24);
      __m512i max24 = _mm512_max_epi16(v23, perm24);
      __m512i v24 = _mm512_mask_mov_epi16(max24, 0x2400, min24);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [13,16], [15,15], [11,14], 
                 [12,12], [10,10], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 13, 15, 11, 16, 12, 14, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm25 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 25, 24, 23, 22, 
                                                21, 20, 19, 18, 17, 13, 15, 
                                                11, 16, 12, 14, 10, 9, 8, 7, 
                                                6, 5, 4, 3, 2, 1, 0), v24);
      __m512i min25 = _mm512_min_epi16(v24, perm25);
      __m512i max25 = _mm512_max_epi16(v24, perm25);
      __m512i v25 = _mm512_mask_mov_epi16(max25, 0x2800, min25);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [13,15], [12,14], 
                 [11,11], [10,10], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 13, 12, 15, 14, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm26 = _mm512_shuffle_epi8(v25, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 27, 26, 25, 
                                           24, 31, 30, 29, 28, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min26 = _mm512_min_epi16(v25, perm26);
      __m512i max26 = _mm512_max_epi16(v25, perm26);
      __m512i v26 = _mm512_mask_mov_epi16(max26, 0x3000, min26);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [13,14], 
                 [12,12], [11,11], [10,10], [9,9], [8,8], [7,7], [6,6], 
                 [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 13, 14, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm27 = _mm512_shuffle_epi8(v26, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 31, 30, 27, 
                                           26, 29, 28, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 15, 14, 13, 
                                           12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min27 = _mm512_min_epi16(v26, perm27);
      __m512i max27 = _mm512_max_epi16(v26, perm27);
      __m512i v27 = _mm512_mask_mov_epi16(max27, 0x2000, min27);
      
      return v27;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bosenelson_28_int16_t(int16_t * const 
                             arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = bosenelson_28_int16_t_vec(v);
      
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


