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
#define N 61
#define SORT_NAME oddeven_61_int8_t

#ifndef _SIMD_SORT_oddeven_61_int8_t_H_
#define _SIMD_SORT_oddeven_61_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 61
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : oddeven
	Network Depth                    : 21
	SIMD Instructions                : 2 / 103
	Optimization Preference          : space
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512bw, AVX512vbmi
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
oddeven_61_int8_t_vec(__m512i v) {
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [58,59], [56,57], 
                 [54,55], [52,53], [50,51], [48,49], [46,47], [44,45], 
                 [42,43], [40,41], [38,39], [36,37], [34,35], [32,33], 
                 [30,31], [28,29], [26,27], [24,25], [22,23], [20,21], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [2,3], [0,1]) */
      /* Perm:  (63, 62, 61, 60, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 
                 49, 46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 
                 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 
                 19, 16, 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  
                 2,  3,  0,  1) */
      __m512i perm0 = _mm512_shuffle_epi8(v, _mm512_set_epi8(63, 62, 61, 60, 
                                          58, 59, 56, 57, 54, 55, 52, 53, 50, 
                                          51, 48, 49, 46, 47, 44, 45, 42, 43, 
                                          40, 41, 38, 39, 36, 37, 34, 35, 32, 
                                          33, 30, 31, 28, 29, 26, 27, 24, 25, 
                                          22, 23, 20, 21, 18, 19, 16, 17, 14, 
                                          15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 
                                          5, 2, 3, 0, 1));
      __m512i min0 = _mm512_min_epi8(v, perm0);
      __m512i max0 = _mm512_max_epi8(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi8(max0, 0x555555555555555, min0);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [57,59], [56,58], 
                 [53,55], [52,54], [49,51], [48,50], [45,47], [44,46], 
                 [41,43], [40,42], [37,39], [36,38], [33,35], [32,34], 
                 [29,31], [28,30], [25,27], [24,26], [21,23], [20,22], 
                 [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], 
                 [4,6], [1,3], [0,2]) */
      /* Perm:  (63, 62, 61, 60, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 
                 50, 45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 
                 35, 34, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 
                 16, 19, 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  
                 1,  0,  3,  2) */
      __m512i perm1 = _mm512_shuffle_epi8(v0, _mm512_set_epi8(63, 62, 61, 60, 
                                          57, 56, 59, 58, 53, 52, 55, 54, 49, 
                                          48, 51, 50, 45, 44, 47, 46, 41, 40, 
                                          43, 42, 37, 36, 39, 38, 33, 32, 35, 
                                          34, 29, 28, 31, 30, 25, 24, 27, 26, 
                                          21, 20, 23, 22, 17, 16, 19, 18, 13, 
                                          12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 
                                          6, 1, 0, 3, 2));
      __m512i min1 = _mm512_min_epi8(v0, perm1);
      __m512i max1 = _mm512_max_epi8(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi8(max1, 0x333333333333333, min1);
      
      /* Pairs: ([63,63], [62,62], [61,61], [56,60], [59,59], [57,58], 
                 [51,55], [53,54], [48,52], [49,50], [43,47], [45,46], 
                 [40,44], [41,42], [35,39], [37,38], [32,36], [33,34], 
                 [27,31], [29,30], [24,28], [25,26], [19,23], [21,22], 
                 [16,20], [17,18], [11,15], [13,14], [8,12], [9,10], [3,7], 
                 [5,6], [0,4], [1,2]) */
      /* Perm:  (63, 62, 61, 56, 59, 57, 58, 60, 51, 53, 54, 48, 55, 49, 50, 
                 52, 43, 45, 46, 40, 47, 41, 42, 44, 35, 37, 38, 32, 39, 33, 
                 34, 36, 27, 29, 30, 24, 31, 25, 26, 28, 19, 21, 22, 16, 23, 
                 17, 18, 20, 11, 13, 14,  8, 15,  9, 10, 12,  3,  5,  6,  0,  
                 7,  1,  2,  4) */
      __m512i perm2 = _mm512_shuffle_epi8(v1, _mm512_set_epi8(63, 62, 61, 56, 
                                          59, 57, 58, 60, 51, 53, 54, 48, 55, 
                                          49, 50, 52, 43, 45, 46, 40, 47, 41, 
                                          42, 44, 35, 37, 38, 32, 39, 33, 34, 
                                          36, 27, 29, 30, 24, 31, 25, 26, 28, 
                                          19, 21, 22, 16, 23, 17, 18, 20, 11, 
                                          13, 14, 8, 15, 9, 10, 12, 3, 5, 6, 
                                          0, 7, 1, 2, 4));
      __m512i min2 = _mm512_min_epi8(v1, perm2);
      __m512i max2 = _mm512_max_epi8(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi8(max2, 0x32b2b2b2b2b2b2b, min2);
      
      /* Pairs: ([63,63], [62,62], [61,61], [58,60], [59,59], [57,57], 
                 [48,56], [55,55], [50,54], [49,53], [52,52], [51,51], 
                 [39,47], [42,46], [41,45], [44,44], [43,43], [32,40], 
                 [34,38], [33,37], [36,36], [35,35], [23,31], [26,30], 
                 [25,29], [28,28], [27,27], [16,24], [18,22], [17,21], 
                 [20,20], [19,19], [7,15], [10,14], [9,13], [12,12], [11,11], 
                 [0,8], [2,6], [1,5], [4,4], [3,3]) */
      /* Perm:  (63, 62, 61, 58, 59, 60, 57, 48, 55, 50, 49, 52, 51, 54, 53, 
                 56, 39, 42, 41, 44, 43, 46, 45, 32, 47, 34, 33, 36, 35, 38, 
                 37, 40, 23, 26, 25, 28, 27, 30, 29, 16, 31, 18, 17, 20, 19, 
                 22, 21, 24,  7, 10,  9, 12, 11, 14, 13,  0, 15,  2,  1,  4,  
                 3,  6,  5,  8) */
      __m512i perm3 = _mm512_shuffle_epi8(v2, _mm512_set_epi8(63, 62, 61, 58, 
                                          59, 60, 57, 48, 55, 50, 49, 52, 51, 
                                          54, 53, 56, 39, 42, 41, 44, 43, 46, 
                                          45, 32, 47, 34, 33, 36, 35, 38, 37, 
                                          40, 23, 26, 25, 28, 27, 30, 29, 16, 
                                          31, 18, 17, 20, 19, 22, 21, 24, 7, 
                                          10, 9, 12, 11, 14, 13, 0, 15, 2, 1, 
                                          4, 3, 6, 5, 8));
      __m512i min3 = _mm512_min_epi8(v2, perm3);
      __m512i max3 = _mm512_max_epi8(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi8(max3, 0x407068706870687, min3);
      
      /* Pairs: ([63,63], [62,62], [61,61], [59,60], [57,58], [56,56], 
                 [55,55], [54,54], [51,53], [50,52], [49,49], [32,48], 
                 [47,47], [46,46], [43,45], [42,44], [41,41], [40,40], 
                 [39,39], [38,38], [35,37], [34,36], [33,33], [15,31], 
                 [30,30], [27,29], [26,28], [25,25], [24,24], [23,23], 
                 [22,22], [19,21], [18,20], [17,17], [0,16], [14,14], 
                 [11,13], [10,12], [9,9], [8,8], [7,7], [6,6], [3,5], [2,4], 
                 [1,1]) */
      /* Perm:  (63, 62, 61, 59, 60, 57, 58, 56, 55, 54, 51, 50, 53, 52, 49, 
                 32, 47, 46, 43, 42, 45, 44, 41, 40, 39, 38, 35, 34, 37, 36, 
                 33, 48, 15, 30, 27, 26, 29, 28, 25, 24, 23, 22, 19, 18, 21, 
                 20, 17,  0, 31, 14, 11, 10, 13, 12,  9,  8,  7,  6,  3,  2,  
                 5,  4,  1, 16) */
      __m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 59, 
                                              60, 57, 58, 56, 55, 54, 51, 50, 
                                              53, 52, 49, 32, 47, 46, 43, 42, 
                                              45, 44, 41, 40, 39, 38, 35, 34, 
                                              37, 36, 33, 48, 15, 30, 27, 26, 
                                              29, 28, 25, 24, 23, 22, 19, 18, 
                                              21, 20, 17, 0, 31, 14, 11, 10, 
                                              13, 12, 9, 8, 7, 6, 3, 2, 5, 4, 
                                              1, 16), v3);
      __m512i min4 = _mm512_min_epi8(v3, perm4);
      __m512i max4 = _mm512_max_epi8(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi8(max4, 0xa0c0c0d0c0c8c0d, min4);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [59,59], [58,58], 
                 [57,57], [56,56], [55,55], [53,54], [51,52], [49,50], 
                 [48,48], [47,47], [45,46], [43,44], [41,42], [40,40], 
                 [39,39], [37,38], [35,36], [33,34], [0,32], [31,31], 
                 [29,30], [27,28], [25,26], [24,24], [23,23], [21,22], 
                 [19,20], [17,18], [16,16], [15,15], [13,14], [11,12], 
                 [9,10], [8,8], [7,7], [5,6], [3,4], [1,2]) */
      /* Perm:  (63, 62, 61, 60, 59, 58, 57, 56, 55, 53, 54, 51, 52, 49, 50, 
                 48, 47, 45, 46, 43, 44, 41, 42, 40, 39, 37, 38, 35, 36, 33, 
                 34,  0, 31, 29, 30, 27, 28, 25, 26, 24, 23, 21, 22, 19, 20, 
                 17, 18, 16, 15, 13, 14, 11, 12,  9, 10,  8,  7,  5,  6,  3,  
                 4,  1,  2, 32) */
      __m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 53, 54, 51, 
                                              52, 49, 50, 48, 47, 45, 46, 43, 
                                              44, 41, 42, 40, 39, 37, 38, 35, 
                                              36, 33, 34, 0, 31, 29, 30, 27, 
                                              28, 25, 26, 24, 23, 21, 22, 19, 
                                              20, 17, 18, 16, 15, 13, 14, 11, 
                                              12, 9, 10, 8, 7, 5, 6, 3, 4, 1, 
                                              2, 32), v4);
      __m512i min5 = _mm512_min_epi8(v4, perm5);
      __m512i max5 = _mm512_max_epi8(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi8(max5, 0x2a2a2a2a2a2a2b, min5);
      
      /* Pairs: ([63,63], [62,62], [61,61], [52,60], [51,59], [50,58], 
                 [49,57], [56,56], [55,55], [54,54], [53,53], [48,48], 
                 [47,47], [38,46], [37,45], [36,44], [35,43], [34,42], 
                 [33,41], [40,40], [39,39], [32,32], [31,31], [22,30], 
                 [21,29], [20,28], [19,27], [18,26], [17,25], [24,24], 
                 [23,23], [16,16], [15,15], [6,14], [5,13], [4,12], [3,11], 
                 [2,10], [1,9], [8,8], [7,7], [0,0]) */
      /* Perm:  (63, 62, 61, 52, 51, 50, 49, 56, 55, 54, 53, 60, 59, 58, 57, 
                 48, 47, 38, 37, 36, 35, 34, 33, 40, 39, 46, 45, 44, 43, 42, 
                 41, 32, 31, 22, 21, 20, 19, 18, 17, 24, 23, 30, 29, 28, 27, 
                 26, 25, 16, 15,  6,  5,  4,  3,  2,  1,  8,  7, 14, 13, 12, 
                 11, 10,  9,  0) */
      __m512i perm6 = _mm512_shuffle_epi8(v5, _mm512_set_epi8(63, 62, 61, 52, 
                                          51, 50, 49, 56, 55, 54, 53, 60, 59, 
                                          58, 57, 48, 47, 38, 37, 36, 35, 34, 
                                          33, 40, 39, 46, 45, 44, 43, 42, 41, 
                                          32, 31, 22, 21, 20, 19, 18, 17, 24, 
                                          23, 30, 29, 28, 27, 26, 25, 16, 15, 
                                          6, 5, 4, 3, 2, 1, 8, 7, 14, 13, 12, 
                                          11, 10, 9, 0));
      __m512i min6 = _mm512_min_epi8(v5, perm6);
      __m512i max6 = _mm512_max_epi8(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi8(max6, 0x1e007e007e007e, min6);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [55,59], [54,58], 
                 [53,57], [52,56], [51,51], [50,50], [49,49], [48,48], 
                 [47,47], [46,46], [45,45], [44,44], [39,43], [38,42], 
                 [37,41], [36,40], [35,35], [34,34], [33,33], [32,32], 
                 [31,31], [30,30], [29,29], [28,28], [23,27], [22,26], 
                 [21,25], [20,24], [19,19], [18,18], [17,17], [16,16], 
                 [15,15], [14,14], [13,13], [12,12], [7,11], [6,10], [5,9], 
                 [4,8], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 55, 54, 53, 52, 59, 58, 57, 56, 51, 50, 49, 
                 48, 47, 46, 45, 44, 39, 38, 37, 36, 43, 42, 41, 40, 35, 34, 
                 33, 32, 31, 30, 29, 28, 23, 22, 21, 20, 27, 26, 25, 24, 19, 
                 18, 17, 16, 15, 14, 13, 12,  7,  6,  5,  4, 11, 10,  9,  8,  
                 3,  2,  1,  0) */
      __m512i perm7 = _mm512_shuffle_epi32(v6, _MM_PERM_ENUM(0xd8));
      __m512i min7 = _mm512_min_epi8(v6, perm7);
      __m512i max7 = _mm512_max_epi8(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi8(max7, 0xf000f000f000f0, min7);
      
      /* Pairs: ([63,63], [62,62], [61,61], [58,60], [59,59], [55,57], 
                 [54,56], [51,53], [50,52], [49,49], [48,48], [47,47], 
                 [46,46], [43,45], [42,44], [39,41], [38,40], [35,37], 
                 [34,36], [33,33], [32,32], [31,31], [30,30], [27,29], 
                 [26,28], [23,25], [22,24], [19,21], [18,20], [17,17], 
                 [16,16], [15,15], [14,14], [11,13], [10,12], [7,9], [6,8], 
                 [3,5], [2,4], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 58, 59, 60, 55, 54, 57, 56, 51, 50, 53, 52, 49, 
                 48, 47, 46, 43, 42, 45, 44, 39, 38, 41, 40, 35, 34, 37, 36, 
                 33, 32, 31, 30, 27, 26, 29, 28, 23, 22, 25, 24, 19, 18, 21, 
                 20, 17, 16, 15, 14, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  
                 5,  4,  1,  0) */
      __m512i perm8 = _mm512_shuffle_epi8(v7, _mm512_set_epi8(63, 62, 61, 58, 
                                          59, 60, 55, 54, 57, 56, 51, 50, 53, 
                                          52, 49, 48, 47, 46, 43, 42, 45, 44, 
                                          39, 38, 41, 40, 35, 34, 37, 36, 33, 
                                          32, 31, 30, 27, 26, 29, 28, 23, 22, 
                                          25, 24, 19, 18, 21, 20, 17, 16, 15, 
                                          14, 11, 10, 13, 12, 7, 6, 9, 8, 3, 
                                          2, 5, 4, 1, 0));
      __m512i min8 = _mm512_min_epi8(v7, perm8);
      __m512i max8 = _mm512_max_epi8(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi8(max8, 0x4cc0ccc0ccc0ccc, min8);
      
      /* Pairs: ([63,63], [62,62], [61,61], [59,60], [57,58], [55,56], 
                 [53,54], [51,52], [49,50], [48,48], [47,47], [45,46], 
                 [43,44], [41,42], [39,40], [37,38], [35,36], [33,34], 
                 [32,32], [31,31], [29,30], [27,28], [25,26], [23,24], 
                 [21,22], [19,20], [17,18], [16,16], [15,15], [13,14], 
                 [11,12], [9,10], [7,8], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (63, 62, 61, 59, 60, 57, 58, 55, 56, 53, 54, 51, 52, 49, 50, 
                 48, 47, 45, 46, 43, 44, 41, 42, 39, 40, 37, 38, 35, 36, 33, 
                 34, 32, 31, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 
                 17, 18, 16, 15, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  
                 4,  1,  2,  0) */
      __m512i perm9 = _mm512_shuffle_epi8(v8, _mm512_set_epi8(63, 62, 61, 59, 
                                          60, 57, 58, 55, 56, 53, 54, 51, 52, 
                                          49, 50, 48, 47, 45, 46, 43, 44, 41, 
                                          42, 39, 40, 37, 38, 35, 36, 33, 34, 
                                          32, 31, 29, 30, 27, 28, 25, 26, 23, 
                                          24, 21, 22, 19, 20, 17, 18, 16, 15, 
                                          13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 
                                          3, 4, 1, 2, 0));
      __m512i min9 = _mm512_min_epi8(v8, perm9);
      __m512i max9 = _mm512_max_epi8(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi8(max9, 0xaaa2aaa2aaa2aaa, min9);
      
      /* Pairs: ([63,63], [62,62], [61,61], [44,60], [43,59], [42,58], 
                 [41,57], [40,56], [39,55], [38,54], [37,53], [36,52], 
                 [35,51], [34,50], [33,49], [48,48], [47,47], [46,46], 
                 [45,45], [32,32], [31,31], [14,30], [13,29], [12,28], 
                 [11,27], [10,26], [9,25], [8,24], [7,23], [6,22], [5,21], 
                 [4,20], [3,19], [2,18], [1,17], [16,16], [15,15], [0,0]) */
      /* Perm:  (63, 62, 61, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 
                 48, 47, 46, 45, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 
                 49, 32, 31, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  
                 2,  1, 16, 15, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 
                 19, 18, 17,  0) */
      __m512i perm10 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               44, 43, 42, 41, 40, 39, 38, 
                                               37, 36, 35, 34, 33, 48, 47, 
                                               46, 45, 60, 59, 58, 57, 56, 
                                               55, 54, 53, 52, 51, 50, 49, 
                                               32, 31, 14, 13, 12, 11, 10, 9, 
                                               8, 7, 6, 5, 4, 3, 2, 1, 16, 
                                               15, 30, 29, 28, 27, 26, 25, 
                                               24, 23, 22, 21, 20, 19, 18, 
                                               17, 0), v9);
      __m512i min10 = _mm512_min_epi8(v9, perm10);
      __m512i max10 = _mm512_max_epi8(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi8(max10, 0x1ffe00007ffe, min10);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [59,59], [58,58], 
                 [57,57], [56,56], [47,55], [46,54], [45,53], [44,52], 
                 [43,51], [42,50], [41,49], [40,48], [39,39], [38,38], 
                 [37,37], [36,36], [35,35], [34,34], [33,33], [32,32], 
                 [31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [15,23], [14,22], [13,21], [12,20], 
                 [11,19], [10,18], [9,17], [8,16], [7,7], [6,6], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 59, 58, 57, 56, 47, 46, 45, 44, 43, 42, 41, 
                 40, 55, 54, 53, 52, 51, 50, 49, 48, 39, 38, 37, 36, 35, 34, 
                 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 15, 14, 13, 12, 11, 
                 10,  9,  8, 23, 22, 21, 20, 19, 18, 17, 16,  7,  6,  5,  4,  
                 3,  2,  1,  0) */
      __m512i perm11 = _mm512_permutex_epi64(v10, 0xd8);
      __m512i min11 = _mm512_min_epi8(v10, perm11);
      __m512i max11 = _mm512_max_epi8(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi8(max11, 0xff000000ff00, min11);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [55,59], [54,58], 
                 [53,57], [52,56], [47,51], [46,50], [45,49], [44,48], 
                 [39,43], [38,42], [37,41], [36,40], [35,35], [34,34], 
                 [33,33], [32,32], [31,31], [30,30], [29,29], [28,28], 
                 [23,27], [22,26], [21,25], [20,24], [15,19], [14,18], 
                 [13,17], [12,16], [7,11], [6,10], [5,9], [4,8], [3,3], 
                 [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 55, 54, 53, 52, 59, 58, 57, 56, 47, 46, 45, 
                 44, 51, 50, 49, 48, 39, 38, 37, 36, 43, 42, 41, 40, 35, 34, 
                 33, 32, 31, 30, 29, 28, 23, 22, 21, 20, 27, 26, 25, 24, 15, 
                 14, 13, 12, 19, 18, 17, 16,  7,  6,  5,  4, 11, 10,  9,  8,  
                 3,  2,  1,  0) */
      __m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 55, 54, 53, 52, 59, 58, 
                                               57, 56, 47, 46, 45, 44, 51, 
                                               50, 49, 48, 39, 38, 37, 36, 
                                               43, 42, 41, 40, 35, 34, 33, 
                                               32, 31, 30, 29, 28, 23, 22, 
                                               21, 20, 27, 26, 25, 24, 15, 
                                               14, 13, 12, 19, 18, 17, 16, 7, 
                                               6, 5, 4, 11, 10, 9, 8, 3, 2, 
                                               1, 0), v11);
      __m512i min12 = _mm512_min_epi8(v11, perm12);
      __m512i max12 = _mm512_max_epi8(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi8(max12, 0xf0f0f000f0f0f0, min12);
      
      /* Pairs: ([63,63], [62,62], [61,61], [58,60], [59,59], [55,57], 
                 [54,56], [51,53], [50,52], [47,49], [46,48], [43,45], 
                 [42,44], [39,41], [38,40], [35,37], [34,36], [33,33], 
                 [32,32], [31,31], [30,30], [27,29], [26,28], [23,25], 
                 [22,24], [19,21], [18,20], [15,17], [14,16], [11,13], 
                 [10,12], [7,9], [6,8], [3,5], [2,4], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 58, 59, 60, 55, 54, 57, 56, 51, 50, 53, 52, 47, 
                 46, 49, 48, 43, 42, 45, 44, 39, 38, 41, 40, 35, 34, 37, 36, 
                 33, 32, 31, 30, 27, 26, 29, 28, 23, 22, 25, 24, 19, 18, 21, 
                 20, 15, 14, 17, 16, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  
                 5,  4,  1,  0) */
      __m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               58, 59, 60, 55, 54, 57, 56, 
                                               51, 50, 53, 52, 47, 46, 49, 
                                               48, 43, 42, 45, 44, 39, 38, 
                                               41, 40, 35, 34, 37, 36, 33, 
                                               32, 31, 30, 27, 26, 29, 28, 
                                               23, 22, 25, 24, 19, 18, 21, 
                                               20, 15, 14, 17, 16, 11, 10, 
                                               13, 12, 7, 6, 9, 8, 3, 2, 5, 
                                               4, 1, 0), v12);
      __m512i min13 = _mm512_min_epi8(v12, perm13);
      __m512i max13 = _mm512_max_epi8(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi8(max13, 0x4cccccc0ccccccc, min13);
      
      /* Pairs: ([63,63], [62,62], [61,61], [59,60], [57,58], [55,56], 
                 [53,54], [51,52], [49,50], [47,48], [45,46], [43,44], 
                 [41,42], [39,40], [37,38], [35,36], [33,34], [32,32], 
                 [31,31], [29,30], [27,28], [25,26], [23,24], [21,22], 
                 [19,20], [17,18], [15,16], [13,14], [11,12], [9,10], [7,8], 
                 [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (63, 62, 61, 59, 60, 57, 58, 55, 56, 53, 54, 51, 52, 49, 50, 
                 47, 48, 45, 46, 43, 44, 41, 42, 39, 40, 37, 38, 35, 36, 33, 
                 34, 32, 31, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 
                 17, 18, 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  
                 4,  1,  2,  0) */
      __m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               59, 60, 57, 58, 55, 56, 53, 
                                               54, 51, 52, 49, 50, 47, 48, 
                                               45, 46, 43, 44, 41, 42, 39, 
                                               40, 37, 38, 35, 36, 33, 34, 
                                               32, 31, 29, 30, 27, 28, 25, 
                                               26, 23, 24, 21, 22, 19, 20, 
                                               17, 18, 15, 16, 13, 14, 11, 
                                               12, 9, 10, 7, 8, 5, 6, 3, 4, 
                                               1, 2, 0), v13);
      __m512i min14 = _mm512_min_epi8(v13, perm14);
      __m512i max14 = _mm512_max_epi8(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi8(max14, 0xaaaaaaa2aaaaaaa, min14);
      
      /* Pairs: ([63,63], [62,62], [61,61], [28,60], [27,59], [26,58], 
                 [25,57], [24,56], [23,55], [22,54], [21,53], [20,52], 
                 [19,51], [18,50], [17,49], [16,48], [15,47], [14,46], 
                 [13,45], [12,44], [11,43], [10,42], [9,41], [8,40], [7,39], 
                 [6,38], [5,37], [4,36], [3,35], [2,34], [1,33], [32,32], 
                 [31,31], [30,30], [29,29], [0,0]) */
      /* Perm:  (63, 62, 61, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1, 32, 31, 30, 29, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 
                 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 
                 35, 34, 33,  0) */
      __m512i perm15 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 16, 15, 
                                               14, 13, 12, 11, 10, 9, 8, 7, 
                                               6, 5, 4, 3, 2, 1, 32, 31, 30, 
                                               29, 60, 59, 58, 57, 56, 55, 
                                               54, 53, 52, 51, 50, 49, 48, 
                                               47, 46, 45, 44, 43, 42, 41, 
                                               40, 39, 38, 37, 36, 35, 34, 
                                               33, 0), v14);
      __m512i min15 = _mm512_min_epi8(v14, perm15);
      __m512i max15 = _mm512_max_epi8(v14, perm15);
      __m512i v15 = _mm512_mask_mov_epi8(max15, 0x1ffffffe, min15);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [59,59], [58,58], 
                 [57,57], [56,56], [55,55], [54,54], [53,53], [52,52], 
                 [51,51], [50,50], [49,49], [48,48], [31,47], [30,46], 
                 [29,45], [28,44], [27,43], [26,42], [25,41], [24,40], 
                 [23,39], [22,38], [21,37], [20,36], [19,35], [18,34], 
                 [17,33], [16,32], [15,15], [14,14], [13,13], [12,12], 
                 [11,11], [10,10], [9,9], [8,8], [7,7], [6,6], [5,5], [4,4], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 
                 48, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 
                 17, 16, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 
                 34, 33, 32, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  
                 3,  2,  1,  0) */
      __m512i perm16 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 31, 
                                               30, 29, 28, 27, 26, 25, 24, 
                                               23, 22, 21, 20, 19, 18, 17, 
                                               16, 47, 46, 45, 44, 43, 42, 
                                               41, 40, 39, 38, 37, 36, 35, 
                                               34, 33, 32, 15, 14, 13, 12, 
                                               11, 10, 9, 8, 7, 6, 5, 4, 3, 
                                               2, 1, 0), v15);
      __m512i min16 = _mm512_min_epi8(v15, perm16);
      __m512i max16 = _mm512_max_epi8(v15, perm16);
      __m512i v16 = _mm512_mask_mov_epi8(max16, 0xffff0000, min16);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [59,59], [58,58], 
                 [57,57], [56,56], [47,55], [46,54], [45,53], [44,52], 
                 [43,51], [42,50], [41,49], [40,48], [31,39], [30,38], 
                 [29,37], [28,36], [27,35], [26,34], [25,33], [24,32], 
                 [15,23], [14,22], [13,21], [12,20], [11,19], [10,18], 
                 [9,17], [8,16], [7,7], [6,6], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 59, 58, 57, 56, 47, 46, 45, 44, 43, 42, 41, 
                 40, 55, 54, 53, 52, 51, 50, 49, 48, 31, 30, 29, 28, 27, 26, 
                 25, 24, 39, 38, 37, 36, 35, 34, 33, 32, 15, 14, 13, 12, 11, 
                 10,  9,  8, 23, 22, 21, 20, 19, 18, 17, 16,  7,  6,  5,  4,  
                 3,  2,  1,  0) */
      __m512i perm17 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 47, 46, 
                                               45, 44, 43, 42, 41, 40, 55, 
                                               54, 53, 52, 51, 50, 49, 48, 
                                               31, 30, 29, 28, 27, 26, 25, 
                                               24, 39, 38, 37, 36, 35, 34, 
                                               33, 32, 15, 14, 13, 12, 11, 
                                               10, 9, 8, 23, 22, 21, 20, 19, 
                                               18, 17, 16, 7, 6, 5, 4, 3, 2, 
                                               1, 0), v16);
      __m512i min17 = _mm512_min_epi8(v16, perm17);
      __m512i max17 = _mm512_max_epi8(v16, perm17);
      __m512i v17 = _mm512_mask_mov_epi8(max17, 0xff00ff00ff00, min17);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [55,59], [54,58], 
                 [53,57], [52,56], [47,51], [46,50], [45,49], [44,48], 
                 [39,43], [38,42], [37,41], [36,40], [31,35], [30,34], 
                 [29,33], [28,32], [23,27], [22,26], [21,25], [20,24], 
                 [15,19], [14,18], [13,17], [12,16], [7,11], [6,10], [5,9], 
                 [4,8], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 55, 54, 53, 52, 59, 58, 57, 56, 47, 46, 45, 
                 44, 51, 50, 49, 48, 39, 38, 37, 36, 43, 42, 41, 40, 31, 30, 
                 29, 28, 35, 34, 33, 32, 23, 22, 21, 20, 27, 26, 25, 24, 15, 
                 14, 13, 12, 19, 18, 17, 16,  7,  6,  5,  4, 11, 10,  9,  8,  
                 3,  2,  1,  0) */
      __m512i perm18 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 55, 54, 53, 52, 59, 58, 
                                               57, 56, 47, 46, 45, 44, 51, 
                                               50, 49, 48, 39, 38, 37, 36, 
                                               43, 42, 41, 40, 31, 30, 29, 
                                               28, 35, 34, 33, 32, 23, 22, 
                                               21, 20, 27, 26, 25, 24, 15, 
                                               14, 13, 12, 19, 18, 17, 16, 7, 
                                               6, 5, 4, 11, 10, 9, 8, 3, 2, 
                                               1, 0), v17);
      __m512i min18 = _mm512_min_epi8(v17, perm18);
      __m512i max18 = _mm512_max_epi8(v17, perm18);
      __m512i v18 = _mm512_mask_mov_epi8(max18, 0xf0f0f0f0f0f0f0, min18);
      
      /* Pairs: ([63,63], [62,62], [61,61], [58,60], [59,59], [55,57], 
                 [54,56], [51,53], [50,52], [47,49], [46,48], [43,45], 
                 [42,44], [39,41], [38,40], [35,37], [34,36], [31,33], 
                 [30,32], [27,29], [26,28], [23,25], [22,24], [19,21], 
                 [18,20], [15,17], [14,16], [11,13], [10,12], [7,9], [6,8], 
                 [3,5], [2,4], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 58, 59, 60, 55, 54, 57, 56, 51, 50, 53, 52, 47, 
                 46, 49, 48, 43, 42, 45, 44, 39, 38, 41, 40, 35, 34, 37, 36, 
                 31, 30, 33, 32, 27, 26, 29, 28, 23, 22, 25, 24, 19, 18, 21, 
                 20, 15, 14, 17, 16, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  
                 5,  4,  1,  0) */
      __m512i perm19 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               58, 59, 60, 55, 54, 57, 56, 
                                               51, 50, 53, 52, 47, 46, 49, 
                                               48, 43, 42, 45, 44, 39, 38, 
                                               41, 40, 35, 34, 37, 36, 31, 
                                               30, 33, 32, 27, 26, 29, 28, 
                                               23, 22, 25, 24, 19, 18, 21, 
                                               20, 15, 14, 17, 16, 11, 10, 
                                               13, 12, 7, 6, 9, 8, 3, 2, 5, 
                                               4, 1, 0), v18);
      __m512i min19 = _mm512_min_epi8(v18, perm19);
      __m512i max19 = _mm512_max_epi8(v18, perm19);
      __m512i v19 = _mm512_mask_mov_epi8(max19, 0x4cccccccccccccc, min19);
      
      /* Pairs: ([63,63], [62,62], [61,61], [59,60], [57,58], [55,56], 
                 [53,54], [51,52], [49,50], [47,48], [45,46], [43,44], 
                 [41,42], [39,40], [37,38], [35,36], [33,34], [31,32], 
                 [29,30], [27,28], [25,26], [23,24], [21,22], [19,20], 
                 [17,18], [15,16], [13,14], [11,12], [9,10], [7,8], [5,6], 
                 [3,4], [1,2], [0,0]) */
      /* Perm:  (63, 62, 61, 59, 60, 57, 58, 55, 56, 53, 54, 51, 52, 49, 50, 
                 47, 48, 45, 46, 43, 44, 41, 42, 39, 40, 37, 38, 35, 36, 33, 
                 34, 31, 32, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 
                 17, 18, 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  
                 4,  1,  2,  0) */
      __m512i perm20 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               59, 60, 57, 58, 55, 56, 53, 
                                               54, 51, 52, 49, 50, 47, 48, 
                                               45, 46, 43, 44, 41, 42, 39, 
                                               40, 37, 38, 35, 36, 33, 34, 
                                               31, 32, 29, 30, 27, 28, 25, 
                                               26, 23, 24, 21, 22, 19, 20, 
                                               17, 18, 15, 16, 13, 14, 11, 
                                               12, 9, 10, 7, 8, 5, 6, 3, 4, 
                                               1, 2, 0), v19);
      __m512i min20 = _mm512_min_epi8(v19, perm20);
      __m512i max20 = _mm512_max_epi8(v19, perm20);
      __m512i v20 = _mm512_mask_mov_epi8(max20, 0xaaaaaaaaaaaaaaa, min20);
      
      return v20;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
oddeven_61_int8_t(int8_t * const arr) 
                             {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = oddeven_61_int8_t_vec(v);
      
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


