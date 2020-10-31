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
#define N 63
#define SORT_NAME batcher_63_int8_t

#ifndef _SIMD_SORT_batcher_63_int8_t_H_
#define _SIMD_SORT_batcher_63_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 63
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 21
	SIMD Instructions                : 2 / 105
	Optimization Preference          : space
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512vbmi, AVX512bw
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
batcher_63_int8_t_vec(__m512i v) {
      
      /* Pairs: ([63,63], [30,62], [29,61], [28,60], [27,59], [26,58], 
                 [25,57], [24,56], [23,55], [22,54], [21,53], [20,52], 
                 [19,51], [18,50], [17,49], [16,48], [15,47], [14,46], 
                 [13,45], [12,44], [11,43], [10,42], [9,41], [8,40], [7,39], 
                 [6,38], [5,37], [4,36], [3,35], [2,34], [1,33], [0,32], 
                 [31,31]) */
      /* Perm:  (63, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  
                 1,  0, 31, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 
                 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 
                 35, 34, 33, 32) */
      __m512i perm0 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 
                                              1, 0, 31, 62, 61, 60, 59, 58, 
                                              57, 56, 55, 54, 53, 52, 51, 50, 
                                              49, 48, 47, 46, 45, 44, 43, 42, 
                                              41, 40, 39, 38, 37, 36, 35, 34, 
                                              33, 32), v);
      __m512i min0 = _mm512_min_epi8(v, perm0);
      __m512i max0 = _mm512_max_epi8(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi8(max0, 0x7fffffff, min0);
      
      /* Pairs: ([63,63], [46,62], [45,61], [44,60], [43,59], [42,58], 
                 [41,57], [40,56], [39,55], [38,54], [37,53], [36,52], 
                 [35,51], [34,50], [33,49], [32,48], [47,47], [15,31], 
                 [14,30], [13,29], [12,28], [11,27], [10,26], [9,25], [8,24], 
                 [7,23], [6,22], [5,21], [4,20], [3,19], [2,18], [1,17], 
                 [0,16]) */
      /* Perm:  (63, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 
                 32, 47, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 
                 49, 48, 15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  
                 2,  1,  0, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 
                 19, 18, 17, 16) */
      __m512i perm1 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 46, 45, 44, 
                                              43, 42, 41, 40, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 47, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 
                                              1, 0, 31, 30, 29, 28, 27, 26, 
                                              25, 24, 23, 22, 21, 20, 19, 18, 
                                              17, 16), v0);
      __m512i min1 = _mm512_min_epi8(v0, perm1);
      __m512i max1 = _mm512_max_epi8(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi8(max1, 0x7fff0000ffff, min1);
      
      /* Pairs: ([63,63], [54,62], [53,61], [52,60], [51,59], [50,58], 
                 [49,57], [48,56], [55,55], [31,47], [30,46], [29,45], 
                 [28,44], [27,43], [26,42], [25,41], [24,40], [23,39], 
                 [22,38], [21,37], [20,36], [19,35], [18,34], [17,33], 
                 [16,32], [7,15], [6,14], [5,13], [4,12], [3,11], [2,10], 
                 [1,9], [0,8]) */
      /* Perm:  (63, 54, 53, 52, 51, 50, 49, 48, 55, 62, 61, 60, 59, 58, 57, 
                 56, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 
                 17, 16, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 
                 34, 33, 32,  7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 
                 11, 10,  9,  8) */
      __m512i perm2 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 54, 53, 52, 
                                              51, 50, 49, 48, 55, 62, 61, 60, 
                                              59, 58, 57, 56, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 7, 6, 5, 4, 3, 
                                              2, 1, 0, 15, 14, 13, 12, 11, 
                                              10, 9, 8), v1);
      __m512i min2 = _mm512_min_epi8(v1, perm2);
      __m512i max2 = _mm512_max_epi8(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi8(max2, 0x7f0000ffff00ff, min2);
      
      /* Pairs: ([63,63], [58,62], [57,61], [56,60], [59,59], [55,55], 
                 [54,54], [53,53], [52,52], [51,51], [50,50], [49,49], 
                 [48,48], [39,47], [38,46], [37,45], [36,44], [35,43], 
                 [34,42], [33,41], [32,40], [23,31], [22,30], [21,29], 
                 [20,28], [19,27], [18,26], [17,25], [16,24], [15,15], 
                 [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], [8,8], 
                 [3,7], [2,6], [1,5], [0,4]) */
      /* Perm:  (63, 58, 57, 56, 59, 62, 61, 60, 55, 54, 53, 52, 51, 50, 49, 
                 48, 39, 38, 37, 36, 35, 34, 33, 32, 47, 46, 45, 44, 43, 42, 
                 41, 40, 23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 
                 26, 25, 24, 15, 14, 13, 12, 11, 10,  9,  8,  3,  2,  1,  0,  
                 7,  6,  5,  4) */
      __m512i perm3 = _mm512_shuffle_epi8(v2, _mm512_set_epi8(63, 58, 57, 56, 
                                          59, 62, 61, 60, 55, 54, 53, 52, 51, 
                                          50, 49, 48, 39, 38, 37, 36, 35, 34, 
                                          33, 32, 47, 46, 45, 44, 43, 42, 41, 
                                          40, 23, 22, 21, 20, 19, 18, 17, 16, 
                                          31, 30, 29, 28, 27, 26, 25, 24, 15, 
                                          14, 13, 12, 11, 10, 9, 8, 3, 2, 1, 
                                          0, 7, 6, 5, 4));
      __m512i min3 = _mm512_min_epi8(v2, perm3);
      __m512i max3 = _mm512_max_epi8(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi8(max3, 0x70000ff00ff000f, min3);
      
      /* Pairs: ([63,63], [60,62], [61,61], [59,59], [58,58], [57,57], 
                 [56,56], [31,55], [30,54], [29,53], [28,52], [27,51], 
                 [26,50], [25,49], [24,48], [47,47], [46,46], [45,45], 
                 [44,44], [43,43], [42,42], [41,41], [40,40], [15,39], 
                 [14,38], [13,37], [12,36], [11,35], [10,34], [9,33], [8,32], 
                 [23,23], [22,22], [21,21], [20,20], [19,19], [18,18], 
                 [17,17], [16,16], [7,7], [6,6], [5,5], [4,4], [1,3], [0,2]) 
                 */
      /* Perm:  (63, 60, 61, 62, 59, 58, 57, 56, 31, 30, 29, 28, 27, 26, 25, 
                 24, 47, 46, 45, 44, 43, 42, 41, 40, 15, 14, 13, 12, 11, 10,  
                 9,  8, 55, 54, 53, 52, 51, 50, 49, 48, 23, 22, 21, 20, 19, 
                 18, 17, 16, 39, 38, 37, 36, 35, 34, 33, 32,  7,  6,  5,  4,  
                 1,  0,  3,  2) */
      __m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 60, 61, 62, 
                                              59, 58, 57, 56, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 7, 6, 5, 4, 1, 
                                              0, 3, 2), v3);
      __m512i min4 = _mm512_min_epi8(v3, perm4);
      __m512i max4 = _mm512_max_epi8(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi8(max4, 0x10000000ff00ff03, min4);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [59,59], [58,58], 
                 [57,57], [56,56], [47,55], [46,54], [45,53], [44,52], 
                 [43,51], [42,50], [41,49], [40,48], [31,39], [30,38], 
                 [29,37], [28,36], [27,35], [26,34], [25,33], [24,32], 
                 [15,23], [14,22], [13,21], [12,20], [11,19], [10,18], 
                 [9,17], [8,16], [7,7], [6,6], [5,5], [4,4], [3,3], [2,2], 
                 [0,1]) */
      /* Perm:  (63, 62, 61, 60, 59, 58, 57, 56, 47, 46, 45, 44, 43, 42, 41, 
                 40, 55, 54, 53, 52, 51, 50, 49, 48, 31, 30, 29, 28, 27, 26, 
                 25, 24, 39, 38, 37, 36, 35, 34, 33, 32, 15, 14, 13, 12, 11, 
                 10,  9,  8, 23, 22, 21, 20, 19, 18, 17, 16,  7,  6,  5,  4,  
                 3,  2,  0,  1) */
      __m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 7, 6, 5, 4, 3, 
                                              2, 0, 1), v4);
      __m512i min5 = _mm512_min_epi8(v4, perm5);
      __m512i max5 = _mm512_max_epi8(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi8(max5, 0xff00ff00ff01, min5);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [59,59], [58,58], 
                 [57,57], [56,56], [51,55], [50,54], [49,53], [48,52], 
                 [43,47], [42,46], [41,45], [40,44], [35,39], [34,38], 
                 [33,37], [32,36], [27,31], [26,30], [25,29], [24,28], 
                 [19,23], [18,22], [17,21], [16,20], [11,15], [10,14], 
                 [9,13], [8,12], [7,7], [6,6], [5,5], [4,4], [3,3], [2,2], 
                 [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 59, 58, 57, 56, 51, 50, 49, 48, 55, 54, 53, 
                 52, 43, 42, 41, 40, 47, 46, 45, 44, 35, 34, 33, 32, 39, 38, 
                 37, 36, 27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 
                 22, 21, 20, 11, 10,  9,  8, 15, 14, 13, 12,  7,  6,  5,  4,  
                 3,  2,  1,  0) */
      __m512i perm6 = _mm512_shuffle_epi8(v5, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 51, 50, 49, 48, 55, 
                                          54, 53, 52, 43, 42, 41, 40, 47, 46, 
                                          45, 44, 35, 34, 33, 32, 39, 38, 37, 
                                          36, 27, 26, 25, 24, 31, 30, 29, 28, 
                                          19, 18, 17, 16, 23, 22, 21, 20, 11, 
                                          10, 9, 8, 15, 14, 13, 12, 7, 6, 5, 
                                          4, 3, 2, 1, 0));
      __m512i min6 = _mm512_min_epi8(v5, perm6);
      __m512i max6 = _mm512_max_epi8(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi8(max6, 0xf0f0f0f0f0f00, min6);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [31,59], [30,58], 
                 [29,57], [28,56], [55,55], [54,54], [53,53], [52,52], 
                 [23,51], [22,50], [21,49], [20,48], [47,47], [46,46], 
                 [45,45], [44,44], [15,43], [14,42], [13,41], [12,40], 
                 [39,39], [38,38], [37,37], [36,36], [7,35], [6,34], [5,33], 
                 [4,32], [27,27], [26,26], [25,25], [24,24], [19,19], 
                 [18,18], [17,17], [16,16], [11,11], [10,10], [9,9], [8,8], 
                 [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 31, 30, 29, 28, 55, 54, 53, 52, 23, 22, 21, 
                 20, 47, 46, 45, 44, 15, 14, 13, 12, 39, 38, 37, 36,  7,  6,  
                 5,  4, 59, 58, 57, 56, 27, 26, 25, 24, 51, 50, 49, 48, 19, 
                 18, 17, 16, 43, 42, 41, 40, 11, 10,  9,  8, 35, 34, 33, 32,  
                 3,  2,  1,  0) */
      __m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              31, 30, 29, 28, 55, 54, 53, 52, 
                                              23, 22, 21, 20, 47, 46, 45, 44, 
                                              15, 14, 13, 12, 39, 38, 37, 36, 
                                              7, 6, 5, 4, 59, 58, 57, 56, 27, 
                                              26, 25, 24, 51, 50, 49, 48, 19, 
                                              18, 17, 16, 43, 42, 41, 40, 11, 
                                              10, 9, 8, 35, 34, 33, 32, 3, 2, 
                                              1, 0), v6);
      __m512i min7 = _mm512_min_epi8(v6, perm7);
      __m512i max7 = _mm512_max_epi8(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi8(max7, 0xf0f0f0f0, min7);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [47,59], [46,58], 
                 [45,57], [44,56], [55,55], [54,54], [53,53], [52,52], 
                 [39,51], [38,50], [37,49], [36,48], [31,43], [30,42], 
                 [29,41], [28,40], [23,35], [22,34], [21,33], [20,32], 
                 [15,27], [14,26], [13,25], [12,24], [7,19], [6,18], [5,17], 
                 [4,16], [11,11], [10,10], [9,9], [8,8], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (63, 62, 61, 60, 47, 46, 45, 44, 55, 54, 53, 52, 39, 38, 37, 
                 36, 59, 58, 57, 56, 31, 30, 29, 28, 51, 50, 49, 48, 23, 22, 
                 21, 20, 43, 42, 41, 40, 15, 14, 13, 12, 35, 34, 33, 32,  7,  
                 6,  5,  4, 27, 26, 25, 24, 11, 10,  9,  8, 19, 18, 17, 16,  
                 3,  2,  1,  0) */
      __m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              47, 46, 45, 44, 55, 54, 53, 52, 
                                              39, 38, 37, 36, 59, 58, 57, 56, 
                                              31, 30, 29, 28, 51, 50, 49, 48, 
                                              23, 22, 21, 20, 43, 42, 41, 40, 
                                              15, 14, 13, 12, 35, 34, 33, 32, 
                                              7, 6, 5, 4, 27, 26, 25, 24, 11, 
                                              10, 9, 8, 19, 18, 17, 16, 3, 2, 
                                              1, 0), v7);
      __m512i min8 = _mm512_min_epi8(v7, perm8);
      __m512i max8 = _mm512_max_epi8(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi8(max8, 0xf0f0f0f0f0f0, min8);
      
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
      __m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              55, 54, 53, 52, 59, 58, 57, 56, 
                                              47, 46, 45, 44, 51, 50, 49, 48, 
                                              39, 38, 37, 36, 43, 42, 41, 40, 
                                              31, 30, 29, 28, 35, 34, 33, 32, 
                                              23, 22, 21, 20, 27, 26, 25, 24, 
                                              15, 14, 13, 12, 19, 18, 17, 16, 
                                              7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 
                                              1, 0), v8);
      __m512i min9 = _mm512_min_epi8(v8, perm9);
      __m512i max9 = _mm512_max_epi8(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi8(max9, 0xf0f0f0f0f0f0f0, min9);
      
      /* Pairs: ([63,63], [62,62], [61,61], [60,60], [57,59], [56,58], 
                 [53,55], [52,54], [49,51], [48,50], [45,47], [44,46], 
                 [41,43], [40,42], [37,39], [36,38], [33,35], [32,34], 
                 [29,31], [28,30], [25,27], [24,26], [21,23], [20,22], 
                 [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], 
                 [4,6], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (63, 62, 61, 60, 57, 56, 59, 58, 53, 52, 55, 54, 49, 48, 51, 
                 50, 45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 
                 35, 34, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 
                 16, 19, 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  
                 3,  2,  1,  0) */
      __m512i perm10 = _mm512_shuffle_epi8(v9, _mm512_set_epi8(63, 62, 61, 
                                           60, 57, 56, 59, 58, 53, 52, 55, 
                                           54, 49, 48, 51, 50, 45, 44, 47, 
                                           46, 41, 40, 43, 42, 37, 36, 39, 
                                           38, 33, 32, 35, 34, 29, 28, 31, 
                                           30, 25, 24, 27, 26, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 5, 4, 7, 6, 3, 
                                           2, 1, 0));
      __m512i min10 = _mm512_min_epi8(v9, perm10);
      __m512i max10 = _mm512_max_epi8(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi8(max10, 0x333333333333330, min10);
      
      /* Pairs: ([63,63], [62,62], [31,61], [30,60], [59,59], [58,58], 
                 [27,57], [26,56], [55,55], [54,54], [23,53], [22,52], 
                 [51,51], [50,50], [19,49], [18,48], [47,47], [46,46], 
                 [15,45], [14,44], [43,43], [42,42], [11,41], [10,40], 
                 [39,39], [38,38], [7,37], [6,36], [35,35], [34,34], [3,33], 
                 [2,32], [29,29], [28,28], [25,25], [24,24], [21,21], 
                 [20,20], [17,17], [16,16], [13,13], [12,12], [9,9], [8,8], 
                 [5,5], [4,4], [1,1], [0,0]) */
      /* Perm:  (63, 62, 31, 30, 59, 58, 27, 26, 55, 54, 23, 22, 51, 50, 19, 
                 18, 47, 46, 15, 14, 43, 42, 11, 10, 39, 38,  7,  6, 35, 34,  
                 3,  2, 61, 60, 29, 28, 57, 56, 25, 24, 53, 52, 21, 20, 49, 
                 48, 17, 16, 45, 44, 13, 12, 41, 40,  9,  8, 37, 36,  5,  4, 
                 33, 32,  1,  0) */
      __m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 31, 
                                               30, 59, 58, 27, 26, 55, 54, 
                                               23, 22, 51, 50, 19, 18, 47, 
                                               46, 15, 14, 43, 42, 11, 10, 
                                               39, 38, 7, 6, 35, 34, 3, 2, 
                                               61, 60, 29, 28, 57, 56, 25, 
                                               24, 53, 52, 21, 20, 49, 48, 
                                               17, 16, 45, 44, 13, 12, 41, 
                                               40, 9, 8, 37, 36, 5, 4, 33, 
                                               32, 1, 0), v10);
      __m512i min11 = _mm512_min_epi8(v10, perm11);
      __m512i max11 = _mm512_max_epi8(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi8(max11, 0xcccccccc, min11);
      
      /* Pairs: ([63,63], [62,62], [47,61], [46,60], [59,59], [58,58], 
                 [43,57], [42,56], [55,55], [54,54], [39,53], [38,52], 
                 [51,51], [50,50], [35,49], [34,48], [31,45], [30,44], 
                 [27,41], [26,40], [23,37], [22,36], [19,33], [18,32], 
                 [15,29], [14,28], [11,25], [10,24], [7,21], [6,20], [3,17], 
                 [2,16], [13,13], [12,12], [9,9], [8,8], [5,5], [4,4], [1,1], 
                 [0,0]) */
      /* Perm:  (63, 62, 47, 46, 59, 58, 43, 42, 55, 54, 39, 38, 51, 50, 35, 
                 34, 61, 60, 31, 30, 57, 56, 27, 26, 53, 52, 23, 22, 49, 48, 
                 19, 18, 45, 44, 15, 14, 41, 40, 11, 10, 37, 36,  7,  6, 33, 
                 32,  3,  2, 29, 28, 13, 12, 25, 24,  9,  8, 21, 20,  5,  4, 
                 17, 16,  1,  0) */
      __m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 47, 
                                               46, 59, 58, 43, 42, 55, 54, 
                                               39, 38, 51, 50, 35, 34, 61, 
                                               60, 31, 30, 57, 56, 27, 26, 
                                               53, 52, 23, 22, 49, 48, 19, 
                                               18, 45, 44, 15, 14, 41, 40, 
                                               11, 10, 37, 36, 7, 6, 33, 32, 
                                               3, 2, 29, 28, 13, 12, 25, 24, 
                                               9, 8, 21, 20, 5, 4, 17, 16, 1, 
                                               0), v11);
      __m512i min12 = _mm512_min_epi8(v11, perm12);
      __m512i max12 = _mm512_max_epi8(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi8(max12, 0xcccccccccccc, min12);
      
      /* Pairs: ([63,63], [62,62], [55,61], [54,60], [59,59], [58,58], 
                 [51,57], [50,56], [47,53], [46,52], [43,49], [42,48], 
                 [39,45], [38,44], [35,41], [34,40], [31,37], [30,36], 
                 [27,33], [26,32], [23,29], [22,28], [19,25], [18,24], 
                 [15,21], [14,20], [11,17], [10,16], [7,13], [6,12], [3,9], 
                 [2,8], [5,5], [4,4], [1,1], [0,0]) */
      /* Perm:  (63, 62, 55, 54, 59, 58, 51, 50, 61, 60, 47, 46, 57, 56, 43, 
                 42, 53, 52, 39, 38, 49, 48, 35, 34, 45, 44, 31, 30, 41, 40, 
                 27, 26, 37, 36, 23, 22, 33, 32, 19, 18, 29, 28, 15, 14, 25, 
                 24, 11, 10, 21, 20,  7,  6, 17, 16,  3,  2, 13, 12,  5,  4,  
                 9,  8,  1,  0) */
      __m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 55, 
                                               54, 59, 58, 51, 50, 61, 60, 
                                               47, 46, 57, 56, 43, 42, 53, 
                                               52, 39, 38, 49, 48, 35, 34, 
                                               45, 44, 31, 30, 41, 40, 27, 
                                               26, 37, 36, 23, 22, 33, 32, 
                                               19, 18, 29, 28, 15, 14, 25, 
                                               24, 11, 10, 21, 20, 7, 6, 17, 
                                               16, 3, 2, 13, 12, 5, 4, 9, 8, 
                                               1, 0), v12);
      __m512i min13 = _mm512_min_epi8(v12, perm13);
      __m512i max13 = _mm512_max_epi8(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi8(max13, 0xcccccccccccccc, min13);
      
      /* Pairs: ([63,63], [62,62], [59,61], [58,60], [55,57], [54,56], 
                 [51,53], [50,52], [47,49], [46,48], [43,45], [42,44], 
                 [39,41], [38,40], [35,37], [34,36], [31,33], [30,32], 
                 [27,29], [26,28], [23,25], [22,24], [19,21], [18,20], 
                 [15,17], [14,16], [11,13], [10,12], [7,9], [6,8], [3,5], 
                 [2,4], [1,1], [0,0]) */
      /* Perm:  (63, 62, 59, 58, 61, 60, 55, 54, 57, 56, 51, 50, 53, 52, 47, 
                 46, 49, 48, 43, 42, 45, 44, 39, 38, 41, 40, 35, 34, 37, 36, 
                 31, 30, 33, 32, 27, 26, 29, 28, 23, 22, 25, 24, 19, 18, 21, 
                 20, 15, 14, 17, 16, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  
                 5,  4,  1,  0) */
      __m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 59, 
                                               58, 61, 60, 55, 54, 57, 56, 
                                               51, 50, 53, 52, 47, 46, 49, 
                                               48, 43, 42, 45, 44, 39, 38, 
                                               41, 40, 35, 34, 37, 36, 31, 
                                               30, 33, 32, 27, 26, 29, 28, 
                                               23, 22, 25, 24, 19, 18, 21, 
                                               20, 15, 14, 17, 16, 11, 10, 
                                               13, 12, 7, 6, 9, 8, 3, 2, 5, 
                                               4, 1, 0), v13);
      __m512i min14 = _mm512_min_epi8(v13, perm14);
      __m512i max14 = _mm512_max_epi8(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi8(max14, 0xccccccccccccccc, min14);
      
      /* Pairs: ([63,63], [62,62], [60,61], [58,59], [56,57], [54,55], 
                 [52,53], [50,51], [48,49], [46,47], [44,45], [42,43], 
                 [40,41], [38,39], [36,37], [34,35], [32,33], [30,31], 
                 [28,29], [26,27], [24,25], [22,23], [20,21], [18,19], 
                 [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], [4,5], 
                 [2,3], [1,1], [0,0]) */
      /* Perm:  (63, 62, 60, 61, 58, 59, 56, 57, 54, 55, 52, 53, 50, 51, 48, 
                 49, 46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 
                 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 
                 19, 16, 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  
                 2,  3,  1,  0) */
      __m512i perm15 = _mm512_shuffle_epi8(v14, _mm512_set_epi8(63, 62, 60, 
                                           61, 58, 59, 56, 57, 54, 55, 52, 
                                           53, 50, 51, 48, 49, 46, 47, 44, 
                                           45, 42, 43, 40, 41, 38, 39, 36, 
                                           37, 34, 35, 32, 33, 30, 31, 28, 
                                           29, 26, 27, 24, 25, 22, 23, 20, 
                                           21, 18, 19, 16, 17, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 1, 0));
      __m512i min15 = _mm512_min_epi8(v14, perm15);
      __m512i max15 = _mm512_max_epi8(v14, perm15);
      __m512i v15 = _mm512_mask_mov_epi8(max15, 0x1555555555555554, min15);
      
      /* Pairs: ([63,63], [31,62], [61,61], [29,60], [59,59], [27,58], 
                 [57,57], [25,56], [55,55], [23,54], [53,53], [21,52], 
                 [51,51], [19,50], [49,49], [17,48], [47,47], [15,46], 
                 [45,45], [13,44], [43,43], [11,42], [41,41], [9,40], 
                 [39,39], [7,38], [37,37], [5,36], [35,35], [3,34], [33,33], 
                 [1,32], [30,30], [28,28], [26,26], [24,24], [22,22], 
                 [20,20], [18,18], [16,16], [14,14], [12,12], [10,10], [8,8], 
                 [6,6], [4,4], [2,2], [0,0]) */
      /* Perm:  (63, 31, 61, 29, 59, 27, 57, 25, 55, 23, 53, 21, 51, 19, 49, 
                 17, 47, 15, 45, 13, 43, 11, 41,  9, 39,  7, 37,  5, 35,  3, 
                 33,  1, 62, 30, 60, 28, 58, 26, 56, 24, 54, 22, 52, 20, 50, 
                 18, 48, 16, 46, 14, 44, 12, 42, 10, 40,  8, 38,  6, 36,  4, 
                 34,  2, 32,  0) */
      __m512i perm16 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 31, 61, 
                                               29, 59, 27, 57, 25, 55, 23, 
                                               53, 21, 51, 19, 49, 17, 47, 
                                               15, 45, 13, 43, 11, 41, 9, 39, 
                                               7, 37, 5, 35, 3, 33, 1, 62, 
                                               30, 60, 28, 58, 26, 56, 24, 
                                               54, 22, 52, 20, 50, 18, 48, 
                                               16, 46, 14, 44, 12, 42, 10, 
                                               40, 8, 38, 6, 36, 4, 34, 2, 
                                               32, 0), v15);
      __m512i min16 = _mm512_min_epi8(v15, perm16);
      __m512i max16 = _mm512_max_epi8(v15, perm16);
      __m512i v16 = _mm512_mask_mov_epi8(max16, 0xaaaaaaaa, min16);
      
      /* Pairs: ([63,63], [47,62], [61,61], [45,60], [59,59], [43,58], 
                 [57,57], [41,56], [55,55], [39,54], [53,53], [37,52], 
                 [51,51], [35,50], [49,49], [33,48], [31,46], [29,44], 
                 [27,42], [25,40], [23,38], [21,36], [19,34], [17,32], 
                 [15,30], [13,28], [11,26], [9,24], [7,22], [5,20], [3,18], 
                 [1,16], [14,14], [12,12], [10,10], [8,8], [6,6], [4,4], 
                 [2,2], [0,0]) */
      /* Perm:  (63, 47, 61, 45, 59, 43, 57, 41, 55, 39, 53, 37, 51, 35, 49, 
                 33, 62, 31, 60, 29, 58, 27, 56, 25, 54, 23, 52, 21, 50, 19, 
                 48, 17, 46, 15, 44, 13, 42, 11, 40,  9, 38,  7, 36,  5, 34,  
                 3, 32,  1, 30, 14, 28, 12, 26, 10, 24,  8, 22,  6, 20,  4, 
                 18,  2, 16,  0) */
      __m512i perm17 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 47, 61, 
                                               45, 59, 43, 57, 41, 55, 39, 
                                               53, 37, 51, 35, 49, 33, 62, 
                                               31, 60, 29, 58, 27, 56, 25, 
                                               54, 23, 52, 21, 50, 19, 48, 
                                               17, 46, 15, 44, 13, 42, 11, 
                                               40, 9, 38, 7, 36, 5, 34, 3, 
                                               32, 1, 30, 14, 28, 12, 26, 10, 
                                               24, 8, 22, 6, 20, 4, 18, 2, 
                                               16, 0), v16);
      __m512i min17 = _mm512_min_epi8(v16, perm17);
      __m512i max17 = _mm512_max_epi8(v16, perm17);
      __m512i v17 = _mm512_mask_mov_epi8(max17, 0xaaaaaaaaaaaa, min17);
      
      /* Pairs: ([63,63], [55,62], [61,61], [53,60], [59,59], [51,58], 
                 [57,57], [49,56], [47,54], [45,52], [43,50], [41,48], 
                 [39,46], [37,44], [35,42], [33,40], [31,38], [29,36], 
                 [27,34], [25,32], [23,30], [21,28], [19,26], [17,24], 
                 [15,22], [13,20], [11,18], [9,16], [7,14], [5,12], [3,10], 
                 [1,8], [6,6], [4,4], [2,2], [0,0]) */
      /* Perm:  (63, 55, 61, 53, 59, 51, 57, 49, 62, 47, 60, 45, 58, 43, 56, 
                 41, 54, 39, 52, 37, 50, 35, 48, 33, 46, 31, 44, 29, 42, 27, 
                 40, 25, 38, 23, 36, 21, 34, 19, 32, 17, 30, 15, 28, 13, 26, 
                 11, 24,  9, 22,  7, 20,  5, 18,  3, 16,  1, 14,  6, 12,  4, 
                 10,  2,  8,  0) */
      __m512i perm18 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 55, 61, 
                                               53, 59, 51, 57, 49, 62, 47, 
                                               60, 45, 58, 43, 56, 41, 54, 
                                               39, 52, 37, 50, 35, 48, 33, 
                                               46, 31, 44, 29, 42, 27, 40, 
                                               25, 38, 23, 36, 21, 34, 19, 
                                               32, 17, 30, 15, 28, 13, 26, 
                                               11, 24, 9, 22, 7, 20, 5, 18, 
                                               3, 16, 1, 14, 6, 12, 4, 10, 2, 
                                               8, 0), v17);
      __m512i min18 = _mm512_min_epi8(v17, perm18);
      __m512i max18 = _mm512_max_epi8(v17, perm18);
      __m512i v18 = _mm512_mask_mov_epi8(max18, 0xaaaaaaaaaaaaaa, min18);
      
      /* Pairs: ([63,63], [59,62], [61,61], [57,60], [55,58], [53,56], 
                 [51,54], [49,52], [47,50], [45,48], [43,46], [41,44], 
                 [39,42], [37,40], [35,38], [33,36], [31,34], [29,32], 
                 [27,30], [25,28], [23,26], [21,24], [19,22], [17,20], 
                 [15,18], [13,16], [11,14], [9,12], [7,10], [5,8], [3,6], 
                 [1,4], [2,2], [0,0]) */
      /* Perm:  (63, 59, 61, 57, 62, 55, 60, 53, 58, 51, 56, 49, 54, 47, 52, 
                 45, 50, 43, 48, 41, 46, 39, 44, 37, 42, 35, 40, 33, 38, 31, 
                 36, 29, 34, 27, 32, 25, 30, 23, 28, 21, 26, 19, 24, 17, 22, 
                 15, 20, 13, 18, 11, 16,  9, 14,  7, 12,  5, 10,  3,  8,  1,  
                 6,  2,  4,  0) */
      __m512i perm19 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 59, 61, 
                                               57, 62, 55, 60, 53, 58, 51, 
                                               56, 49, 54, 47, 52, 45, 50, 
                                               43, 48, 41, 46, 39, 44, 37, 
                                               42, 35, 40, 33, 38, 31, 36, 
                                               29, 34, 27, 32, 25, 30, 23, 
                                               28, 21, 26, 19, 24, 17, 22, 
                                               15, 20, 13, 18, 11, 16, 9, 14, 
                                               7, 12, 5, 10, 3, 8, 1, 6, 2, 
                                               4, 0), v18);
      __m512i min19 = _mm512_min_epi8(v18, perm19);
      __m512i max19 = _mm512_max_epi8(v18, perm19);
      __m512i v19 = _mm512_mask_mov_epi8(max19, 0xaaaaaaaaaaaaaaa, min19);
      
      /* Pairs: ([63,63], [61,62], [59,60], [57,58], [55,56], [53,54], 
                 [51,52], [49,50], [47,48], [45,46], [43,44], [41,42], 
                 [39,40], [37,38], [35,36], [33,34], [31,32], [29,30], 
                 [27,28], [25,26], [23,24], [21,22], [19,20], [17,18], 
                 [15,16], [13,14], [11,12], [9,10], [7,8], [5,6], [3,4], 
                 [1,2], [0,0]) */
      /* Perm:  (63, 61, 62, 59, 60, 57, 58, 55, 56, 53, 54, 51, 52, 49, 50, 
                 47, 48, 45, 46, 43, 44, 41, 42, 39, 40, 37, 38, 35, 36, 33, 
                 34, 31, 32, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 
                 17, 18, 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  
                 4,  1,  2,  0) */
      __m512i perm20 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 61, 62, 
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
      __m512i v20 = _mm512_mask_mov_epi8(max20, 0x2aaaaaaaaaaaaaaa, min20);
      
      return v20;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
batcher_63_int8_t(int8_t * const arr) 
                             {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = batcher_63_int8_t_vec(v);
      
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


