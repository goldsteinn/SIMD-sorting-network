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
#define N 24
#define SORT_NAME bitonic_24_int8_t

#ifndef _SIMD_SORT_bitonic_24_int8_t_H_
#define _SIMD_SORT_bitonic_24_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 24
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 3 / 112
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX2, SSE2, AVX
	SIMD Instruction Set(s) Excluded : AVX512*
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
bitonic_24_int8_t_vec(__m256i v) {
      
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
      __m256i min0 = _mm256_min_epi8(v, perm0);
      __m256i max0 = _mm256_max_epi8(v, perm0);
      __m256i v0 = _mm256_blendv_epi8(max0, min0, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 128, 0, 0, 128, 0, 0, 
                                      128, 0, 0, 128, 0, 0, 128, 0, 0, 128, 
                                      0, 0, 128, 0, 0, 128, 0));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [21,23], [22,22], [20,20], [18,19], 
                 [17,17], [15,16], [12,14], [13,13], [11,11], [9,10], [6,8], 
                 [7,7], [3,5], [4,4], [2,2], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 21, 22, 23, 20, 18, 19, 17, 
                 15, 16, 12, 13, 14, 11,  9, 10,  6,  7,  8,  3,  4,  5,  2,  
                 0,  1) */
      __m256i _tmp1 = _mm256_permute4x64_epi64(v0, 0x4e);
      __m256i _tmp2 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(15, 14, 13, 12, 
                                          11, 10, 9, 8, 5, 6, 7, 4, 2, 3, 1, 
                                          128, 128, 12, 13, 14, 11, 9, 10, 6, 
                                          7, 8, 3, 4, 5, 2, 0, 1));
      __m256i _tmp3 = _mm256_shuffle_epi8(_tmp1, _mm256_set_epi8(128, 128, 
                                          128, 128, 128, 128, 128, 128, 128, 
                                          128, 128, 128, 128, 128, 128, 15, 
                                          0, 128, 128, 128, 128, 128, 128, 
                                          128, 128, 128, 128, 128, 128, 128, 
                                          128, 128));
      __m256i perm1 = _mm256_or_si256(_tmp2, _tmp3);
      __m256i min1 = _mm256_min_epi8(v0, perm1);
      __m256i max1 = _mm256_max_epi8(v0, perm1);
      __m256i v1 = _mm256_blendv_epi8(max1, min1, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 0, 0, 
                                      128, 0, 0, 128, 0, 0, 128, 0, 0, 128, 
                                      0, 0, 128, 0, 0, 128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [21,22], [19,20], [18,18], 
                 [16,17], [15,15], [14,14], [12,13], [10,11], [9,9], [8,8], 
                 [6,7], [5,5], [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 21, 22, 19, 20, 18, 16, 
                 17, 15, 14, 12, 13, 10, 11,  9,  8,  6,  7,  5,  3,  4,  1,  
                 2,  0) */
      __m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 21, 22, 19, 20, 
                                          18, 16, 17, 15, 14, 12, 13, 10, 11, 
                                          9, 8, 6, 7, 5, 3, 4, 1, 2, 0));
      __m256i min2 = _mm256_min_epi8(v1, perm2);
      __m256i max2 = _mm256_max_epi8(v1, perm2);
      __m256i v2 = _mm256_blendv_epi8(max2, min2, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 128, 0, 128, 0, 0, 
                                      128, 0, 0, 0, 128, 0, 128, 0, 0, 0, 
                                      128, 0, 0, 128, 0, 128, 0));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [19,23], [20,22], [21,21], [18,18], 
                 [17,17], [12,16], [13,15], [14,14], [11,11], [6,10], [7,9], 
                 [8,8], [1,5], [2,4], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 19, 20, 21, 22, 23, 18, 17, 
                 12, 13, 14, 15, 16, 11,  6,  7,  8,  9, 10,  1,  2,  3,  4,  
                 5,  0) */
      __m256i _tmp4 = _mm256_permute4x64_epi64(v2, 0x4e);
      __m256i _tmp5 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(15, 14, 13, 12, 
                                          11, 10, 9, 8, 3, 4, 5, 6, 7, 2, 1, 
                                          128, 13, 14, 15, 128, 11, 6, 7, 8, 
                                          9, 10, 1, 2, 3, 4, 5, 0));
      __m256i _tmp6 = _mm256_shuffle_epi8(_tmp4, _mm256_set_epi8(128, 128, 
                                          128, 128, 128, 128, 128, 128, 128, 
                                          128, 128, 128, 128, 128, 128, 12, 
                                          128, 128, 128, 0, 128, 128, 128, 
                                          128, 128, 128, 128, 128, 128, 128, 
                                          128, 128));
      __m256i perm3 = _mm256_or_si256(_tmp5, _tmp6);
      __m256i min3 = _mm256_min_epi8(v2, perm3);
      __m256i max3 = _mm256_max_epi8(v2, perm3);
      __m256i v3 = _mm256_blendv_epi8(max3, min3, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 
                                      0, 0, 128, 128, 0, 0, 0, 0, 128, 128, 
                                      0, 0, 0, 128, 128, 0));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [22,23], [19,21], [18,20], [15,17], 
                 [14,16], [12,13], [9,11], [8,10], [6,7], [4,5], [1,3], 
                 [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 19, 18, 21, 20, 15, 
                 14, 17, 16, 12, 13,  9,  8, 11, 10,  6,  7,  4,  5,  1,  0,  
                 3,  2) */
      __m256i _tmp7 = _mm256_permute4x64_epi64(v3, 0x4e);
      __m256i _tmp8 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(15, 14, 13, 12, 
                                          11, 10, 9, 8, 6, 7, 3, 2, 5, 4, 
                                          128, 128, 128, 128, 12, 13, 9, 8, 
                                          11, 10, 6, 7, 4, 5, 1, 0, 3, 2));
      __m256i _tmp9 = _mm256_shuffle_epi8(_tmp7, _mm256_set_epi8(128, 128, 
                                          128, 128, 128, 128, 128, 128, 128, 
                                          128, 128, 128, 128, 128, 15, 14, 1, 
                                          0, 128, 128, 128, 128, 128, 128, 
                                          128, 128, 128, 128, 128, 128, 128, 
                                          128));
      __m256i perm4 = _mm256_or_si256(_tmp8, _tmp9);
      __m256i min4 = _mm256_min_epi8(v3, perm4);
      __m256i max4 = _mm256_max_epi8(v3, perm4);
      __m256i v4 = _mm256_blendv_epi8(max4, min4, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 128, 0, 0, 128, 128, 0, 
                                      0, 128, 128, 0, 128, 0, 0, 128, 128, 0, 
                                      128, 0, 128, 0, 0, 128, 128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [20,21], [18,19], 
                 [16,17], [14,15], [13,13], [12,12], [10,11], [8,9], [7,7], 
                 [6,6], [5,5], [4,4], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 18, 19, 16, 
                 17, 14, 15, 13, 12, 10, 11,  8,  9,  7,  6,  5,  4,  2,  3,  
                 0,  1) */
      __m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 20, 21, 18, 
                                          19, 16, 17, 14, 15, 13, 12, 10, 11, 
                                          8, 9, 7, 6, 5, 4, 2, 3, 0, 1));
      __m256i min5 = _mm256_min_epi8(v4, perm5);
      __m256i max5 = _mm256_max_epi8(v4, perm5);
      __m256i v5 = _mm256_blendv_epi8(max5, min5, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 128, 0, 128, 0, 
                                      128, 0, 128, 0, 0, 0, 128, 0, 128, 0, 
                                      0, 0, 0, 0, 128, 0, 128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [14,23], [15,22], [16,21], [17,20], 
                 [19,19], [18,18], [13,13], [12,12], [11,11], [10,10], [0,9], 
                 [1,8], [2,7], [3,6], [5,5], [4,4]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 14, 15, 16, 17, 19, 18, 20, 
                 21, 22, 23, 13, 12, 11, 10,  0,  1,  2,  3,  5,  4,  6,  7,  
                 8,  9) */
      __m256i _tmp10 = _mm256_permute4x64_epi64(v5, 0x4e);
      __m256i _tmp11 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(15, 14, 13, 
                                           12, 11, 10, 9, 8, 128, 128, 0, 1, 
                                           3, 2, 4, 5, 128, 128, 13, 12, 11, 
                                           10, 0, 1, 2, 3, 5, 4, 6, 7, 8, 
                                           9));
      __m256i _tmp12 = _mm256_shuffle_epi8(_tmp10, _mm256_set_epi8(128, 128, 
                                           128, 128, 128, 128, 128, 128, 14, 
                                           15, 128, 128, 128, 128, 128, 128, 
                                           6, 7, 128, 128, 128, 128, 128, 
                                           128, 128, 128, 128, 128, 128, 128, 
                                           128, 128));
      __m256i perm6 = _mm256_or_si256(_tmp11, _tmp12);
      __m256i min6 = _mm256_min_epi8(v5, perm6);
      __m256i max6 = _mm256_max_epi8(v5, perm6);
      __m256i v6 = _mm256_blendv_epi8(max6, min6, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 
                                      128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                      128, 128, 128, 128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [21,23], [20,22], [14,19], [15,18], 
                 [13,17], [12,16], [7,11], [6,10], [4,9], [5,8], [1,3], 
                 [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 21, 20, 23, 22, 14, 15, 13, 
                 12, 18, 19, 17, 16,  7,  6,  4,  5, 11, 10,  8,  9,  1,  0,  
                 3,  2) */
      __m256i _tmp13 = _mm256_permute4x64_epi64(v6, 0x4e);
      __m256i _tmp14 = _mm256_shuffle_epi8(v6, _mm256_set_epi8(15, 14, 13, 
                                           12, 11, 10, 9, 8, 5, 4, 7, 6, 128, 
                                           128, 128, 128, 128, 128, 128, 128, 
                                           7, 6, 4, 5, 11, 10, 8, 9, 1, 0, 3, 
                                           2));
      __m256i _tmp15 = _mm256_shuffle_epi8(_tmp13, _mm256_set_epi8(128, 128, 
                                           128, 128, 128, 128, 128, 128, 128, 
                                           128, 128, 128, 14, 15, 13, 12, 2, 
                                           3, 1, 0, 128, 128, 128, 128, 128, 
                                           128, 128, 128, 128, 128, 128, 
                                           128));
      __m256i perm7 = _mm256_or_si256(_tmp14, _tmp15);
      __m256i min7 = _mm256_min_epi8(v6, perm7);
      __m256i max7 = _mm256_max_epi8(v6, perm7);
      __m256i v7 = _mm256_blendv_epi8(max7, min7, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 
                                      128, 128, 128, 128, 0, 0, 0, 0, 128, 
                                      128, 128, 128, 0, 0, 128, 128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [22,23], [20,21], [16,19], [17,18], 
                 [13,15], [12,14], [9,11], [8,10], [4,7], [5,6], [2,3], 
                 [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 21, 16, 17, 18, 
                 19, 13, 12, 15, 14,  9,  8, 11, 10,  4,  5,  6,  7,  2,  3,  
                 0,  1) */
      __m256i perm8 = _mm256_shuffle_epi8(v7, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 22, 23, 20, 21, 16, 
                                          17, 18, 19, 13, 12, 15, 14, 9, 8, 
                                          11, 10, 4, 5, 6, 7, 2, 3, 0, 1));
      __m256i min8 = _mm256_min_epi8(v7, perm8);
      __m256i max8 = _mm256_max_epi8(v7, perm8);
      __m256i v8 = _mm256_blendv_epi8(max8, min8, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 128, 0, 128, 0, 0, 128, 
                                      128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 
                                      0, 128, 128, 0, 128, 0, 128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  3,  2,  
                 1,  0) */
      __m256i perm9 = _mm256_shuffle_epi8(v8, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 18, 
                                          19, 16, 17, 14, 15, 12, 13, 10, 11, 
                                          8, 9, 6, 7, 4, 5, 3, 2, 1, 0));
      __m256i min9 = _mm256_min_epi8(v8, perm9);
      __m256i max9 = _mm256_max_epi8(v8, perm9);
      __m256i v9 = _mm256_blendv_epi8(max9, min9, _mm256_set_epi8(0, 0, 0, 0, 
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 128, 
                                      0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 
                                      0, 128, 0, 0, 0, 0));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [4,23], [5,22], [6,21], [7,20], [8,19], 
                 [9,18], [10,17], [11,16], [15,15], [14,14], [13,13], 
                 [12,12], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24,  4,  5,  6,  7,  8,  9, 10, 
                 11, 15, 14, 13, 12, 16, 17, 18, 19, 20, 21, 22, 23,  3,  2,  
                 1,  0) */
      __m256i _tmp16 = _mm256_permute4x64_epi64(v9, 0x4e);
      __m256i _tmp17 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(15, 14, 13, 
                                           12, 11, 10, 9, 8, 128, 128, 128, 
                                           128, 128, 128, 128, 128, 15, 14, 
                                           13, 12, 128, 128, 128, 128, 128, 
                                           128, 128, 128, 3, 2, 1, 0));
      __m256i _tmp18 = _mm256_shuffle_epi8(_tmp16, _mm256_set_epi8(128, 128, 
                                           128, 128, 128, 128, 128, 128, 4, 
                                           5, 6, 7, 8, 9, 10, 11, 128, 128, 
                                           128, 128, 0, 1, 2, 3, 4, 5, 6, 7, 
                                           128, 128, 128, 128));
      __m256i perm10 = _mm256_or_si256(_tmp17, _tmp18);
      __m256i min10 = _mm256_min_epi8(v9, perm10);
      __m256i max10 = _mm256_max_epi8(v9, perm10);
      __m256i v10 = _mm256_blend_epi32(max10, min10, 0x6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [19,23], [18,22], [17,21], [16,20], 
                 [4,15], [5,14], [6,13], [7,12], [3,11], [2,10], [1,9], 
                 [0,8]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 19, 18, 17, 16, 23, 22, 21, 
                 20,  4,  5,  6,  7,  3,  2,  1,  0, 12, 13, 14, 15, 11, 10,  
                 9,  8) */
      __m256i perm11 = _mm256_shuffle_epi8(v10, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 19, 18, 17, 
                                           16, 23, 22, 21, 20, 4, 5, 6, 7, 3, 
                                           2, 1, 0, 12, 13, 14, 15, 11, 10, 
                                           9, 8));
      __m256i min11 = _mm256_min_epi8(v10, perm11);
      __m256i max11 = _mm256_max_epi8(v10, perm11);
      __m256i v11 = _mm256_blend_epi32(max11, min11, 0x13);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [21,23], [20,22], [17,19], [16,18], 
                 [8,15], [9,14], [10,13], [11,12], [3,7], [2,6], [1,5], 
                 [0,4]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 21, 20, 23, 22, 17, 16, 19, 
                 18,  8,  9, 10, 11, 12, 13, 14, 15,  3,  2,  1,  0,  7,  6,  
                 5,  4) */
      __m256i perm12 = _mm256_shuffle_epi8(v11, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 8, 9, 10, 11, 
                                           12, 13, 14, 15, 3, 2, 1, 0, 7, 6, 
                                           5, 4));
      __m256i min12 = _mm256_min_epi8(v11, perm12);
      __m256i max12 = _mm256_max_epi8(v11, perm12);
      __m256i v12 = _mm256_blendv_epi8(max12, min12, _mm256_set_epi8(0, 0, 0, 
                                       0, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 
                                       128, 128, 0, 0, 0, 0, 128, 128, 128, 
                                       128, 0, 0, 0, 0, 128, 128, 128, 
                                       128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [22,23], [20,21], [18,19], [16,17], 
                 [13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], 
                 [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 21, 18, 19, 16, 
                 17, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m256i perm13 = _mm256_shuffle_epi8(v12, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 22, 23, 20, 
                                           21, 18, 19, 16, 17, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 
                                           0, 3, 2));
      __m256i min13 = _mm256_min_epi8(v12, perm13);
      __m256i max13 = _mm256_max_epi8(v12, perm13);
      __m256i v13 = _mm256_blendv_epi8(max13, min13, _mm256_set_epi8(0, 0, 0, 
                                       0, 0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 
                                       0, 128, 0, 0, 128, 128, 0, 0, 128, 
                                       128, 0, 0, 128, 128, 0, 0, 128, 
                                       128));
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [14,15], [12,13], 
                 [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm14 = _mm256_shuffle_epi8(v13, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 21, 
                                           20, 19, 18, 17, 16, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 0, 1));
      __m256i min14 = _mm256_min_epi8(v13, perm14);
      __m256i max14 = _mm256_max_epi8(v13, perm14);
      __m256i v14 = _mm256_blendv_epi8(max14, min14, _mm256_set_epi8(0, 0, 0, 
                                       0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                                       0, 128, 0, 128, 0, 128, 0, 128, 0, 
                                       128, 0, 128, 0, 128, 0, 128));
      
      return v14;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bitonic_24_int8_t(int8_t * const arr) 
                             {
      
      __m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
      asm volatile("vpblendd %[load_mask], (%[arr]), %[fill_v], %[fill_v]\n"
                   : [ fill_v ] "+x" (_tmp0)
                   : [ arr ] "r" (arr), [ load_mask ] "i" (0x3f)
                   :);
      __m256i v = _tmp0;
      fill_works(v);
      v = bitonic_24_int8_t_vec(v);
      
      fill_works(v);_mm256_maskstore_epi32((int32_t * const)arr, 
                                            _mm256_set_epi32(0x0, 0x0, 
                                            0x80000000, 0x80000000, 
                                            0x80000000, 0x80000000, 
                                            0x80000000, 0x80000000), v);
      
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


