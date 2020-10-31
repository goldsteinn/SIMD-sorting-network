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
#define N 30
#define SORT_NAME bitonic_32_int8_t

#ifndef _SIMD_SORT_bitonic_32_int8_t_H_
#define _SIMD_SORT_bitonic_32_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 32
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 3 / 72
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX2, AVX512vbmi, AVX
	SIMD Instruction Set(s) Excluded : None
	Aligned Load & Store             : True
	Integer Aligned Load & Store     : True
	Full Load & Store                : True
	Scaled Sorting Network           : True

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
 __m256i __attribute__((const)) 
bitonic_32_int8_t_vec(__m256i v) {
      
      /* Pairs: ([30,31], [28,29], [26,27], [24,25], [22,23], [20,21], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [2,3], [0,1]) */
      /* Perm:  (30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(30, 31, 28, 29, 
                                          26, 27, 24, 25, 22, 23, 20, 21, 18, 
                                          19, 16, 17, 14, 15, 12, 13, 10, 11, 
                                          8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
      __m256i min0 = _mm256_min_epi8(v, perm0);
      __m256i max0 = _mm256_max_epi8(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi8(max0, 0x55555555, min0);
      
      /* Pairs: ([28,31], [29,30], [24,27], [25,26], [20,23], [21,22], 
                 [16,19], [17,18], [12,15], [13,14], [8,11], [9,10], [4,7], 
                 [5,6], [0,3], [1,2]) */
      /* Perm:  (28, 29, 30, 31, 24, 25, 26, 27, 20, 21, 22, 23, 16, 17, 18, 
                 19, 12, 13, 14, 15,  8,  9, 10, 11,  4,  5,  6,  7,  0,  1,  
                 2,  3) */
      __m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(28, 29, 30, 31, 
                                          24, 25, 26, 27, 20, 21, 22, 23, 16, 
                                          17, 18, 19, 12, 13, 14, 15, 8, 9, 
                                          10, 11, 4, 5, 6, 7, 0, 1, 2, 3));
      __m256i min1 = _mm256_min_epi8(v0, perm1);
      __m256i max1 = _mm256_max_epi8(v0, perm1);
      __m256i v1 = _mm256_blend_epi16(max1, min1, 0x55);
      
      /* Pairs: ([30,31], [28,29], [26,27], [24,25], [22,23], [20,21], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [2,3], [0,1]) */
      /* Perm:  (30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(30, 31, 28, 29, 
                                          26, 27, 24, 25, 22, 23, 20, 21, 18, 
                                          19, 16, 17, 14, 15, 12, 13, 10, 11, 
                                          8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
      __m256i min2 = _mm256_min_epi8(v1, perm2);
      __m256i max2 = _mm256_max_epi8(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi8(max2, 0x55555555, min2);
      
      /* Pairs: ([24,31], [25,30], [26,29], [27,28], [16,23], [17,22], 
                 [18,21], [19,20], [8,15], [9,14], [10,13], [11,12], [0,7], 
                 [1,6], [2,5], [3,4]) */
      /* Perm:  (24, 25, 26, 27, 28, 29, 30, 31, 16, 17, 18, 19, 20, 21, 22, 
                 23,  8,  9, 10, 11, 12, 13, 14, 15,  0,  1,  2,  3,  4,  5,  
                 6,  7) */
      __m256i perm3 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(24, 25, 26, 27, 
                                          28, 29, 30, 31, 16, 17, 18, 19, 20, 
                                          21, 22, 23, 8, 9, 10, 11, 12, 13, 
                                          14, 15, 0, 1, 2, 3, 4, 5, 6, 7));
      __m256i min3 = _mm256_min_epi8(v2, perm3);
      __m256i max3 = _mm256_max_epi8(v2, perm3);
      __m256i v3 = _mm256_blend_epi32(max3, min3, 0x55);
      
      /* Pairs: ([29,31], [28,30], [25,27], [24,26], [21,23], [20,22], 
                 [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], 
                 [4,6], [1,3], [0,2]) */
      /* Perm:  (29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 
                 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m256i perm4 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(v3, 
                                             0xb1), 0xb1);
      __m256i min4 = _mm256_min_epi8(v3, perm4);
      __m256i max4 = _mm256_max_epi8(v3, perm4);
      __m256i v4 = _mm256_blend_epi16(max4, min4, 0x55);
      
      /* Pairs: ([30,31], [28,29], [26,27], [24,25], [22,23], [20,21], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [2,3], [0,1]) */
      /* Perm:  (30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(30, 31, 28, 29, 
                                          26, 27, 24, 25, 22, 23, 20, 21, 18, 
                                          19, 16, 17, 14, 15, 12, 13, 10, 11, 
                                          8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
      __m256i min5 = _mm256_min_epi8(v4, perm5);
      __m256i max5 = _mm256_max_epi8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x55555555, min5);
      
      /* Pairs: ([16,31], [17,30], [18,29], [19,28], [20,27], [21,26], 
                 [22,25], [23,24], [0,15], [1,14], [2,13], [3,12], [4,11], 
                 [5,10], [6,9], [7,8]) */
      /* Perm:  (16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 
                 31,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 
                 14, 15) */
      __m256i perm6 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(16, 17, 18, 19, 
                                          20, 21, 22, 23, 24, 25, 26, 27, 28, 
                                          29, 30, 31, 0, 1, 2, 3, 4, 5, 6, 7, 
                                          8, 9, 10, 11, 12, 13, 14, 15));
      __m256i min6 = _mm256_min_epi8(v5, perm6);
      __m256i max6 = _mm256_max_epi8(v5, perm6);
      __m256i v6 = _mm256_blend_epi32(max6, min6, 0x33);
      
      /* Pairs: ([27,31], [26,30], [25,29], [24,28], [19,23], [18,22], 
                 [17,21], [16,20], [11,15], [10,14], [9,13], [8,12], [3,7], 
                 [2,6], [1,5], [0,4]) */
      /* Perm:  (27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 
                 20, 11, 10,  9,  8, 15, 14, 13, 12,  3,  2,  1,  0,  7,  6,  
                 5,  4) */
      __m256i perm7 = _mm256_shuffle_epi32(v6, uint8_t(0xb1));
      __m256i min7 = _mm256_min_epi8(v6, perm7);
      __m256i max7 = _mm256_max_epi8(v6, perm7);
      __m256i v7 = _mm256_blend_epi32(max7, min7, 0x55);
      
      /* Pairs: ([29,31], [28,30], [25,27], [24,26], [21,23], [20,22], 
                 [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], 
                 [4,6], [1,3], [0,2]) */
      /* Perm:  (29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 
                 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m256i perm8 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(v7, 
                                             0xb1), 0xb1);
      __m256i min8 = _mm256_min_epi8(v7, perm8);
      __m256i max8 = _mm256_max_epi8(v7, perm8);
      __m256i v8 = _mm256_blend_epi16(max8, min8, 0x55);
      
      /* Pairs: ([30,31], [28,29], [26,27], [24,25], [22,23], [20,21], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [2,3], [0,1]) */
      /* Perm:  (30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm9 = _mm256_shuffle_epi8(v8, _mm256_set_epi8(30, 31, 28, 29, 
                                          26, 27, 24, 25, 22, 23, 20, 21, 18, 
                                          19, 16, 17, 14, 15, 12, 13, 10, 11, 
                                          8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
      __m256i min9 = _mm256_min_epi8(v8, perm9);
      __m256i max9 = _mm256_max_epi8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0x55555555, min9);
      
      /* Pairs: ([0,31], [1,30], [2,29], [3,28], [4,27], [5,26], [6,25], 
                 [7,24], [8,23], [9,22], [10,21], [11,20], [12,19], [13,18], 
                 [14,17], [15,16]) */
      /* Perm:  ( 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 
                 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 
                 30, 31) */
      __m256i perm10 = _mm256_permutexvar_epi8(_mm256_set_epi8(0, 1, 2, 3, 4, 
                                               5, 6, 7, 8, 9, 10, 11, 12, 13, 
                                               14, 15, 16, 17, 18, 19, 20, 
                                               21, 22, 23, 24, 25, 26, 27, 
                                               28, 29, 30, 31), v9);
      __m256i min10 = _mm256_min_epi8(v9, perm10);
      __m256i max10 = _mm256_max_epi8(v9, perm10);
      __m256i v10 = _mm256_blend_epi32(max10, min10, 0xf);
      
      /* Pairs: ([23,31], [22,30], [21,29], [20,28], [19,27], [18,26], 
                 [17,25], [16,24], [7,15], [6,14], [5,13], [4,12], [3,11], 
                 [2,10], [1,9], [0,8]) */
      /* Perm:  (23, 22, 21, 20, 19, 18, 17, 16, 31, 30, 29, 28, 27, 26, 25, 
                 24,  7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  
                 9,  8) */
      __m256i perm11 = _mm256_shuffle_epi32(v10, uint8_t(0x4e));
      __m256i min11 = _mm256_min_epi8(v10, perm11);
      __m256i max11 = _mm256_max_epi8(v10, perm11);
      __m256i v11 = _mm256_blend_epi32(max11, min11, 0x33);
      
      /* Pairs: ([27,31], [26,30], [25,29], [24,28], [19,23], [18,22], 
                 [17,21], [16,20], [11,15], [10,14], [9,13], [8,12], [3,7], 
                 [2,6], [1,5], [0,4]) */
      /* Perm:  (27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 
                 20, 11, 10,  9,  8, 15, 14, 13, 12,  3,  2,  1,  0,  7,  6,  
                 5,  4) */
      __m256i perm12 = _mm256_shuffle_epi32(v11, uint8_t(0xb1));
      __m256i min12 = _mm256_min_epi8(v11, perm12);
      __m256i max12 = _mm256_max_epi8(v11, perm12);
      __m256i v12 = _mm256_blend_epi32(max12, min12, 0x55);
      
      /* Pairs: ([29,31], [28,30], [25,27], [24,26], [21,23], [20,22], 
                 [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], 
                 [4,6], [1,3], [0,2]) */
      /* Perm:  (29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 
                 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  
                 3,  2) */
      __m256i perm13 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(v12, 
                                              0xb1), 0xb1);
      __m256i min13 = _mm256_min_epi8(v12, perm13);
      __m256i max13 = _mm256_max_epi8(v12, perm13);
      __m256i v13 = _mm256_blend_epi16(max13, min13, 0x55);
      
      /* Pairs: ([30,31], [28,29], [26,27], [24,25], [22,23], [20,21], 
                 [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], 
                 [4,5], [2,3], [0,1]) */
      /* Perm:  (30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  
                 0,  1) */
      __m256i perm14 = _mm256_shuffle_epi8(v13, _mm256_set_epi8(30, 31, 28, 
                                           29, 26, 27, 24, 25, 22, 23, 20, 
                                           21, 18, 19, 16, 17, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 0, 1));
      __m256i min14 = _mm256_min_epi8(v13, perm14);
      __m256i max14 = _mm256_max_epi8(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi8(max14, 0x55555555, min14);
      
      return v14;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bitonic_32_int8_t(int8_t * const arr) 
                             {
      
      __m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
      __m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x3fffffff, arr);
      
      v = bitonic_32_int8_t_vec(v);
      
      _mm256_mask_storeu_epi8((void *)arr, 0x3fffffff, v);
      
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


