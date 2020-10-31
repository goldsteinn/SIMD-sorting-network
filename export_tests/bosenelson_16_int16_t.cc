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
#define N 16
#define SORT_NAME bosenelson_16_int16_t

#ifndef _SIMD_SORT_bosenelson_16_int16_t_H_
#define _SIMD_SORT_bosenelson_16_int16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 16
	Underlying Sort Type             : int16_t
	Network Generation Algorithm     : bosenelson
	Network Depth                    : 15
	SIMD Instructions                : 2 / 74
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX2, SSE2, AVX512bw, AVX512vl, AVX
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
      int i = N;for (; i < 16; ++i) {
          assert(t.arr[i] == int16_t(0x7fff));
 }
}

/* SIMD Sort */
 __m256i __attribute__((const)) 
bosenelson_16_int16_t_vec(__m256i v) {
      
      /* Pairs: ([14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], 
                 [0,1]) */
      /* Perm:  (14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  
                 1) */
      __m256i perm0 = _mm256_shufflehi_epi16(_mm256_shufflelo_epi16(v, 0xb1), 
                                             0xb1);
      __m256i min0 = _mm256_min_epi16(v, perm0);
      __m256i max0 = _mm256_max_epi16(v, perm0);
      __m256i v0 = _mm256_blend_epi16(max0, min0, 0x55);
      
      /* Pairs: ([13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], 
                 [0,2]) */
      /* Perm:  (13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  
                 2) */
      __m256i perm1 = _mm256_shuffle_epi32(v0, uint8_t(0xb1));
      __m256i min1 = _mm256_min_epi16(v0, perm1);
      __m256i max1 = _mm256_max_epi16(v0, perm1);
      __m256i v1 = _mm256_blend_epi32(max1, min1, 0x55);
      
      /* Pairs: ([11,15], [13,14], [8,12], [9,10], [3,7], [5,6], [0,4], 
                 [1,2]) */
      /* Perm:  (11, 13, 14,  8, 15,  9, 10, 12,  3,  5,  6,  0,  7,  1,  2,  
                 4) */
      __m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(23, 22, 27, 26, 
                                          29, 28, 17, 16, 31, 30, 19, 18, 21, 
                                          20, 25, 24, 7, 6, 11, 10, 13, 12, 
                                          1, 0, 15, 14, 3, 2, 5, 4, 9, 8));
      __m256i min2 = _mm256_min_epi16(v1, perm2);
      __m256i max2 = _mm256_max_epi16(v1, perm2);
      __m256i v2 = _mm256_blend_epi16(max2, min2, 0x2b);
      
      /* Pairs: ([7,15], [10,14], [9,13], [12,12], [11,11], [0,8], [2,6], 
                 [1,5], [4,4], [3,3]) */
      /* Perm:  ( 7, 10,  9, 12, 11, 14, 13,  0, 15,  2,  1,  4,  3,  6,  5,  
                 8) */
      __m256i perm3 = _mm256_permutexvar_epi16(_mm256_set_epi16(7, 10, 9, 12, 
                                               11, 14, 13, 0, 15, 2, 1, 4, 3, 
                                               6, 5, 8), v2);
      __m256i min3 = _mm256_min_epi16(v2, perm3);
      __m256i max3 = _mm256_max_epi16(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi16(max3, 0x687, min3);
      
      /* Pairs: ([15,15], [11,14], [13,13], [9,12], [10,10], [8,8], [7,7], 
                 [3,6], [5,5], [1,4], [2,2], [0,0]) */
      /* Perm:  (15, 11, 13,  9, 14, 10, 12,  8,  7,  3,  5,  1,  6,  2,  4,  
                 0) */
      __m256i perm4 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(31, 30, 23, 22, 
                                          27, 26, 19, 18, 29, 28, 21, 20, 25, 
                                          24, 17, 16, 15, 14, 7, 6, 11, 10, 
                                          3, 2, 13, 12, 5, 4, 9, 8, 1, 0));
      __m256i min4 = _mm256_min_epi16(v3, perm4);
      __m256i max4 = _mm256_max_epi16(v3, perm4);
      __m256i v4 = _mm256_blend_epi16(max4, min4, 0xa);
      
      /* Pairs: ([15,15], [6,14], [11,13], [10,12], [1,9], [8,8], [7,7], 
                 [3,5], [2,4], [0,0]) */
      /* Perm:  (15,  6, 11, 10, 13, 12,  1,  8,  7, 14,  3,  2,  5,  4,  9,  
                 0) */
      __m256i perm5 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 6, 11, 
                                               10, 13, 12, 1, 8, 7, 14, 3, 2, 
                                               5, 4, 9, 0), v4);
      __m256i min5 = _mm256_min_epi16(v4, perm5);
      __m256i max5 = _mm256_max_epi16(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi16(max5, 0xc4e, min5);
      
      /* Pairs: ([15,15], [7,14], [5,13], [11,12], [2,10], [9,9], [1,8], 
                 [6,6], [3,4], [0,0]) */
      /* Perm:  (15,  7,  5, 11, 12,  2,  9,  1, 14,  6, 13,  3,  4, 10,  8,  
                 0) */
      __m256i perm6 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 7, 5, 11, 
                                               12, 2, 9, 1, 14, 6, 13, 3, 4, 
                                               10, 8, 0), v5);
      __m256i min6 = _mm256_min_epi16(v5, perm6);
      __m256i max6 = _mm256_max_epi16(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi16(max6, 0x8ae, min6);
      
      /* Pairs: ([15,15], [14,14], [7,13], [4,12], [3,11], [10,10], [9,9], 
                 [2,8], [6,6], [5,5], [1,1], [0,0]) */
      /* Perm:  (15, 14,  7,  4,  3, 10,  9,  2, 13,  6,  5, 12, 11,  8,  1,  
                 0) */
      __m256i perm7 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 7, 4, 
                                               3, 10, 9, 2, 13, 6, 5, 12, 11, 
                                               8, 1, 0), v6);
      __m256i min7 = _mm256_min_epi16(v6, perm7);
      __m256i max7 = _mm256_max_epi16(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi16(max7, 0x9c, min7);
      
      /* Pairs: ([15,15], [14,14], [13,13], [5,12], [11,11], [3,10], [9,9], 
                 [8,8], [7,7], [6,6], [4,4], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13,  5, 11,  3,  9,  8,  7,  6, 12,  4, 10,  2,  1,  
                 0) */
      __m256i perm8 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                               5, 11, 3, 9, 8, 7, 6, 12, 4, 
                                               10, 2, 1, 0), v7);
      __m256i min8 = _mm256_min_epi16(v7, perm8);
      __m256i max8 = _mm256_max_epi16(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi16(max8, 0x28, min8);
      
      /* Pairs: ([15,15], [14,14], [13,13], [6,12], [11,11], [10,10], [3,9], 
                 [8,8], [7,7], [5,5], [4,4], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13,  6, 11, 10,  3,  8,  7, 12,  5,  4,  9,  2,  1,  
                 0) */
      __m256i perm9 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                               6, 11, 10, 3, 8, 7, 12, 5, 4, 
                                               9, 2, 1, 0), v8);
      __m256i min9 = _mm256_min_epi16(v8, perm9);
      __m256i max9 = _mm256_max_epi16(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi16(max9, 0x48, min9);
      
      /* Pairs: ([15,15], [14,14], [13,13], [7,12], [11,11], [6,10], [5,9], 
                 [3,8], [4,4], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13,  7, 11,  6,  5,  3, 12, 10,  9,  4,  8,  2,  1,  
                 0) */
      __m256i perm10 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                                7, 11, 6, 5, 3, 12, 10, 9, 4, 
                                                8, 2, 1, 0), v9);
      __m256i min10 = _mm256_min_epi16(v9, perm10);
      __m256i max10 = _mm256_max_epi16(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi16(max10, 0xe8, min10);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [7,11], [10,10], [9,9], 
                 [4,8], [6,6], [5,5], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12,  7, 10,  9,  4, 11,  6,  5,  8,  3,  2,  1,  
                 0) */
      __m256i perm11 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                                12, 7, 10, 9, 4, 11, 6, 5, 8, 
                                                3, 2, 1, 0), v10);
      __m256i min11 = _mm256_min_epi16(v10, perm11);
      __m256i max11 = _mm256_max_epi16(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi16(max11, 0x90, min11);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [7,10], [9,9], 
                 [5,8], [6,6], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11,  7,  9,  5, 10,  6,  8,  4,  3,  2,  1,  
                 0) */
      __m256i perm12 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                                12, 11, 7, 9, 5, 10, 6, 8, 4, 
                                                3, 2, 1, 0), v11);
      __m256i min12 = _mm256_min_epi16(v11, perm12);
      __m256i max12 = _mm256_max_epi16(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi16(max12, 0xa0, min12);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [7,9], 
                 [6,8], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  7,  6,  9,  8,  5,  4,  3,  2,  1,  
                 0) */
      __m256i perm13 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                                12, 11, 10, 7, 6, 9, 8, 5, 4, 
                                                3, 2, 1, 0), v12);
      __m256i min13 = _mm256_min_epi16(v12, perm13);
      __m256i max13 = _mm256_max_epi16(v12, perm13);
      __m256i v13 = _mm256_blend_epi32(max13, min13, 0x8);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [7,8], [6,6], [5,5], [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  7,  8,  6,  5,  4,  3,  2,  1,  
                 0) */
      __m256i perm14 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                                12, 11, 10, 9, 7, 8, 6, 5, 4, 
                                                3, 2, 1, 0), v13);
      __m256i min14 = _mm256_min_epi16(v13, perm14);
      __m256i max14 = _mm256_max_epi16(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi16(max14, 0x80, min14);
      
      return v14;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
bosenelson_16_int16_t(int16_t * const 
                             arr) {
      
      __m256i _tmp0 = _mm256_set1_epi16(int16_t(0x7fff));
      asm volatile("vpblendd %[load_mask], (%[arr]), %[fill_v], %[fill_v]\n"
                   : [ fill_v ] "+x" (_tmp0)
                   : [ arr ] "r" (arr), [ load_mask ] "i" (0xff)
                   :);
      __m256i v = _tmp0;
      fill_works(v);
      v = bosenelson_16_int16_t_vec(v);
      
      fill_works(v);_mm256_mask_storeu_epi16((void *)arr, 0xffff, v);
      
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


