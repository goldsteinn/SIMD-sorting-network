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
#define N 9
#define SORT_NAME oddeven_9_uint16_t

#ifndef _SIMD_SORT_oddeven_9_uint16_t_H_
#define _SIMD_SORT_oddeven_9_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 9
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : oddeven
	Network Depth                    : 9
	SIMD Instructions                : 3 / 45
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX2, AVX
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
          assert(t.arr[i] == uint16_t(0xffff));
 }
}

/* SIMD Sort */
 __m256i __attribute__((const)) 
oddeven_9_uint16_t_vec(__m256i v) {
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [6,7], [4,5], [2,3], [0,1]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  6,  7,  4,  5,  2,  3,  0,  
                 1) */
      __m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 13, 12, 15, 14, 9, 8, 
                                          11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
      __m256i min0 = _mm256_min_epu16(v, perm0);
      __m256i max0 = _mm256_max_epu16(v, perm0);
      __m256i v0 = _mm256_mask_mov_epi16(max0, 0x55, min0);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [5,7], [4,6], [1,3], [0,2]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  5,  4,  7,  6,  1,  0,  3,  
                 2) */
      __m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 11, 10, 9, 8, 15, 14, 
                                          13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
      __m256i min1 = _mm256_min_epu16(v0, perm1);
      __m256i max1 = _mm256_max_epu16(v0, perm1);
      __m256i v1 = _mm256_blend_epi32(max1, min1, 0x5);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [3,7], [5,6], [0,4], [1,2]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  3,  5,  6,  0,  7,  1,  2,  
                 4) */
      __m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 7, 6, 11, 10, 13, 12, 
                                          1, 0, 15, 14, 3, 2, 5, 4, 9, 8));
      __m256i min2 = _mm256_min_epu16(v1, perm2);
      __m256i max2 = _mm256_max_epu16(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi16(max2, 0x2b, min2);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [0,8], [7,7], [2,6], [1,5], [4,4], [3,3]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  0,  7,  2,  1,  4,  3,  6,  5,  
                 8) */
      __m256i perm3 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                               12, 11, 10, 9, 0, 7, 2, 1, 4, 
                                               3, 6, 5, 8), v2);
      __m256i min3 = _mm256_min_epu16(v2, perm3);
      __m256i max3 = _mm256_max_epu16(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi16(max3, 0x7, min3);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [7,7], [6,6], [3,5], [2,4], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  3,  2,  5,  4,  1,  
                 0) */
      __m256i perm4 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 15, 14, 13, 12, 7, 6, 
                                          5, 4, 11, 10, 9, 8, 3, 2, 1, 0));
      __m256i min4 = _mm256_min_epu16(v3, perm4);
      __m256i max4 = _mm256_max_epu16(v3, perm4);
      __m256i v4 = _mm256_blend_epi32(max4, min4, 0x2);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [7,7], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  7,  5,  6,  3,  4,  1,  2,  
                 0) */
      __m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 15, 14, 11, 10, 13, 12, 
                                          7, 6, 9, 8, 3, 2, 5, 4, 1, 0));
      __m256i min5 = _mm256_min_epu16(v4, perm5);
      __m256i max5 = _mm256_max_epu16(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi16(max5, 0x2a, min5);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [4,8], [7,7], [6,6], [3,5], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  4,  7,  6,  3,  8,  5,  2,  1,  
                 0) */
      __m256i perm6 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                               12, 11, 10, 9, 4, 7, 6, 3, 8, 
                                               5, 2, 1, 0), v5);
      __m256i min6 = _mm256_min_epu16(v5, perm6);
      __m256i max6 = _mm256_max_epu16(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi16(max6, 0x18, min6);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [6,8], [7,7], [5,5], [2,4], [3,3], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  6,  7,  8,  5,  2,  3,  4,  1,  
                 0) */
      __m256i perm7 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                               12, 11, 10, 9, 6, 7, 8, 5, 2, 
                                               3, 4, 1, 0), v6);
      __m256i min7 = _mm256_min_epu16(v6, perm7);
      __m256i max7 = _mm256_max_epu16(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi16(max7, 0x44, min7);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [7,8], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  7,  8,  5,  6,  3,  4,  1,  2,  
                 0) */
      __m256i perm8 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 
                                               12, 11, 10, 9, 7, 8, 5, 6, 3, 
                                               4, 1, 2, 0), v7);
      __m256i min8 = _mm256_min_epu16(v7, perm8);
      __m256i max8 = _mm256_max_epu16(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi16(max8, 0xaa, min8);
      
      return v8;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
oddeven_9_uint16_t(uint16_t * const 
                             arr) {
      
      __m256i _tmp0 = _mm256_set1_epi16(uint16_t(0xffff));
      __m256i v = _mm256_mask_loadu_epi16(_tmp0, 0x1ff, arr);
      fill_works(v);
      v = oddeven_9_uint16_t_vec(v);
      
      fill_works(v);_mm256_mask_storeu_epi16((void *)arr, 0x1ff, v);
      
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


