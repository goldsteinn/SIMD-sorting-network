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
#define N 22
#define SORT_NAME batcher_22_uint8_t

#ifndef _SIMD_SORT_batcher_22_uint8_t_H_
#define _SIMD_SORT_batcher_22_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 22
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 15
	SIMD Instructions                : 2 / 75
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX, AVX512vbmi, AVX512vl, AVX2, AVX512bw
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
     __m256i __attribute__((const)) 

batcher_22_uint8_t_vec(__m256i v) {
      
      __m256i perm0 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 5, 4, 
                                              3, 2, 1, 0, 7, 6, 13, 12, 11, 
                                              10, 9, 8, 15, 14, 21, 20, 19, 
                                              18, 17, 16), v);
      __m256i min0 = _mm256_min_epu8(v, perm0);
      __m256i max0 = _mm256_max_epu8(v, perm0);
      __m256i v0 = _mm256_blend_epi32(max0, min0, 0x3);
      
      __m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 19, 
                                          18, 17, 16, 15, 14, 5, 4, 3, 2, 1, 
                                          0, 7, 6, 13, 12, 11, 10, 9, 8));
      __m256i min1 = _mm256_min_epu8(v0, perm1);
      __m256i max1 = _mm256_max_epu8(v0, perm1);
      __m256i v1 = _mm256_mask_mov_epi8(max1, 0x3f, min1);
      
      __m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 13, 12, 
                                              11, 10, 9, 8, 15, 14, 21, 20, 
                                              19, 18, 17, 16, 3, 2, 1, 0, 7, 
                                              6, 5, 4), v1);
      __m256i min2 = _mm256_min_epu8(v1, perm2);
      __m256i max2 = _mm256_max_epu8(v1, perm2);
      __m256i v2 = _mm256_mask_mov_epi8(max2, 0x3f0f, min2);
      
      __m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 17, 16, 
                                              7, 6, 21, 20, 11, 10, 9, 8, 15, 
                                              14, 13, 12, 19, 18, 5, 4, 1, 0, 
                                              3, 2), v2);
      __m256i min3 = _mm256_min_epu8(v2, perm3);
      __m256i max3 = _mm256_max_epu8(v2, perm3);
      __m256i v3 = _mm256_mask_mov_epi8(max3, 0x30fc3, min3);
      
      __m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              15, 14, 5, 4, 19, 18, 13, 12, 
                                              7, 6, 9, 8, 11, 10, 17, 16, 3, 
                                              2, 0, 1), v3);
      __m256i min4 = _mm256_min_epu8(v3, perm4);
      __m256i max4 = _mm256_max_epu8(v3, perm4);
      __m256i v4 = _mm256_mask_mov_epi8(max4, 0xc0f1, min4);
      
      __m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 13, 12, 15, 14, 17, 16, 
                                              11, 10, 5, 4, 7, 6, 9, 8, 3, 2, 
                                              1, 0), v4);
      __m256i min5 = _mm256_min_epu8(v4, perm5);
      __m256i max5 = _mm256_max_epu8(v4, perm5);
      __m256i v5 = _mm256_mask_mov_epi8(max5, 0x3030, min5);
      
      __m256i perm6 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 23, 22, 21, 20, 17, 
                                          16, 19, 18, 13, 12, 15, 14, 9, 8, 
                                          11, 10, 5, 4, 7, 6, 3, 2, 1, 0));
      __m256i min6 = _mm256_min_epu8(v5, perm6);
      __m256i max6 = _mm256_max_epu8(v5, perm6);
      __m256i v6 = _mm256_mask_mov_epi8(max6, 0x33330, min6);
      
      __m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 7, 6, 
                                              19, 18, 3, 2, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 21, 20, 5, 4, 17, 
                                              16, 1, 0), v6);
      __m256i min7 = _mm256_min_epu8(v6, perm7);
      __m256i max7 = _mm256_max_epu8(v6, perm7);
      __m256i v7 = _mm256_mask_mov_epi8(max7, 0xcc, min7);
      
      __m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 15, 14, 
                                              19, 18, 11, 10, 21, 20, 7, 6, 
                                              17, 16, 3, 2, 13, 12, 5, 4, 9, 
                                              8, 1, 0), v7);
      __m256i min8 = _mm256_min_epu8(v7, perm8);
      __m256i max8 = _mm256_max_epu8(v7, perm8);
      __m256i v8 = _mm256_mask_mov_epi8(max8, 0xcccc, min8);
      
      __m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 19, 18, 
                                              21, 20, 15, 14, 17, 16, 11, 10, 
                                              13, 12, 7, 6, 9, 8, 3, 2, 5, 4, 
                                              1, 0), v8);
      __m256i min9 = _mm256_min_epu8(v8, perm9);
      __m256i max9 = _mm256_max_epu8(v8, perm9);
      __m256i v9 = _mm256_mask_mov_epi8(max9, 0xccccc, min9);
      
      __m256i perm10 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(31, 30, 29, 
                                           28, 27, 26, 25, 24, 23, 22, 20, 
                                           21, 18, 19, 16, 17, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 1, 0));
      __m256i min10 = _mm256_min_epu8(v9, perm10);
      __m256i max10 = _mm256_max_epu8(v9, perm10);
      __m256i v10 = _mm256_mask_mov_epi8(max10, 0x155554, min10);
      
      __m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 5, 19, 3, 17, 1, 15, 7, 
                                               13, 12, 11, 10, 9, 8, 14, 6, 
                                               20, 4, 18, 2, 16, 0), v10);
      __m256i min11 = _mm256_min_epu8(v10, perm11);
      __m256i max11 = _mm256_max_epu8(v10, perm11);
      __m256i v11 = _mm256_mask_mov_epi8(max11, 0xaa, min11);
      
      __m256i perm12 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 13, 19, 11, 17, 9, 15, 14, 
                                               20, 5, 18, 3, 16, 1, 7, 6, 12, 
                                               4, 10, 2, 8, 0), v11);
      __m256i min12 = _mm256_min_epu8(v11, perm12);
      __m256i max12 = _mm256_max_epu8(v11, perm12);
      __m256i v12 = _mm256_mask_mov_epi8(max12, 0x2a2a, min12);
      
      __m256i perm13 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 17, 19, 15, 20, 13, 18, 
                                               11, 16, 9, 14, 7, 12, 5, 10, 
                                               3, 8, 1, 6, 2, 4, 0), v12);
      __m256i min13 = _mm256_min_epu8(v12, perm13);
      __m256i max13 = _mm256_max_epu8(v12, perm13);
      __m256i v13 = _mm256_mask_mov_epi8(max13, 0x2aaaa, min13);
      
      __m256i perm14 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 19, 20, 17, 18, 15, 16, 
                                               13, 14, 11, 12, 9, 10, 7, 8, 
                                               5, 6, 3, 4, 1, 2, 0), v13);
      __m256i min14 = _mm256_min_epu8(v13, perm14);
      __m256i max14 = _mm256_max_epu8(v13, perm14);
      __m256i v14 = _mm256_mask_mov_epi8(max14, 0xaaaaa, min14);
      
      return v14;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

batcher_22_uint8_t(uint8_t * const arr) 
                                 {
      
      __m256i v = _mm256_load_si256((__m256i *)arr);
      
      v = batcher_22_uint8_t_vec(v);
      
      _mm256_store_si256((__m256i *)arr, v);
      
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


