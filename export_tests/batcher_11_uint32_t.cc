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

#define TYPE uint32_t
#define N 11
#define SORT_NAME batcher_11_uint32_t

#ifndef _SIMD_SORT_batcher_11_uint32_t_H_
#define _SIMD_SORT_batcher_11_uint32_t_H_

/*

Sorting Network Information:
	Sort Size                        : 11
	Underlying Sort Type             : uint32_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 10
	SIMD Instructions                : 2 / 50
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

batcher_11_uint32_t_vec(__m512i v) {
      
      __m512i perm0 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 2, 1, 0, 3, 6, 5, 4, 
                                               7, 10, 9, 8), v);
      __m512i min0 = _mm512_min_epu32(v, perm0);
      __m512i max0 = _mm512_max_epu32(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi32(max0, 0xf, min0);
      
      __m512i perm1 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 10, 9, 8, 7, 2, 1, 0, 
                                               3, 6, 5, 4), v0);
      __m512i min1 = _mm512_min_epu32(v0, perm1);
      __m512i max1 = _mm512_max_epu32(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi32(max1, 0x7, min1);
      
      __m512i perm2 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 6, 5, 4, 7, 10, 9, 8, 
                                               1, 0, 3, 2), v1);
      __m512i min2 = _mm512_min_epu32(v1, perm2);
      __m512i max2 = _mm512_max_epu32(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi32(max2, 0x73, min2);
      
      __m512i perm3 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 8, 3, 10, 5, 4, 7, 6, 
                                               9, 2, 0, 1), v2);
      __m512i min3 = _mm512_min_epu32(v2, perm3);
      __m512i max3 = _mm512_max_epu32(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi32(max3, 0x139, min3);
      
      __m512i perm4 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 10, 7, 2, 9, 6, 3, 4, 
                                               5, 8, 1, 0), v3);
      __m512i min4 = _mm512_min_epu32(v3, perm4);
      __m512i max4 = _mm512_max_epu32(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi32(max4, 0x8c, min4);
      
      __m512i perm5 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 10, 9, 6, 7, 8, 5, 2, 
                                               3, 4, 1, 0), v4);
      __m512i min5 = _mm512_min_epu32(v4, perm5);
      __m512i max5 = _mm512_max_epu32(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi32(max5, 0x44, min5);
      
      __m512i perm6 = _mm512_shuffle_epi8(v5, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 55, 54, 53, 52, 51, 
                                          50, 49, 48, 47, 46, 45, 44, 43, 42, 
                                          41, 40, 35, 34, 33, 32, 39, 38, 37, 
                                          36, 27, 26, 25, 24, 31, 30, 29, 28, 
                                          19, 18, 17, 16, 23, 22, 21, 20, 11, 
                                          10, 9, 8, 15, 14, 13, 12, 7, 6, 5, 
                                          4, 3, 2, 1, 0));
      __m512i min6 = _mm512_min_epu32(v5, perm6);
      __m512i max6 = _mm512_max_epu32(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi32(max6, 0x154, min6);
      
      __m512i perm7 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 3, 9, 1, 7, 6, 5, 4, 
                                               10, 2, 8, 0), v6);
      __m512i min7 = _mm512_min_epu32(v6, perm7);
      __m512i max7 = _mm512_max_epu32(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi32(max7, 0xa, min7);
      
      __m512i perm8 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 7, 9, 5, 10, 3, 8, 1, 
                                               6, 2, 4, 0), v7);
      __m512i min8 = _mm512_min_epu32(v7, perm8);
      __m512i max8 = _mm512_max_epu32(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi32(max8, 0xaa, min8);
      
      __m512i perm9 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 11, 9, 10, 7, 8, 5, 6, 3, 
                                               4, 1, 2, 0), v8);
      __m512i min9 = _mm512_min_epu32(v8, perm9);
      __m512i max9 = _mm512_max_epu32(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi32(max9, 0x2aa, min9);
      
      return v9;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

batcher_11_uint32_t(uint32_t * const 
                                 arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = batcher_11_uint32_t_vec(v);
      
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

