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
#define N 27
#define SORT_NAME bitonic_27_uint16_t

#ifndef _SIMD_SORT_bitonic_27_uint16_t_H_
#define _SIMD_SORT_bitonic_27_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 27
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 2 / 75
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

bitonic_27_uint16_t_vec(__m512i v) {
      
      __m512i perm0 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 25, 26, 23, 24, 21, 
                                               22, 20, 18, 19, 16, 17, 14, 
                                               15, 13, 11, 12, 9, 10, 7, 8, 
                                               6, 4, 5, 3, 1, 2, 0), v);
      __m512i min0 = _mm512_min_epu16(v, perm0);
      __m512i max0 = _mm512_max_epu16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x2a54a92, min0);
      
      __m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 23, 24, 25, 26, 22, 
                                               20, 21, 16, 17, 18, 19, 13, 
                                               14, 15, 9, 10, 11, 12, 6, 7, 
                                               8, 3, 4, 5, 2, 0, 1), v0);
      __m512i min1 = _mm512_min_epu16(v0, perm1);
      __m512i max1 = _mm512_max_epu16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x1932649, min1);
      
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 25, 26, 23, 24, 21, 
                                               22, 20, 18, 19, 16, 17, 15, 
                                               13, 14, 11, 12, 9, 10, 8, 6, 
                                               7, 5, 3, 4, 1, 2, 0), v1);
      __m512i min2 = _mm512_min_epu16(v1, perm2);
      __m512i max2 = _mm512_max_epu16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0x2a52a4a, min2);
      
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 20, 21, 22, 23, 24, 
                                               25, 26, 19, 13, 14, 15, 16, 
                                               17, 18, 12, 6, 7, 8, 9, 10, 
                                               11, 1, 2, 3, 4, 5, 0), v2);
      __m512i min3 = _mm512_min_epu16(v2, perm3);
      __m512i max3 = _mm512_max_epu16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x70e1c6, min3);
      
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 24, 25, 26, 21, 20, 
                                               23, 22, 17, 16, 19, 18, 13, 
                                               14, 15, 10, 9, 12, 11, 6, 7, 
                                               8, 4, 5, 1, 0, 3, 2), v3);
      __m512i min4 = _mm512_min_epu16(v3, perm4);
      __m512i max4 = _mm512_max_epu16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0x1332653, min4);
      
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 24, 25, 22, 23, 
                                               20, 21, 18, 19, 16, 17, 14, 
                                               15, 13, 11, 12, 9, 10, 7, 8, 
                                               4, 5, 6, 2, 3, 0, 1), v4);
      __m512i min5 = _mm512_min_epu16(v4, perm5);
      __m512i max5 = _mm512_max_epu16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0x1554a95, min5);
      
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 14, 15, 16, 17, 18, 
                                               19, 20, 21, 22, 23, 24, 25, 
                                               26, 13, 12, 11, 0, 1, 2, 3, 6, 
                                               5, 4, 7, 8, 9, 10), v5);
      __m512i min6 = _mm512_min_epu16(v5, perm6);
      __m512i max6 = _mm512_max_epu16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0xfc00f, min6);
      
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 22, 21, 24, 23, 26, 
                                               25, 16, 15, 14, 13, 20, 19, 
                                               18, 17, 8, 7, 6, 5, 12, 11, 
                                               10, 9, 0, 1, 2, 3, 4), v6);
      __m512i min7 = _mm512_min_epu16(v6, perm7);
      __m512i max7 = _mm512_max_epu16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x61e1e3, min7);
      
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 25, 26, 22, 21, 24, 
                                               23, 18, 17, 20, 19, 14, 13, 
                                               16, 15, 10, 9, 12, 11, 6, 5, 
                                               8, 7, 2, 3, 4, 1, 0), v7);
      __m512i min8 = _mm512_min_epu16(v7, perm8);
      __m512i max8 = _mm512_max_epu16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x2666664, min8);
      
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 23, 24, 21, 
                                               22, 19, 20, 17, 18, 15, 16, 
                                               13, 14, 11, 12, 9, 10, 7, 8, 
                                               5, 6, 3, 4, 1, 2, 0), v8);
      __m512i min9 = _mm512_min_epu16(v8, perm9);
      __m512i max9 = _mm512_max_epu16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0xaaaaaa, min9);
      
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 2, 3, 4, 5, 6, 7, 8, 
                                                9, 10, 11, 12, 15, 14, 13, 
                                                16, 17, 18, 19, 20, 21, 22, 
                                                23, 24, 25, 26, 1, 0), v9);
      __m512i min10 = _mm512_min_epu16(v9, perm10);
      __m512i max10 = _mm512_max_epu16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x1ffc, min10);
      
      __m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 18, 17, 16, 19, 22, 
                                                21, 20, 23, 26, 25, 24, 5, 6, 
                                                7, 4, 3, 2, 1, 0, 13, 14, 15, 
                                                12, 11, 10, 9, 8), v10);
      __m512i min11 = _mm512_min_epu16(v10, perm11);
      __m512i max11 = _mm512_max_epu16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0xf00ff, min11);
      
      __m512i perm12 = _mm512_shuffle_epi8(v11, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 49, 
                                           48, 51, 50, 53, 52, 47, 46, 37, 
                                           36, 35, 34, 33, 32, 39, 38, 45, 
                                           44, 43, 42, 41, 40, 19, 18, 21, 
                                           20, 23, 22, 17, 16, 27, 26, 29, 
                                           28, 31, 30, 25, 24, 7, 6, 5, 4, 3, 
                                           2, 1, 0, 15, 14, 13, 12, 11, 10, 
                                           9, 8));
      __m512i min12 = _mm512_min_epu16(v11, perm12);
      __m512i max12 = _mm512_max_epu16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x1070f0f, min12);
      
      __m512i perm13 = _mm512_shuffle_epi8(v12, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 49, 48, 51, 50, 43, 42, 41, 
                                           40, 47, 46, 45, 44, 35, 34, 33, 
                                           32, 39, 38, 37, 36, 27, 26, 25, 
                                           24, 31, 30, 29, 28, 19, 18, 17, 
                                           16, 23, 22, 21, 20, 11, 10, 9, 8, 
                                           15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 
                                           5, 4));
      __m512i min13 = _mm512_min_epu16(v12, perm13);
      __m512i max13 = _mm512_max_epu16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0x1333333, min13);
      
      __m512i perm14 = _mm512_shuffle_epi8(v13, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 45, 44, 47, 
                                           46, 41, 40, 43, 42, 37, 36, 39, 
                                           38, 33, 32, 35, 34, 29, 28, 31, 
                                           30, 25, 24, 27, 26, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 
                                           0, 3, 2));
      __m512i min14 = _mm512_min_epu16(v13, perm14);
      __m512i max14 = _mm512_max_epu16(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi16(max14, 0x555555, min14);
      
      return v14;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

bitonic_27_uint16_t(uint16_t * const 
                                 arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = bitonic_27_uint16_t_vec(v);
      
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


