#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_27_uint8_t_H_
#define _SIMD_SORT_bitonic_27_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 27
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 3 / 75
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, AVX, AVX2, AVX512vbmi
	SIMD Instruction Set(s) Excluded : None
	Aligned Load & Store             : False
	Full Load & Store                : False

Performance Notes:
1) If you are sorting an array where there IS valid memory up to 
   the nearest sizeof a SIMD register, you will get an improvement enable
   "EXTRA_MEMORY" (this turns on "Full Load & Store". Note that enabling
   "Full Load & Store" will not modify any of the memory not being sorted
   and will not affect the sort in any way. i.e sort(3) [4, 3, 2, 1]
   with full load will still return [2, 3, 4, 1].

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
__m256i __attribute__((const)) bitonic_27_uint8_t_vec(__m256i v) {

__m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(31, 30, 29, 28, 27, 25, 26, 23, 24, 21, 22, 20, 18, 19, 16, 17, 14, 15, 13, 11, 12, 9, 10, 7, 8, 6, 4, 5, 3, 1, 2, 0));
__m256i min0 = _mm256_min_epu8(v, perm0);
__m256i max0 = _mm256_max_epu8(v, perm0);
__m256i v0 = _mm256_mask_mov_epi8(max0, 0x2a54a92, min0);

__m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 27, 23, 24, 25, 26, 22, 20, 21, 16, 17, 18, 19, 13, 14, 15, 9, 10, 11, 12, 6, 7, 8, 3, 4, 5, 2, 0, 1));
__m256i min1 = _mm256_min_epu8(v0, perm1);
__m256i max1 = _mm256_max_epu8(v0, perm1);
__m256i v1 = _mm256_mask_mov_epi8(max1, 0x1932649, min1);

__m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(31, 30, 29, 28, 27, 25, 26, 23, 24, 21, 22, 20, 18, 19, 16, 17, 15, 13, 14, 11, 12, 9, 10, 8, 6, 7, 5, 3, 4, 1, 2, 0));
__m256i min2 = _mm256_min_epu8(v1, perm2);
__m256i max2 = _mm256_max_epu8(v1, perm2);
__m256i v2 = _mm256_mask_mov_epi8(max2, 0x2a52a4a, min2);

__m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 20, 21, 22, 23, 24, 25, 26, 19, 13, 14, 15, 16, 17, 18, 12, 6, 7, 8, 9, 10, 11, 1, 2, 3, 4, 5, 0), v2);
__m256i min3 = _mm256_min_epu8(v2, perm3);
__m256i max3 = _mm256_max_epu8(v2, perm3);
__m256i v3 = _mm256_mask_mov_epi8(max3, 0x70e1c6, min3);

__m256i perm4 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(31, 30, 29, 28, 27, 24, 25, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 14, 15, 10, 9, 12, 11, 6, 7, 8, 4, 5, 1, 0, 3, 2));
__m256i min4 = _mm256_min_epu8(v3, perm4);
__m256i max4 = _mm256_max_epu8(v3, perm4);
__m256i v4 = _mm256_mask_mov_epi8(max4, 0x1332653, min4);

__m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 13, 11, 12, 9, 10, 7, 8, 4, 5, 6, 2, 3, 0, 1));
__m256i min5 = _mm256_min_epu8(v4, perm5);
__m256i max5 = _mm256_max_epu8(v4, perm5);
__m256i v5 = _mm256_mask_mov_epi8(max5, 0x1554a95, min5);

__m256i perm6 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 13, 12, 11, 0, 1, 2, 3, 6, 5, 4, 7, 8, 9, 10), v5);
__m256i min6 = _mm256_min_epu8(v5, perm6);
__m256i max6 = _mm256_max_epu8(v5, perm6);
__m256i v6 = _mm256_mask_mov_epi8(max6, 0xfc00f, min6);

__m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 22, 21, 24, 23, 26, 25, 16, 15, 14, 13, 20, 19, 18, 17, 8, 7, 6, 5, 12, 11, 10, 9, 0, 1, 2, 3, 4), v6);
__m256i min7 = _mm256_min_epu8(v6, perm7);
__m256i max7 = _mm256_max_epu8(v6, perm7);
__m256i v7 = _mm256_mask_mov_epi8(max7, 0x61e1e3, min7);

__m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 25, 26, 22, 21, 24, 23, 18, 17, 20, 19, 14, 13, 16, 15, 10, 9, 12, 11, 6, 5, 8, 7, 2, 3, 4, 1, 0), v7);
__m256i min8 = _mm256_min_epu8(v7, perm8);
__m256i max8 = _mm256_max_epu8(v7, perm8);
__m256i v8 = _mm256_mask_mov_epi8(max8, 0x2666664, min8);

__m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 23, 24, 21, 22, 19, 20, 17, 18, 15, 16, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0), v8);
__m256i min9 = _mm256_min_epu8(v8, perm9);
__m256i max9 = _mm256_max_epu8(v8, perm9);
__m256i v9 = _mm256_mask_mov_epi8(max9, 0xaaaaaa, min9);

__m256i perm10 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 15, 14, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 1, 0), v9);
__m256i min10 = _mm256_min_epu8(v9, perm10);
__m256i max10 = _mm256_max_epu8(v9, perm10);
__m256i v10 = _mm256_mask_mov_epi8(max10, 0x1ffc, min10);

__m256i perm11 = _mm256_shuffle_epi8(v10, _mm256_set_epi8(31, 30, 29, 28, 27, 18, 17, 16, 19, 22, 21, 20, 23, 26, 25, 24, 5, 6, 7, 4, 3, 2, 1, 0, 13, 14, 15, 12, 11, 10, 9, 8));
__m256i min11 = _mm256_min_epu8(v10, perm11);
__m256i max11 = _mm256_max_epu8(v10, perm11);
__m256i v11 = _mm256_blend_epi32(max11, min11, 0x13);

__m256i perm12 = _mm256_shuffle_epi8(v11, _mm256_set_epi8(31, 30, 29, 28, 27, 24, 25, 26, 23, 18, 17, 16, 19, 22, 21, 20, 9, 10, 11, 8, 13, 14, 15, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m256i min12 = _mm256_min_epu8(v11, perm12);
__m256i max12 = _mm256_max_epu8(v11, perm12);
__m256i v12 = _mm256_mask_mov_epi8(max12, 0x1070f0f, min12);

__m256i perm13 = _mm256_shuffle_epi8(v12, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 24, 25, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min13 = _mm256_min_epu8(v12, perm13);
__m256i max13 = _mm256_max_epu8(v12, perm13);
__m256i v13 = _mm256_mask_mov_epi8(max13, 0x1333333, min13);

__m256i perm14 = _mm256_shuffle_epi8(v13, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m256i min14 = _mm256_min_epu8(v13, perm14);
__m256i max14 = _mm256_max_epu8(v13, perm14);
__m256i v14 = _mm256_mask_mov_epi8(max14, 0x555555, min14);

return v14;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_27_uint8_t(uint8_t * const arr) {

__m256i v = _mm256_mask_loadu_epi8(_mm256_set1_epi8(uint8_t(0xff)), 0x7ffffff, arr);
v = bitonic_27_uint8_t_vec(v);
_mm256_mask_storeu_epi8((void *)arr, 0x7ffffff, v);

}


#endif


#define TYPE uint8_t
#define N 27
#define SORT_NAME bitonic_27_uint8_t

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


