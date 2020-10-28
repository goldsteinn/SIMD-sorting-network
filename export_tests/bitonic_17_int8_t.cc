#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_17_int8_t_H_
#define _SIMD_SORT_bitonic_17_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 17
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 12
	SIMD Instructions                : 3 / 60
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, AVX, AVX512vbmi, AVX2
	SIMD Instruction Set(s) Excluded : None
	Aligned Load & Store             : True
	Full Load & Store                : True

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
__m256i __attribute__((const)) bitonic_17_int8_t_vec(__m256i v) {

__m256i perm0 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 15, 16, 14, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1), v);
__m256i min0 = _mm256_min_epi8(v, perm0);
__m256i max0 = _mm256_max_epi8(v, perm0);
__m256i v0 = _mm256_mask_mov_epi8(max0, 0x9555, min0);

__m256i perm1 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 14, 15, 16, 13, 12, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3), v0);
__m256i min1 = _mm256_min_epi8(v0, perm1);
__m256i max1 = _mm256_max_epi8(v0, perm1);
__m256i v1 = _mm256_mask_mov_epi8(max1, 0x4333, min1);

__m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 13, 14, 15, 16, 12, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1), v1);
__m256i min2 = _mm256_min_epi8(v1, perm2);
__m256i max2 = _mm256_max_epi8(v1, perm2);
__m256i v2 = _mm256_mask_mov_epi8(max2, 0x6555, min2);

__m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 11, 12, 13, 14, 15, 16, 10, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7), v2);
__m256i min3 = _mm256_min_epi8(v2, perm3);
__m256i max3 = _mm256_max_epi8(v2, perm3);
__m256i v3 = _mm256_mask_mov_epi8(max3, 0x380f, min3);

__m256i perm4 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 15, 12, 13, 11, 10, 9, 8, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min4 = _mm256_min_epi8(v3, perm4);
__m256i max4 = _mm256_max_epi8(v3, perm4);
__m256i v4 = _mm256_mask_mov_epi8(max4, 0x5033, min4);

__m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 8, 9, 10, 11, 12, 13, 14, 15, 6, 7, 4, 5, 2, 3, 0, 1));
__m256i min5 = _mm256_min_epi8(v4, perm5);
__m256i max5 = _mm256_max_epi8(v4, perm5);
__m256i v5 = _mm256_mask_mov_epi8(max5, 0xf55, min5);

__m256i perm6 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 7, 13, 12, 15, 14, 9, 8, 11, 10, 16, 6, 5, 4, 3, 2, 1, 0), v5);
__m256i min6 = _mm256_min_epi8(v5, perm6);
__m256i max6 = _mm256_max_epi8(v5, perm6);
__m256i v6 = _mm256_mask_mov_epi8(max6, 0x3380, min6);

__m256i perm7 = _mm256_shuffle_epi8(v6, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 15, 12, 13, 10, 11, 8, 9, 7, 6, 5, 4, 3, 2, 1, 0));
__m256i min7 = _mm256_min_epi8(v6, perm7);
__m256i max7 = _mm256_max_epi8(v6, perm7);
__m256i v7 = _mm256_mask_mov_epi8(max7, 0x5500, min7);

__m256i perm8 = _mm256_shuffle_epi8(v7, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15));
__m256i min8 = _mm256_min_epi8(v7, perm8);
__m256i max8 = _mm256_max_epi8(v7, perm8);
__m256i v8 = _mm256_blend_epi32(max8, min8, 0x3);

__m256i perm9 = _mm256_shuffle_epi8(v8, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m256i min9 = _mm256_min_epi8(v8, perm9);
__m256i max9 = _mm256_max_epi8(v8, perm9);
__m256i v9 = _mm256_blend_epi32(max9, min9, 0x5);

__m256i perm10 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min10 = _mm256_min_epi8(v9, perm10);
__m256i max10 = _mm256_max_epi8(v9, perm10);
__m256i v10 = _mm256_mask_mov_epi8(max10, 0x3333, min10);

__m256i perm11 = _mm256_shuffle_epi8(v10, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m256i min11 = _mm256_min_epi8(v10, perm11);
__m256i max11 = _mm256_max_epi8(v10, perm11);
__m256i v11 = _mm256_mask_mov_epi8(max11, 0x5555, min11);

return v11;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_17_int8_t(int8_t * const arr) {

__m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
__m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x1ffff, arr);

v = bitonic_17_int8_t_vec(v);

_mm256_mask_storeu_epi8((void *)arr, 0x1ffff, v);

}


#endif


#define TYPE int8_t
#define N 17
#define SORT_NAME bitonic_17_int8_t

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


