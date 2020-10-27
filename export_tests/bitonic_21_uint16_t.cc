#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_21_uint16_t_H_
#define _SIMD_SORT_bitonic_21_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 21
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 3 / 75
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512bw, AVX, AVX512vl
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
__m512i __attribute__((const)) bitonic_21_uint16_t_vec(__m512i v) {

__m512i perm0 = _mm512_shuffle_epi8(v, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 39, 38, 41, 40, 37, 36, 33, 32, 35, 34, 31, 30, 27, 26, 29, 28, 25, 24, 21, 20, 23, 22, 17, 16, 19, 18, 15, 14, 11, 10, 13, 12, 7, 6, 9, 8, 5, 4, 1, 0, 3, 2));
__m512i min0 = _mm512_min_epu16(v, perm0);
__m512i max0 = _mm512_max_epu16(v, perm0);
__m512i v0 = _mm512_mask_mov_epi16(max0, 0x92529, min0);

__m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 18, 19, 20, 17, 15, 16, 14, 12, 13, 11, 10, 9, 7, 8, 6, 5, 2, 3, 4, 1, 0), v0);
__m512i min1 = _mm512_min_epu16(v0, perm1);
__m512i max1 = _mm512_max_epu16(v0, perm1);
__m512i v1 = _mm512_mask_mov_epi16(max1, 0x49084, min1);

__m512i perm2 = _mm512_shuffle_epi8(v1, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 37, 36, 39, 38, 33, 32, 35, 34, 31, 30, 27, 26, 29, 28, 21, 20, 23, 22, 25, 24, 17, 16, 19, 18, 11, 10, 13, 12, 15, 14, 3, 2, 5, 4, 7, 6, 9, 8, 1, 0));
__m512i min2 = _mm512_min_epu16(v1, perm2);
__m512i max2 = _mm512_max_epu16(v1, perm2);
__m512i v2 = _mm512_mask_mov_epi16(max2, 0x52526, min2);

__m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 16, 17, 18, 19, 20, 15, 12, 11, 14, 13, 10, 7, 6, 9, 8, 5, 4, 0, 1, 2, 3), v2);
__m512i min3 = _mm512_min_epu16(v2, perm3);
__m512i max3 = _mm512_max_epu16(v2, perm3);
__m512i v3 = _mm512_mask_mov_epi16(max3, 0x318c3, min3);

__m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 19, 20, 16, 15, 18, 17, 13, 14, 11, 12, 10, 8, 9, 6, 7, 5, 4, 2, 3, 0, 1), v3);
__m512i min4 = _mm512_min_epu16(v3, perm4);
__m512i max4 = _mm512_max_epu16(v3, perm4);
__m512i v4 = _mm512_mask_mov_epi16(max4, 0x9a945, min4);

__m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 12, 13, 17, 18, 15, 16, 14, 19, 20, 11, 10, 9, 2, 3, 0, 1, 4, 7, 8, 5, 6), v4);
__m512i min5 = _mm512_min_epu16(v4, perm5);
__m512i max5 = _mm512_max_epu16(v4, perm5);
__m512i v5 = _mm512_mask_mov_epi16(max5, 0x2b00f, min5);

__m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 14, 11, 12, 13, 18, 15, 16, 17, 10, 5, 8, 7, 4, 9, 6, 3, 2, 0, 1), v5);
__m512i min6 = _mm512_min_epu16(v5, perm6);
__m512i max6 = _mm512_max_epu16(v5, perm6);
__m512i v6 = _mm512_mask_mov_epi16(max6, 0x7831, min6);

__m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 18, 19, 20, 15, 16, 17, 10, 11, 12, 13, 14, 7, 6, 9, 8, 3, 2, 5, 4, 1, 0), v6);
__m512i min7 = _mm512_min_epu16(v6, perm7);
__m512i max7 = _mm512_max_epu16(v6, perm7);
__m512i v7 = _mm512_mask_mov_epi16(max7, 0x48ccc, min7);

__m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 18, 19, 17, 14, 15, 16, 13, 10, 11, 12, 8, 9, 6, 7, 4, 5, 2, 3, 1, 0), v7);
__m512i min8 = _mm512_min_epu16(v7, perm8);
__m512i max8 = _mm512_max_epu16(v7, perm8);
__m512i v8 = _mm512_mask_mov_epi16(max8, 0x44554, min8);

__m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 5, 6, 7, 16, 17, 14, 15, 12, 13, 10, 11, 9, 8, 18, 19, 20, 4, 3, 2, 1, 0), v8);
__m512i min9 = _mm512_min_epu16(v8, perm9);
__m512i max9 = _mm512_max_epu16(v8, perm9);
__m512i v9 = _mm512_mask_mov_epi16(max9, 0x154e0, min9);

__m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 8, 9, 2, 3, 4, 5, 6, 7, 16, 17, 10, 11, 12, 13, 14, 15, 1, 0), v9);
__m512i min10 = _mm512_min_epu16(v9, perm10);
__m512i max10 = _mm512_max_epu16(v9, perm10);
__m512i v10 = _mm512_mask_mov_epi16(max10, 0x3fc, min10);

__m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 16, 17, 18, 19, 20, 11, 10, 13, 12, 15, 14, 1, 0, 3, 2, 5, 4, 7, 6, 9, 8), v10);
__m512i min11 = _mm512_min_epu16(v10, perm11);
__m512i max11 = _mm512_max_epu16(v10, perm11);
__m512i v11 = _mm512_mask_mov_epi16(max11, 0x30c0f, min11);

__m512i perm12 = _mm512_shuffle_epi8(v11, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 33, 32, 35, 34, 37, 36, 31, 30, 29, 28, 17, 16, 19, 18, 23, 22, 21, 20, 25, 24, 27, 26, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8));
__m512i min12 = _mm512_min_epu16(v11, perm12);
__m512i max12 = _mm512_max_epu16(v11, perm12);
__m512i v12 = _mm512_mask_mov_epi16(max12, 0x10303, min12);

__m512i perm13 = _mm512_shuffle_epi8(v12, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 37, 36, 39, 38, 33, 32, 35, 34, 27, 26, 25, 24, 31, 30, 29, 28, 17, 16, 19, 18, 21, 20, 23, 22, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m512i min13 = _mm512_min_epu16(v12, perm13);
__m512i max13 = _mm512_max_epu16(v12, perm13);
__m512i v13 = _mm512_mask_mov_epi16(max13, 0x53333, min13);

__m512i perm14 = _mm512_shuffle_epi8(v13, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m512i min14 = _mm512_min_epu16(v13, perm14);
__m512i max14 = _mm512_max_epu16(v13, perm14);
__m512i v14 = _mm512_mask_mov_epi16(max14, 0x5555, min14);

return v14;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_21_uint16_t(uint16_t * const arr) {

__m512i v = _mm512_mask_loadu_epi16(_mm512_set1_epi16(uint16_t(0xffff)), 0x1fffff, arr);
v = bitonic_21_uint16_t_vec(v);
_mm512_mask_storeu_epi16((void *)arr, 0x1fffff, v);

}


#endif


#define TYPE uint16_t
#define N 21
#define SORT_NAME bitonic_21_uint16_t

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


