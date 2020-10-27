#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_50_uint8_t_H_
#define _SIMD_SORT_bitonic_50_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 50
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 21
	SIMD Instructions                : 3 / 105
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512bw, AVX, AVX512vbmi, AVX512vl
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
__m512i __attribute__((const)) bitonic_50_uint8_t_vec(__m512i v) {

__m512i perm0 = _mm512_shuffle_epi8(v, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 48, 49, 46, 47, 44, 45, 43, 41, 42, 40, 38, 39, 37, 35, 36, 34, 32, 33, 31, 29, 30, 28, 26, 27, 25, 23, 24, 21, 22, 19, 20, 18, 16, 17, 15, 13, 14, 12, 10, 11, 9, 7, 8, 6, 4, 5, 3, 1, 2, 0));
__m512i min0 = _mm512_min_epu8(v, perm0);
__m512i max0 = _mm512_max_epu8(v, perm0);
__m512i v0 = _mm512_mask_mov_epi8(max0, 0x1524924a92492, min0);

__m512i perm1 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 46, 47, 48, 49, 45, 43, 44, 42, 40, 41, 37, 38, 39, 36, 34, 35, 31, 32, 33, 28, 29, 30, 27, 25, 26, 21, 22, 23, 24, 18, 19, 20, 15, 16, 17, 14, 12, 13, 9, 10, 11, 8, 6, 7, 5, 3, 4, 0, 1, 2), v0);
__m512i min1 = _mm512_min_epu8(v0, perm1);
__m512i max1 = _mm512_max_epu8(v0, perm1);
__m512i v1 = _mm512_mask_mov_epi8(max1, 0xc92492649249, min1);

__m512i perm2 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 48, 49, 46, 47, 44, 45, 43, 41, 42, 40, 39, 37, 38, 35, 36, 34, 33, 31, 32, 30, 28, 29, 26, 27, 25, 23, 24, 21, 22, 20, 18, 19, 17, 15, 16, 13, 14, 12, 11, 9, 10, 7, 8, 6, 4, 5, 3, 2, 0, 1), v1);
__m512i min2 = _mm512_min_epu8(v1, perm2);
__m512i max2 = _mm512_max_epu8(v1, perm2);
__m512i v2 = _mm512_mask_mov_epi8(max2, 0x1522894a4a291, min2);

__m512i perm3 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 43, 44, 45, 46, 47, 48, 49, 42, 37, 38, 39, 40, 41, 36, 31, 32, 33, 34, 35, 26, 27, 28, 29, 30, 25, 24, 18, 19, 20, 21, 22, 23, 13, 14, 15, 16, 17, 12, 7, 8, 9, 10, 11, 6, 5, 0, 1, 2, 3, 4), v2);
__m512i min3 = _mm512_min_epu8(v2, perm3);
__m512i max3 = _mm512_max_epu8(v2, perm3);
__m512i v3 = _mm512_mask_mov_epi8(max3, 0x38618c1c6183, min3);

__m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 47, 48, 49, 44, 43, 46, 45, 40, 39, 42, 41, 37, 38, 34, 33, 36, 35, 31, 32, 29, 30, 26, 25, 28, 27, 22, 21, 24, 23, 18, 19, 20, 16, 17, 13, 12, 15, 14, 10, 11, 7, 6, 9, 8, 3, 2, 5, 4, 0, 1), v3);
__m512i min4 = _mm512_min_epu8(v3, perm4);
__m512i max4 = _mm512_max_epu8(v3, perm4);
__m512i v4 = _mm512_mask_mov_epi8(max4, 0x99a6a66534cd, min4);

__m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 38, 47, 48, 45, 46, 43, 44, 41, 42, 39, 40, 49, 37, 35, 36, 33, 34, 32, 31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 16, 17, 18, 14, 15, 12, 13, 11, 10, 8, 9, 6, 7, 4, 5, 2, 3, 1, 0), v4);
__m512i min5 = _mm512_min_epu8(v4, perm5);
__m512i max5 = _mm512_max_epu8(v4, perm5);
__m512i v5 = _mm512_mask_mov_epi8(max5, 0xaaca0aa95154, min5);

__m512i perm6 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 39, 40, 41, 42, 44, 43, 45, 46, 47, 48, 38, 37, 36, 35, 25, 26, 27, 28, 30, 29, 31, 32, 33, 34, 24, 23, 12, 13, 14, 15, 18, 17, 16, 19, 20, 21, 22, 2, 3, 4, 5, 7, 6, 8, 9, 10, 11, 1, 0), v5);
__m512i min6 = _mm512_min_epu8(v5, perm6);
__m512i max6 = _mm512_max_epu8(v5, perm6);
__m512i v6 = _mm512_mask_mov_epi8(max6, 0x7801e00f03c, min6);

__m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 45, 46, 47, 48, 49, 39, 40, 38, 37, 43, 44, 42, 41, 32, 31, 29, 30, 36, 35, 33, 34, 26, 25, 28, 27, 20, 19, 18, 17, 24, 23, 22, 21, 12, 13, 14, 15, 16, 9, 8, 11, 10, 2, 3, 1, 0, 6, 7, 5, 4), v6);
__m512i min7 = _mm512_min_epu8(v6, perm7);
__m512i max7 = _mm512_max_epu8(v6, perm7);
__m512i v7 = _mm512_mask_mov_epi8(max7, 0x61e1e61e330f, min7);

__m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 45, 46, 47, 41, 42, 43, 44, 38, 37, 40, 39, 34, 33, 36, 35, 29, 30, 31, 32, 27, 28, 25, 26, 22, 21, 24, 23, 18, 17, 20, 19, 14, 15, 16, 13, 12, 10, 11, 8, 9, 4, 5, 6, 7, 1, 0, 3, 2), v7);
__m512i min8 = _mm512_min_epu8(v7, perm8);
__m512i max8 = _mm512_max_epu8(v7, perm8);
__m512i v8 = _mm512_mask_mov_epi8(max8, 0x26666a664533, min8);

__m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 28, 47, 48, 45, 46, 43, 44, 41, 42, 39, 40, 37, 38, 35, 36, 33, 34, 31, 32, 29, 30, 49, 27, 26, 25, 23, 24, 21, 22, 19, 20, 17, 18, 15, 16, 13, 14, 8, 11, 10, 9, 12, 6, 7, 4, 5, 2, 3, 0, 1), v8);
__m512i min9 = _mm512_min_epu8(v8, perm9);
__m512i max9 = _mm512_max_epu8(v8, perm9);
__m512i v9 = _mm512_mask_mov_epi8(max9, 0xaaaab0aaa155, min9);

__m512i perm10 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 29, 30, 31, 32, 33, 34, 35, 36, 40, 39, 38, 37, 41, 42, 43, 44, 45, 46, 47, 48, 28, 27, 26, 25, 24, 23, 22, 21, 0, 1, 2, 3, 4, 5, 6, 7, 12, 11, 10, 9, 8, 13, 14, 15, 16, 17, 18, 19, 20), v9);
__m512i min10 = _mm512_min_epu8(v9, perm10);
__m512i max10 = _mm512_max_epu8(v9, perm10);
__m512i v10 = _mm512_mask_mov_epi8(max10, 0x1fe00000ff, min10);

__m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 41, 44, 43, 42, 45, 48, 47, 46, 49, 29, 30, 31, 32, 28, 27, 26, 25, 37, 38, 39, 40, 36, 35, 34, 33, 16, 15, 14, 13, 12, 9, 10, 11, 24, 23, 22, 21, 20, 17, 18, 19, 0, 3, 2, 1, 4, 7, 6, 5, 8), v10);
__m512i min11 = _mm512_min_epu8(v10, perm11);
__m512i max11 = _mm512_max_epu8(v10, perm11);
__m512i v11 = _mm512_mask_mov_epi8(max11, 0x1e01fe01fe0f, min11);

__m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 46, 47, 48, 41, 42, 43, 44, 45, 33, 34, 35, 36, 37, 38, 39, 40, 28, 27, 26, 25, 32, 31, 30, 29, 20, 19, 18, 17, 24, 23, 22, 21, 12, 9, 10, 11, 16, 13, 14, 15, 4, 5, 6, 7, 8, 1, 2, 3, 0), v11);
__m512i min12 = _mm512_min_epu8(v11, perm12);
__m512i max12 = _mm512_max_epu8(v11, perm12);
__m512i v12 = _mm512_mask_mov_epi8(max12, 0x461e1e1e1e32, min12);

__m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 45, 46, 47, 44, 41, 42, 43, 38, 37, 40, 39, 34, 33, 36, 35, 30, 29, 32, 31, 26, 25, 28, 27, 22, 21, 24, 23, 18, 17, 20, 19, 14, 13, 16, 15, 10, 9, 12, 11, 6, 7, 8, 5, 2, 3, 4, 1, 0), v12);
__m512i min13 = _mm512_min_epu8(v12, perm13);
__m512i max13 = _mm512_max_epu8(v12, perm13);
__m512i v13 = _mm512_mask_mov_epi8(max13, 0x226666666644, min13);

__m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 47, 48, 45, 46, 43, 44, 41, 42, 39, 40, 37, 38, 35, 36, 33, 34, 31, 32, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 15, 16, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0), v13);
__m512i min14 = _mm512_min_epu8(v13, perm14);
__m512i max14 = _mm512_max_epu8(v13, perm14);
__m512i v14 = _mm512_mask_mov_epi8(max14, 0xaaaaaaaaaaaa, min14);

__m512i perm15 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 31, 30, 29, 28, 27, 26, 25, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 6, 5, 4, 3, 2, 1, 0), v14);
__m512i min15 = _mm512_min_epu8(v14, perm15);
__m512i max15 = _mm512_max_epu8(v14, perm15);
__m512i v15 = _mm512_mask_mov_epi8(max15, 0x1ffff80, min15);

__m512i perm16 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 33, 32, 39, 38, 37, 36, 35, 34, 41, 40, 47, 46, 45, 44, 43, 42, 49, 48, 9, 10, 11, 12, 13, 14, 15, 8, 7, 6, 5, 4, 3, 2, 1, 0, 25, 26, 27, 28, 29, 30, 31, 24, 23, 22, 21, 20, 19, 18, 17, 16), v15);
__m512i min16 = _mm512_min_epu8(v15, perm16);
__m512i max16 = _mm512_max_epu8(v15, perm16);
__m512i v16 = _mm512_mask_mov_epi8(max16, 0xff0000ffff, min16);

__m512i perm17 = _mm512_shuffle_epi8(v16, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 48, 49, 43, 42, 45, 44, 47, 46, 33, 32, 35, 34, 37, 36, 39, 38, 41, 40, 17, 18, 19, 20, 21, 22, 23, 16, 25, 26, 27, 28, 29, 30, 31, 24, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8));
__m512i min17 = _mm512_min_epu8(v16, perm17);
__m512i max17 = _mm512_max_epu8(v16, perm17);
__m512i v17 = _mm512_mask_mov_epi8(max17, 0x10c0f00ff00ff, min17);

__m512i perm18 = _mm512_shuffle_epi8(v17, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 41, 40, 43, 42, 45, 44, 39, 38, 33, 32, 35, 34, 37, 36, 27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m512i min18 = _mm512_min_epu8(v17, perm18);
__m512i max18 = _mm512_max_epu8(v17, perm18);
__m512i v18 = _mm512_mask_mov_epi8(max18, 0x3030f0f0f0f, min18);

__m512i perm19 = _mm512_shuffle_epi8(v18, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 45, 44, 47, 46, 41, 40, 43, 42, 37, 36, 39, 38, 33, 32, 35, 34, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m512i min19 = _mm512_min_epu8(v18, perm19);
__m512i max19 = _mm512_max_epu8(v18, perm19);
__m512i v19 = _mm512_mask_mov_epi8(max19, 0x333333333333, min19);

__m512i perm20 = _mm512_shuffle_epi8(v19, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 46, 47, 44, 45, 42, 43, 40, 41, 38, 39, 36, 37, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m512i min20 = _mm512_min_epu8(v19, perm20);
__m512i max20 = _mm512_max_epu8(v19, perm20);
__m512i v20 = _mm512_mask_mov_epi8(max20, 0x555555555555, min20);

return v20;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_50_uint8_t(uint8_t * const arr) {

__m512i v = _mm512_mask_loadu_epi8(_mm512_set1_epi8(uint8_t(0xff)), 0x3ffffffffffff, arr);
v = bitonic_50_uint8_t_vec(v);
_mm512_mask_storeu_epi8((void *)arr, 0x3ffffffffffff, v);

}


#endif


#define TYPE uint8_t
#define N 50
#define SORT_NAME bitonic_50_uint8_t

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


