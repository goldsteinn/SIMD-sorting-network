#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_36_int8_t_H_
#define _SIMD_SORT_bitonic_36_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 36
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 19
	SIMD Instructions                : 3 / 95
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
__m512i __attribute__((const)) bitonic_36_int8_t_vec(__m512i v) {

__m512i perm0 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 34, 35, 33, 31, 32, 29, 30, 27, 28, 25, 26, 24, 22, 23, 20, 21, 18, 19, 16, 17, 15, 13, 14, 11, 12, 9, 10, 7, 8, 6, 4, 5, 2, 3, 0, 1), v);
__m512i min0 = _mm512_min_epi8(v, perm0);
__m512i max0 = _mm512_max_epi8(v, perm0);
__m512i v0 = _mm512_mask_mov_epi8(max0, 0x4aa552a95, min0);

__m512i perm1 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 33, 34, 35, 32, 31, 27, 28, 29, 30, 26, 24, 25, 23, 22, 18, 19, 20, 21, 17, 15, 16, 14, 13, 9, 10, 11, 12, 6, 7, 8, 5, 4, 0, 1, 2, 3), v0);
__m512i min1 = _mm512_min_epi8(v0, perm1);
__m512i max1 = _mm512_max_epi8(v0, perm1);
__m512i v1 = _mm512_mask_mov_epi8(max1, 0x2190c8643, min1);

__m512i perm2 = _mm512_shuffle_epi8(v1, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 32, 33, 34, 35, 31, 29, 30, 27, 28, 25, 26, 22, 23, 24, 20, 21, 18, 19, 16, 17, 13, 14, 15, 11, 12, 9, 10, 5, 6, 7, 8, 4, 2, 3, 0, 1));
__m512i min2 = _mm512_min_epi8(v1, perm2);
__m512i max2 = _mm512_max_epi8(v1, perm2);
__m512i v2 = _mm512_mask_mov_epi8(max2, 0x32a552a65, min2);

__m512i perm3 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 30, 31, 32, 33, 34, 35, 29, 28, 27, 24, 23, 26, 25, 18, 21, 20, 19, 22, 15, 14, 17, 16, 9, 12, 11, 10, 13, 3, 4, 5, 6, 7, 8, 2, 1, 0), v2);
__m512i min3 = _mm512_min_epi8(v2, perm3);
__m512i max3 = _mm512_max_epi8(v2, perm3);
__m512i v3 = _mm512_mask_mov_epi8(max3, 0x1c184c238, min3);

__m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 33, 34, 31, 32, 30, 29, 28, 27, 25, 26, 23, 24, 22, 21, 20, 19, 18, 16, 17, 14, 15, 13, 12, 11, 10, 9, 8, 6, 7, 4, 5, 3, 2, 1, 0), v3);
__m512i min4 = _mm512_min_epi8(v3, perm4);
__m512i max4 = _mm512_max_epi8(v3, perm4);
__m512i v4 = _mm512_mask_mov_epi8(max4, 0x282814050, min4);

__m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 27, 28, 29, 30, 31, 32, 33, 34, 22, 19, 20, 21, 26, 23, 24, 25, 18, 13, 10, 11, 12, 17, 14, 15, 16, 9, 8, 0, 1, 2, 3, 4, 5, 6, 7), v4);
__m512i min5 = _mm512_min_epi8(v4, perm5);
__m512i max5 = _mm512_max_epi8(v4, perm5);
__m512i v5 = _mm512_mask_mov_epi8(max5, 0x78783c0f, min5);

__m512i perm6 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 32, 31, 34, 33, 28, 27, 30, 29, 24, 23, 26, 25, 20, 19, 22, 21, 18, 15, 14, 17, 16, 11, 10, 13, 12, 9, 8, 5, 4, 7, 6, 1, 0, 3, 2), v5);
__m512i min6 = _mm512_min_epi8(v5, perm6);
__m512i max6 = _mm512_max_epi8(v5, perm6);
__m512i v6 = _mm512_mask_mov_epi8(max6, 0x19998cc33, min6);

__m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 33, 34, 31, 32, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 18, 16, 17, 14, 15, 12, 13, 10, 11, 9, 8, 6, 7, 4, 5, 2, 3, 0, 1), v6);
__m512i min7 = _mm512_min_epi8(v6, perm7);
__m512i max7 = _mm512_max_epi8(v6, perm7);
__m512i v7 = _mm512_mask_mov_epi8(max7, 0x2aaa95455, min7);

__m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 25, 26, 19, 20, 21, 22, 23, 24, 27, 34, 35, 28, 29, 30, 31, 32, 33, 18, 17, 2, 3, 4, 5, 6, 7, 0, 1, 8, 11, 12, 13, 14, 15, 16, 9, 10), v7);
__m512i min8 = _mm512_min_epi8(v7, perm8);
__m512i max8 = _mm512_max_epi8(v7, perm8);
__m512i v8 = _mm512_mask_mov_epi8(max8, 0x7f800ff, min8);

__m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 34, 35, 29, 28, 31, 30, 33, 32, 25, 18, 27, 20, 19, 22, 21, 24, 23, 26, 9, 12, 11, 14, 13, 16, 15, 8, 17, 10, 3, 2, 5, 4, 7, 6, 0, 1), v8);
__m512i min9 = _mm512_min_epi8(v8, perm9);
__m512i max9 = _mm512_max_epi8(v8, perm9);
__m512i v9 = _mm512_mask_mov_epi8(max9, 0x4321c1b0d, min9);

__m512i perm10 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 27, 26, 29, 28, 31, 30, 21, 24, 23, 18, 25, 20, 19, 22, 13, 16, 15, 10, 17, 12, 11, 14, 5, 4, 7, 6, 9, 8, 3, 2, 1, 0), v9);
__m512i min10 = _mm512_min_epi8(v9, perm10);
__m512i max10 = _mm512_max_epi8(v9, perm10);
__m512i v10 = _mm512_mask_mov_epi8(max10, 0xc242430, min10);

__m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 31, 30, 33, 32, 27, 26, 29, 28, 23, 22, 25, 24, 19, 18, 21, 20, 15, 14, 17, 16, 11, 10, 13, 12, 7, 6, 9, 8, 3, 2, 5, 4, 1, 0), v10);
__m512i min11 = _mm512_min_epi8(v10, perm11);
__m512i max11 = _mm512_max_epi8(v10, perm11);
__m512i v11 = _mm512_mask_mov_epi8(max11, 0xcccccccc, min11);

__m512i perm12 = _mm512_shuffle_epi8(v11, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 1, 0));
__m512i min12 = _mm512_min_epi8(v11, perm12);
__m512i max12 = _mm512_max_epi8(v11, perm12);
__m512i v12 = _mm512_mask_mov_epi8(max12, 0x155555554, min12);

__m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 14, 15, 16, 17, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 19, 18, 32, 33, 34, 35, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 1, 0), v12);
__m512i min13 = _mm512_min_epi8(v12, perm13);
__m512i max13 = _mm512_max_epi8(v12, perm13);
__m512i v13 = _mm512_mask_mov_epi8(max13, 0x3fffc, min13);

__m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 33, 32, 35, 34, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28, 14, 15, 1, 0, 18, 19, 5, 4, 3, 2, 9, 8, 7, 6, 13, 12, 11, 10, 17, 16), v13);
__m512i min14 = _mm512_min_epi8(v13, perm14);
__m512i max14 = _mm512_max_epi8(v13, perm14);
__m512i v14 = _mm512_mask_mov_epi8(max14, 0x300f0c03f, min14);

__m512i perm15 = _mm512_shuffle_epi8(v14, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 34, 35, 32, 33, 31, 30, 29, 28, 19, 18, 16, 17, 23, 22, 21, 20, 27, 26, 24, 25, 7, 6, 13, 12, 11, 10, 1, 0, 15, 14, 5, 4, 3, 2, 9, 8));
__m512i min15 = _mm512_min_epi8(v14, perm15);
__m512i max15 = _mm512_max_epi8(v14, perm15);
__m512i v15 = _mm512_mask_mov_epi8(max15, 0x5000f00c3, min15);

__m512i perm16 = _mm512_shuffle_epi8(v15, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 27, 26, 25, 24, 31, 30, 29, 28, 19, 18, 16, 17, 23, 22, 20, 21, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m512i min16 = _mm512_min_epi8(v15, perm16);
__m512i max16 = _mm512_max_epi8(v15, perm16);
__m512i v16 = _mm512_mask_mov_epi8(max16, 0xf0f0f0f, min16);

__m512i perm17 = _mm512_shuffle_epi8(v16, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 16, 17, 18, 19, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m512i min17 = _mm512_min_epi8(v16, perm17);
__m512i max17 = _mm512_max_epi8(v16, perm17);
__m512i v17 = _mm512_mask_mov_epi8(max17, 0x33333333, min17);

__m512i perm18 = _mm512_shuffle_epi8(v17, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m512i min18 = _mm512_min_epi8(v17, perm18);
__m512i max18 = _mm512_max_epi8(v17, perm18);
__m512i v18 = _mm512_mask_mov_epi8(max18, 0x55555555, min18);

return v18;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_36_int8_t(int8_t * const arr) {

__m512i v = _mm512_mask_loadu_epi8(_mm512_set1_epi8(int8_t(0x7f)), 0xfffffffff, arr);
v = bitonic_36_int8_t_vec(v);
_mm512_mask_storeu_epi8((void *)arr, 0xfffffffff, v);

}


#endif


#define TYPE int8_t
#define N 36
#define SORT_NAME bitonic_36_int8_t

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


