#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_35_uint8_t_H_
#define _SIMD_SORT_bitonic_35_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 35
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 19
	SIMD Instructions                : 2 / 95
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512vbmi, AVX512bw
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
__m512i __attribute__((const)) bitonic_35_uint8_t_vec(__m512i v) {

__m512i perm0 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 33, 34, 32, 30, 31, 28, 29, 26, 27, 24, 25, 23, 21, 22, 19, 20, 17, 18, 15, 16, 14, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1), v);
__m512i min0 = _mm512_min_epu8(v, perm0);
__m512i max0 = _mm512_max_epu8(v, perm0);
__m512i v0 = _mm512_mask_mov_epi8(max0, 0x2552a9555, min0);

__m512i perm1 = _mm512_shuffle_epi8(v0, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 32, 33, 34, 31, 30, 26, 27, 28, 29, 25, 23, 24, 22, 21, 17, 18, 19, 20, 16, 14, 15, 13, 12, 8, 9, 10, 11, 4, 5, 6, 7, 0, 1, 2, 3));
__m512i min1 = _mm512_min_epu8(v0, perm1);
__m512i max1 = _mm512_max_epu8(v0, perm1);
__m512i v1 = _mm512_mask_mov_epi8(max1, 0x10c864333, min1);

__m512i perm2 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 31, 32, 33, 34, 30, 28, 29, 26, 27, 24, 25, 21, 22, 23, 19, 20, 17, 18, 15, 16, 12, 13, 14, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1), v1);
__m512i min2 = _mm512_min_epu8(v1, perm2);
__m512i max2 = _mm512_max_epu8(v1, perm2);
__m512i v2 = _mm512_mask_mov_epi8(max2, 0x1952a9555, min2);

__m512i perm3 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 29, 30, 31, 32, 33, 34, 28, 27, 26, 23, 22, 25, 24, 17, 20, 19, 18, 21, 14, 13, 16, 15, 8, 11, 10, 9, 12, 0, 1, 2, 3, 4, 5, 6, 7), v2);
__m512i min3 = _mm512_min_epu8(v2, perm3);
__m512i max3 = _mm512_max_epu8(v2, perm3);
__m512i v3 = _mm512_mask_mov_epi8(max3, 0xe0c2610f, min3);

__m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 32, 33, 30, 31, 29, 28, 27, 26, 24, 25, 22, 23, 21, 20, 19, 18, 17, 15, 16, 13, 14, 12, 11, 10, 9, 8, 5, 4, 7, 6, 1, 0, 3, 2), v3);
__m512i min4 = _mm512_min_epu8(v3, perm4);
__m512i max4 = _mm512_max_epu8(v3, perm4);
__m512i v4 = _mm512_mask_mov_epi8(max4, 0x14140a033, min4);

__m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 26, 27, 28, 29, 30, 31, 32, 33, 21, 18, 19, 20, 25, 22, 23, 24, 17, 12, 9, 10, 11, 16, 13, 14, 15, 8, 6, 7, 4, 5, 2, 3, 0, 1), v4);
__m512i min5 = _mm512_min_epu8(v4, perm5);
__m512i max5 = _mm512_max_epu8(v4, perm5);
__m512i v5 = _mm512_mask_mov_epi8(max5, 0x3c3c1e55, min5);

__m512i perm6 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 31, 30, 33, 32, 27, 26, 29, 28, 23, 22, 25, 24, 19, 18, 21, 20, 17, 14, 13, 16, 15, 10, 9, 12, 11, 0, 7, 6, 5, 4, 3, 2, 1, 8), v5);
__m512i min6 = _mm512_min_epu8(v5, perm6);
__m512i max6 = _mm512_max_epu8(v5, perm6);
__m512i v6 = _mm512_mask_mov_epi8(max6, 0xcccc6601, min6);

__m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 17, 15, 16, 13, 14, 11, 12, 9, 10, 8, 7, 6, 5, 4, 3, 2, 1, 0), v6);
__m512i min7 = _mm512_min_epu8(v6, perm7);
__m512i max7 = _mm512_max_epu8(v6, perm7);
__m512i v7 = _mm512_mask_mov_epi8(max7, 0x15554aa00, min7);

__m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 24, 25, 18, 19, 20, 21, 22, 23, 26, 33, 34, 27, 28, 29, 30, 31, 32, 17, 8, 1, 2, 3, 4, 5, 6, 7, 16, 9, 10, 11, 12, 13, 14, 15, 0), v7);
__m512i min8 = _mm512_min_epu8(v7, perm8);
__m512i max8 = _mm512_max_epu8(v7, perm8);
__m512i v8 = _mm512_mask_mov_epi8(max8, 0x3fc01fe, min8);

__m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 33, 34, 28, 27, 30, 29, 32, 31, 24, 17, 26, 19, 18, 21, 20, 23, 22, 25, 12, 11, 10, 9, 16, 15, 14, 13, 4, 3, 2, 1, 8, 7, 6, 5, 0), v8);
__m512i min9 = _mm512_min_epu8(v8, perm9);
__m512i max9 = _mm512_max_epu8(v8, perm9);
__m512i v9 = _mm512_mask_mov_epi8(max9, 0x2190e1e1e, min9);

__m512i perm10 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 26, 25, 28, 27, 30, 29, 20, 23, 22, 17, 24, 19, 18, 21, 14, 13, 16, 15, 10, 9, 12, 11, 6, 5, 8, 7, 2, 1, 4, 3, 0), v9);
__m512i min10 = _mm512_min_epu8(v9, perm10);
__m512i max10 = _mm512_max_epu8(v9, perm10);
__m512i v10 = _mm512_mask_mov_epi8(max10, 0x6126666, min10);

__m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 30, 29, 32, 31, 26, 25, 28, 27, 22, 21, 24, 23, 18, 17, 20, 19, 15, 16, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0), v10);
__m512i min11 = _mm512_min_epu8(v10, perm11);
__m512i max11 = _mm512_max_epu8(v10, perm11);
__m512i v11 = _mm512_mask_mov_epi8(max11, 0x6666aaaa, min11);

__m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 14, 15, 31, 32, 29, 30, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 16, 33, 34, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), v11);
__m512i min12 = _mm512_min_epu8(v11, perm12);
__m512i max12 = _mm512_max_epu8(v11, perm12);
__m512i v12 = _mm512_mask_mov_epi8(max12, 0xaaaac000, min12);

__m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 0), v12);
__m512i min13 = _mm512_min_epu8(v12, perm13);
__m512i max13 = _mm512_max_epu8(v12, perm13);
__m512i v13 = _mm512_mask_mov_epi8(max13, 0x1fffe, min13);

__m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 32, 33, 34, 23, 22, 21, 20, 19, 18, 17, 24, 31, 30, 29, 28, 27, 26, 25, 0, 7, 6, 5, 4, 3, 2, 1, 8, 15, 14, 13, 12, 11, 10, 9, 16), v13);
__m512i min14 = _mm512_min_epu8(v13, perm14);
__m512i max14 = _mm512_max_epu8(v13, perm14);
__m512i v14 = _mm512_mask_mov_epi8(max14, 0x100fe00ff, min14);

__m512i perm15 = _mm512_shuffle_epi8(v14, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 32, 33, 27, 26, 25, 28, 31, 30, 29, 16, 19, 18, 17, 20, 23, 22, 21, 24, 11, 10, 9, 12, 15, 14, 13, 0, 3, 2, 1, 4, 7, 6, 5, 8));
__m512i min15 = _mm512_min_epu8(v14, perm15);
__m512i max15 = _mm512_max_epu8(v14, perm15);
__m512i v15 = _mm512_mask_mov_epi8(max15, 0x10e0f0e0f, min15);

__m512i perm16 = _mm512_shuffle_epi8(v15, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 30, 31, 24, 25, 26, 27, 28, 21, 22, 23, 16, 17, 18, 19, 20, 13, 14, 15, 8, 9, 10, 11, 12, 5, 6, 7, 0, 1, 2, 3, 4));
__m512i min16 = _mm512_min_epu8(v15, perm16);
__m512i max16 = _mm512_max_epu8(v15, perm16);
__m512i v16 = _mm512_mask_mov_epi8(max16, 0x23232323, min16);

__m512i perm17 = _mm512_shuffle_epi8(v16, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 28, 29, 30, 27, 24, 25, 26, 23, 20, 21, 22, 19, 16, 17, 18, 15, 12, 13, 14, 11, 8, 9, 10, 7, 4, 5, 6, 3, 0, 1, 2));
__m512i min17 = _mm512_min_epu8(v16, perm17);
__m512i max17 = _mm512_max_epu8(v16, perm17);
__m512i v17 = _mm512_mask_mov_epi8(max17, 0x11111111, min17);

__m512i perm18 = _mm512_shuffle_epi8(v17, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m512i min18 = _mm512_min_epu8(v17, perm18);
__m512i max18 = _mm512_max_epu8(v17, perm18);
__m512i v18 = _mm512_mask_mov_epi8(max18, 0x55555555, min18);

return v18;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_35_uint8_t(uint8_t * const arr) {

__m512i v = _mm512_load_si512((__m512i *)arr);

v = bitonic_35_uint8_t_vec(v);

_mm512_store_si512((__m512i *)arr, v);

}


#endif


#define TYPE uint8_t
#define N 35
#define SORT_NAME bitonic_35_uint8_t

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


