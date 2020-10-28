#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_41_int8_t_H_
#define _SIMD_SORT_bitonic_41_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 41
	Underlying Sort Type             : int8_t
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
__m512i __attribute__((const)) bitonic_41_int8_t_vec(__m512i v) {

__m512i perm0 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 39, 40, 38, 36, 37, 35, 33, 34, 32, 30, 31, 28, 29, 27, 25, 26, 23, 24, 22, 20, 21, 18, 19, 17, 15, 16, 13, 14, 12, 10, 11, 8, 9, 7, 5, 6, 3, 4, 2, 0, 1), v);
__m512i min0 = _mm512_min_epi8(v, perm0);
__m512i max0 = _mm512_max_epi8(v, perm0);
__m512i v0 = _mm512_mask_mov_epi8(max0, 0x925294a529, min0);

__m512i perm1 = _mm512_shuffle_epi8(v0, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 38, 39, 40, 37, 35, 36, 34, 32, 33, 31, 30, 29, 27, 28, 26, 25, 22, 23, 24, 21, 20, 19, 17, 18, 16, 15, 12, 13, 14, 11, 10, 7, 8, 9, 6, 5, 4, 2, 3, 1, 0));
__m512i min1 = _mm512_min_epi8(v0, perm1);
__m512i max1 = _mm512_max_epi8(v0, perm1);
__m512i v1 = _mm512_mask_mov_epi8(max1, 0x4908421084, min1);

__m512i perm2 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 38, 39, 36, 37, 35, 33, 34, 30, 31, 32, 28, 29, 25, 26, 27, 21, 22, 23, 24, 20, 18, 19, 15, 16, 17, 11, 12, 13, 14, 10, 6, 7, 8, 9, 5, 3, 4, 0, 1, 2), v1);
__m512i min2 = _mm512_min_epi8(v1, perm2);
__m512i max2 = _mm512_max_epi8(v1, perm2);
__m512i v2 = _mm512_mask_mov_epi8(max2, 0x52526498c9, min2);

__m512i perm3 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 36, 37, 38, 39, 40, 35, 32, 31, 34, 33, 30, 27, 26, 29, 28, 25, 24, 20, 21, 22, 23, 17, 16, 19, 18, 15, 14, 10, 11, 12, 13, 9, 5, 6, 7, 8, 2, 1, 4, 3, 0), v2);
__m512i min3 = _mm512_min_epi8(v2, perm3);
__m512i max3 = _mm512_max_epi8(v2, perm3);
__m512i v3 = _mm512_mask_mov_epi8(max3, 0x318c330c66, min3);

__m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 39, 40, 36, 35, 38, 37, 33, 34, 31, 32, 30, 28, 29, 26, 27, 25, 24, 22, 23, 20, 21, 18, 19, 16, 17, 15, 14, 12, 13, 10, 11, 9, 7, 8, 5, 6, 3, 4, 1, 2, 0), v3);
__m512i min4 = _mm512_min_epi8(v3, perm4);
__m512i max4 = _mm512_max_epi8(v3, perm4);
__m512i v4 = _mm512_mask_mov_epi8(max4, 0x9a945514aa, min4);

__m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 32, 33, 37, 38, 35, 36, 34, 39, 40, 31, 30, 29, 22, 23, 20, 21, 24, 27, 28, 25, 26, 19, 12, 13, 10, 11, 14, 17, 18, 15, 16, 3, 4, 1, 2, 5, 8, 9, 6, 7, 0), v4);
__m512i min5 = _mm512_min_epi8(v4, perm5);
__m512i max5 = _mm512_max_epi8(v4, perm5);
__m512i v5 = _mm512_mask_mov_epi8(max5, 0x2b00f03c1e, min5);

__m512i perm6 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 34, 31, 32, 33, 38, 35, 36, 37, 30, 25, 28, 27, 24, 29, 26, 23, 22, 20, 21, 15, 18, 17, 14, 19, 16, 13, 12, 10, 11, 8, 9, 7, 6, 3, 0, 5, 2, 1, 4), v5);
__m512i min6 = _mm512_min_epi8(v5, perm6);
__m512i max6 = _mm512_max_epi8(v5, perm6);
__m512i v6 = _mm512_mask_mov_epi8(max6, 0x78310c509, min6);

__m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 38, 39, 40, 35, 36, 37, 30, 31, 32, 33, 34, 27, 26, 29, 28, 23, 22, 25, 24, 21, 20, 17, 16, 19, 18, 13, 12, 15, 14, 11, 10, 9, 8, 5, 4, 7, 6, 1, 0, 3, 2), v6);
__m512i min7 = _mm512_min_epi8(v6, perm7);
__m512i max7 = _mm512_max_epi8(v6, perm7);
__m512i v7 = _mm512_mask_mov_epi8(max7, 0x48ccc33033, min7);

__m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 38, 39, 37, 34, 35, 36, 33, 30, 31, 32, 28, 29, 26, 27, 24, 25, 22, 23, 21, 20, 18, 19, 16, 17, 14, 15, 12, 13, 11, 10, 9, 8, 6, 7, 4, 5, 2, 3, 0, 1), v7);
__m512i min8 = _mm512_min_epi8(v7, perm8);
__m512i max8 = _mm512_max_epi8(v7, perm8);
__m512i v8 = _mm512_mask_mov_epi8(max8, 0x4455455055, min8);

__m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 25, 26, 27, 36, 37, 34, 35, 32, 33, 30, 31, 29, 28, 38, 39, 40, 24, 23, 22, 21, 20, 19, 18, 4, 5, 6, 7, 0, 1, 2, 3, 9, 8, 14, 15, 16, 17, 10, 11, 12, 13), v8);
__m512i min9 = _mm512_min_epi8(v8, perm9);
__m512i max9 = _mm512_max_epi8(v8, perm9);
__m512i v9 = _mm512_mask_mov_epi8(max9, 0x154e0000ff, min9);

__m512i perm10 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 28, 29, 22, 23, 24, 25, 26, 27, 36, 37, 30, 31, 32, 33, 34, 35, 21, 20, 11, 10, 17, 16, 15, 14, 8, 9, 19, 18, 12, 13, 7, 6, 5, 4, 1, 0, 3, 2), v9);
__m512i min10 = _mm512_min_epi8(v9, perm10);
__m512i max10 = _mm512_max_epi8(v9, perm10);
__m512i v10 = _mm512_mask_mov_epi8(max10, 0x3fc00f03, min10);

__m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 36, 37, 38, 39, 40, 31, 30, 33, 32, 35, 34, 21, 20, 23, 22, 25, 24, 27, 26, 29, 28, 15, 14, 13, 12, 19, 18, 17, 16, 6, 7, 5, 4, 10, 11, 9, 8, 2, 3, 0, 1), v10);
__m512i min11 = _mm512_min_epi8(v10, perm11);
__m512i max11 = _mm512_max_epi8(v10, perm11);
__m512i v11 = _mm512_mask_mov_epi8(max11, 0x30c0f0f0f5, min11);

__m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 36, 37, 38, 35, 34, 28, 29, 31, 30, 32, 33, 27, 26, 21, 20, 23, 22, 25, 24, 17, 16, 19, 18, 13, 12, 15, 14, 8, 9, 10, 11, 5, 4, 7, 6, 3, 2, 1, 0), v11);
__m512i min12 = _mm512_min_epi8(v11, perm12);
__m512i max12 = _mm512_max_epi8(v11, perm12);
__m512i v12 = _mm512_mask_mov_epi8(max12, 0x1030333330, min12);

__m512i perm13 = _mm512_shuffle_epi8(v12, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 38, 39, 36, 37, 33, 32, 35, 34, 28, 29, 30, 31, 25, 24, 27, 26, 21, 20, 23, 22, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 3, 2, 1, 0));
__m512i min13 = _mm512_min_epi8(v12, perm13);
__m512i max13 = _mm512_max_epi8(v12, perm13);
__m512i v13 = _mm512_mask_mov_epi8(max13, 0x5333355550, min13);

__m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 11, 12, 13, 14, 15, 34, 35, 32, 33, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 19, 18, 17, 16, 36, 37, 38, 39, 40, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0), v13);
__m512i min14 = _mm512_min_epi8(v13, perm14);
__m512i max14 = _mm512_max_epi8(v13, perm14);
__m512i v14 = _mm512_mask_mov_epi8(max14, 0x55550f800, min14);

__m512i perm15 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 16, 17, 18, 19, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 32, 33, 34, 35, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 3, 2, 1, 0), v14);
__m512i min15 = _mm512_min_epi8(v14, perm15);
__m512i max15 = _mm512_max_epi8(v14, perm15);
__m512i v15 = _mm512_mask_mov_epi8(max15, 0xffff0, min15);

__m512i perm16 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 32, 35, 34, 33, 36, 39, 38, 37, 40, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16), v15);
__m512i min16 = _mm512_min_epi8(v15, perm16);
__m512i max16 = _mm512_max_epi8(v15, perm16);
__m512i v16 = _mm512_mask_mov_epi8(max16, 0xf00f000ff, min16);

__m512i perm17 = _mm512_shuffle_epi8(v16, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 37, 38, 39, 32, 33, 34, 35, 36, 31, 30, 29, 28, 16, 17, 18, 19, 23, 22, 21, 20, 24, 25, 26, 27, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8));
__m512i min17 = _mm512_min_epi8(v16, perm17);
__m512i max17 = _mm512_max_epi8(v16, perm17);
__m512i v17 = _mm512_mask_mov_epi8(max17, 0x23000f000f, min17);

__m512i perm18 = _mm512_shuffle_epi8(v17, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 36, 37, 38, 35, 32, 33, 34, 27, 26, 25, 24, 31, 30, 29, 28, 16, 17, 18, 19, 20, 21, 22, 23, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m512i min18 = _mm512_min_epi8(v17, perm18);
__m512i max18 = _mm512_max_epi8(v17, perm18);
__m512i v18 = _mm512_mask_mov_epi8(max18, 0x110f0f0f0f, min18);

__m512i perm19 = _mm512_shuffle_epi8(v18, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 38, 39, 36, 37, 34, 35, 32, 33, 29, 28, 31, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m512i min19 = _mm512_min_epi8(v18, perm19);
__m512i max19 = _mm512_max_epi8(v18, perm19);
__m512i v19 = _mm512_mask_mov_epi8(max19, 0x5533333333, min19);

__m512i perm20 = _mm512_shuffle_epi8(v19, _mm512_set_epi8(63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 30, 31, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m512i min20 = _mm512_min_epi8(v19, perm20);
__m512i max20 = _mm512_max_epi8(v19, perm20);
__m512i v20 = _mm512_mask_mov_epi8(max20, 0x55555555, min20);

return v20;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_41_int8_t(int8_t * const arr) {

__m512i _tmp0 = _mm512_set1_epi8(int8_t(0x7f));
__m512i v = _mm512_mask_loadu_epi8(_tmp0, 0x1ffffffffff, arr);

v = bitonic_41_int8_t_vec(v);

_mm512_mask_storeu_epi8((void *)arr, 0x1ffffffffff, v);

}


#endif


#define TYPE int8_t
#define N 41
#define SORT_NAME bitonic_41_int8_t

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


