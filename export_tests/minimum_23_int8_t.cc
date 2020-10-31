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

#define TYPE int8_t
#define N 23
#define SORT_NAME minimum_23_int8_t

#ifndef _SIMD_SORT_minimum_23_int8_t_H_
#define _SIMD_SORT_minimum_23_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 23
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 12
	SIMD Instructions                : 3 / 60
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX2, AVX512vbmi, AVX
	SIMD Instruction Set(s) Excluded : None
	Aligned Load & Store             : False
	Integer Aligned Load & Store     : False
	Full Load & Store                : False
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



void fill_works(__m256i v) {
sarr<TYPE, N> t;
memcpy(t.arr, &v, 32);
int i = N;for (; i < 32; ++i) {
assert(t.arr[i] == int8_t(0x7f));
}
}

/* SIMD Sort */
__m256i __attribute__((const)) minimum_23_int8_t_vec(__m256i v) {

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [20,21], [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1) */
__m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m256i min0 = _mm256_min_epi8(v, perm0);
__m256i max0 = _mm256_max_epi8(v, perm0);
__m256i v0 = _mm256_mask_mov_epi8(max0, 0x155555, min0);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [20,22], [21,21], [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 21, 22, 17, 16, 19, 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2) */
__m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 21, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min1 = _mm256_min_epi8(v0, perm1);
__m256i max1 = _mm256_max_epi8(v0, perm1);
__m256i v1 = _mm256_mask_mov_epi8(max1, 0x133333, min1);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [18,22], [15,21], [14,20], [19,19], [13,17], [12,16], [7,11], [6,10], [3,9], [2,8], [1,5], [0,4]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 18, 15, 14, 19, 22, 13, 12, 21, 20, 17, 16,  7,  6,  3,  2, 11, 10,  1,  0,  9,  8,  5,  4) */
__m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 18, 15, 14, 19, 22, 13, 12, 21, 20, 17, 16, 7, 6, 3, 2, 11, 10, 1, 0, 9, 8, 5, 4), v1);
__m256i min2 = _mm256_min_epi8(v1, perm2);
__m256i max2 = _mm256_max_epi8(v1, perm2);
__m256i v2 = _mm256_mask_mov_epi8(max2, 0x4f0cf, min2);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [20,22], [21,21], [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 21, 22, 17, 16, 19, 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2) */
__m256i perm3 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 21, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min3 = _mm256_min_epi8(v2, perm3);
__m256i max3 = _mm256_max_epi8(v2, perm3);
__m256i v3 = _mm256_mask_mov_epi8(max3, 0x133333, min3);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [10,22], [19,21], [18,20], [15,17], [14,16], [1,13], [0,12], [11,11], [7,9], [6,8], [3,5], [2,4]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 10, 19, 18, 21, 20, 15, 14, 17, 16,  1,  0, 11, 22,  7,  6,  9,  8,  3,  2,  5,  4, 13, 12) */
__m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 10, 19, 18, 21, 20, 15, 14, 17, 16, 1, 0, 11, 22, 7, 6, 9, 8, 3, 2, 5, 4, 13, 12), v3);
__m256i min4 = _mm256_min_epi8(v3, perm4);
__m256i max4 = _mm256_max_epi8(v3, perm4);
__m256i v4 = _mm256_mask_mov_epi8(max4, 0xcc4cf, min4);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [11,22], [9,21], [8,20], [7,19], [6,18], [5,17], [4,16], [3,15], [2,14], [13,13], [1,12], [10,10], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 11,  9,  8,  7,  6,  5,  4,  3,  2, 13,  1, 22, 10, 21, 20, 19, 18, 17, 16, 15, 14, 12,  0) */
__m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 11, 9, 8, 7, 6, 5, 4, 3, 2, 13, 1, 22, 10, 21, 20, 19, 18, 17, 16, 15, 14, 12, 0), v4);
__m256i min5 = _mm256_min_epi8(v4, perm5);
__m256i max5 = _mm256_max_epi8(v4, perm5);
__m256i v5 = _mm256_mask_mov_epi8(max5, 0xbfe, min5);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [21,22], [9,20], [17,19], [16,18], [10,15], [3,14], [8,13], [12,12], [11,11], [5,7], [4,6], [1,2], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 21, 22,  9, 17, 16, 19, 18, 10,  3,  8, 12, 11, 15, 20, 13,  5,  4,  7,  6, 14,  1,  2,  0) */
__m256i perm6 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 21, 22, 9, 17, 16, 19, 18, 10, 3, 8, 12, 11, 15, 20, 13, 5, 4, 7, 6, 14, 1, 2, 0), v5);
__m256i min6 = _mm256_min_epi8(v5, perm6);
__m256i max6 = _mm256_max_epi8(v5, perm6);
__m256i v6 = _mm256_mask_mov_epi8(max6, 0x23073a, min6);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [21,21], [17,20], [19,19], [7,18], [5,16], [11,15], [10,14], [9,13], [8,12], [3,6], [4,4], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 17, 19,  7, 20,  5, 11, 10,  9,  8, 15, 14, 13, 12, 18,  3, 16,  4,  6,  2,  1,  0) */
__m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 17, 19, 7, 20, 5, 11, 10, 9, 8, 15, 14, 13, 12, 18, 3, 16, 4, 6, 2, 1, 0), v6);
__m256i min7 = _mm256_min_epi8(v6, perm7);
__m256i max7 = _mm256_max_epi8(v6, perm7);
__m256i v7 = _mm256_mask_mov_epi8(max7, 0x20fa8, min7);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [20,21], [15,19], [11,18], [13,17], [9,16], [7,14], [5,12], [6,10], [4,8], [2,3], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 15, 11, 13,  9, 19,  7, 17,  5, 18,  6, 16,  4, 14, 10, 12,  8,  2,  3,  1,  0) */
__m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 15, 11, 13, 9, 19, 7, 17, 5, 18, 6, 16, 4, 14, 10, 12, 8, 2, 3, 1, 0), v7);
__m256i min8 = _mm256_min_epi8(v7, perm8);
__m256i max8 = _mm256_max_epi8(v7, perm8);
__m256i v8 = _mm256_mask_mov_epi8(max8, 0x10aaf4, min8);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [19,21], [20,20], [15,18], [17,17], [14,16], [11,13], [10,12], [7,9], [5,8], [6,6], [2,4], [3,3], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 19, 20, 21, 15, 17, 14, 18, 16, 11, 10, 13, 12,  7,  5,  9,  6,  8,  2,  3,  4,  1,  0) */
__m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 19, 20, 21, 15, 17, 14, 18, 16, 11, 10, 13, 12, 7, 5, 9, 6, 8, 2, 3, 4, 1, 0), v8);
__m256i min9 = _mm256_min_epi8(v8, perm9);
__m256i max9 = _mm256_max_epi8(v8, perm9);
__m256i v9 = _mm256_mask_mov_epi8(max9, 0x8cca4, min9);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [21,21], [18,20], [19,19], [15,17], [13,16], [11,14], [9,12], [7,10], [6,8], [3,5], [4,4], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 18, 19, 20, 15, 13, 17, 11, 16,  9, 14,  7, 12,  6, 10,  8,  3,  4,  5,  2,  1,  0) */
__m256i perm10 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 18, 19, 20, 15, 13, 17, 11, 16, 9, 14, 7, 12, 6, 10, 8, 3, 4, 5, 2, 1, 0), v9);
__m256i min10 = _mm256_min_epi8(v9, perm10);
__m256i max10 = _mm256_max_epi8(v9, perm10);
__m256i v10 = _mm256_mask_mov_epi8(max10, 0x4aac8, min10);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [21,21], [19,20], [17,18], [15,16], [13,14], [11,12], [9,10], [7,8], [5,6], [3,4], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 19, 20, 17, 18, 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  2,  1,  0) */
__m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 19, 20, 17, 18, 15, 16, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 2, 1, 0), v10);
__m256i min11 = _mm256_min_epi8(v10, perm11);
__m256i max11 = _mm256_max_epi8(v10, perm11);
__m256i v11 = _mm256_mask_mov_epi8(max11, 0xaaaa8, min11);

return v11;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) minimum_23_int8_t(int8_t * const arr) {

__m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
__m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x7fffff, arr);
fill_works(v);
v = minimum_23_int8_t_vec(v);

fill_works(v);_mm256_mask_storeu_epi8((void *)arr, 0x7fffff, v);

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


