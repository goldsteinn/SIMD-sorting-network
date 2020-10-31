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
#define N 31
#define SORT_NAME minimum_31_int8_t

#ifndef _SIMD_SORT_minimum_31_int8_t_H_
#define _SIMD_SORT_minimum_31_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 31
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 14
	SIMD Instructions                : 3 / 70
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
__m256i __attribute__((const)) minimum_31_int8_t_vec(__m256i v) {

/* Pairs: ([31,31], [30,30], [28,29], [26,27], [24,25], [22,23], [20,21], [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], [0,1]) */
/* Perm:  (31, 30, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  1) */
__m256i perm0 = _mm256_shuffle_epi8(v, _mm256_set_epi8(31, 30, 28, 29, 26, 27, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m256i min0 = _mm256_min_epi8(v, perm0);
__m256i max0 = _mm256_max_epi8(v, perm0);
__m256i v0 = _mm256_mask_mov_epi8(max0, 0x15555555, min0);

/* Pairs: ([31,31], [28,30], [29,29], [25,27], [24,26], [21,23], [20,22], [17,19], [16,18], [13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], [0,2]) */
/* Perm:  (31, 28, 29, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  2) */
__m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 28, 29, 30, 25, 24, 27, 26, 21, 20, 23, 22, 17, 16, 19, 18, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min1 = _mm256_min_epi8(v0, perm1);
__m256i max1 = _mm256_max_epi8(v0, perm1);
__m256i v1 = _mm256_mask_mov_epi8(max1, 0x13333333, min1);

/* Pairs: ([31,31], [26,30], [25,29], [24,28], [27,27], [19,23], [18,22], [17,21], [16,20], [11,15], [10,14], [9,13], [8,12], [3,7], [2,6], [1,5], [0,4]) */
/* Perm:  (31, 26, 25, 24, 27, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 11, 10,  9,  8, 15, 14, 13, 12,  3,  2,  1,  0,  7,  6,  5,  4) */
__m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(31, 26, 25, 24, 27, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 11, 10, 9, 8, 15, 14, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m256i min2 = _mm256_min_epi8(v1, perm2);
__m256i max2 = _mm256_max_epi8(v1, perm2);
__m256i v2 = _mm256_mask_mov_epi8(max2, 0x70f0f0f, min2);

/* Pairs: ([31,31], [22,30], [21,29], [20,28], [19,27], [18,26], [17,25], [16,24], [23,23], [7,15], [6,14], [5,13], [4,12], [3,11], [2,10], [1,9], [0,8]) */
/* Perm:  (31, 22, 21, 20, 19, 18, 17, 16, 23, 30, 29, 28, 27, 26, 25, 24,  7,  6,  5,  4,  3,  2,  1,  0, 15, 14, 13, 12, 11, 10,  9,  8) */
__m256i perm3 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(31, 22, 21, 20, 19, 18, 17, 16, 23, 30, 29, 28, 27, 26, 25, 24, 7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8));
__m256i min3 = _mm256_min_epi8(v2, perm3);
__m256i max3 = _mm256_max_epi8(v2, perm3);
__m256i v3 = _mm256_mask_mov_epi8(max3, 0x7f00ff, min3);

/* Pairs: ([31,31], [23,30], [27,29], [19,28], [21,26], [22,25], [17,24], [18,20], [0,16], [15,15], [7,14], [11,13], [3,12], [5,10], [6,9], [1,8], [2,4]) */
/* Perm:  (31, 23, 27, 19, 29, 21, 22, 17, 30, 25, 26, 18, 28, 20, 24,  0, 15,  7, 11,  3, 13,  5,  6,  1, 14,  9, 10,  2, 12,  4,  8, 16) */
__m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 23, 27, 19, 29, 21, 22, 17, 30, 25, 26, 18, 28, 20, 24, 0, 15, 7, 11, 3, 13, 5, 6, 1, 14, 9, 10, 2, 12, 4, 8, 16), v3);
__m256i min4 = _mm256_min_epi8(v3, perm4);
__m256i max4 = _mm256_max_epi8(v3, perm4);
__m256i v4 = _mm256_mask_mov_epi8(max4, 0x8ee08ef, min4);

/* Pairs: ([31,31], [29,30], [26,28], [23,27], [9,25], [20,24], [6,22], [19,21], [17,18], [16,16], [15,15], [13,14], [10,12], [7,11], [4,8], [3,5], [1,2], [0,0]) */
/* Perm:  (31, 29, 30, 26, 23, 28,  9, 20, 27,  6, 19, 24, 21, 17, 18, 16, 15, 13, 14, 10,  7, 12, 25,  4, 11, 22,  3,  8,  5,  1,  2,  0) */
__m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 29, 30, 26, 23, 28, 9, 20, 27, 6, 19, 24, 21, 17, 18, 16, 15, 13, 14, 10, 7, 12, 25, 4, 11, 22, 3, 8, 5, 1, 2, 0), v4);
__m256i min5 = _mm256_min_epi8(v4, perm5);
__m256i max5 = _mm256_max_epi8(v4, perm5);
__m256i v5 = _mm256_mask_mov_epi8(max5, 0x249a26da, min5);

/* Pairs: ([31,31], [14,30], [13,29], [12,28], [11,27], [21,26], [25,25], [8,24], [7,23], [22,22], [4,20], [3,19], [2,18], [1,17], [16,16], [15,15], [5,10], [9,9], [6,6], [0,0]) */
/* Perm:  (31, 14, 13, 12, 11, 21, 25,  8,  7, 22, 26,  4,  3,  2,  1, 16, 15, 30, 29, 28, 27,  5,  9, 24, 23,  6, 10, 20, 19, 18, 17,  0) */
__m256i perm6 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 14, 13, 12, 11, 21, 25, 8, 7, 22, 26, 4, 3, 2, 1, 16, 15, 30, 29, 28, 27, 5, 9, 24, 23, 6, 10, 20, 19, 18, 17, 0), v5);
__m256i min6 = _mm256_min_epi8(v5, perm6);
__m256i max6 = _mm256_max_epi8(v5, perm6);
__m256i v6 = _mm256_mask_mov_epi8(max6, 0x2079be, min6);

/* Pairs: ([31,31], [30,30], [29,29], [14,28], [15,27], [10,26], [13,25], [22,24], [11,23], [5,21], [8,20], [19,19], [6,18], [3,17], [4,16], [12,12], [7,9], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 14, 15, 10, 13, 22, 11, 24,  5,  8, 19,  6,  3,  4, 27, 28, 25, 12, 23, 26,  7, 20,  9, 18, 21, 16, 17,  2,  1,  0) */
__m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 14, 15, 10, 13, 22, 11, 24, 5, 8, 19, 6, 3, 4, 27, 28, 25, 12, 23, 26, 7, 20, 9, 18, 21, 16, 17, 2, 1, 0), v6);
__m256i min7 = _mm256_min_epi8(v6, perm7);
__m256i max7 = _mm256_max_epi8(v6, perm7);
__m256i v7 = _mm256_mask_mov_epi8(max7, 0x40edf8, min7);

/* Pairs: ([31,31], [27,30], [29,29], [23,28], [15,26], [25,25], [14,24], [10,22], [9,21], [12,20], [11,19], [18,18], [7,17], [5,16], [13,13], [3,8], [6,6], [1,4], [2,2], [0,0]) */
/* Perm:  (31, 27, 29, 23, 30, 15, 25, 14, 28, 10,  9, 12, 11, 18,  7,  5, 26, 24, 13, 20, 19, 22, 21,  3, 17,  6, 16,  1,  8,  2,  4,  0) */
__m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 27, 29, 23, 30, 15, 25, 14, 28, 10, 9, 12, 11, 18, 7, 5, 26, 24, 13, 20, 19, 22, 21, 3, 17, 6, 16, 1, 8, 2, 4, 0), v7);
__m256i min8 = _mm256_min_epi8(v7, perm8);
__m256i max8 = _mm256_max_epi8(v7, perm8);
__m256i v8 = _mm256_mask_mov_epi8(max8, 0x880deaa, min8);

/* Pairs: ([31,31], [30,30], [26,29], [28,28], [27,27], [25,25], [23,24], [13,22], [21,21], [14,20], [15,19], [9,18], [11,17], [12,16], [10,10], [7,8], [6,6], [2,5], [4,4], [3,3], [1,1], [0,0]) */
/* Perm:  (31, 30, 26, 28, 27, 29, 25, 23, 24, 13, 21, 14, 15,  9, 11, 12, 19, 20, 22, 16, 17, 10, 18,  7,  8,  6,  2,  4,  3,  5,  1,  0) */
__m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 26, 28, 27, 29, 25, 23, 24, 13, 21, 14, 15, 9, 11, 12, 19, 20, 22, 16, 17, 10, 18, 7, 8, 6, 2, 4, 3, 5, 1, 0), v8);
__m256i min9 = _mm256_min_epi8(v8, perm9);
__m256i max9 = _mm256_max_epi8(v8, perm9);
__m256i v9 = _mm256_mask_mov_epi8(max9, 0x480fa84, min9);

/* Pairs: ([31,31], [30,30], [27,29], [28,28], [26,26], [19,25], [24,24], [23,23], [15,22], [20,21], [14,18], [13,17], [9,16], [6,12], [10,11], [8,8], [7,7], [5,5], [2,4], [3,3], [1,1], [0,0]) */
/* Perm:  (31, 30, 27, 28, 29, 26, 19, 24, 23, 15, 20, 21, 25, 14, 13,  9, 22, 18, 17,  6, 10, 11, 16,  8,  7, 12,  5,  2,  3,  4,  1,  0) */
__m256i perm10 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 27, 28, 29, 26, 19, 24, 23, 15, 20, 21, 25, 14, 13, 9, 22, 18, 17, 6, 10, 11, 16, 8, 7, 12, 5, 2, 3, 4, 1, 0), v9);
__m256i min10 = _mm256_min_epi8(v9, perm10);
__m256i max10 = _mm256_max_epi8(v9, perm10);
__m256i v10 = _mm256_mask_mov_epi8(max10, 0x818e644, min10);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [25,26], [24,24], [19,23], [21,22], [18,20], [15,17], [14,16], [11,13], [8,12], [9,10], [7,7], [5,6], [4,4], [3,3], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 25, 26, 24, 19, 21, 22, 18, 23, 20, 15, 14, 17, 16, 11,  8, 13,  9, 10, 12,  7,  5,  6,  4,  3,  2,  1,  0) */
__m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 25, 26, 24, 19, 21, 22, 18, 23, 20, 15, 14, 17, 16, 11, 8, 13, 9, 10, 12, 7, 5, 6, 4, 3, 2, 1, 0), v10);
__m256i min11 = _mm256_min_epi8(v10, perm11);
__m256i max11 = _mm256_max_epi8(v10, perm11);
__m256i v11 = _mm256_mask_mov_epi8(max11, 0x22ccb20, min11);

/* Pairs: ([31,31], [30,30], [29,29], [26,28], [27,27], [24,25], [22,23], [19,21], [17,20], [15,18], [13,16], [11,14], [10,12], [8,9], [6,7], [3,5], [4,4], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 26, 27, 28, 24, 25, 22, 23, 19, 17, 21, 15, 20, 13, 18, 11, 16, 10, 14, 12,  8,  9,  6,  7,  3,  4,  5,  2,  1,  0) */
__m256i perm12 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 26, 27, 28, 24, 25, 22, 23, 19, 17, 21, 15, 20, 13, 18, 11, 16, 10, 14, 12, 8, 9, 6, 7, 3, 4, 5, 2, 1, 0), v11);
__m256i min12 = _mm256_min_epi8(v11, perm12);
__m256i max12 = _mm256_max_epi8(v11, perm12);
__m256i v12 = _mm256_mask_mov_epi8(max12, 0x54aad48, min12);

/* Pairs: ([31,31], [30,30], [29,29], [27,28], [25,26], [23,24], [21,22], [19,20], [17,18], [15,16], [13,14], [11,12], [9,10], [7,8], [5,6], [3,4], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  2,  1,  0) */
__m256i perm13 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 27, 28, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 15, 16, 13, 14, 11, 12, 9, 10, 7, 8, 5, 6, 3, 4, 2, 1, 0), v12);
__m256i min13 = _mm256_min_epi8(v12, perm13);
__m256i max13 = _mm256_max_epi8(v12, perm13);
__m256i v13 = _mm256_mask_mov_epi8(max13, 0xaaaaaa8, min13);

return v13;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) minimum_31_int8_t(int8_t * const arr) {

__m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
__m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x7fffffff, arr);
fill_works(v);
v = minimum_31_int8_t_vec(v);

fill_works(v);_mm256_mask_storeu_epi8((void *)arr, 0x7fffffff, v);

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


