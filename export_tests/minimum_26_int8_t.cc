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
#define N 26
#define SORT_NAME minimum_26_int8_t

#ifndef _SIMD_SORT_minimum_26_int8_t_H_
#define _SIMD_SORT_minimum_26_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 26
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 13
	SIMD Instructions                : 3 / 65
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX512vbmi, AVX, AVX2
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
__m256i __attribute__((const)) minimum_26_int8_t_vec(__m256i v) {

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [12,25], [19,24], [17,23], [5,22], [18,21], [3,20], [9,16], [10,15], [11,14], [0,13], [2,8], [4,7], [1,6]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 12, 19, 17,  5, 18,  3, 24, 21, 23,  9, 10, 11,  0, 25, 14, 15, 16,  2,  4,  1, 22,  7, 20,  8,  6, 13) */
__m256i perm0 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 12, 19, 17, 5, 18, 3, 24, 21, 23, 9, 10, 11, 0, 25, 14, 15, 16, 2, 4, 1, 22, 7, 20, 8, 6, 13), v);
__m256i min0 = _mm256_min_epi8(v, perm0);
__m256i max0 = _mm256_max_epi8(v, perm0);
__m256i v0 = _mm256_mask_mov_epi8(max0, 0xe1e3f, min0);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [22,25], [6,24], [23,23], [7,21], [13,20], [1,19], [4,18], [9,17], [8,16], [14,15], [5,12], [10,11], [0,3], [2,2]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 22,  6, 23, 25,  7, 13,  1,  4,  9,  8, 14, 15, 20,  5, 10, 11, 17, 16, 21, 24, 12, 18,  0,  2, 19,  3) */
__m256i perm1 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 22, 6, 23, 25, 7, 13, 1, 4, 9, 8, 14, 15, 20, 5, 10, 11, 17, 16, 21, 24, 12, 18, 0, 2, 19, 3), v0);
__m256i min1 = _mm256_min_epi8(v0, perm1);
__m256i max1 = _mm256_max_epi8(v0, perm1);
__m256i v1 = _mm256_mask_mov_epi8(max1, 0x4067f3, min1);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [15,25], [20,24], [8,23], [13,22], [21,21], [14,19], [9,18], [2,17], [7,16], [3,12], [6,11], [0,10], [1,5], [4,4]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 15, 20,  8, 13, 21, 24, 14,  9,  2,  7, 25, 19, 22,  3,  6,  0, 18, 23, 16, 11,  1,  4, 12, 17,  5, 10) */
__m256i perm2 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 15, 20, 8, 13, 21, 24, 14, 9, 2, 7, 25, 19, 22, 3, 6, 0, 18, 23, 16, 11, 1, 4, 12, 17, 5, 10), v1);
__m256i min2 = _mm256_min_epi8(v1, perm2);
__m256i max2 = _mm256_max_epi8(v1, perm2);
__m256i v2 = _mm256_mask_mov_epi8(max2, 0x10e3cf, min2);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [24,25], [16,23], [11,22], [17,21], [15,20], [12,19], [7,18], [3,14], [6,13], [5,10], [2,9], [4,8], [0,1]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 24, 25, 16, 11, 17, 15, 12,  7, 21, 23, 20,  3,  6, 19, 22,  5,  2,  4, 18, 13, 10,  8, 14,  9,  0,  1) */
__m256i perm3 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 24, 25, 16, 11, 17, 15, 12, 7, 21, 23, 20, 3, 6, 19, 22, 5, 2, 4, 18, 13, 10, 8, 14, 9, 0, 1), v2);
__m256i min3 = _mm256_min_epi8(v2, perm3);
__m256i max3 = _mm256_max_epi8(v2, perm3);
__m256i v3 = _mm256_mask_mov_epi8(max3, 0x10398fd, min3);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [20,24], [21,23], [19,22], [16,18], [8,17], [10,15], [11,14], [12,13], [7,9], [3,6], [1,5], [2,4], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 20, 21, 19, 23, 24, 22, 16,  8, 18, 10, 11, 12, 13, 14, 15,  7, 17,  9,  3,  1,  2,  6,  4,  5,  0) */
__m256i perm4 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 20, 21, 19, 23, 24, 22, 16, 8, 18, 10, 11, 12, 13, 14, 15, 7, 17, 9, 3, 1, 2, 6, 4, 5, 0), v3);
__m256i min4 = _mm256_min_epi8(v3, perm4);
__m256i max4 = _mm256_max_epi8(v3, perm4);
__m256i v4 = _mm256_mask_mov_epi8(max4, 0x391d8e, min4);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [23,25], [22,24], [18,21], [19,20], [15,17], [9,16], [13,14], [11,12], [8,10], [4,7], [5,6], [1,3], [0,2]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 23, 22, 25, 24, 18, 19, 20, 21, 15,  9, 17, 13, 14, 11, 12,  8, 16, 10,  4,  5,  6,  7,  1,  0,  3,  2) */
__m256i perm5 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 23, 22, 25, 24, 18, 19, 20, 21, 15, 9, 17, 13, 14, 11, 12, 8, 16, 10, 4, 5, 6, 7, 1, 0, 3, 2), v4);
__m256i min5 = _mm256_min_epi8(v4, perm5);
__m256i max5 = _mm256_max_epi8(v4, perm5);
__m256i v5 = _mm256_mask_mov_epi8(max5, 0xccab33, min5);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [23,24], [21,22], [20,20], [7,19], [6,18], [14,17], [13,16], [10,15], [9,12], [8,11], [5,5], [3,4], [1,2], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 23, 24, 21, 22, 20,  7,  6, 14, 13, 10, 17, 16,  9,  8, 15, 12, 11, 19, 18,  5,  3,  4,  1,  2,  0) */
__m256i perm6 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 23, 24, 21, 22, 20, 7, 6, 14, 13, 10, 17, 16, 9, 8, 15, 12, 11, 19, 18, 5, 3, 4, 1, 2, 0), v5);
__m256i min6 = _mm256_min_epi8(v5, perm6);
__m256i max6 = _mm256_max_epi8(v5, perm6);
__m256i v6 = _mm256_mask_mov_epi8(max6, 0xa067ca, min6);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [16,24], [14,23], [22,22], [19,21], [18,20], [17,17], [13,15], [10,12], [2,11], [1,9], [8,8], [5,7], [4,6], [3,3], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 16, 14, 22, 19, 18, 21, 20, 17, 24, 13, 23, 15, 10,  2, 12,  1,  8,  5,  4,  7,  6,  3, 11,  9,  0) */
__m256i perm7 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 16, 14, 22, 19, 18, 21, 20, 17, 24, 13, 23, 15, 10, 2, 12, 1, 8, 5, 4, 7, 6, 3, 11, 9, 0), v6);
__m256i min7 = _mm256_min_epi8(v6, perm7);
__m256i max7 = _mm256_max_epi8(v6, perm7);
__m256i v7 = _mm256_mask_mov_epi8(max7, 0xd6436, min7);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [22,24], [17,23], [21,21], [20,20], [11,19], [12,18], [15,16], [6,14], [7,13], [9,10], [2,8], [5,5], [4,4], [1,3], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 22, 17, 24, 21, 20, 11, 12, 23, 15, 16,  6,  7, 18, 19,  9, 10,  2, 13, 14,  5,  4,  1,  8,  3,  0) */
__m256i perm8 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 22, 17, 24, 21, 20, 11, 12, 23, 15, 16, 6, 7, 18, 19, 9, 10, 2, 13, 14, 5, 4, 1, 8, 3, 0), v7);
__m256i min8 = _mm256_min_epi8(v7, perm8);
__m256i max8 = _mm256_max_epi8(v7, perm8);
__m256i v8 = _mm256_mask_mov_epi8(max8, 0x429ac6, min8);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [20,23], [22,22], [16,21], [15,19], [17,18], [12,14], [11,13], [6,10], [4,9], [7,8], [2,5], [3,3], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 20, 22, 16, 23, 15, 17, 18, 21, 19, 12, 11, 14, 13,  6,  4,  7,  8, 10,  2,  9,  3,  5,  1,  0) */
__m256i perm9 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 20, 22, 16, 23, 15, 17, 18, 21, 19, 12, 11, 14, 13, 6, 4, 7, 8, 10, 2, 9, 3, 5, 1, 0), v8);
__m256i min9 = _mm256_min_epi8(v8, perm9);
__m256i max9 = _mm256_max_epi8(v8, perm9);
__m256i v9 = _mm256_mask_mov_epi8(max9, 0x1398d4, min9);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [20,22], [18,21], [14,19], [16,17], [13,15], [10,12], [6,11], [8,9], [4,7], [3,5], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 18, 22, 14, 21, 16, 17, 13, 19, 15, 10,  6, 12,  8,  9,  4, 11,  3,  7,  5,  2,  1,  0) */
__m256i perm10 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 18, 22, 14, 21, 16, 17, 13, 19, 15, 10, 6, 12, 8, 9, 4, 11, 3, 7, 5, 2, 1, 0), v9);
__m256i min10 = _mm256_min_epi8(v9, perm10);
__m256i max10 = _mm256_max_epi8(v9, perm10);
__m256i v10 = _mm256_mask_mov_epi8(max10, 0x156558, min10);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [22,23], [21,21], [18,20], [17,19], [14,16], [12,15], [10,13], [9,11], [6,8], [5,7], [4,4], [2,3], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 21, 18, 17, 20, 19, 14, 12, 16, 10, 15,  9, 13, 11,  6,  5,  8,  7,  4,  2,  3,  1,  0) */
__m256i perm11 = _mm256_permutexvar_epi8(_mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 21, 18, 17, 20, 19, 14, 12, 16, 10, 15, 9, 13, 11, 6, 5, 8, 7, 4, 2, 3, 1, 0), v10);
__m256i min11 = _mm256_min_epi8(v10, perm11);
__m256i max11 = _mm256_max_epi8(v10, perm11);
__m256i v11 = _mm256_mask_mov_epi8(max11, 0x465664, min11);

/* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], [25,25], [24,24], [23,23], [22,22], [20,21], [18,19], [16,17], [14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [3,3], [2,2], [1,1], [0,0]) */
/* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  3,  2,  1,  0) */
__m256i perm12 = _mm256_shuffle_epi8(v11, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 3, 2, 1, 0));
__m256i min12 = _mm256_min_epi8(v11, perm12);
__m256i max12 = _mm256_max_epi8(v11, perm12);
__m256i v12 = _mm256_mask_mov_epi8(max12, 0x155550, min12);

return v12;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) minimum_26_int8_t(int8_t * const arr) {

__m256i _tmp0 = _mm256_set1_epi8(int8_t(0x7f));
__m256i v = _mm256_mask_loadu_epi8(_tmp0, 0x3ffffff, arr);
fill_works(v);
v = minimum_26_int8_t_vec(v);

fill_works(v);_mm256_mask_storeu_epi8((void *)arr, 0x3ffffff, v);

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


