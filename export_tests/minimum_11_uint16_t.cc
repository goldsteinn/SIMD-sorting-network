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

#define TYPE uint16_t
#define N 11
#define SORT_NAME minimum_11_uint16_t

#ifndef _SIMD_SORT_minimum_11_uint16_t_H_
#define _SIMD_SORT_minimum_11_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 11
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 8
	SIMD Instructions                : 3 / 40
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX512vl, AVX512bw, SSE2, AVX, AVX2
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
int i = N;for (; i < 16; ++i) {
assert(t.arr[i] == uint16_t(0xffff));
}
}

/* SIMD Sort */
__m256i __attribute__((const)) minimum_11_uint16_t_vec(__m256i v) {

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [0,9], [5,8], [3,7], [1,6], [2,4]) */
/* Perm:  (15, 14, 13, 12, 11, 10,  0,  5,  3,  1,  8,  2,  7,  4,  6,  9) */
__m256i perm0 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 12, 11, 10, 0, 5, 3, 1, 8, 2, 7, 4, 6, 9), v);
__m256i min0 = _mm256_min_epu16(v, perm0);
__m256i max0 = _mm256_max_epu16(v, perm0);
__m256i v0 = _mm256_mask_mov_epi16(max0, 0x2f, min0);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [4,10], [6,9], [7,8], [3,5], [2,2], [0,1]) */
/* Perm:  (15, 14, 13, 12, 11,  4,  6,  7,  8,  9,  3, 10,  5,  2,  0,  1) */
__m256i perm1 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 12, 11, 4, 6, 7, 8, 9, 3, 10, 5, 2, 0, 1), v0);
__m256i min1 = _mm256_min_epu16(v0, perm1);
__m256i max1 = _mm256_max_epu16(v0, perm1);
__m256i v1 = _mm256_mask_mov_epi16(max1, 0xd9, min1);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [8,10], [9,9], [4,7], [6,6], [2,5], [1,3], [0,0]) */
/* Perm:  (15, 14, 13, 12, 11,  8,  9, 10,  4,  6,  2,  7,  1,  5,  3,  0) */
__m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 17, 16, 19, 18, 21, 20, 9, 8, 13, 12, 5, 4, 15, 14, 3, 2, 11, 10, 7, 6, 1, 0));
__m256i min2 = _mm256_min_epu16(v1, perm2);
__m256i max2 = _mm256_max_epu16(v1, perm2);
__m256i v2 = _mm256_mask_mov_epi16(max2, 0x116, min2);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [5,9], [6,8], [3,7], [0,4], [1,2]) */
/* Perm:  (15, 14, 13, 12, 11, 10,  5,  6,  3,  8,  9,  0,  7,  1,  2,  4) */
__m256i perm3 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 12, 11, 10, 5, 6, 3, 8, 9, 0, 7, 1, 2, 4), v2);
__m256i min3 = _mm256_min_epu16(v2, perm3);
__m256i max3 = _mm256_max_epu16(v2, perm3);
__m256i v3 = _mm256_mask_mov_epi16(max3, 0x6b, min3);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [9,10], [7,8], [2,6], [4,5], [3,3], [0,1]) */
/* Perm:  (15, 14, 13, 12, 11,  9, 10,  7,  8,  2,  4,  5,  3,  6,  0,  1) */
__m256i perm4 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 12, 11, 9, 10, 7, 8, 2, 4, 5, 3, 6, 0, 1), v3);
__m256i min4 = _mm256_min_epu16(v3, perm4);
__m256i max4 = _mm256_max_epu16(v3, perm4);
__m256i v4 = _mm256_mask_mov_epi16(max4, 0x295, min4);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [8,9], [5,7], [3,6], [2,4], [1,1], [0,0]) */
/* Perm:  (15, 14, 13, 12, 11, 10,  8,  9,  5,  3,  7,  2,  6,  4,  1,  0) */
__m256i perm5 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 17, 16, 19, 18, 11, 10, 7, 6, 15, 14, 5, 4, 13, 12, 9, 8, 3, 2, 1, 0));
__m256i min5 = _mm256_min_epu16(v4, perm5);
__m256i max5 = _mm256_max_epu16(v4, perm5);
__m256i v5 = _mm256_mask_mov_epi16(max5, 0x12c, min5);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], [7,8], [5,6], [3,4], [1,2], [0,0]) */
/* Perm:  (15, 14, 13, 12, 11, 10,  9,  7,  8,  5,  6,  3,  4,  1,  2,  0) */
__m256i perm6 = _mm256_permutexvar_epi16(_mm256_set_epi16(15, 14, 13, 12, 11, 10, 9, 7, 8, 5, 6, 3, 4, 1, 2, 0), v5);
__m256i min6 = _mm256_min_epu16(v5, perm6);
__m256i max6 = _mm256_max_epu16(v5, perm6);
__m256i v6 = _mm256_mask_mov_epi16(max6, 0xaa, min6);

/* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], [8,8], [6,7], [4,5], [2,3], [1,1], [0,0]) */
/* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  6,  7,  4,  5,  2,  3,  1,  0) */
__m256i perm7 = _mm256_shuffle_epi8(v6, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 13, 12, 15, 14, 9, 8, 11, 10, 5, 4, 7, 6, 3, 2, 1, 0));
__m256i min7 = _mm256_min_epu16(v6, perm7);
__m256i max7 = _mm256_max_epu16(v6, perm7);
__m256i v7 = _mm256_mask_mov_epi16(max7, 0x54, min7);

return v7;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) minimum_11_uint16_t(uint16_t * const arr) {

__m256i _tmp0 = _mm256_set1_epi16(uint16_t(0xffff));
__m256i v = _mm256_mask_loadu_epi16(_tmp0, 0x7ff, arr);
fill_works(v);
v = minimum_11_uint16_t_vec(v);

fill_works(v);_mm256_mask_storeu_epi16((void *)arr, 0x7ff, v);

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


