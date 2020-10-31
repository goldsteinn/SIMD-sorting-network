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

#define TYPE int16_t
#define N 18
#define SORT_NAME minimum_18_int16_t

#ifndef _SIMD_SORT_minimum_18_int16_t_H_
#define _SIMD_SORT_minimum_18_int16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 18
	Underlying Sort Type             : int16_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 11
	SIMD Instructions                : 2 / 55
	Optimization Preference          : space
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512bw
	SIMD Instruction Set(s) Excluded : None
	Aligned Load & Store             : True
	Integer Aligned Load & Store     : True
	Full Load & Store                : True
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



/* SIMD Sort */
 __m512i __attribute__((const)) 
minimum_18_int16_t_vec(__m512i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [11,17], [7,16], [2,15], [12,14], [8,13], 
                 [1,10], [4,9], [0,6], [3,5]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 11,  
                 7,  2, 12,  8, 14, 17,  1,  4, 13, 16,  0,  3,  9,  5, 15, 
                 10,  6) */
      __m512i perm0 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 11, 7, 2, 12, 
                                               8, 14, 17, 1, 4, 13, 16, 0, 3, 
                                               9, 5, 15, 10, 6), v);
      __m512i min0 = _mm512_min_epi16(v, perm0);
      __m512i max0 = _mm512_max_epi16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x199f, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [5,17], [13,16], [15,15], [6,14], [0,12], 
                 [3,11], [9,10], [7,8], [1,4], [2,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18,  5, 
                 13, 15,  6, 16,  0,  3,  9, 10,  7,  8, 14, 17,  1, 11,  2,  
                 4, 12) */
      __m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 5, 13, 15, 6, 
                                               16, 0, 3, 9, 10, 7, 8, 14, 17, 
                                               1, 11, 2, 4, 12), v0);
      __m512i min1 = _mm512_min_epi16(v0, perm1);
      __m512i max1 = _mm512_max_epi16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x22eb, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [4,16], [10,15], [14,14], [1,13], 
                 [12,12], [8,11], [6,9], [2,7], [5,5], [3,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  
                 4, 10, 14,  1, 12,  8, 15,  6, 11,  2,  9,  5, 16,  3,  7, 
                 13,  0) */
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 4, 10, 14, 
                                               1, 12, 8, 15, 6, 11, 2, 9, 5, 
                                               16, 3, 7, 13, 0), v1);
      __m512i min2 = _mm512_min_epi16(v1, perm2);
      __m512i max2 = _mm512_max_epi16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0x556, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [16,17], [14,15], [5,13], [4,12], [11,11], 
                 [8,10], [7,9], [6,6], [2,3], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 16, 
                 17, 14, 15,  5,  4, 11,  8,  7, 10,  9,  6, 13, 12,  2,  3,  
                 0,  1) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 16, 17, 14, 
                                               15, 5, 4, 11, 8, 7, 10, 9, 6, 
                                               13, 12, 2, 3, 0, 1), v2);
      __m512i min3 = _mm512_min_epi16(v2, perm3);
      __m512i max3 = _mm512_max_epi16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x141b5, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [15,17], [6,16], [13,14], [10,12], [1,11], 
                 [9,9], [8,8], [5,7], [3,4], [0,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 15,  
                 6, 17, 13, 14, 10,  1, 12,  9,  8,  5, 16,  7,  3,  4,  0, 
                 11,  2) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 15, 6, 17, 13, 
                                               14, 10, 1, 12, 9, 8, 5, 16, 7, 
                                               3, 4, 0, 11, 2), v3);
      __m512i min4 = _mm512_min_epi16(v3, perm4);
      __m512i max4 = _mm512_max_epi16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0xa46b, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [9,16], [15,15], [14,14], [7,13], 
                 [11,12], [4,10], [1,8], [5,6], [3,3], [2,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,  
                 9, 15, 14,  7, 11, 12,  4, 16,  1, 13,  5,  6, 10,  3,  2,  
                 8,  0) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 9, 15, 14, 
                                               7, 11, 12, 4, 16, 1, 13, 5, 6, 
                                               10, 3, 2, 8, 0), v4);
      __m512i min5 = _mm512_min_epi16(v4, perm5);
      __m512i max5 = _mm512_max_epi16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0xab2, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [14,16], [12,15], [10,13], 
                 [9,11], [6,8], [4,7], [2,5], [1,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 14, 12, 16, 10, 15,  9, 13, 11,  6,  4,  8,  2,  7,  1,  5,  
                 3,  0) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 14, 12, 
                                               16, 10, 15, 9, 13, 11, 6, 4, 
                                               8, 2, 7, 1, 5, 3, 0), v5);
      __m512i min6 = _mm512_min_epi16(v5, perm6);
      __m512i max6 = _mm512_max_epi16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0x5656, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [15,16], [12,14], [11,13], 
                 [8,10], [7,9], [4,6], [3,5], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 15, 16, 12, 11, 14, 13,  8,  7, 10,  9,  4,  3,  6,  5,  1,  
                 2,  0) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 15, 16, 
                                               12, 11, 14, 13, 8, 7, 10, 9, 
                                               4, 3, 6, 5, 1, 2, 0), v6);
      __m512i min7 = _mm512_min_epi16(v6, perm7);
      __m512i max7 = _mm512_max_epi16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x999a, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [14,15], [13,13], 
                 [9,12], [10,11], [5,8], [6,7], [4,4], [2,3], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 14, 15, 13,  9, 10, 11, 12,  5,  6,  7,  8,  4,  2,  3,  
                 1,  0) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 16, 14, 
                                               15, 13, 9, 10, 11, 12, 5, 6, 
                                               7, 8, 4, 2, 3, 1, 0), v7);
      __m512i min8 = _mm512_min_epi16(v7, perm8);
      __m512i max8 = _mm512_max_epi16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x4664, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [13,14], 
                 [11,12], [9,10], [7,8], [5,6], [3,4], [2,2], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  2,  
                 1,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               21, 20, 19, 18, 17, 16, 15, 
                                               13, 14, 11, 12, 9, 10, 7, 8, 
                                               5, 6, 3, 4, 2, 1, 0), v8);
      __m512i min9 = _mm512_min_epi16(v8, perm9);
      __m512i max9 = _mm512_max_epi16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0x2aa8, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [21,21], [20,20], 
                 [19,19], [18,18], [17,17], [16,16], [15,15], [14,14], 
                 [12,13], [10,11], [8,9], [6,7], [4,5], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 
                 16, 15, 14, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  3,  2,  
                 1,  0) */
      __m512i perm10 = _mm512_shuffle_epi8(v9, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 39, 38, 37, 
                                           36, 35, 34, 33, 32, 31, 30, 29, 
                                           28, 25, 24, 27, 26, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 7, 6, 5, 4, 3, 
                                           2, 1, 0));
      __m512i min10 = _mm512_min_epi16(v9, perm10);
      __m512i max10 = _mm512_max_epi16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x1550, min10);
      
      return v10;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
minimum_18_int16_t(int16_t * const arr) 
                             {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = minimum_18_int16_t_vec(v);
      
      _mm512_store_si512((__m512i *)arr, v);
      
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


