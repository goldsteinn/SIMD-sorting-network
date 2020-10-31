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
#define N 27
#define SORT_NAME minimum_27_uint16_t

#ifndef _SIMD_SORT_minimum_27_uint16_t_H_
#define _SIMD_SORT_minimum_27_uint16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 27
	Underlying Sort Type             : uint16_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 14
	SIMD Instructions                : 2 / 70
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
minimum_27_uint16_t_vec(__m512i v) {
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [21,26], 
                 [22,25], [14,24], [13,23], [19,20], [17,18], [11,16], 
                 [12,15], [10,10], [0,9], [5,8], [3,7], [1,6], [2,4]) */
      /* Perm:  (31, 30, 29, 28, 27, 21, 22, 14, 13, 25, 26, 19, 20, 17, 18, 
                 11, 12, 24, 23, 15, 16, 10,  0,  5,  3,  1,  8,  2,  7,  4,  
                 6,  9) */
      __m512i perm0 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 21, 22, 14, 13, 25, 
                                               26, 19, 20, 17, 18, 11, 12, 
                                               24, 23, 15, 16, 10, 0, 5, 3, 
                                               1, 8, 2, 7, 4, 6, 9), v);
      __m512i min0 = _mm512_min_epu16(v, perm0);
      __m512i max0 = _mm512_max_epu16(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi16(max0, 0x6a782f, min0);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [24,26], 
                 [16,25], [20,23], [19,22], [12,21], [15,18], [14,17], 
                 [11,13], [4,10], [6,9], [7,8], [3,5], [2,2], [0,1]) */
      /* Perm:  (31, 30, 29, 28, 27, 24, 16, 26, 20, 19, 12, 23, 22, 15, 14, 
                 25, 18, 17, 11, 21, 13,  4,  6,  7,  8,  9,  3, 10,  5,  2,  
                 0,  1) */
      __m512i perm1 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 24, 16, 26, 20, 19, 
                                               12, 23, 22, 15, 14, 25, 18, 
                                               17, 11, 21, 13, 4, 6, 7, 8, 9, 
                                               3, 10, 5, 2, 0, 1), v0);
      __m512i min1 = _mm512_min_epu16(v0, perm1);
      __m512i max1 = _mm512_max_epu16(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi16(max1, 0x119d8d9, min1);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [18,26], 
                 [23,25], [15,24], [13,22], [17,21], [16,20], [11,19], 
                 [12,14], [8,10], [9,9], [4,7], [6,6], [2,5], [1,3], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 18, 23, 15, 25, 13, 17, 16, 11, 26, 21, 
                 20, 24, 12, 22, 14, 19,  8,  9, 10,  4,  6,  2,  7,  1,  5,  
                 3,  0) */
      __m512i perm2 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 18, 23, 15, 25, 13, 
                                               17, 16, 11, 26, 21, 20, 24, 
                                               12, 22, 14, 19, 8, 9, 10, 4, 
                                               6, 2, 7, 1, 5, 3, 0), v1);
      __m512i min2 = _mm512_min_epu16(v1, perm2);
      __m512i max2 = _mm512_max_epu16(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi16(max2, 0x87b916, min2);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [25,26], 
                 [22,24], [18,23], [20,21], [14,19], [16,17], [13,15], 
                 [11,12], [10,10], [5,9], [6,8], [3,7], [0,4], [1,2]) */
      /* Perm:  (31, 30, 29, 28, 27, 25, 26, 22, 18, 24, 20, 21, 14, 23, 16, 
                 17, 13, 19, 15, 11, 12, 10,  5,  6,  3,  8,  9,  0,  7,  1,  
                 2,  4) */
      __m512i perm3 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 25, 26, 22, 18, 24, 
                                               20, 21, 14, 23, 16, 17, 13, 
                                               19, 15, 11, 12, 10, 5, 6, 3, 
                                               8, 9, 0, 7, 1, 2, 4), v2);
      __m512i min3 = _mm512_min_epu16(v2, perm3);
      __m512i max3 = _mm512_max_epu16(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi16(max3, 0x255686b, min3);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [23,25], [21,24], [18,22], [17,20], [15,19], [13,16], 
                 [12,14], [11,11], [9,10], [7,8], [2,6], [4,5], [3,3], [0,1]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 23, 21, 25, 18, 24, 17, 15, 22, 20, 
                 13, 19, 12, 16, 14, 11,  9, 10,  7,  8,  2,  4,  5,  3,  6,  
                 0,  1) */
      __m512i perm4 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 23, 21, 25, 18, 
                                               24, 17, 15, 22, 20, 13, 19, 
                                               12, 16, 14, 11, 9, 10, 7, 8, 
                                               2, 4, 5, 3, 6, 0, 1), v3);
      __m512i min4 = _mm512_min_epu16(v3, perm4);
      __m512i max4 = _mm512_max_epu16(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi16(max4, 0xa6b295, min4);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [24,25], [21,23], [15,22], [18,20], [17,19], [14,16], 
                 [12,13], [0,11], [10,10], [8,9], [5,7], [3,6], [2,4], [1,1]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 26, 24, 25, 21, 15, 23, 18, 17, 20, 19, 
                 14, 22, 16, 12, 13,  0, 10,  8,  9,  5,  3,  7,  2,  6,  4,  
                 1, 11) */
      __m512i perm5 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 24, 25, 21, 15, 
                                               23, 18, 17, 20, 19, 14, 22, 
                                               16, 12, 13, 0, 10, 8, 9, 5, 3, 
                                               7, 2, 6, 4, 1, 11), v4);
      __m512i min5 = _mm512_min_epu16(v4, perm5);
      __m512i max5 = _mm512_max_epu16(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi16(max5, 0x126d12d, min5);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [23,24], [21,22], [19,20], [17,18], [15,16], 
                 [13,14], [12,12], [11,11], [10,10], [9,9], [7,8], [5,6], 
                 [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 13, 14, 12, 11, 10,  9,  7,  8,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m512i perm6 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 23, 24, 21, 
                                               22, 19, 20, 17, 18, 15, 16, 
                                               13, 14, 12, 11, 10, 9, 7, 8, 
                                               5, 6, 3, 4, 1, 2, 0), v5);
      __m512i min6 = _mm512_min_epu16(v5, perm6);
      __m512i max6 = _mm512_max_epu16(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi16(max6, 0xaaa0aa, min6);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [20,22], [19,21], [16,18], 
                 [15,17], [14,14], [13,13], [1,12], [11,11], [10,10], [9,9], 
                 [8,8], [6,7], [4,5], [2,3], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 20, 19, 22, 21, 16, 15, 
                 18, 17, 14, 13,  1, 11, 10,  9,  8,  6,  7,  4,  5,  2,  3, 
                 12,  0) */
      __m512i perm7 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 20, 
                                               19, 22, 21, 16, 15, 18, 17, 
                                               14, 13, 1, 11, 10, 9, 8, 6, 7, 
                                               4, 5, 2, 3, 12, 0), v6);
      __m512i min7 = _mm512_min_epu16(v6, perm7);
      __m512i max7 = _mm512_max_epu16(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi16(max7, 0x198056, min7);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [22,23], [20,21], [18,19], [16,17], 
                 [14,15], [2,13], [12,12], [11,11], [10,10], [9,9], [8,8], 
                 [7,7], [6,6], [5,5], [4,4], [3,3], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 21, 18, 19, 16, 
                 17, 14, 15,  2, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3, 13,  
                 1,  0) */
      __m512i perm8 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 22, 23, 
                                               20, 21, 18, 19, 16, 17, 14, 
                                               15, 2, 12, 11, 10, 9, 8, 7, 6, 
                                               5, 4, 3, 13, 1, 0), v7);
      __m512i min8 = _mm512_min_epu16(v7, perm8);
      __m512i max8 = _mm512_max_epu16(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi16(max8, 0x554004, min8);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [25,25], [24,24], [23,23], [22,22], [10,21], [9,20], [8,19], 
                 [7,18], [6,17], [5,16], [4,15], [3,14], [13,13], [12,12], 
                 [11,11], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 10,  9,  8,  7,  6,  
                 5,  4,  3, 13, 12, 11, 21, 20, 19, 18, 17, 16, 15, 14,  2,  
                 1,  0) */
      __m512i perm9 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                               28, 27, 26, 25, 24, 23, 22, 
                                               10, 9, 8, 7, 6, 5, 4, 3, 13, 
                                               12, 11, 21, 20, 19, 18, 17, 
                                               16, 15, 14, 2, 1, 0), v8);
      __m512i min9 = _mm512_min_epu16(v8, perm9);
      __m512i max9 = _mm512_max_epu16(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi16(max9, 0x7f8, min9);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [18,26], 
                 [17,25], [16,24], [15,23], [14,22], [21,21], [20,20], 
                 [19,19], [10,13], [9,12], [8,11], [7,7], [6,6], [5,5], 
                 [4,4], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 18, 17, 16, 15, 14, 21, 20, 19, 26, 25, 
                 24, 23, 22, 10,  9,  8, 13, 12, 11,  7,  6,  5,  4,  3,  2,  
                 1,  0) */
      __m512i perm10 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 18, 17, 16, 15, 14, 
                                                21, 20, 19, 26, 25, 24, 23, 
                                                22, 10, 9, 8, 13, 12, 11, 7, 
                                                6, 5, 4, 3, 2, 1, 0), v9);
      __m512i min10 = _mm512_min_epu16(v9, perm10);
      __m512i max10 = _mm512_max_epu16(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi16(max10, 0x7c700, min10);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [26,26], 
                 [21,25], [20,24], [19,23], [18,22], [13,17], [12,16], 
                 [11,15], [7,14], [6,10], [5,9], [4,8], [3,3], [2,2], [1,1], 
                 [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 26, 21, 20, 19, 18, 25, 24, 23, 22, 13, 
                 12, 11,  7, 17, 16, 15,  6,  5,  4, 14, 10,  9,  8,  3,  2,  
                 1,  0) */
      __m512i perm11 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 26, 21, 20, 19, 18, 
                                                25, 24, 23, 22, 13, 12, 11, 
                                                7, 17, 16, 15, 6, 5, 4, 14, 
                                                10, 9, 8, 3, 2, 1, 0), v10);
      __m512i min11 = _mm512_min_epu16(v10, perm11);
      __m512i max11 = _mm512_max_epu16(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi16(max11, 0x3c38f0, min11);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [24,26], 
                 [25,25], [21,23], [20,22], [17,19], [16,18], [13,15], 
                 [12,14], [10,11], [7,9], [6,8], [3,5], [2,4], [1,1], [0,0]) 
                 */
      /* Perm:  (31, 30, 29, 28, 27, 24, 25, 26, 21, 20, 23, 22, 17, 16, 19, 
                 18, 13, 12, 15, 14, 10, 11,  7,  6,  9,  8,  3,  2,  5,  4,  
                 1,  0) */
      __m512i perm12 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 24, 25, 26, 21, 20, 
                                                23, 22, 17, 16, 19, 18, 13, 
                                                12, 15, 14, 10, 11, 7, 6, 9, 
                                                8, 3, 2, 5, 4, 1, 0), v11);
      __m512i min12 = _mm512_min_epu16(v11, perm12);
      __m512i max12 = _mm512_max_epu16(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi16(max12, 0x13334cc, min12);
      
      /* Pairs: ([31,31], [30,30], [29,29], [28,28], [27,27], [25,26], 
                 [23,24], [21,22], [19,20], [17,18], [15,16], [13,14], 
                 [11,12], [9,10], [7,8], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (31, 30, 29, 28, 27, 25, 26, 23, 24, 21, 22, 19, 20, 17, 18, 
                 15, 16, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  1,  
                 2,  0) */
      __m512i perm13 = _mm512_permutexvar_epi16(_mm512_set_epi16(31, 30, 29, 
                                                28, 27, 25, 26, 23, 24, 21, 
                                                22, 19, 20, 17, 18, 15, 16, 
                                                13, 14, 11, 12, 9, 10, 7, 8, 
                                                5, 6, 3, 4, 1, 2, 0), v12);
      __m512i min13 = _mm512_min_epu16(v12, perm13);
      __m512i max13 = _mm512_max_epu16(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi16(max13, 0x2aaaaaa, min13);
      
      return v13;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
minimum_27_uint16_t(uint16_t * const 
                             arr) {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = minimum_27_uint16_t_vec(v);
      
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


