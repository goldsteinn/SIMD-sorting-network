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
#define N 14
#define SORT_NAME minimum_14_int8_t

#ifndef _SIMD_SORT_minimum_14_int8_t_H_
#define _SIMD_SORT_minimum_14_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 14
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 9
	SIMD Instructions                : 2 / 45
	Optimization Preference          : space
	SIMD Type                        : __m128i
	SIMD Instruction Set(s) Used     : SSE2, SSSE3, SSE4.1, AVX512vl, AVX512bw
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
 __m128i __attribute__((const)) 
minimum_14_int8_t_vec(__m128i v) {
      
      /* Pairs: ([15,15], [14,14], [8,13], [4,12], [7,11], [5,10], [1,9], 
                 [2,6], [0,3]) */
      /* Perm:  (15, 14,  8,  4,  7,  5,  1, 13, 11,  2, 10, 12,  0,  6,  9,  
                 3) */
      __m128i perm0 = _mm_shuffle_epi8(v, _mm_set_epi8(15, 14, 8, 4, 7, 5, 1, 
                                       13, 11, 2, 10, 12, 0, 6, 9, 3));
      __m128i min0 = _mm_min_epi8(v, perm0);
      __m128i max0 = _mm_max_epi8(v, perm0);
      __m128i v0 = _mm_mask_mov_epi8(max0, 0x1b7, min0);
      
      /* Pairs: ([15,15], [14,14], [11,13], [3,12], [6,10], [9,9], [7,8], 
                 [4,5], [0,2], [1,1]) */
      /* Perm:  (15, 14, 11,  3, 13,  6,  9,  7,  8, 10,  4,  5, 12,  0,  1,  
                 2) */
      __m128i perm1 = _mm_shuffle_epi8(v0, _mm_set_epi8(15, 14, 11, 3, 13, 6, 
                                       9, 7, 8, 10, 4, 5, 12, 0, 1, 2));
      __m128i min1 = _mm_min_epi8(v0, perm1);
      __m128i max1 = _mm_max_epi8(v0, perm1);
      __m128i v1 = _mm_mask_mov_epi8(max1, 0x8d9, min1);
      
      /* Pairs: ([15,15], [14,14], [13,13], [10,12], [2,11], [5,9], [8,8], 
                 [4,7], [3,6], [0,1]) */
      /* Perm:  (15, 14, 13, 10,  2, 12,  5,  8,  4,  3,  9,  7,  6, 11,  0,  
                 1) */
      __m128i perm2 = _mm_shuffle_epi8(v1, _mm_set_epi8(15, 14, 13, 10, 2, 
                                       12, 5, 8, 4, 3, 9, 7, 6, 11, 0, 1));
      __m128i min2 = _mm_min_epi8(v1, perm2);
      __m128i max2 = _mm_max_epi8(v1, perm2);
      __m128i v2 = _mm_mask_mov_epi8(max2, 0x43d, min2);
      
      /* Pairs: ([15,15], [14,14], [6,13], [12,12], [9,11], [10,10], [3,8], 
                 [1,7], [2,5], [0,4]) */
      /* Perm:  (15, 14,  6, 12,  9, 10, 11,  3,  1, 13,  2,  0,  8,  5,  7,  
                 4) */
      __m128i perm3 = _mm_shuffle_epi8(v2, _mm_set_epi8(15, 14, 6, 12, 9, 10, 
                                       11, 3, 1, 13, 2, 0, 8, 5, 7, 4));
      __m128i min3 = _mm_min_epi8(v2, perm3);
      __m128i max3 = _mm_max_epi8(v2, perm3);
      __m128i v3 = _mm_mask_mov_epi8(max3, 0x24f, min3);
      
      /* Pairs: ([15,15], [14,14], [12,13], [11,11], [8,10], [6,9], [5,7], 
                 [3,4], [1,2], [0,0]) */
      /* Perm:  (15, 14, 12, 13, 11,  8,  6, 10,  5,  9,  7,  3,  4,  1,  2,  
                 0) */
      __m128i perm4 = _mm_shuffle_epi8(v3, _mm_set_epi8(15, 14, 12, 13, 11, 
                                       8, 6, 10, 5, 9, 7, 3, 4, 1, 2, 0));
      __m128i min4 = _mm_min_epi8(v3, perm4);
      __m128i max4 = _mm_max_epi8(v3, perm4);
      __m128i v4 = _mm_mask_mov_epi8(max4, 0x116a, min4);
      
      /* Pairs: ([15,15], [14,14], [13,13], [11,12], [6,10], [5,9], [7,8], 
                 [2,4], [1,3], [0,0]) */
      /* Perm:  (15, 14, 13, 11, 12,  6,  5,  7,  8, 10,  9,  2,  1,  4,  3,  
                 0) */
      __m128i perm5 = _mm_shuffle_epi8(v4, _mm_set_epi8(15, 14, 13, 11, 12, 
                                       6, 5, 7, 8, 10, 9, 2, 1, 4, 3, 0));
      __m128i min5 = _mm_min_epi8(v4, perm5);
      __m128i max5 = _mm_max_epi8(v4, perm5);
      __m128i v5 = _mm_mask_mov_epi8(max5, 0x8e6, min5);
      
      /* Pairs: ([15,15], [14,14], [12,13], [8,11], [9,10], [6,7], [4,5], 
                 [2,3], [1,1], [0,0]) */
      /* Perm:  (15, 14, 12, 13,  8,  9, 10, 11,  6,  7,  4,  5,  2,  3,  1,  
                 0) */
      __m128i perm6 = _mm_shuffle_epi8(v5, _mm_set_epi8(15, 14, 12, 13, 8, 9, 
                                       10, 11, 6, 7, 4, 5, 2, 3, 1, 0));
      __m128i min6 = _mm_min_epi8(v5, perm6);
      __m128i max6 = _mm_max_epi8(v5, perm6);
      __m128i v6 = _mm_mask_mov_epi8(max6, 0x1354, min6);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [10,11], [8,9], [5,7], 
                 [4,6], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 10, 11,  8,  9,  5,  4,  7,  6,  3,  2,  1,  
                 0) */
      __m128i perm7 = _mm_shuffle_epi8(v6, _mm_set_epi8(15, 14, 13, 12, 10, 
                                       11, 8, 9, 5, 4, 7, 6, 3, 2, 1, 0));
      __m128i min7 = _mm_min_epi8(v6, perm7);
      __m128i max7 = _mm_max_epi8(v6, perm7);
      __m128i v7 = _mm_mask_mov_epi8(max7, 0x530, min7);
      
      /* Pairs: ([15,15], [14,14], [13,13], [11,12], [9,10], [7,8], [5,6], 
                 [3,4], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  2,  1,  
                 0) */
      __m128i perm8 = _mm_shuffle_epi8(v7, _mm_set_epi8(15, 14, 13, 11, 12, 
                                       9, 10, 7, 8, 5, 6, 3, 4, 2, 1, 0));
      __m128i min8 = _mm_min_epi8(v7, perm8);
      __m128i max8 = _mm_max_epi8(v7, perm8);
      __m128i v8 = _mm_mask_mov_epi8(max8, 0xaa8, min8);
      
      return v8;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
minimum_14_int8_t(int8_t * const arr) 
                             {
      
      __m128i v = _mm_load_si128((__m128i *)arr);
      
      v = minimum_14_int8_t_vec(v);
      
      _mm_store_si128((__m128i *)arr, v);
      
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


