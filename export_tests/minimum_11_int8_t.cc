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
#define N 11
#define SORT_NAME minimum_11_int8_t

#ifndef _SIMD_SORT_minimum_11_int8_t_H_
#define _SIMD_SORT_minimum_11_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 11
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 8
	SIMD Instructions                : 2 / 40
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
minimum_11_int8_t_vec(__m128i v) {
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [0,9], 
                 [5,8], [3,7], [1,6], [2,4]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  0,  5,  3,  1,  8,  2,  7,  4,  6,  
                 9) */
      __m128i perm0 = _mm_shuffle_epi8(v, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 0, 5, 3, 1, 8, 2, 7, 4, 6, 9));
      __m128i min0 = _mm_min_epi8(v, perm0);
      __m128i max0 = _mm_max_epi8(v, perm0);
      __m128i v0 = _mm_mask_mov_epi8(max0, 0x2f, min0);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [4,10], [6,9], 
                 [7,8], [3,5], [2,2], [0,1]) */
      /* Perm:  (15, 14, 13, 12, 11,  4,  6,  7,  8,  9,  3, 10,  5,  2,  0,  
                 1) */
      __m128i perm1 = _mm_shuffle_epi8(v0, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       4, 6, 7, 8, 9, 3, 10, 5, 2, 0, 1));
      __m128i min1 = _mm_min_epi8(v0, perm1);
      __m128i max1 = _mm_max_epi8(v0, perm1);
      __m128i v1 = _mm_mask_mov_epi8(max1, 0xd9, min1);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [8,10], [9,9], 
                 [4,7], [6,6], [2,5], [1,3], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11,  8,  9, 10,  4,  6,  2,  7,  1,  5,  3,  
                 0) */
      __m128i perm2 = _mm_shuffle_epi8(v1, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       8, 9, 10, 4, 6, 2, 7, 1, 5, 3, 0));
      __m128i min2 = _mm_min_epi8(v1, perm2);
      __m128i max2 = _mm_max_epi8(v1, perm2);
      __m128i v2 = _mm_mask_mov_epi8(max2, 0x116, min2);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [5,9], 
                 [6,8], [3,7], [0,4], [1,2]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  5,  6,  3,  8,  9,  0,  7,  1,  2,  
                 4) */
      __m128i perm3 = _mm_shuffle_epi8(v2, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 5, 6, 3, 8, 9, 0, 7, 1, 2, 4));
      __m128i min3 = _mm_min_epi8(v2, perm3);
      __m128i max3 = _mm_max_epi8(v2, perm3);
      __m128i v3 = _mm_mask_mov_epi8(max3, 0x6b, min3);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [9,10], [7,8], 
                 [2,6], [4,5], [3,3], [0,1]) */
      /* Perm:  (15, 14, 13, 12, 11,  9, 10,  7,  8,  2,  4,  5,  3,  6,  0,  
                 1) */
      __m128i perm4 = _mm_shuffle_epi8(v3, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       9, 10, 7, 8, 2, 4, 5, 3, 6, 0, 1));
      __m128i min4 = _mm_min_epi8(v3, perm4);
      __m128i max4 = _mm_max_epi8(v3, perm4);
      __m128i v4 = _mm_mask_mov_epi8(max4, 0x295, min4);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [8,9], 
                 [5,7], [3,6], [2,4], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  8,  9,  5,  3,  7,  2,  6,  4,  1,  
                 0) */
      __m128i perm5 = _mm_shuffle_epi8(v4, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 8, 9, 5, 3, 7, 2, 6, 4, 1, 0));
      __m128i min5 = _mm_min_epi8(v4, perm5);
      __m128i max5 = _mm_max_epi8(v4, perm5);
      __m128i v5 = _mm_mask_mov_epi8(max5, 0x12c, min5);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [7,8], [5,6], [3,4], [1,2], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  7,  8,  5,  6,  3,  4,  1,  2,  
                 0) */
      __m128i perm6 = _mm_shuffle_epi8(v5, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 9, 7, 8, 5, 6, 3, 4, 1, 2, 0));
      __m128i min6 = _mm_min_epi8(v5, perm6);
      __m128i max6 = _mm_max_epi8(v5, perm6);
      __m128i v6 = _mm_mask_mov_epi8(max6, 0xaa, min6);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [6,7], [4,5], [2,3], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  6,  7,  4,  5,  2,  3,  1,  
                 0) */
      __m128i perm7 = _mm_shuffle_epi8(v6, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 9, 8, 6, 7, 4, 5, 2, 3, 1, 0));
      __m128i min7 = _mm_min_epi8(v6, perm7);
      __m128i max7 = _mm_max_epi8(v6, perm7);
      __m128i v7 = _mm_mask_mov_epi8(max7, 0x54, min7);
      
      return v7;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
minimum_11_int8_t(int8_t * const arr) 
                             {
      
      __m128i v = _mm_load_si128((__m128i *)arr);
      
      v = minimum_11_int8_t_vec(v);
      
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


