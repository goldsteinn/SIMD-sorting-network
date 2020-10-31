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

#define TYPE uint32_t
#define N 16
#define SORT_NAME oddeven_16_uint32_t

#ifndef _SIMD_SORT_oddeven_16_uint32_t_H_
#define _SIMD_SORT_oddeven_16_uint32_t_H_

/*

Sorting Network Information:
	Sort Size                        : 16
	Underlying Sort Type             : uint32_t
	Network Generation Algorithm     : oddeven
	Network Depth                    : 10
	SIMD Instructions                : 3 / 47
	Optimization Preference          : space
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512vl
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



 void fill_works(__m512i v) {
      sarr<TYPE, N> t;
      memcpy(t.arr, &v, 64);
      int i = N;for (; i < 16; ++i) {
          assert(t.arr[i] == uint32_t(0xffffffff));
 }
}

/* SIMD Sort */
 __m512i __attribute__((const)) 
oddeven_16_uint32_t_vec(__m512i v) {
      
      /* Pairs: ([14,15], [12,13], [10,11], [8,9], [6,7], [4,5], [2,3], 
                 [0,1]) */
      /* Perm:  (14, 15, 12, 13, 10, 11,  8,  9,  6,  7,  4,  5,  2,  3,  0,  
                 1) */
      __m512i perm0 = _mm512_shuffle_epi32(v, _MM_PERM_ENUM(0xb1));
      __m512i min0 = _mm512_min_epu32(v, perm0);
      __m512i max0 = _mm512_max_epu32(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi32(max0, 0x5555, min0);
      
      /* Pairs: ([13,15], [12,14], [9,11], [8,10], [5,7], [4,6], [1,3], 
                 [0,2]) */
      /* Perm:  (13, 12, 15, 14,  9,  8, 11, 10,  5,  4,  7,  6,  1,  0,  3,  
                 2) */
      __m512i perm1 = _mm512_shuffle_epi32(v0, _MM_PERM_ENUM(0x4e));
      __m512i min1 = _mm512_min_epu32(v0, perm1);
      __m512i max1 = _mm512_max_epu32(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi32(max1, 0x3333, min1);
      
      /* Pairs: ([11,15], [13,14], [8,12], [9,10], [3,7], [5,6], [0,4], 
                 [1,2]) */
      /* Perm:  (11, 13, 14,  8, 15,  9, 10, 12,  3,  5,  6,  0,  7,  1,  2,  
                 4) */
      __m512i perm2 = _mm512_permutexvar_epi32(_mm512_set_epi32(11, 13, 14, 
                                               8, 15, 9, 10, 12, 3, 5, 6, 0, 
                                               7, 1, 2, 4), v1);
      __m512i min2 = _mm512_min_epu32(v1, perm2);
      __m512i max2 = _mm512_max_epu32(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi32(max2, 0x2b2b, min2);
      
      /* Pairs: ([7,15], [10,14], [9,13], [12,12], [11,11], [0,8], [2,6], 
                 [1,5], [4,4], [3,3]) */
      /* Perm:  ( 7, 10,  9, 12, 11, 14, 13,  0, 15,  2,  1,  4,  3,  6,  5,  
                 8) */
      __m512i perm3 = _mm512_permutexvar_epi32(_mm512_set_epi32(7, 10, 9, 12, 
                                               11, 14, 13, 0, 15, 2, 1, 4, 3, 
                                               6, 5, 8), v2);
      __m512i min3 = _mm512_min_epu32(v2, perm3);
      __m512i max3 = _mm512_max_epu32(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi32(max3, 0x687, min3);
      
      /* Pairs: ([15,15], [14,14], [11,13], [10,12], [9,9], [8,8], [7,7], 
                 [6,6], [3,5], [2,4], [1,1], [0,0]) */
      /* Perm:  (15, 14, 11, 10, 13, 12,  9,  8,  7,  6,  3,  2,  5,  4,  1,  
                 0) */
      __m512i perm4 = _mm512_permutex_epi64(v3, 0xd8);
      __m512i min4 = _mm512_min_epu32(v3, perm4);
      __m512i max4 = _mm512_max_epu32(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi32(max4, 0xc0c, min4);
      
      /* Pairs: ([15,15], [13,14], [11,12], [9,10], [8,8], [7,7], [5,6], 
                 [3,4], [1,2], [0,0]) */
      /* Perm:  (15, 13, 14, 11, 12,  9, 10,  8,  7,  5,  6,  3,  4,  1,  2,  
                 0) */
      __m512i perm5 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 13, 14, 
                                               11, 12, 9, 10, 8, 7, 5, 6, 3, 
                                               4, 1, 2, 0), v4);
      __m512i min5 = _mm512_min_epu32(v4, perm5);
      __m512i max5 = _mm512_max_epu32(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi32(max5, 0x2a2a, min5);
      
      /* Pairs: ([15,15], [6,14], [5,13], [4,12], [3,11], [2,10], [1,9], 
                 [8,8], [7,7], [0,0]) */
      /* Perm:  (15,  6,  5,  4,  3,  2,  1,  8,  7, 14, 13, 12, 11, 10,  9,  
                 0) */
      __m512i perm6 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 6, 5, 4, 
                                               3, 2, 1, 8, 7, 14, 13, 12, 11, 
                                               10, 9, 0), v5);
      __m512i min6 = _mm512_min_epu32(v5, perm6);
      __m512i max6 = _mm512_max_epu32(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi32(max6, 0x7e, min6);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [7,11], [6,10], [5,9], 
                 [4,8], [3,3], [2,2], [1,1], [0,0]) */
      /* Perm:  (15, 14, 13, 12,  7,  6,  5,  4, 11, 10,  9,  8,  3,  2,  1,  
                 0) */
      __m512i perm7 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 13, 
                                               12, 7, 6, 5, 4, 11, 10, 9, 8, 
                                               3, 2, 1, 0), v6);
      __m512i min7 = _mm512_min_epu32(v6, perm7);
      __m512i max7 = _mm512_max_epu32(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi32(max7, 0xf0, min7);
      
      /* Pairs: ([15,15], [14,14], [11,13], [10,12], [7,9], [6,8], [3,5], 
                 [2,4], [1,1], [0,0]) */
      /* Perm:  (15, 14, 11, 10, 13, 12,  7,  6,  9,  8,  3,  2,  5,  4,  1,  
                 0) */
      __m512i perm8 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 14, 11, 
                                               10, 13, 12, 7, 6, 9, 8, 3, 2, 
                                               5, 4, 1, 0), v7);
      __m512i min8 = _mm512_min_epu32(v7, perm8);
      __m512i max8 = _mm512_max_epu32(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi32(max8, 0xccc, min8);
      
      /* Pairs: ([15,15], [13,14], [11,12], [9,10], [7,8], [5,6], [3,4], 
                 [1,2], [0,0]) */
      /* Perm:  (15, 13, 14, 11, 12,  9, 10,  7,  8,  5,  6,  3,  4,  1,  2,  
                 0) */
      __m512i perm9 = _mm512_permutexvar_epi32(_mm512_set_epi32(15, 13, 14, 
                                               11, 12, 9, 10, 7, 8, 5, 6, 3, 
                                               4, 1, 2, 0), v8);
      __m512i min9 = _mm512_min_epu32(v8, perm9);
      __m512i max9 = _mm512_max_epu32(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi32(max9, 0x2aaa, min9);
      
      return v9;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
oddeven_16_uint32_t(uint32_t * const 
                             arr) {
      
      __m512i _tmp0 = _mm512_set1_epi32(uint32_t(0xffffffff));
      __m512i v = _mm512_mask_load_epi32(_tmp0, 0xffff, arr);
      fill_works(v);
      v = oddeven_16_uint32_t_vec(v);
      
      fill_works(v);_mm512_mask_store_epi32((void *)arr, 0xffff, v);
      
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


