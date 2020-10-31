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
#define SORT_NAME batcher_11_int8_t

#ifndef _SIMD_SORT_batcher_11_int8_t_H_
#define _SIMD_SORT_batcher_11_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 11
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 10
	SIMD Instructions                : 5 / 59
	Optimization Preference          : space
	SIMD Type                        : __m128i
	SIMD Instruction Set(s) Used     : SSE4.1, SSE2, SSSE3, AVX2
	SIMD Instruction Set(s) Excluded : AVX512*
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



 void fill_works(__m128i v) {
      sarr<TYPE, N> t;
      memcpy(t.arr, &v, 16);
      int i = N;for (; i < 16; ++i) {
          assert(t.arr[i] == int8_t(0x7f));
 }
}

/* SIMD Sort */
 __m128i __attribute__((const)) 
batcher_11_int8_t_vec(__m128i v) {
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [2,10], [1,9], 
                 [0,8], [3,7], [6,6], [5,5], [7,3], [10,2], [9,1], [8,0]) */
      /* Perm:  (15, 14, 13, 12, 11,  2,  1,  0,  3,  6,  5,  7, 10,  9,  8) 
                 */
      __m128i perm0 = _mm_shuffle_epi8(v, _mm_set_epi8(15, 14, 13, 12, 11, 2, 
                                       1, 0, 3, 6, 5, 4, 7, 10, 9, 8));
      __m128i min0 = _mm_min_epi8(v, perm0);
      __m128i max0 = _mm_max_epi8(v, perm0);
      __m128i v0 = _mm_blend_epi32(max0, min0, 0x1);
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [8,8], [2,6], [1,5], [0,4], [6,2], [5,1], [4,0]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  8,  2,  1,  0,  6,  5,  4) */
      __m128i perm1 = _mm_shuffle_epi8(v0, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 9, 8, 7, 2, 1, 0, 3, 6, 5, 4));
      __m128i min1 = _mm_min_epi8(v0, perm1);
      __m128i max1 = _mm_max_epi8(v0, perm1);
      __m128i v1 = _mm_blendv_epi8(max1, min1, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 0, 0, 0, 0, 0, 0, 128, 128, 128));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [6,10], [4,8], 
                 [7,7], [10,6], [8,4], [1,3], [0,2], [3,1], [2,0]) */
      /* Perm:  (15, 14, 13, 12, 11,  6,  4,  7, 10,  8,  1,  0,  3,  2) */
      __m128i perm2 = _mm_shuffle_epi8(v1, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       6, 5, 4, 7, 10, 9, 8, 1, 0, 3, 2));
      __m128i min2 = _mm_min_epi8(v1, perm2);
      __m128i max2 = _mm_max_epi8(v1, perm2);
      __m128i v2 = _mm_blendv_epi8(max2, min2, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 0, 0, 128, 128, 128, 0, 0, 128, 128));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [8,10], [3,9], 
                 [10,8], [4,6], [6,4], [9,3], [0,1], [1,0]) */
      /* Perm:  (15, 14, 13, 12, 11,  8,  3, 10,  4,  6,  9,  0,  1) */
      __m128i perm3 = _mm_shuffle_epi8(v2, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       8, 3, 10, 5, 4, 7, 6, 9, 2, 0, 1));
      __m128i min3 = _mm_min_epi8(v2, perm3);
      __m128i max3 = _mm_max_epi8(v2, perm3);
      __m128i v3 = _mm_blendv_epi8(max3, min3, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 128, 0, 0, 128, 128, 128, 0, 0, 128));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [7,9], 
                 [2,8], [9,7], [6,6], [3,5], [5,3], [8,2]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  7,  2,  9,  6,  3,  5,  8) */
      __m128i perm4 = _mm_shuffle_epi8(v3, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 7, 2, 9, 6, 3, 4, 5, 8, 1, 0));
      __m128i min4 = _mm_min_epi8(v3, perm4);
      __m128i max4 = _mm_max_epi8(v3, perm4);
      __m128i v4 = _mm_blendv_epi8(max4, min4, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 0, 128, 0, 0, 0, 128, 128, 0, 0));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [9,9], 
                 [6,8], [7,7], [8,6], [2,4], [4,2]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  9,  6,  7,  8,  2,  4) */
      __m128i perm5 = _mm_shuffle_epi8(v4, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 9, 6, 7, 8, 5, 2, 3, 4, 1, 0));
      __m128i min5 = _mm_min_epi8(v4, perm5);
      __m128i max5 = _mm_max_epi8(v4, perm5);
      __m128i v5 = _mm_blendv_epi8(max5, min5, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 0, 0, 128, 0, 0, 0, 128, 0, 0));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [10,10], [8,9], 
                 [9,8], [4,5], [5,4], [2,3], [3,2]) */
      /* Perm:  (15, 14, 13, 12, 11, 10,  8,  9,  4,  5,  2,  3) */
      __m128i perm6 = _mm_shuffle_epi8(v5, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       10, 8, 9, 6, 7, 4, 5, 2, 3, 1, 0));
      __m128i min6 = _mm_min_epi8(v5, perm6);
      __m128i max6 = _mm_max_epi8(v5, perm6);
      __m128i v6 = _mm_blendv_epi8(max6, min6, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 128, 0, 128, 0, 128, 0, 128, 0, 0));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [3,10], [9,9], 
                 [1,8], [7,7], [5,5], [10,3], [8,1]) */
      /* Perm:  (15, 14, 13, 12, 11,  3,  9,  1,  7,  5, 10,  8) */
      __m128i perm7 = _mm_shuffle_epi8(v6, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       3, 9, 1, 7, 6, 5, 4, 10, 2, 8, 0));
      __m128i min7 = _mm_min_epi8(v6, perm7);
      __m128i max7 = _mm_max_epi8(v6, perm7);
      __m128i v7 = _mm_blendv_epi8(max7, min7, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 0, 0, 0, 0, 0, 128, 0, 128, 0));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [7,10], [9,9], 
                 [10,7], [3,6], [1,4], [6,3], [4,1]) */
      /* Perm:  (15, 14, 13, 12, 11,  7,  9, 10,  3,  1,  6,  4) */
      __m128i perm8 = _mm_shuffle_epi8(v7, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       7, 9, 5, 10, 3, 8, 1, 6, 2, 4, 0));
      __m128i min8 = _mm_min_epi8(v7, perm8);
      __m128i max8 = _mm_max_epi8(v7, perm8);
      __m128i v8 = _mm_blendv_epi8(max8, min8, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   0, 0, 128, 0, 128, 0, 128, 0, 128, 0));
      
      /* Pairs: ([15,15], [14,14], [13,13], [12,12], [11,11], [9,10], [10,9], 
                 [7,8], [8,7], [3,4], [4,3], [1,2], [2,1]) */
      /* Perm:  (15, 14, 13, 12, 11,  9, 10,  7,  8,  3,  4,  1,  2) */
      __m128i perm9 = _mm_shuffle_epi8(v8, _mm_set_epi8(15, 14, 13, 12, 11, 
                                       9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 0));
      __m128i min9 = _mm_min_epi8(v8, perm9);
      __m128i max9 = _mm_max_epi8(v8, perm9);
      __m128i v9 = _mm_blendv_epi8(max9, min9, _mm_set_epi8(0, 0, 0, 0, 0, 0, 
                                   128, 0, 128, 0, 128, 0, 128, 0, 128, 0));
      
      return v9;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
batcher_11_int8_t(int8_t * const arr) 
                             {
      
      __m128i _tmp0 = _mm_set1_epi8(int8_t(0x7f));
      __m128i _tmp1 = _mm_set_epi8(0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 
                                   128, 128, 128, 128, 128, 128);
      asm volatile("vpblendvb %[load_mask], (%[arr]), %[fill_v], %[fill_v]\n"
                   : [ fill_v ] "+x" (_tmp0)
                   : [ arr ] "r" (arr), [ load_mask ] "x" (_tmp1)
                   :);
      __m128i v = _tmp0;
      fill_works(v);
      v = batcher_11_int8_t_vec(v);
      
      fill_works(v);_mm_maskstore_epi32((int32_t * const)arr, 
                                         _mm_set_epi32(0x0, 0x0, 0x80000000, 
                                         0x80000000), v);
      const uint32_t _tmp2 = _mm_extract_epi32(v, 2);
      __builtin_memcpy(arr + 8, &_tmp2, 3);;
      
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


