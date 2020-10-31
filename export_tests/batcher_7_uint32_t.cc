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
#define N 7
#define SORT_NAME batcher_7_uint32_t

#ifndef _SIMD_SORT_batcher_7_uint32_t_H_
#define _SIMD_SORT_batcher_7_uint32_t_H_

/*

Sorting Network Information:
	Sort Size                        : 7
	Underlying Sort Type             : uint32_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 6
	SIMD Instructions                : 3 / 30
	Optimization Preference          : space
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX2, SSE2, AVX
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



 void fill_works(__m256i v) {
      sarr<TYPE, N> t;
      memcpy(t.arr, &v, 32);
      int i = N;for (; i < 8; ++i) {
          assert(t.arr[i] == uint32_t(0xffffffff));
 }
}

/* SIMD Sort */
 __m256i __attribute__((const)) 
batcher_7_uint32_t_vec(__m256i v) {
      
      /* Pairs: ([7, 7], [2, 6], [0, 4], [3, 3], [6, 2]) */
      /* Perm:  ( 7,  2,  0,  3,  6) */
      __m256i perm0 = _mm256_permutevar8x32_epi32(v, _mm256_set_epi32(7, 2, 
                                                  1, 0, 3, 6, 5, 4));
      __m256i min0 = _mm256_min_epu32(v, perm0);
      __m256i max0 = _mm256_max_epu32(v, perm0);
      __m256i v0 = _mm256_blend_epi32(max0, min0, 0x7);
      
      /* Pairs: ([7, 7], [4, 6], [5, 5], [6, 4]) */
      /* Perm:  ( 7,  4,  5,  6) */
      __m256i perm1 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(31, 30, 29, 28, 
                                          19, 18, 17, 16, 23, 22, 21, 20, 27, 
                                          26, 25, 24, 7, 6, 5, 4, 3, 2, 1, 0, 
                                          15, 14, 13, 12, 11, 10, 9, 8));
      __m256i min1 = _mm256_min_epu32(v0, perm1);
      __m256i max1 = _mm256_max_epu32(v0, perm1);
      __m256i v1 = _mm256_blend_epi32(max1, min1, 0x13);
      
      /* Pairs: ([7, 7], [6, 6], [3, 5], [5, 3]) */
      /* Perm:  ( 7,  6,  3,  5) */
      __m256i perm2 = _mm256_permutevar8x32_epi32(v1, _mm256_set_epi32(7, 6, 
                                                  3, 2, 5, 4, 0, 1));
      __m256i min2 = _mm256_min_epu32(v1, perm2);
      __m256i max2 = _mm256_max_epu32(v1, perm2);
      __m256i v2 = _mm256_blend_epi32(max2, min2, 0xd);
      
      /* Pairs: ([7, 7], [6, 6], [4, 5], [5, 4]) */
      /* Perm:  ( 7,  6,  4,  5) */
      __m256i perm3 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(31, 30, 29, 28, 
                                          27, 26, 25, 24, 19, 18, 17, 16, 23, 
                                          22, 21, 20, 11, 10, 9, 8, 15, 14, 
                                          13, 12, 7, 6, 5, 4, 3, 2, 1, 0));
      __m256i min3 = _mm256_min_epu32(v2, perm3);
      __m256i max3 = _mm256_max_epu32(v2, perm3);
      __m256i v3 = _mm256_blend_epi32(max3, min3, 0x14);
      
      /* Pairs: ([7, 7], [3, 6], [5, 5], [6, 3]) */
      /* Perm:  ( 7,  3,  5,  6) */
      __m256i perm4 = _mm256_permutevar8x32_epi32(v3, _mm256_set_epi32(7, 3, 
                                                  5, 1, 6, 2, 4, 0));
      __m256i min4 = _mm256_min_epu32(v3, perm4);
      __m256i max4 = _mm256_max_epu32(v3, perm4);
      __m256i v4 = _mm256_blend_epi32(max4, min4, 0xa);
      
      /* Pairs: ([7, 7], [5, 6], [6, 5], [3, 4], [4, 3]) */
      /* Perm:  ( 7,  5,  6,  3,  4) */
      __m256i perm5 = _mm256_permutevar8x32_epi32(v4, _mm256_set_epi32(7, 5, 
                                                  6, 3, 4, 1, 2, 0));
      __m256i min5 = _mm256_min_epu32(v4, perm5);
      __m256i max5 = _mm256_max_epu32(v4, perm5);
      __m256i v5 = _mm256_blend_epi32(max5, min5, 0x2a);
      
      return v5;
 }



/* Wrapper For SIMD Sort */
 void inline __attribute__((always_inline)) 
batcher_7_uint32_t(uint32_t * const 
                             arr) {
      
      __m256i _tmp0 = _mm256_set1_epi32(uint32_t(0xffffffff));
      asm volatile("vpblendd %[load_mask], (%[arr]), %[fill_v], %[fill_v]\n"
                   : [ fill_v ] "+x" (_tmp0)
                   : [ arr ] "r" (arr), [ load_mask ] "i" (0x7f)
                   :);
      __m256i v = _tmp0;
      fill_works(v);
      v = batcher_7_uint32_t_vec(v);
      
      fill_works(v);_mm256_maskstore_epi32((int32_t * const)arr, 
                                            _mm256_set_epi32(0x0, 0x80000000, 
                                            0x80000000, 0x80000000, 
                                            0x80000000, 0x80000000, 
                                            0x80000000, 0x80000000), v);
      
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


