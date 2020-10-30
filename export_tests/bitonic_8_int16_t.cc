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
#define N 8
#define SORT_NAME bitonic_8_int16_t

#ifndef _SIMD_SORT_bitonic_8_int16_t_H_
#define _SIMD_SORT_bitonic_8_int16_t_H_

/*

Sorting Network Information:
	Sort Size                        : 8
	Underlying Sort Type             : int16_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 6
	SIMD Instructions                : 2 / 29
	Optimization Preference          : space
	SIMD Type                        : __m128i
	SIMD Instruction Set(s) Used     : SSE2, SSE4.1, AVX2, SSSE3
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

bitonic_8_int16_t_vec(__m128i v) {
      
      __m128i perm0 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(v, 0xb1), 
                                          0xb1);
      __m128i min0 = _mm_min_epi16(v, perm0);
      __m128i max0 = _mm_max_epi16(v, perm0);
      __m128i v0 = _mm_blend_epi16(max0, min0, 0x55);
      
      __m128i perm1 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(v0, 0x1b), 
                                          0x1b);
      __m128i min1 = _mm_min_epi16(v0, perm1);
      __m128i max1 = _mm_max_epi16(v0, perm1);
      __m128i v1 = _mm_blend_epi32(max1, min1, 0x5);
      
      __m128i perm2 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(v1, 0xb1), 
                                          0xb1);
      __m128i min2 = _mm_min_epi16(v1, perm2);
      __m128i max2 = _mm_max_epi16(v1, perm2);
      __m128i v2 = _mm_blend_epi16(max2, min2, 0x55);
      
      __m128i perm3 = _mm_shuffle_epi8(v2, _mm_set_epi8(1, 0, 3, 2, 5, 4, 7, 
                                       6, 9, 8, 11, 10, 13, 12, 15, 14));
      __m128i min3 = _mm_min_epi16(v2, perm3);
      __m128i max3 = _mm_max_epi16(v2, perm3);
      __m128i v3 = _mm_blend_epi32(max3, min3, 0x3);
      
      __m128i perm4 = _mm_shuffle_epi32(v3, uint8_t(0xb1));
      __m128i min4 = _mm_min_epi16(v3, perm4);
      __m128i max4 = _mm_max_epi16(v3, perm4);
      __m128i v4 = _mm_blend_epi32(max4, min4, 0x5);
      
      __m128i perm5 = _mm_shufflehi_epi16(_mm_shufflelo_epi16(v4, 0xb1), 
                                          0xb1);
      __m128i min5 = _mm_min_epi16(v4, perm5);
      __m128i max5 = _mm_max_epi16(v4, perm5);
      __m128i v5 = _mm_blend_epi16(max5, min5, 0x55);
      
      return v5;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

bitonic_8_int16_t(int16_t * const arr) 
                                 {
      
      __m128i v = _mm_load_si128((__m128i *)arr);
      
      v = bitonic_8_int16_t_vec(v);
      
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


