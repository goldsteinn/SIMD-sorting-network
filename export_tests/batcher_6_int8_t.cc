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
#define N 6
#define SORT_NAME batcher_6_int8_t

#ifndef _SIMD_SORT_batcher_6_int8_t_H_
#define _SIMD_SORT_batcher_6_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 6
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 6
	SIMD Instructions                : 0 / 76
	Optimization Preference          : space
	SIMD Type                        : __m64
	SIMD Instruction Set(s) Used     : MMX, SSSE3
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

#include <xmmintrin.h>
#include <immintrin.h>
#include <stdint.h>

typedef __m64 _aliasing_m64_ __attribute__((aligned(8), may_alias));


/* SIMD Sort */
     __m64 __attribute__((const)) 

batcher_6_int8_t_vec(__m64 v) {
      
      __m64 perm0 = _mm_shuffle_pi16(v, 0xc6);
      __m64 _tmp0 = _mm_cmpgt_pi8(perm0, v);
      __m64 min0 = _mm_or_si64(_mm_and_si64(_tmp0, v), _mm_andnot_si64(_tmp0, 
                                            perm0));
      __m64 _tmp1 = _mm_cmpgt_pi8(v, perm0);
      __m64 max0 = _mm_or_si64(_mm_and_si64(_tmp1, v), _mm_andnot_si64(_tmp1, 
                                            perm0));
      __m64 _tmp2 = (__m64)(0xffffUL);
      __m64 v0 = _mm_or_si64(_mm_and_si64(_tmp2, min0), 
                                          _mm_andnot_si64(_tmp2, max0));
      
      __m64 perm1 = _mm_shuffle_pi16(v0, 0xe1);
      __m64 _tmp3 = _mm_cmpgt_pi8(perm1, v0);
      __m64 min1 = _mm_or_si64(_mm_and_si64(_tmp3, v0), 
                                            _mm_andnot_si64(_tmp3, perm1));
      __m64 _tmp4 = _mm_cmpgt_pi8(v0, perm1);
      __m64 max1 = _mm_or_si64(_mm_and_si64(_tmp4, v0), 
                                            _mm_andnot_si64(_tmp4, perm1));
      __m64 _tmp5 = (__m64)(0xffffUL);
      __m64 v1 = _mm_or_si64(_mm_and_si64(_tmp5, min1), 
                                          _mm_andnot_si64(_tmp5, max1));
      
      __m64 perm2 = _mm_shuffle_pi8(v1, _mm_set_pi8(7, 6, 3, 2, 5, 4, 0, 
                                                    1));
      __m64 _tmp6 = _mm_cmpgt_pi8(perm2, v1);
      __m64 min2 = _mm_or_si64(_mm_and_si64(_tmp6, v1), 
                                            _mm_andnot_si64(_tmp6, perm2));
      __m64 _tmp7 = _mm_cmpgt_pi8(v1, perm2);
      __m64 max2 = _mm_or_si64(_mm_and_si64(_tmp7, v1), 
                                            _mm_andnot_si64(_tmp7, perm2));
      __m64 _tmp8 = (__m64)(0xffff00ffUL);
      __m64 v2 = _mm_or_si64(_mm_and_si64(_tmp8, min2), 
                                          _mm_andnot_si64(_tmp8, max2));
      
      __m64 perm3 = _mm_shuffle_pi8(v2, _mm_set_pi8(7, 6, 4, 5, 2, 3, 1, 
                                                    0));
      __m64 _tmp9 = _mm_cmpgt_pi8(perm3, v2);
      __m64 min3 = _mm_or_si64(_mm_and_si64(_tmp9, v2), 
                                            _mm_andnot_si64(_tmp9, perm3));
      __m64 _tmp10 = _mm_cmpgt_pi8(v2, perm3);
      __m64 max3 = _mm_or_si64(_mm_and_si64(_tmp10, v2), 
                                            _mm_andnot_si64(_tmp10, perm3));
      __m64 _tmp11 = (__m64)(0xff00ff0000UL);
      __m64 v3 = _mm_or_si64(_mm_and_si64(_tmp11, min3), 
                                          _mm_andnot_si64(_tmp11, max3));
      
      __m64 perm4 = _mm_shuffle_pi8(v3, _mm_set_pi8(7, 6, 5, 1, 3, 2, 4, 
                                                    0));
      __m64 _tmp12 = _mm_cmpgt_pi8(perm4, v3);
      __m64 min4 = _mm_or_si64(_mm_and_si64(_tmp12, v3), 
                                            _mm_andnot_si64(_tmp12, perm4));
      __m64 _tmp13 = _mm_cmpgt_pi8(v3, perm4);
      __m64 max4 = _mm_or_si64(_mm_and_si64(_tmp13, v3), 
                                            _mm_andnot_si64(_tmp13, perm4));
      __m64 _tmp14 = (__m64)(0xff00UL);
      __m64 v4 = _mm_or_si64(_mm_and_si64(_tmp14, min4), 
                                          _mm_andnot_si64(_tmp14, max4));
      
      __m64 perm5 = _mm_shuffle_pi8(v4, _mm_set_pi8(7, 6, 5, 3, 4, 1, 2, 
                                                    0));
      __m64 _tmp15 = _mm_cmpgt_pi8(perm5, v4);
      __m64 min5 = _mm_or_si64(_mm_and_si64(_tmp15, v4), 
                                            _mm_andnot_si64(_tmp15, perm5));
      __m64 _tmp16 = _mm_cmpgt_pi8(v4, perm5);
      __m64 max5 = _mm_or_si64(_mm_and_si64(_tmp16, v4), 
                                            _mm_andnot_si64(_tmp16, perm5));
      __m64 _tmp17 = (__m64)(0xff00ff00UL);
      __m64 v5 = _mm_or_si64(_mm_and_si64(_tmp17, min5), 
                                          _mm_andnot_si64(_tmp17, max5));
      
      return v5;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

batcher_6_int8_t(int8_t * const arr) {
      
      __m64 v = (*((_aliasing_m64_ *)arr));
      
      v = batcher_6_int8_t_vec(v);
      
      (*((_aliasing_m64_ *)arr)) = v;
      
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

