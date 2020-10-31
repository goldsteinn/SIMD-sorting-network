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

#define TYPE uint8_t
#define N 7
#define SORT_NAME minimum_7_uint8_t

#ifndef _SIMD_SORT_minimum_7_uint8_t_H_
#define _SIMD_SORT_minimum_7_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 7
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : minimum
	Network Depth                    : 6
	SIMD Instructions                : 1 / 42
	Optimization Preference          : space
	SIMD Type                        : __m64
	SIMD Instruction Set(s) Used     : MMX, SSSE3, SSE
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

#include <xmmintrin.h>
#include <immintrin.h>
#include <stdint.h>



void fill_works(__m64 v) {
sarr<TYPE, N> t;
memcpy(t.arr, &v, 8);
int i = N;for (; i < 8; ++i) {
assert(t.arr[i] == uint8_t(0xff));
}
}

/* SIMD Sort */
__m64 __attribute__((const)) minimum_7_uint8_t_vec(__m64 v) {

/* Pairs: ([7,7], [0,6], [4,5], [2,3], [1,1]) */
/* Perm:  ( 7,  0,  4,  5,  2,  3,  1,  6) */
__m64 perm0 = _mm_shuffle_pi8(v, _mm_set_pi8(7, 0, 4, 5, 2, 3, 1, 6));
__m64 min0 = _mm_min_pu8(v, perm0);
__m64 max0 = _mm_max_pu8(v, perm0);
__m64 _tmp1 = (__m64)(0xff00ff00ffUL);
__m64 v0 = _mm_or_si64(_mm_and_si64(_tmp1, min0), _mm_andnot_si64(_tmp1, max0));

/* Pairs: ([7,7], [3,6], [5,5], [1,4], [0,2]) */
/* Perm:  ( 7,  3,  5,  1,  6,  0,  4,  2) */
__m64 perm1 = _mm_shuffle_pi8(v0, _mm_set_pi8(7, 3, 5, 1, 6, 0, 4, 2));
__m64 min1 = _mm_min_pu8(v0, perm1);
__m64 max1 = _mm_max_pu8(v0, perm1);
__m64 _tmp2 = (__m64)(0xff00ffffUL);
__m64 v1 = _mm_or_si64(_mm_and_si64(_tmp2, min1), _mm_andnot_si64(_tmp2, max1));

/* Pairs: ([7,7], [6,6], [2,5], [3,4], [0,1]) */
/* Perm:  ( 7,  6,  2,  3,  4,  5,  0,  1) */
__m64 perm2 = _mm_shuffle_pi8(v1, _mm_set_pi8(7, 6, 2, 3, 4, 5, 0, 1));
__m64 min2 = _mm_min_pu8(v1, perm2);
__m64 max2 = _mm_max_pu8(v1, perm2);
__m64 _tmp3 = (__m64)(0xffff00ffUL);
__m64 v2 = _mm_or_si64(_mm_and_si64(_tmp3, min2), _mm_andnot_si64(_tmp3, max2));

/* Pairs: ([7,7], [4,6], [5,5], [3,3], [1,2], [0,0]) */
/* Perm:  ( 7,  4,  5,  6,  3,  1,  2,  0) */
__m64 perm3 = _mm_shuffle_pi8(v2, _mm_set_pi8(7, 4, 5, 6, 3, 1, 2, 0));
__m64 min3 = _mm_min_pu8(v2, perm3);
__m64 max3 = _mm_max_pu8(v2, perm3);
__m64 _tmp4 = (__m64)(0xff0000ff00UL);
__m64 v3 = _mm_or_si64(_mm_and_si64(_tmp4, min3), _mm_andnot_si64(_tmp4, max3));

/* Pairs: ([7,7], [6,6], [4,5], [2,3], [1,1], [0,0]) */
/* Perm:  ( 7,  6,  4,  5,  2,  3,  1,  0) */
__m64 perm4 = _mm_shuffle_pi8(v3, _mm_set_pi8(7, 6, 4, 5, 2, 3, 1, 0));
__m64 min4 = _mm_min_pu8(v3, perm4);
__m64 max4 = _mm_max_pu8(v3, perm4);
__m64 _tmp5 = (__m64)(0xff00ff0000UL);
__m64 v4 = _mm_or_si64(_mm_and_si64(_tmp5, min4), _mm_andnot_si64(_tmp5, max4));

/* Pairs: ([7,7], [5,6], [3,4], [1,2], [0,0]) */
/* Perm:  ( 7,  5,  6,  3,  4,  1,  2,  0) */
__m64 perm5 = _mm_shuffle_pi8(v4, _mm_set_pi8(7, 5, 6, 3, 4, 1, 2, 0));
__m64 min5 = _mm_min_pu8(v4, perm5);
__m64 max5 = _mm_max_pu8(v4, perm5);
__m64 _tmp6 = (__m64)(0xff00ff00ff00UL);
__m64 v5 = _mm_or_si64(_mm_and_si64(_tmp6, min5), _mm_andnot_si64(_tmp6, max5));

return v5;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) minimum_7_uint8_t(uint8_t * const arr) {

__m64 _tmp0 = _mm_set1_pi8(uint8_t(0xff));
__builtin_memcpy(&_tmp0, arr, 7);
__m64 v = _tmp0;
fill_works(v);
v = minimum_7_uint8_t_vec(v);

fill_works(v);__builtin_memcpy(arr, &v, 7);

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


