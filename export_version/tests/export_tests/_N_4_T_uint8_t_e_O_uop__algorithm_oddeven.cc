#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    ninit() {
        for (uint32_t i = 0; i < n; ++i) {
            T ni = 0;
            T _i = i;
            ni -= i;
            arr[i] = ni;
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

#define TYPE      uint8_t
#define N         4
#define SORT_NAME oddeven_4_uint8_t

#ifndef _SIMD_SORT_oddeven_4_uint8_t_H_
#define _SIMD_SORT_oddeven_4_uint8_t_H_

/*

Sorting Network Information:
        Sort Size                        : 4
        Underlying Sort Type             : uint8_t
        Network Generation Algorithm     : oddeven
        Network Depth                    : 3
        SIMD Instructions                : 0 / 20
        Optimization Preference          : uop
        SIMD Type                        : __m64
        SIMD Instruction Set(s) Used     : MMX, SSSE3, SSE
        SIMD Instruction Set(s) Excluded : None
        Aligned Load & Store             : False
        Integer Aligned Load & Store     : False
        Full Load & Store                : True

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
   is best is probably application dependent. Note that while
"Optimization.SPACE" will save .rodata memory it will often cost more in .text
memory.

 */

#include <immintrin.h>
#include <stdint.h>
#include <xmmintrin.h>

typedef __m64 _aliasing_m64_ __attribute__((aligned(1), may_alias));


/* SIMD Sort */
__m64 __attribute__((const)) oddeven_4_uint8_t_vec(__m64 v) {

    /* Pairs: ([7,7], [6,6], [5,5], [4,4], [2,3], [0,1]) */
    /* Perm:  ( 7,  6,  5,  4,  2,  3,  0,  1) */
    __m64 perm0 = _mm_shuffle_pi8(v, _mm_set_pi8(7, 6, 5, 4, 2, 3, 0, 1));
    __m64 min0  = _mm_min_pu8(v, perm0);
    __m64 max0  = _mm_max_pu8(v, perm0);
    __m64 _tmp0 = (__m64)(0xff00ffUL);
    __m64 v0 =
        _mm_or_si64(_mm_and_si64(_tmp0, min0), _mm_andnot_si64(_tmp0, max0));

    /* Pairs: ([7,7], [6,6], [5,5], [4,4], [1,3], [0,2]) */
    /* Perm:  ( 7,  6,  5,  4,  1,  0,  3,  2) */
    __m64 perm1 = _mm_shuffle_pi16(v0, 0xe1);
    __m64 min1  = _mm_min_pu8(v0, perm1);
    __m64 max1  = _mm_max_pu8(v0, perm1);
    __m64 _tmp1 = (__m64)(0xffffUL);
    __m64 v1 =
        _mm_or_si64(_mm_and_si64(_tmp1, min1), _mm_andnot_si64(_tmp1, max1));

    /* Pairs: ([7,7], [6,6], [5,5], [4,4], [3,3], [1,2], [0,0]) */
    /* Perm:  ( 7,  6,  5,  4,  3,  1,  2,  0) */
    __m64 perm2 = _mm_shuffle_pi8(v1, _mm_set_pi8(7, 6, 5, 4, 3, 1, 2, 0));
    __m64 min2  = _mm_min_pu8(v1, perm2);
    __m64 max2  = _mm_max_pu8(v1, perm2);
    __m64 _tmp2 = (__m64)(0xff00UL);
    __m64 v2 =
        _mm_or_si64(_mm_and_si64(_tmp2, min2), _mm_andnot_si64(_tmp2, max2));

    return v2;
}


/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline))
oddeven_4_uint8_t(uint8_t * const arr) {

    __m64 v = (*((_aliasing_m64_ *)arr));

    v = oddeven_4_uint8_t_vec(v);

    (*((_aliasing_m64_ *)arr)) = v;
}


#endif



#define TSIZE 1000
    void
    test() {
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

        s1.ninit();
        memcpy(s2.arr, s1.arr, 64);

        std::sort(s1.arr, s1.arr + N);
        SORT_NAME(s2.arr);
        assert(!memcmp(s1.arr, s2.arr, 64));

        for (uint32_t i = 0; i < TSIZE; ++i) {
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
