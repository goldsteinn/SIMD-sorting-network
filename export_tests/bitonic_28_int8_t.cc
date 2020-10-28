#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifndef _SIMD_SORT_bitonic_28_int8_t_H_
#define _SIMD_SORT_bitonic_28_int8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 28
	Underlying Sort Type             : int8_t
	Network Generation Algorithm     : bitonic
	Network Depth                    : 15
	SIMD Instructions                : 6 / 125
	SIMD Type                        : __m256i
	SIMD Instruction Set(s) Used     : AVX2, AVX
	SIMD Instruction Set(s) Excluded : AVX512*
	Aligned Load & Store             : True
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
   is best is probably application dependent. Note that while "Optimization.SPACE"
   will save .rodata memory it will often cost more in .text memory.

 */

#include <immintrin.h>
#include <stdint.h>



/* SIMD Sort */
__m256i __attribute__((const)) bitonic_28_int8_t_vec(__m256i v) {

__m256i _tmp2 = _mm256_permute4x64_epi64(v, 0x4e);
__m256i _tmp3 = _mm256_shuffle_epi8(v, _mm256_set_epi8(15, 14, 13, 12, 10, 11, 8, 9, 6, 7, 5, 3, 4, 1, 2, 128, 128, 14, 12, 13, 10, 11, 8, 9, 7, 5, 6, 3, 4, 1, 2, 0));
__m256i _tmp4 = _mm256_shuffle_epi8(_tmp2, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 15, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm0 = _mm256_or_si256(_tmp3, _tmp4);
__m256i min0 = _mm256_min_epi8(v, perm0);
__m256i max0 = _mm256_max_epi8(v, perm0);
__m256i v0 = _mm256_blendv_epi8(max0, min0, _mm256_set_epi8(0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 128, 0));

__m256i _tmp5 = _mm256_permute4x64_epi64(v0, 0x4e);
__m256i _tmp6 = _mm256_shuffle_epi8(v0, _mm256_set_epi8(15, 14, 13, 12, 8, 9, 10, 11, 7, 5, 6, 1, 2, 3, 4, 128, 15, 128, 10, 11, 12, 13, 7, 8, 9, 3, 4, 5, 6, 2, 0, 1));
__m256i _tmp7 = _mm256_shuffle_epi8(_tmp5, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 14, 128, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm1 = _mm256_or_si256(_tmp6, _tmp7);
__m256i min1 = _mm256_min_epi8(v0, perm1);
__m256i max1 = _mm256_max_epi8(v0, perm1);
__m256i v1 = _mm256_blendv_epi8(max1, min1, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 128, 0, 0, 128, 128, 0, 0, 128, 0, 0, 128, 128, 0, 0, 128, 0, 0, 128, 128, 0, 0, 128));

__m256i perm2 = _mm256_shuffle_epi8(v1, _mm256_set_epi8(31, 30, 29, 28, 26, 27, 24, 25, 22, 23, 21, 19, 20, 17, 18, 16, 14, 15, 12, 13, 10, 11, 9, 7, 8, 5, 6, 3, 4, 1, 2, 0));
__m256i min2 = _mm256_min_epi8(v1, perm2);
__m256i max2 = _mm256_max_epi8(v1, perm2);
__m256i v2 = _mm256_blendv_epi8(max2, min2, _mm256_set_epi8(0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 128, 0, 128, 0));

__m256i _tmp8 = _mm256_permute4x64_epi64(v2, 0x4e);
__m256i _tmp9 = _mm256_shuffle_epi8(v2, _mm256_set_epi8(15, 14, 13, 12, 5, 6, 7, 8, 9, 10, 11, 4, 128, 128, 0, 1, 128, 128, 13, 7, 8, 9, 10, 11, 12, 0, 1, 2, 3, 4, 5, 6));
__m256i _tmp10 = _mm256_shuffle_epi8(_tmp8, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 14, 15, 128, 128, 2, 3, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm3 = _mm256_or_si256(_tmp9, _tmp10);
__m256i min3 = _mm256_min_epi8(v2, perm3);
__m256i max3 = _mm256_max_epi8(v2, perm3);
__m256i v3 = _mm256_blendv_epi8(max3, min3, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128));

__m256i _tmp11 = _mm256_permute4x64_epi64(v3, 0x4e);
__m256i _tmp12 = _mm256_shuffle_epi8(v3, _mm256_set_epi8(15, 14, 13, 12, 9, 10, 11, 6, 5, 8, 7, 2, 1, 4, 3, 128, 15, 128, 11, 10, 13, 12, 7, 8, 9, 4, 5, 6, 1, 0, 3, 2));
__m256i _tmp13 = _mm256_shuffle_epi8(_tmp11, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 14, 128, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm4 = _mm256_or_si256(_tmp12, _tmp13);
__m256i min4 = _mm256_min_epi8(v3, perm4);
__m256i max4 = _mm256_max_epi8(v3, perm4);
__m256i v4 = _mm256_blendv_epi8(max4, min4, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 0, 0, 128, 128, 0, 0, 128, 0, 0, 128, 0, 0, 128, 128));

__m256i _tmp14 = _mm256_permute4x64_epi64(v4, 0x4e);
__m256i _tmp15 = _mm256_shuffle_epi8(v4, _mm256_set_epi8(15, 14, 13, 12, 11, 9, 10, 7, 8, 5, 6, 3, 4, 1, 2, 128, 128, 14, 12, 13, 10, 11, 8, 9, 7, 6, 4, 5, 2, 3, 0, 1));
__m256i _tmp16 = _mm256_shuffle_epi8(_tmp14, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 15, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm5 = _mm256_or_si256(_tmp15, _tmp16);
__m256i min5 = _mm256_min_epi8(v4, perm5);
__m256i max5 = _mm256_max_epi8(v4, perm5);
__m256i v5 = _mm256_blendv_epi8(max5, min5, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 0, 128, 0, 128, 0, 128, 0, 0, 0, 128, 0, 128, 0, 128));

__m256i _tmp17 = _mm256_permute4x64_epi64(v5, 0x4e);
__m256i _tmp18 = _mm256_shuffle_epi8(v5, _mm256_set_epi8(15, 14, 13, 12, 128, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 128, 14, 13, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12));
__m256i _tmp19 = _mm256_shuffle_epi8(_tmp17, _mm256_set_epi8(128, 128, 128, 128, 15, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 11, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm6 = _mm256_or_si256(_tmp18, _tmp19);
__m256i min6 = _mm256_min_epi8(v5, perm6);
__m256i max6 = _mm256_max_epi8(v5, perm6);
__m256i v6 = _mm256_blendv_epi8(max6, min6, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128));

__m256i _tmp20 = _mm256_permute4x64_epi64(v6, 0x4e);
__m256i _tmp21 = _mm256_shuffle_epi8(v6, _mm256_set_epi8(15, 14, 13, 12, 7, 6, 9, 8, 11, 10, 1, 0, 128, 128, 5, 4, 128, 128, 9, 8, 7, 6, 13, 12, 11, 10, 1, 0, 3, 2, 5, 4));
__m256i _tmp22 = _mm256_shuffle_epi8(_tmp20, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 15, 14, 128, 128, 3, 2, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm7 = _mm256_or_si256(_tmp21, _tmp22);
__m256i min7 = _mm256_min_epi8(v6, perm7);
__m256i max7 = _mm256_max_epi8(v6, perm7);
__m256i v7 = _mm256_blendv_epi8(max7, min7, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 128, 128));

__m256i _tmp23 = _mm256_permute4x64_epi64(v7, 0x4e);
__m256i _tmp24 = _mm256_shuffle_epi8(v7, _mm256_set_epi8(15, 14, 13, 12, 10, 11, 7, 6, 9, 8, 3, 2, 5, 4, 128, 128, 128, 128, 11, 10, 13, 12, 7, 6, 9, 8, 3, 2, 5, 4, 0, 1));
__m256i _tmp25 = _mm256_shuffle_epi8(_tmp23, _mm256_set_epi8(128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 15, 14, 1, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128));
__m256i perm8 = _mm256_or_si256(_tmp24, _tmp25);
__m256i min8 = _mm256_min_epi8(v7, perm8);
__m256i max8 = _mm256_max_epi8(v7, perm8);
__m256i v8 = _mm256_blendv_epi8(max8, min8, _mm256_set_epi8(0, 0, 0, 0, 0, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 128));

__m256i perm9 = _mm256_shuffle_epi8(v8, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 24, 25, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 1, 0));
__m256i min9 = _mm256_min_epi8(v8, perm9);
__m256i max9 = _mm256_max_epi8(v8, perm9);
__m256i v9 = _mm256_blendv_epi8(max9, min9, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 0));

__m256i _tmp26 = _mm256_permute4x64_epi64(v9, 0x4e);
__m256i _tmp27 = _mm256_shuffle_epi8(v9, _mm256_set_epi8(15, 14, 13, 12, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 15, 14, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 1, 0));
__m256i _tmp28 = _mm256_shuffle_epi8(_tmp26, _mm256_set_epi8(128, 128, 128, 128, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 128, 128, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 128, 128));
__m256i perm10 = _mm256_or_si256(_tmp27, _tmp28);
__m256i min10 = _mm256_min_epi8(v9, perm10);
__m256i max10 = _mm256_max_epi8(v9, perm10);
__m256i v10 = _mm256_blendv_epi8(max10, min10, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 0, 0));

__m256i perm11 = _mm256_shuffle_epi8(v10, _mm256_set_epi8(31, 30, 29, 28, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 6, 7, 5, 4, 3, 2, 1, 0, 14, 15, 13, 12, 11, 10, 9, 8));
__m256i min11 = _mm256_min_epi8(v10, perm11);
__m256i max11 = _mm256_max_epi8(v10, perm11);
__m256i v11 = _mm256_blend_epi32(max11, min11, 0x13);

__m256i perm12 = _mm256_shuffle_epi8(v11, _mm256_set_epi8(31, 30, 29, 28, 25, 24, 27, 26, 19, 18, 17, 16, 23, 22, 21, 20, 10, 11, 9, 8, 14, 15, 13, 12, 3, 2, 1, 0, 7, 6, 5, 4));
__m256i min12 = _mm256_min_epi8(v11, perm12);
__m256i max12 = _mm256_max_epi8(v11, perm12);
__m256i v12 = _mm256_blendv_epi8(max12, min12, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 128, 128, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128, 128, 0, 0, 0, 0, 128, 128, 128, 128));

__m256i perm13 = _mm256_shuffle_epi8(v12, _mm256_set_epi8(31, 30, 29, 28, 26, 27, 24, 25, 21, 20, 23, 22, 17, 16, 19, 18, 12, 13, 14, 15, 9, 8, 11, 10, 5, 4, 7, 6, 1, 0, 3, 2));
__m256i min13 = _mm256_min_epi8(v12, perm13);
__m256i max13 = _mm256_max_epi8(v12, perm13);
__m256i v13 = _mm256_blendv_epi8(max13, min13, _mm256_set_epi8(0, 0, 0, 0, 0, 128, 0, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128, 0, 0, 128, 128));

__m256i perm14 = _mm256_shuffle_epi8(v13, _mm256_set_epi8(31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 21, 18, 19, 16, 17, 14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1));
__m256i min14 = _mm256_min_epi8(v13, perm14);
__m256i max14 = _mm256_max_epi8(v13, perm14);
__m256i v14 = _mm256_blendv_epi8(max14, min14, _mm256_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128, 0, 128));

return v14;
}



/* Wrapper For SIMD Sort */
void inline __attribute__((always_inline)) bitonic_28_int8_t(int8_t * const arr) {

__m256i _tmp0 = _mm256_maskload_epi32((int32_t * const)arr, _mm256_set_epi32(0x0, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000));
__m256i _tmp1 = _mm256_set1_epi8(int8_t(0x7f));
__m256i v = _mm256_blend_epi32(_tmp1, _tmp0, 0x7f);

v = bitonic_28_int8_t_vec(v);

_mm256_maskstore_epi32((int32_t * const)arr, _mm256_set_epi32(0x0, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000), v);

}


#endif


#define TYPE int8_t
#define N 28
#define SORT_NAME bitonic_28_int8_t

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


