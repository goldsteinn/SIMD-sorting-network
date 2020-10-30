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
#define N 49
#define SORT_NAME batcher_49_uint8_t

#ifndef _SIMD_SORT_batcher_49_uint8_t_H_
#define _SIMD_SORT_batcher_49_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 49
	Underlying Sort Type             : uint8_t
	Network Generation Algorithm     : batcher
	Network Depth                    : 21
	SIMD Instructions                : 2 / 105
	Optimization Preference          : space
	SIMD Type                        : __m512i
	SIMD Instruction Set(s) Used     : AVX512f, AVX512vbmi, AVX512bw
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
     __m512i __attribute__((const)) 

batcher_49_uint8_t_vec(__m512i v) {
      
      __m512i perm0 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 16, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 
                                              1, 0, 31, 30, 29, 28, 27, 26, 
                                              25, 24, 23, 22, 21, 20, 19, 18, 
                                              17, 48, 47, 46, 45, 44, 43, 42, 
                                              41, 40, 39, 38, 37, 36, 35, 34, 
                                              33, 32), v);
      __m512i min0 = _mm512_min_epu8(v, perm0);
      __m512i max0 = _mm512_max_epu8(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi8(max0, 0x1ffff, min0);
      
      __m512i perm1 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 32, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 39, 38, 37, 36, 
                                              35, 34, 33, 48, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 
                                              1, 0, 31, 30, 29, 28, 27, 26, 
                                              25, 24, 23, 22, 21, 20, 19, 18, 
                                              17, 16), v0);
      __m512i min1 = _mm512_min_epu8(v0, perm1);
      __m512i max1 = _mm512_max_epu8(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi8(max1, 0x10000ffff, min1);
      
      __m512i perm2 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 7, 6, 5, 4, 3, 
                                              2, 1, 0, 15, 14, 13, 12, 11, 
                                              10, 9, 8), v1);
      __m512i min2 = _mm512_min_epu8(v1, perm2);
      __m512i max2 = _mm512_max_epu8(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi8(max2, 0xffff00ff, min2);
      
      __m512i perm3 = _mm512_shuffle_epi8(v2, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 55, 54, 53, 52, 51, 
                                          50, 49, 48, 39, 38, 37, 36, 35, 34, 
                                          33, 32, 47, 46, 45, 44, 43, 42, 41, 
                                          40, 23, 22, 21, 20, 19, 18, 17, 16, 
                                          31, 30, 29, 28, 27, 26, 25, 24, 15, 
                                          14, 13, 12, 11, 10, 9, 8, 3, 2, 1, 
                                          0, 7, 6, 5, 4));
      __m512i min3 = _mm512_min_epu8(v2, perm3);
      __m512i max3 = _mm512_max_epu8(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi8(max3, 0xff00ff000f, min3);
      
      __m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 24, 43, 42, 41, 44, 
                                              47, 46, 45, 40, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 31, 30, 29, 28, 
                                              27, 26, 25, 48, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 7, 6, 5, 4, 1, 
                                              0, 3, 2), v3);
      __m512i min4 = _mm512_min_epu8(v3, perm4);
      __m512i max4 = _mm512_max_epu8(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi8(max4, 0xe000100ff03, min4);
      
      __m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 40, 45, 46, 47, 44, 
                                              43, 42, 41, 48, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 15, 14, 13, 12, 
                                              11, 10, 9, 8, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 7, 6, 5, 4, 3, 
                                              2, 0, 1), v4);
      __m512i min5 = _mm512_min_epu8(v4, perm5);
      __m512i max5 = _mm512_max_epu8(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi8(max5, 0x2100ff00ff01, min5);
      
      __m512i perm6 = _mm512_shuffle_epi8(v5, _mm512_set_epi8(63, 62, 61, 60, 
                                          59, 58, 57, 56, 55, 54, 53, 52, 51, 
                                          50, 49, 48, 47, 46, 45, 40, 43, 42, 
                                          41, 44, 35, 34, 33, 32, 39, 38, 37, 
                                          36, 27, 26, 25, 24, 31, 30, 29, 28, 
                                          19, 18, 17, 16, 23, 22, 21, 20, 11, 
                                          10, 9, 8, 15, 14, 13, 12, 7, 6, 5, 
                                          4, 3, 2, 1, 0));
      __m512i min6 = _mm512_min_epu8(v5, perm6);
      __m512i max6 = _mm512_max_epu8(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi8(max6, 0x10f0f0f0f00, min6);
      
      __m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 20, 47, 46, 45, 44, 
                                              15, 14, 13, 12, 39, 38, 37, 36, 
                                              7, 6, 5, 4, 31, 30, 29, 28, 27, 
                                              26, 25, 24, 23, 22, 21, 48, 19, 
                                              18, 17, 16, 43, 42, 41, 40, 11, 
                                              10, 9, 8, 35, 34, 33, 32, 3, 2, 
                                              1, 0), v6);
      __m512i min7 = _mm512_min_epu8(v6, perm7);
      __m512i max7 = _mm512_max_epu8(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi8(max7, 0x10f0f0, min7);
      
      __m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 36, 47, 46, 45, 44, 
                                              31, 30, 29, 28, 39, 38, 37, 48, 
                                              23, 22, 21, 20, 43, 42, 41, 40, 
                                              15, 14, 13, 12, 35, 34, 33, 32, 
                                              7, 6, 5, 4, 27, 26, 25, 24, 11, 
                                              10, 9, 8, 19, 18, 17, 16, 3, 2, 
                                              1, 0), v7);
      __m512i min8 = _mm512_min_epu8(v7, perm8);
      __m512i max8 = _mm512_max_epu8(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi8(max8, 0x10f0f0f0f0, min8);
      
      __m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 44, 47, 46, 45, 48, 
                                              39, 38, 37, 36, 43, 42, 41, 40, 
                                              31, 30, 29, 28, 35, 34, 33, 32, 
                                              23, 22, 21, 20, 27, 26, 25, 24, 
                                              15, 14, 13, 12, 19, 18, 17, 16, 
                                              7, 6, 5, 4, 11, 10, 9, 8, 3, 2, 
                                              1, 0), v8);
      __m512i min9 = _mm512_min_epu8(v8, perm9);
      __m512i max9 = _mm512_max_epu8(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi8(max9, 0x10f0f0f0f0f0, min9);
      
      __m512i perm10 = _mm512_shuffle_epi8(v9, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 44, 45, 
                                           46, 41, 40, 43, 42, 37, 36, 39, 
                                           38, 33, 32, 35, 34, 29, 28, 31, 
                                           30, 25, 24, 27, 26, 21, 20, 23, 
                                           22, 17, 16, 19, 18, 13, 12, 15, 
                                           14, 9, 8, 11, 10, 5, 4, 7, 6, 3, 
                                           2, 1, 0));
      __m512i min10 = _mm512_min_epu8(v9, perm10);
      __m512i max10 = _mm512_max_epu8(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi8(max10, 0x133333333330, min10);
      
      __m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 18, 47, 
                                               46, 15, 14, 43, 42, 11, 10, 
                                               39, 38, 7, 6, 35, 34, 3, 2, 
                                               31, 30, 29, 28, 27, 26, 25, 
                                               24, 23, 22, 21, 20, 19, 48, 
                                               17, 16, 45, 44, 13, 12, 41, 
                                               40, 9, 8, 37, 36, 5, 4, 33, 
                                               32, 1, 0), v10);
      __m512i min11 = _mm512_min_epu8(v10, perm11);
      __m512i max11 = _mm512_max_epu8(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi8(max11, 0x4cccc, min11);
      
      __m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 34, 47, 
                                               46, 31, 30, 43, 42, 27, 26, 
                                               39, 38, 23, 22, 35, 48, 19, 
                                               18, 45, 44, 15, 14, 41, 40, 
                                               11, 10, 37, 36, 7, 6, 33, 32, 
                                               3, 2, 29, 28, 13, 12, 25, 24, 
                                               9, 8, 21, 20, 5, 4, 17, 16, 1, 
                                               0), v11);
      __m512i min12 = _mm512_min_epu8(v11, perm12);
      __m512i max12 = _mm512_max_epu8(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi8(max12, 0x4cccccccc, min12);
      
      __m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 42, 47, 
                                               46, 39, 38, 43, 48, 35, 34, 
                                               45, 44, 31, 30, 41, 40, 27, 
                                               26, 37, 36, 23, 22, 33, 32, 
                                               19, 18, 29, 28, 15, 14, 25, 
                                               24, 11, 10, 21, 20, 7, 6, 17, 
                                               16, 3, 2, 13, 12, 5, 4, 9, 8, 
                                               1, 0), v12);
      __m512i min13 = _mm512_min_epu8(v12, perm13);
      __m512i max13 = _mm512_max_epu8(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi8(max13, 0x4cccccccccc, min13);
      
      __m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 46, 47, 
                                               48, 43, 42, 45, 44, 39, 38, 
                                               41, 40, 35, 34, 37, 36, 31, 
                                               30, 33, 32, 27, 26, 29, 28, 
                                               23, 22, 25, 24, 19, 18, 21, 
                                               20, 15, 14, 17, 16, 11, 10, 
                                               13, 12, 7, 6, 9, 8, 3, 2, 5, 
                                               4, 1, 0), v13);
      __m512i min14 = _mm512_min_epu8(v13, perm14);
      __m512i max14 = _mm512_max_epu8(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi8(max14, 0x4ccccccccccc, min14);
      
      __m512i perm15 = _mm512_shuffle_epi8(v14, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 46, 47, 44, 
                                           45, 42, 43, 40, 41, 38, 39, 36, 
                                           37, 34, 35, 32, 33, 30, 31, 28, 
                                           29, 26, 27, 24, 25, 22, 23, 20, 
                                           21, 18, 19, 16, 17, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 1, 0));
      __m512i min15 = _mm512_min_epu8(v14, perm15);
      __m512i max15 = _mm512_max_epu8(v14, perm15);
      __m512i v15 = _mm512_mask_mov_epi8(max15, 0x555555555554, min15);
      
      __m512i perm16 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 17, 47, 
                                               15, 45, 13, 43, 11, 41, 9, 39, 
                                               7, 37, 5, 35, 3, 33, 1, 31, 
                                               30, 29, 28, 27, 26, 25, 24, 
                                               23, 22, 21, 20, 19, 18, 48, 
                                               16, 46, 14, 44, 12, 42, 10, 
                                               40, 8, 38, 6, 36, 4, 34, 2, 
                                               32, 0), v15);
      __m512i min16 = _mm512_min_epu8(v15, perm16);
      __m512i max16 = _mm512_max_epu8(v15, perm16);
      __m512i v16 = _mm512_mask_mov_epi8(max16, 0x2aaaa, min16);
      
      __m512i perm17 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 33, 47, 
                                               31, 45, 29, 43, 27, 41, 25, 
                                               39, 23, 37, 21, 35, 19, 48, 
                                               17, 46, 15, 44, 13, 42, 11, 
                                               40, 9, 38, 7, 36, 5, 34, 3, 
                                               32, 1, 30, 14, 28, 12, 26, 10, 
                                               24, 8, 22, 6, 20, 4, 18, 2, 
                                               16, 0), v16);
      __m512i min17 = _mm512_min_epu8(v16, perm17);
      __m512i max17 = _mm512_max_epu8(v16, perm17);
      __m512i v17 = _mm512_mask_mov_epi8(max17, 0x2aaaaaaaa, min17);
      
      __m512i perm18 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 41, 47, 
                                               39, 45, 37, 43, 35, 48, 33, 
                                               46, 31, 44, 29, 42, 27, 40, 
                                               25, 38, 23, 36, 21, 34, 19, 
                                               32, 17, 30, 15, 28, 13, 26, 
                                               11, 24, 9, 22, 7, 20, 5, 18, 
                                               3, 16, 1, 14, 6, 12, 4, 10, 2, 
                                               8, 0), v17);
      __m512i min18 = _mm512_min_epu8(v17, perm18);
      __m512i max18 = _mm512_max_epu8(v17, perm18);
      __m512i v18 = _mm512_mask_mov_epi8(max18, 0x2aaaaaaaaaa, min18);
      
      __m512i perm19 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 45, 47, 
                                               43, 48, 41, 46, 39, 44, 37, 
                                               42, 35, 40, 33, 38, 31, 36, 
                                               29, 34, 27, 32, 25, 30, 23, 
                                               28, 21, 26, 19, 24, 17, 22, 
                                               15, 20, 13, 18, 11, 16, 9, 14, 
                                               7, 12, 5, 10, 3, 8, 1, 6, 2, 
                                               4, 0), v18);
      __m512i min19 = _mm512_min_epu8(v18, perm19);
      __m512i max19 = _mm512_max_epu8(v18, perm19);
      __m512i v19 = _mm512_mask_mov_epi8(max19, 0x2aaaaaaaaaaa, min19);
      
      __m512i perm20 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 47, 48, 
                                               45, 46, 43, 44, 41, 42, 39, 
                                               40, 37, 38, 35, 36, 33, 34, 
                                               31, 32, 29, 30, 27, 28, 25, 
                                               26, 23, 24, 21, 22, 19, 20, 
                                               17, 18, 15, 16, 13, 14, 11, 
                                               12, 9, 10, 7, 8, 5, 6, 3, 4, 
                                               1, 2, 0), v19);
      __m512i min20 = _mm512_min_epu8(v19, perm20);
      __m512i max20 = _mm512_max_epu8(v19, perm20);
      __m512i v20 = _mm512_mask_mov_epi8(max20, 0xaaaaaaaaaaaa, min20);
      
      return v20;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

batcher_49_uint8_t(uint8_t * const arr) 
                                 {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = batcher_49_uint8_t_vec(v);
      
      _mm512_store_si512((__m512i *)arr, v);
      
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


