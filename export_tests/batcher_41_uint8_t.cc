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
#define N 41
#define SORT_NAME batcher_41_uint8_t

#ifndef _SIMD_SORT_batcher_41_uint8_t_H_
#define _SIMD_SORT_batcher_41_uint8_t_H_

/*

Sorting Network Information:
	Sort Size                        : 41
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

batcher_41_uint8_t_vec(__m512i v) {
      
      __m512i perm0 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 8, 7, 6, 5, 4, 3, 
                                              2, 1, 0, 15, 14, 13, 12, 11, 
                                              10, 9, 24, 23, 22, 21, 20, 19, 
                                              18, 17, 16, 31, 30, 29, 28, 27, 
                                              26, 25, 40, 39, 38, 37, 36, 35, 
                                              34, 33, 32), v);
      __m512i min0 = _mm512_min_epu8(v, perm0);
      __m512i max0 = _mm512_max_epu8(v, perm0);
      __m512i v0 = _mm512_mask_mov_epi8(max0, 0xffff, min0);
      
      __m512i perm1 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 31, 30, 29, 28, 
                                              27, 26, 25, 8, 7, 6, 5, 4, 3, 
                                              2, 1, 0, 15, 14, 13, 12, 11, 
                                              10, 9, 24, 23, 22, 21, 20, 19, 
                                              18, 17, 16), v0);
      __m512i min1 = _mm512_min_epu8(v0, perm1);
      __m512i max1 = _mm512_max_epu8(v0, perm1);
      __m512i v1 = _mm512_mask_mov_epi8(max1, 0x1ff, min1);
      
      __m512i perm2 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 24, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 31, 30, 29, 28, 
                                              27, 26, 25, 40, 39, 38, 37, 36, 
                                              35, 34, 33, 32, 7, 6, 5, 4, 3, 
                                              2, 1, 0, 15, 14, 13, 12, 11, 
                                              10, 9, 8), v1);
      __m512i min2 = _mm512_min_epu8(v1, perm2);
      __m512i max2 = _mm512_max_epu8(v1, perm2);
      __m512i v2 = _mm512_mask_mov_epi8(max2, 0x1ff00ff, min2);
      
      __m512i perm3 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 32, 15, 14, 13, 12, 
                                              11, 10, 9, 40, 23, 22, 21, 20, 
                                              19, 18, 17, 16, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 39, 38, 37, 36, 
                                              35, 34, 33, 8, 3, 2, 1, 0, 7, 
                                              6, 5, 4), v2);
      __m512i min3 = _mm512_min_epu8(v2, perm3);
      __m512i max3 = _mm512_max_epu8(v2, perm3);
      __m512i v3 = _mm512_mask_mov_epi8(max3, 0x100fffe0f, min3);
      
      __m512i perm4 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 31, 30, 29, 28, 
                                              27, 26, 25, 8, 39, 38, 37, 36, 
                                              35, 34, 33, 24, 15, 14, 13, 12, 
                                              11, 10, 9, 16, 23, 22, 21, 20, 
                                              19, 18, 17, 32, 7, 6, 5, 4, 1, 
                                              0, 3, 2), v3);
      __m512i min4 = _mm512_min_epu8(v3, perm4);
      __m512i max4 = _mm512_max_epu8(v3, perm4);
      __m512i v4 = _mm512_mask_mov_epi8(max4, 0xfe00ff03, min4);
      
      __m512i perm5 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 35, 34, 33, 36, 
                                              39, 38, 37, 24, 27, 26, 25, 28, 
                                              31, 30, 29, 32, 19, 18, 17, 20, 
                                              23, 22, 21, 8, 11, 10, 9, 12, 
                                              15, 14, 13, 16, 7, 6, 5, 4, 3, 
                                              2, 0, 1), v4);
      __m512i min5 = _mm512_min_epu8(v4, perm5);
      __m512i max5 = _mm512_max_epu8(v4, perm5);
      __m512i v5 = _mm512_mask_mov_epi8(max5, 0xe0f0e0f01, min5);
      
      __m512i perm6 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 40, 37, 38, 39, 32, 
                                              7, 6, 5, 36, 31, 30, 29, 24, 
                                              15, 14, 13, 28, 23, 22, 21, 16, 
                                              19, 18, 17, 20, 27, 26, 25, 8, 
                                              11, 10, 9, 12, 35, 34, 33, 4, 
                                              3, 2, 1, 0), v5);
      __m512i min6 = _mm512_min_epu8(v5, perm6);
      __m512i max6 = _mm512_max_epu8(v5, perm6);
      __m512i v6 = _mm512_mask_mov_epi8(max6, 0x210101e1e0, min6);
      
      __m512i perm7 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 12, 39, 38, 37, 36, 
                                              23, 22, 21, 4, 31, 30, 29, 28, 
                                              27, 26, 25, 24, 35, 34, 33, 20, 
                                              7, 6, 5, 16, 15, 14, 13, 40, 
                                              11, 10, 9, 8, 19, 18, 17, 32, 
                                              3, 2, 1, 0), v6);
      __m512i min7 = _mm512_min_epu8(v6, perm7);
      __m512i max7 = _mm512_max_epu8(v6, perm7);
      __m512i v7 = _mm512_mask_mov_epi8(max7, 0xe010f0, min7);
      
      __m512i perm8 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 28, 39, 38, 37, 36, 
                                              31, 30, 29, 20, 35, 34, 33, 40, 
                                              23, 22, 21, 12, 27, 26, 25, 32, 
                                              15, 14, 13, 4, 19, 18, 17, 24, 
                                              7, 6, 5, 8, 11, 10, 9, 16, 3, 
                                              2, 1, 0), v7);
      __m512i min8 = _mm512_min_epu8(v7, perm8);
      __m512i max8 = _mm512_max_epu8(v7, perm8);
      __m512i v8 = _mm512_mask_mov_epi8(max8, 0xf0f0f0f0, min8);
      
      __m512i perm9 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 60, 
                                              59, 58, 57, 56, 55, 54, 53, 52, 
                                              51, 50, 49, 48, 47, 46, 45, 44, 
                                              43, 42, 41, 36, 39, 38, 37, 40, 
                                              33, 34, 35, 28, 29, 30, 31, 32, 
                                              25, 26, 27, 20, 21, 22, 23, 24, 
                                              17, 18, 19, 12, 13, 14, 15, 16, 
                                              9, 10, 11, 4, 5, 6, 7, 8, 3, 2, 
                                              1, 0), v8);
      __m512i min9 = _mm512_min_epu8(v8, perm9);
      __m512i max9 = _mm512_max_epu8(v8, perm9);
      __m512i v9 = _mm512_mask_mov_epi8(max9, 0x1232323230, min9);
      
      __m512i perm10 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 40, 
                                               39, 36, 7, 38, 35, 32, 3, 34, 
                                               31, 28, 15, 30, 27, 24, 11, 
                                               26, 23, 20, 21, 22, 19, 16, 
                                               17, 18, 29, 12, 13, 14, 25, 8, 
                                               9, 10, 37, 4, 5, 6, 33, 2, 1, 
                                               0), v9);
      __m512i min10 = _mm512_min_epu8(v9, perm10);
      __m512i max10 = _mm512_max_epu8(v9, perm10);
      __m512i v10 = _mm512_mask_mov_epi8(max10, 0x1111119998, min10);
      
      __m512i perm11 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 10, 
                                               39, 38, 23, 6, 35, 34, 19, 2, 
                                               31, 30, 29, 14, 27, 26, 25, 
                                               24, 37, 22, 7, 20, 33, 18, 3, 
                                               16, 15, 28, 13, 12, 11, 40, 9, 
                                               8, 21, 36, 5, 4, 17, 32, 1, 
                                               0), v10);
      __m512i min11 = _mm512_min_epu8(v10, perm11);
      __m512i max11 = _mm512_max_epu8(v10, perm11);
      __m512i v11 = _mm512_mask_mov_epi8(max11, 0x8844cc, min11);
      
      __m512i perm12 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 26, 
                                               39, 38, 31, 22, 35, 34, 27, 
                                               18, 37, 30, 23, 28, 33, 40, 
                                               19, 10, 29, 36, 15, 6, 25, 32, 
                                               11, 2, 21, 14, 7, 12, 17, 24, 
                                               3, 8, 13, 20, 5, 4, 9, 16, 1, 
                                               0), v11);
      __m512i min12 = _mm512_min_epu8(v11, perm12);
      __m512i max12 = _mm512_max_epu8(v11, perm12);
      __m512i v12 = _mm512_mask_mov_epi8(max12, 0x8ccc8ccc, min12);
      
      __m512i perm13 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 34, 
                                               39, 38, 35, 30, 37, 40, 31, 
                                               26, 33, 36, 27, 22, 29, 32, 
                                               23, 18, 25, 28, 19, 14, 21, 
                                               24, 15, 10, 17, 20, 11, 6, 13, 
                                               16, 7, 2, 9, 12, 3, 4, 5, 8, 
                                               1, 0), v12);
      __m512i min13 = _mm512_min_epu8(v12, perm13);
      __m512i max13 = _mm512_max_epu8(v12, perm13);
      __m512i v13 = _mm512_mask_mov_epi8(max13, 0xccccccccc, min13);
      
      __m512i perm14 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 38, 
                                               39, 40, 37, 34, 35, 36, 33, 
                                               30, 31, 32, 29, 26, 27, 28, 
                                               25, 22, 23, 24, 21, 18, 19, 
                                               20, 17, 14, 15, 16, 13, 10, 
                                               11, 12, 9, 6, 7, 8, 5, 2, 3, 
                                               4, 1, 0), v13);
      __m512i min14 = _mm512_min_epu8(v13, perm14);
      __m512i max14 = _mm512_max_epu8(v13, perm14);
      __m512i v14 = _mm512_mask_mov_epi8(max14, 0x4444444444, min14);
      
      __m512i perm15 = _mm512_shuffle_epi8(v14, _mm512_set_epi8(63, 62, 61, 
                                           60, 59, 58, 57, 56, 55, 54, 53, 
                                           52, 51, 50, 49, 48, 47, 46, 45, 
                                           44, 43, 42, 41, 40, 38, 39, 36, 
                                           37, 34, 35, 32, 33, 30, 31, 28, 
                                           29, 26, 27, 24, 25, 22, 23, 20, 
                                           21, 18, 19, 16, 17, 14, 15, 12, 
                                           13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 
                                           3, 1, 0));
      __m512i min15 = _mm512_min_epu8(v14, perm15);
      __m512i max15 = _mm512_max_epu8(v14, perm15);
      __m512i v15 = _mm512_mask_mov_epi8(max15, 0x5555555554, min15);
      
      __m512i perm16 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 9, 39, 
                                               7, 37, 5, 35, 3, 33, 1, 31, 
                                               15, 29, 13, 27, 11, 25, 24, 
                                               23, 22, 21, 20, 19, 18, 17, 
                                               16, 30, 14, 28, 12, 26, 10, 
                                               40, 8, 38, 6, 36, 4, 34, 2, 
                                               32, 0), v15);
      __m512i min16 = _mm512_min_epu8(v15, perm16);
      __m512i max16 = _mm512_max_epu8(v15, perm16);
      __m512i v16 = _mm512_mask_mov_epi8(max16, 0xaaaa, min16);
      
      __m512i perm17 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 25, 
                                               39, 23, 37, 21, 35, 19, 33, 
                                               17, 31, 30, 29, 28, 27, 26, 
                                               40, 9, 38, 7, 36, 5, 34, 3, 
                                               32, 1, 15, 14, 13, 12, 11, 10, 
                                               24, 8, 22, 6, 20, 4, 18, 2, 
                                               16, 0), v16);
      __m512i min17 = _mm512_min_epu8(v16, perm17);
      __m512i max17 = _mm512_max_epu8(v16, perm17);
      __m512i v17 = _mm512_mask_mov_epi8(max17, 0x2aa02aa, min17);
      
      __m512i perm18 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 33, 
                                               39, 31, 37, 29, 35, 27, 40, 
                                               25, 38, 23, 36, 21, 34, 19, 
                                               32, 17, 30, 15, 28, 13, 26, 
                                               11, 24, 9, 22, 7, 20, 5, 18, 
                                               3, 16, 1, 14, 6, 12, 4, 10, 2, 
                                               8, 0), v17);
      __m512i min18 = _mm512_min_epu8(v17, perm18);
      __m512i max18 = _mm512_max_epu8(v17, perm18);
      __m512i v18 = _mm512_mask_mov_epi8(max18, 0x2aaaaaaaa, min18);
      
      __m512i perm19 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 37, 
                                               39, 35, 40, 33, 38, 31, 36, 
                                               29, 34, 27, 32, 25, 30, 23, 
                                               28, 21, 26, 19, 24, 17, 22, 
                                               15, 20, 13, 18, 11, 16, 9, 14, 
                                               7, 12, 5, 10, 3, 8, 1, 6, 2, 
                                               4, 0), v18);
      __m512i min19 = _mm512_min_epu8(v18, perm19);
      __m512i max19 = _mm512_max_epu8(v18, perm19);
      __m512i v19 = _mm512_mask_mov_epi8(max19, 0x2aaaaaaaaa, min19);
      
      __m512i perm20 = _mm512_permutexvar_epi8(_mm512_set_epi8(63, 62, 61, 
                                               60, 59, 58, 57, 56, 55, 54, 
                                               53, 52, 51, 50, 49, 48, 47, 
                                               46, 45, 44, 43, 42, 41, 39, 
                                               40, 37, 38, 35, 36, 33, 34, 
                                               31, 32, 29, 30, 27, 28, 25, 
                                               26, 23, 24, 21, 22, 19, 20, 
                                               17, 18, 15, 16, 13, 14, 11, 
                                               12, 9, 10, 7, 8, 5, 6, 3, 4, 
                                               1, 2, 0), v19);
      __m512i min20 = _mm512_min_epu8(v19, perm20);
      __m512i max20 = _mm512_max_epu8(v19, perm20);
      __m512i v20 = _mm512_mask_mov_epi8(max20, 0xaaaaaaaaaa, min20);
      
      return v20;
 }



/* Wrapper For SIMD Sort */
     void inline __attribute__((always_inline)) 

batcher_41_uint8_t(uint8_t * const arr) 
                                 {
      
      __m512i v = _mm512_load_si512((__m512i *)arr);
      
      v = batcher_41_uint8_t_vec(v);
      
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


