#ifndef _VEC_SORT_H_
#define _VEC_SORT_H_

#include <immintrin.h>
#include <stdint.h>
#include "vec_sort_primitives.h"


template<typename T, uint32_t n>
struct vsort;

template<typename T>
struct vsort<T, 32> {
    static constexpr uint32_t n = 32;
    using v_type                = vec_t<T, n>;

    static void
    sort(T * arr) {
        v_type v = vec_load<T, n>(arr);
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 14, 13, 12, 
                                        11, 10, 9 , 8 , 
                                        7 , 6 , 5 , 4 , 
                                        3 , 2 , 1 , 0 , 
                                        31, 30, 29, 28, 
                                        27, 26, 25, 24, 
                                        23, 22, 21, 20, 
                                        19, 18, 17, 16);
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xffff);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(23, 22, 21, 20, 
                                        19, 18, 17, 16, 
                                        31, 30, 29, 28, 
                                        27, 26, 25, 24, 
                                        7 , 6 , 5 , 4 , 
                                        3 , 2 , 1 , 0 , 
                                        15, 14, 13, 12, 
                                        11, 10, 9 , 8 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xff00ff);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(27, 26, 25, 24, 
                                        31, 30, 29, 28, 
                                        15, 14, 13, 12, 
                                        11, 10, 9 , 8 , 
                                        23, 22, 21, 20, 
                                        19, 18, 17, 16, 
                                        3 , 2 , 1 , 0 , 
                                        7 , 6 , 5 , 4 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xf00ff0f);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(29, 28, 31, 30, 
                                        27, 26, 25, 24, 
                                        19, 18, 17, 16, 
                                        23, 22, 21, 20, 
                                        11, 10, 9 , 8 , 
                                        15, 14, 13, 12, 
                                        7 , 6 , 5 , 4 , 
                                        1 , 0 , 3 , 2 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x300f0f03);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(30, 31, 29, 28, 
                                        15, 14, 13, 12, 
                                        23, 22, 21, 20, 
                                        7 , 6 , 5 , 4 , 
                                        27, 26, 25, 24, 
                                        11, 10, 9 , 8 , 
                                        19, 18, 17, 16, 
                                        3 , 2 , 0 , 1 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x4000f0f1);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 30, 29, 28, 
                                        23, 22, 21, 20, 
                                        27, 26, 25, 24, 
                                        15, 14, 13, 12, 
                                        19, 18, 17, 16, 
                                        7 , 6 , 5 , 4 , 
                                        11, 10, 9 , 8 , 
                                        3 , 2 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xf0f0f0);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 30, 29, 28, 
                                        25, 24, 27, 26, 
                                        21, 20, 23, 22, 
                                        17, 16, 19, 18, 
                                        13, 12, 15, 14, 
                                        9 , 8 , 11, 10, 
                                        5 , 4 , 7 , 6 , 
                                        3 , 2 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x3333330);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 30, 15, 14, 
                                        27, 26, 11, 10, 
                                        23, 22, 7 , 6 , 
                                        19, 18, 3 , 2 , 
                                        29, 28, 13, 12, 
                                        25, 24, 9 , 8 , 
                                        21, 20, 5 , 4 , 
                                        17, 16, 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xcccc);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 30, 23, 22, 
                                        27, 26, 19, 18, 
                                        29, 28, 15, 14, 
                                        25, 24, 11, 10, 
                                        21, 20, 7 , 6 , 
                                        17, 16, 3 , 2 , 
                                        13, 12, 5 , 4 , 
                                        9 , 8 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xcccccc);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 30, 27, 26, 
                                        29, 28, 23, 22, 
                                        25, 24, 19, 18, 
                                        21, 20, 15, 14, 
                                        17, 16, 11, 10, 
                                        13, 12, 7 , 6 , 
                                        9 , 8 , 3 , 2 , 
                                        5 , 4 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xccccccc);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 30, 28, 29, 
                                        26, 27, 24, 25, 
                                        22, 23, 20, 21, 
                                        18, 19, 16, 17, 
                                        14, 15, 12, 13, 
                                        10, 11, 8 , 9 , 
                                        6 , 7 , 4 , 5 , 
                                        2 , 3 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x15555554);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 15, 29, 13, 
                                        27, 11, 25, 9 , 
                                        23, 7 , 21, 5 , 
                                        19, 3 , 17, 1 , 
                                        30, 14, 28, 12, 
                                        26, 10, 24, 8 , 
                                        22, 6 , 20, 4 , 
                                        18, 2 , 16, 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xaaaa);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 23, 29, 21, 
                                        27, 19, 25, 17, 
                                        30, 15, 28, 13, 
                                        26, 11, 24, 9 , 
                                        22, 7 , 20, 5 , 
                                        18, 3 , 16, 1 , 
                                        14, 6 , 12, 4 , 
                                        10, 2 , 8 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xaaaaaa);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 27, 29, 25, 
                                        30, 23, 28, 21, 
                                        26, 19, 24, 17, 
                                        22, 15, 20, 13, 
                                        18, 11, 16, 9 , 
                                        14, 7 , 12, 5 , 
                                        10, 3 , 8 , 1 , 
                                        6 , 2 , 4 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xaaaaaaa);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(31, 29, 30, 27, 
                                        28, 25, 26, 23, 
                                        24, 21, 22, 19, 
                                        20, 17, 18, 15, 
                                        16, 13, 14, 11, 
                                        12, 9 , 10, 7 , 
                                        8 , 5 , 6 , 3 , 
                                        4 , 1 , 2 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x2aaaaaaa);
        }
        vec_store<T, n>(arr, v);
    }
};
template<typename T>
struct vsort<T, 16> {
    static constexpr uint32_t n = 16;
    using v_type                = vec_t<T, n>;

    static void
    sort(T * arr) {
        v_type v = vec_load<T, n>(arr);
        {
            // clang-format off
            v_type perm = vec_set<T, n>(7, 6 , 5 , 4 , 
                                        3 , 2 , 1 , 0 , 
                                        15, 14, 13, 12, 
                                        11, 10, 9 , 8 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xff);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(11, 10, 9 , 8 , 
                                        15, 14, 13, 12, 
                                        3 , 2 , 1 , 0 , 
                                        7 , 6 , 5 , 4 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xf0f);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(13, 12, 15, 14, 
                                        7 , 6 , 5 , 4 , 
                                        11, 10, 9 , 8 , 
                                        1 , 0 , 3 , 2 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x30f3);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(14, 15, 13, 12, 
                                        9 , 8 , 11, 10, 
                                        5 , 4 , 7 , 6 , 
                                        3 , 2 , 0 , 1 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x4331);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 14, 7 , 6 , 
                                        11, 10, 3 , 2 , 
                                        13, 12, 5 , 4 , 
                                        9 , 8 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xcc);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 14, 11, 10, 
                                        13, 12, 7 , 6 , 
                                        9 , 8 , 3 , 2 , 
                                        5 , 4 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xccc);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 14, 12, 13, 
                                        10, 11, 8 , 9 , 
                                        6 , 7 , 4 , 5 , 
                                        2 , 3 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x1554);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 7 , 13, 5 , 
                                        11, 3 , 9 , 1 , 
                                        14, 6 , 12, 4 , 
                                        10, 2 , 8 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xaa);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 11, 13, 9 , 
                                        14, 7 , 12, 5 , 
                                        10, 3 , 8 , 1 , 
                                        6 , 2 , 4 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xaaa);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(15, 13, 14, 11, 
                                        12, 9 , 10, 7 , 
                                        8 , 5 , 6 , 3 , 
                                        4 , 1 , 2 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x2aaa);
        }
        vec_store<T, n>(arr, v);
    }
};
template<typename T>
struct vsort<T, 8> {
    static constexpr uint32_t n = 8;
    using v_type                = vec_t<T, n>;

    static void
    sort(T * arr) {
        v_type v = vec_load<T, n>(arr);
        {
            // clang-format off
            v_type perm = vec_set<T, n>(3, 2 , 1 , 0 , 
                                        7 , 6 , 5 , 4 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xf);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(5, 4 , 7 , 6 , 
                                        1 , 0 , 3 , 2 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x33);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(6, 7 , 3 , 2 , 
                                        5 , 4 , 0 , 1 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x4d);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(7, 6 , 4 , 5 , 
                                        2 , 3 , 1 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x14);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(7, 3 , 5 , 1 , 
                                        6 , 2 , 4 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0xa);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(7, 5 , 6 , 3 , 
                                        4 , 1 , 2 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x2a);
        }
        vec_store<T, n>(arr, v);
    }
};
template<typename T>
struct vsort<T, 4> {
    static constexpr uint32_t n = 4;
    using v_type                = vec_t<T, n>;

    static void
    sort(T * arr) {
        v_type v = vec_load<T, n>(arr);
        {
            // clang-format off
            v_type perm = vec_set<T, n>(1, 0 , 3 , 2 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x3);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(2, 3 , 0 , 1 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x5);
        }
        {
            // clang-format off
            v_type perm = vec_set<T, n>(3, 1 , 2 , 0 );
            // clang-format on
            v_type cmp   = vec_perm<T, n>(perm, v);
            v_type s_min = vec_min<T, n>(v, cmp);
            v_type s_max = vec_max<T, n>(v, cmp);
            v            = vec_blend<T, n>(s_max, s_min, 0x2);
        }
        vec_store<T, n>(arr, v);
    }
};


#endif
