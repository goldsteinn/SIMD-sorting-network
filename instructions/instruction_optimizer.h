#ifndef _INSTRUCTION_OPTIMIZER_H_
#define _INSTRUCTION_OPTIMIZER_H_

#include <immintrin.h>
#include <stdint.h>

#include <instructions/instruction_sets.h>
#include <instructions/instruction_types.h>


static constexpr uint32_t lane_size = sizeof(__m128i);
//////////////////////////////////////////////////////////////////////
// Shuffle optimization is to reduce code side.

template<typename T, uint32_t n, uint32_t offset, T... e>
constexpr uint64_t
_get_shuffle_mask_2() {


    constexpr uint32_t ele_per_lane = 4;
    constexpr uint32_t perms[n]     = { static_cast<uint32_t>(e)... };
    uint32_t           mask         = 0;
    for (uint32_t i = 0; i < n; i += lane_size) {
        uint32_t new_mask = 0;
        for (uint32_t _i = 0; _i < ele_per_lane; ++_i) {

            uint32_t p  = perms[_i + i + offset];
            uint32_t lb = n - (i + ele_per_lane + offset);
            uint32_t hb = n - (i + offset);
            if (p >= lb && p < hb) {
                uint32_t idx  = p - lb;
                uint32_t slot = (ele_per_lane - (_i + 1)) * ulog2(ele_per_lane);
                new_mask |= idx << slot;
            }
            else {
                return 0;
            }
        }
        if (new_mask == 0) {
            return 0;
        }
        if (i != 0 && new_mask != mask) {
            return 0;
        }
        mask = new_mask;
    }
    return mask;
}

template<typename T, uint32_t n, T... e>
constexpr uint64_t
get_shuffle_mask_2() {
    uint64_t mask_lo = _get_shuffle_mask_2<T, n, 0, e...>();
    if (mask_lo == 0) {
        return 0;
    }
    uint64_t mask_hi = _get_shuffle_mask_2<T, n, 4, e...>();
    if (mask_hi == 0) {
        return 0;
    }
    return mask_lo | (mask_hi << 32);
}

template<typename T, uint32_t n, T... e>
constexpr uint64_t
get_shuffle_mask_4() {
    constexpr uint32_t ele_per_lane = 4;
    constexpr uint32_t perms[n]     = { static_cast<uint32_t>(e)... };
    uint32_t           mask         = 0;
    for (uint32_t i = 0; i < n; i += ele_per_lane) {
        uint32_t new_mask = 0;
        for (uint32_t _i = 0; _i < ele_per_lane; ++_i) {

            uint32_t p  = perms[_i + i];
            uint32_t lb = n - (i + ele_per_lane);
            uint32_t hb = n - i;
            if (p >= lb && p < hb) {
                uint32_t idx  = p - lb;
                uint32_t slot = (ele_per_lane - (_i + 1)) * ulog2(ele_per_lane);
                new_mask |= idx << slot;
            }
            else {
                return 0;
            }
        }
        if (new_mask == 0) {
            return 0;
        }
        if (i != 0 && new_mask != mask) {
            return 0;
        }
        mask = new_mask;
    }
    return mask;
}


template<typename T, uint32_t n, T... e>
constexpr uint64_t
get_shuffle_mask_8() {
    constexpr uint32_t ele_per_lane = 2;
    constexpr uint32_t perms[n]     = { static_cast<uint32_t>(e)... };
    uint32_t           mask         = 0;
    for (uint32_t i = 0; i < n; i += ele_per_lane) {
        uint32_t new_mask = 0;
        for (uint32_t _i = 0; _i < ele_per_lane; _i++) {

            uint32_t p  = perms[_i + i];
            uint32_t lb = n - (i + ele_per_lane);
            uint32_t hb = n - i;
            if (p >= lb && p < hb) {
                uint32_t idx  = p - lb;
                uint32_t slot = (ele_per_lane - (_i + 1)) * ulog2(ele_per_lane);
                new_mask |= idx << slot;
            }
            else {
                return 0;
            }
        }
        if (new_mask == 0) {
            return 0;
        }
        if (i != 0 && new_mask != mask) {
            return 0;
        }
        mask = new_mask;
    }
    // this is the only mask we could ever use (reverse uint64_t's in the lane)
    return 0x4e;
}


template<typename T, uint32_t n, T... e>
constexpr uint64_t
get_shuffle_mask() {
    if constexpr (sizeof(T) == sizeof(uint16_t)) {
        return 0;  // get_shuffle_mask_2<T, n, e...>();
    }
    else if constexpr (sizeof(T) == sizeof(uint32_t)) {
        return get_shuffle_mask_4<T, n, e...>();
    }
    else if constexpr (sizeof(T) == sizeof(uint64_t)) {
        return get_shuffle_mask_8<T, n, e...>();
    }
    else {
        return 0;
    }
}


template<typename T, uint32_t n, T... e>
constexpr uint64_t
get_blend_mask() {
    constexpr uint32_t perm[n] = { static_cast<uint32_t>(e)... };

    uint64_t blend_mask = 0;
    for (uint32_t i = 0; i < n; ++i) {
        if (perm[i] < (n - (i + 1))) {
            blend_mask |= ((1UL) << i);
        }
    }
    return blend_mask;
}

template<uint64_t mask, uint64_t i>
constexpr uint8_t
vec_v() {
    return !!(mask & (1UL << i));
}

template<typename T, uint32_t n, uint64_t mask>
constexpr vec_t<T, n>
get_blend_vec() {
    if constexpr (n == sizeof(__m128i)) {
        return _mm_set_epi8(vec_v<mask, 0>(),
                            vec_v<mask, 1>(),
                            vec_v<mask, 2>(),
                            vec_v<mask, 3>(),
                            vec_v<mask, 4>(),
                            vec_v<mask, 5>(),
                            vec_v<mask, 6>(),
                            vec_v<mask, 7>(),
                            vec_v<mask, 8>(),
                            vec_v<mask, 9>(),
                            vec_v<mask, 10>(),
                            vec_v<mask, 11>(),
                            vec_v<mask, 12>(),
                            vec_v<mask, 13>(),
                            vec_v<mask, 14>(),
                            vec_v<mask, 15>());
    }
    if constexpr (n == sizeof(__m256i)) {
        return _mm256_set_epi8(vec_v<mask, 0>(),
                               vec_v<mask, 1>(),
                               vec_v<mask, 2>(),
                               vec_v<mask, 3>(),
                               vec_v<mask, 4>(),
                               vec_v<mask, 5>(),
                               vec_v<mask, 6>(),
                               vec_v<mask, 7>(),
                               vec_v<mask, 8>(),
                               vec_v<mask, 9>(),
                               vec_v<mask, 10>(),
                               vec_v<mask, 11>(),
                               vec_v<mask, 12>(),
                               vec_v<mask, 13>(),
                               vec_v<mask, 14>(),
                               vec_v<mask, 15>(),
                               vec_v<mask, 16>(),
                               vec_v<mask, 17>(),
                               vec_v<mask, 18>(),
                               vec_v<mask, 19>(),
                               vec_v<mask, 20>(),
                               vec_v<mask, 21>(),
                               vec_v<mask, 22>(),
                               vec_v<mask, 23>(),
                               vec_v<mask, 24>(),
                               vec_v<mask, 25>(),
                               vec_v<mask, 26>(),
                               vec_v<mask, 27>(),
                               vec_v<mask, 28>(),
                               vec_v<mask, 29>(),
                               vec_v<mask, 30>(),
                               vec_v<mask, 31>());
    }
    else {
        return _mm256_set_epi8(vec_v<mask, 0>(),
                               vec_v<mask, 1>(),
                               vec_v<mask, 2>(),
                               vec_v<mask, 3>(),
                               vec_v<mask, 4>(),
                               vec_v<mask, 5>(),
                               vec_v<mask, 6>(),
                               vec_v<mask, 7>(),
                               vec_v<mask, 8>(),
                               vec_v<mask, 9>(),
                               vec_v<mask, 10>(),
                               vec_v<mask, 11>(),
                               vec_v<mask, 12>(),
                               vec_v<mask, 13>(),
                               vec_v<mask, 14>(),
                               vec_v<mask, 15>(),
                               vec_v<mask, 16>(),
                               vec_v<mask, 17>(),
                               vec_v<mask, 18>(),
                               vec_v<mask, 19>(),
                               vec_v<mask, 20>(),
                               vec_v<mask, 21>(),
                               vec_v<mask, 22>(),
                               vec_v<mask, 23>(),
                               vec_v<mask, 24>(),
                               vec_v<mask, 25>(),
                               vec_v<mask, 26>(),
                               vec_v<mask, 27>(),
                               vec_v<mask, 28>(),
                               vec_v<mask, 29>(),
                               vec_v<mask, 30>(),
                               vec_v<mask, 31>(),
                               vec_v<mask, 32>(),
                               vec_v<mask, 33>(),
                               vec_v<mask, 34>(),
                               vec_v<mask, 35>(),
                               vec_v<mask, 36>(),
                               vec_v<mask, 37>(),
                               vec_v<mask, 38>(),
                               vec_v<mask, 39>(),
                               vec_v<mask, 40>(),
                               vec_v<mask, 41>(),
                               vec_v<mask, 42>(),
                               vec_v<mask, 43>(),
                               vec_v<mask, 44>(),
                               vec_v<mask, 45>(),
                               vec_v<mask, 46>(),
                               vec_v<mask, 47>(),
                               vec_v<mask, 48>(),
                               vec_v<mask, 49>(),
                               vec_v<mask, 50>(),
                               vec_v<mask, 51>(),
                               vec_v<mask, 52>(),
                               vec_v<mask, 53>(),
                               vec_v<mask, 54>(),
                               vec_v<mask, 55>(),
                               vec_v<mask, 56>(),
                               vec_v<mask, 57>(),
                               vec_v<mask, 58>(),
                               vec_v<mask, 59>(),
                               vec_v<mask, 60>(),
                               vec_v<mask, 61>(),
                               vec_v<mask, 62>(),
                               vec_v<mask, 63>());
    }
}


#endif
