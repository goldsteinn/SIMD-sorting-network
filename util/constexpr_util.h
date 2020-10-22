#ifndef _CONSTEXPR_UTIL_H_
#define _CONSTEXPR_UTIL_H_

#include <stdint.h>

constexpr uint32_t
is_pow2(uint32_t v) {
    return !(v & (v - 1));
}


constexpr uint32_t
next_p2(uint32_t v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


constexpr uint32_t
ulog2(uint32_t v) {
    uint32_t r = 0, s = 0;
    r = (v > 0xffff) << 4;
    v >>= r;
    s = (v > 0xff) << 3;
    v >>= s;
    r |= s;
    s = (v > 0xf) << 2;
    v >>= s;
    r |= s;
    s = (v > 0x3) << 1;
    v >>= s;
    r |= s;
    return r | (v >> 1);
}

template<typename T>
constexpr T ALWAYS_INLINE CONST_ATTR
get_max() {
    if constexpr (std::is_signed<T>::value) {
        return (T)(((1UL) << (8 * sizeof(T) - 1)) - 1);
    }
    else {
        return (T)(((1UL) << (8 * sizeof(T))) - 1);
    }
}


#endif
