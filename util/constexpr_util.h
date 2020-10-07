#ifndef _CONSTEXPR_UTIL_H_
#define _CONSTEXPR_UTIL_H_

#include <stdint.h>

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

#endif