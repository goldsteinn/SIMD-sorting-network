#ifndef _INSTRUCTION_TYPES_H_
#define _INSTRUCTION_TYPES_H_

struct __m128_wrapper {
    typedef __m128i type;
} ALIGN_ATTR(sizeof(__m128i));

struct __m256_wrapper {
    typedef __m256i type;
} ALIGN_ATTR(sizeof(__m256i));

struct __m512_wrapper {
    typedef __m512i type;
} ALIGN_ATTR(sizeof(__m512i));

struct uint32_t_wrapper {
    typedef uint32_t type;
} ALIGN_ATTR(sizeof(uint32_t));


template<typename T, uint32_t n>
using get_vec_t =
    typename std::conditional_t<n * sizeof(T) <= 32,
                                typename std::conditional_t<n * sizeof(T) <= 16,
                                                            __m128_wrapper,
                                                            __m256_wrapper>,
                                __m512_wrapper>;

template<typename T, uint32_t n>
using get_blend_t = typename std::conditional_t<sizeof(T) == sizeof(uint8_t),
                                                get_vec_t<T, n>,
                                                uint32_t_wrapper>;


template<typename T, uint32_t n>
using vec_t = typename get_vec_t<T, n>::type;

template<typename T, uint32_t n>
using blend_t = typename get_blend_t<T, n>::type;

template<typename T, uint32_t n>
using shuffle_t = typename get_blend_t<T, n>::type;


#endif
