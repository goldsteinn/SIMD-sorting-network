#ifndef _TESTING_H_
#define _TESTING_H_


#include <stdio.h>
#include <type_traits>
#include <utility>


template<uint32_t... seq1, uint32_t... seq2>
constexpr bool
equal_sequences(std::integer_sequence<uint32_t, seq1...>,
                std::integer_sequence<uint32_t, seq2...>) {

    static_assert(std::is_same<decltype(seq1), decltype(seq2)>::value);
    return true;
}

template<uint32_t... seq>
void
show(std::integer_sequence<uint32_t, seq...> _seq) {
    uint32_t arr[sizeof...(seq)] = { seq... };
    fprintf(stderr, "seq(%zu): [%d", get_size<uint32_t>(_seq), arr[0]);
    for (uint32_t i = 1; i < sizeof...(seq); ++i) {
        fprintf(stderr, ", %d", arr[i]);
    }
    fprintf(stderr, "]\n");
}
