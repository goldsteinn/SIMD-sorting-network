#ifndef _INTEGER_RANGE_H_
#define _INTEGER_RANGE_H_

#include <stdint.h>
#include <stdio.h>
#include <array>
#include <utility>


// Taken from:
// https://stackoverflow.com/questions/20874388/error-spliting-an-stdindex-sequence
// Taken from:
// https://stackoverflow.com/questions/20874388/error-spliting-an-stdindex-sequence
template<typename T, typename seq, T begin>
struct make_integer_range_impl;

template<typename T, T... v, T begin>
struct make_integer_range_impl<T, std::integer_sequence<T, v...>, begin> {
    using type = std::integer_sequence<T, begin + v...>;
};


template<typename T, T begin, T end>
using make_integer_range =
    typename make_integer_range_impl<T,
                                     std::make_integer_sequence<T, end - begin>,
                                     begin>::type;

template<typename T, T... seq1, T... seq2>
constexpr decltype(auto)
slice_impl(std::integer_sequence<T, seq1...>,
           std::integer_sequence<T, seq2...>) {
    using seq_array = std::array<T, sizeof...(seq1)>;
    return std::integer_sequence<T,
                                 std::get<seq2>(seq_array{ { seq1... } })...>();
}


template<typename T, T begin, T end, T... seq>
constexpr decltype(auto)
slice(std::integer_sequence<T, seq...> _seq) {
    return slice_impl(_seq, make_integer_range<T, begin, end>());
}


template<typename T, T point, T... seq>
constexpr decltype(auto)
split_back(std::integer_sequence<T, seq...> _seq) {
    return slice<T, point, sizeof...(seq)>(_seq);
}

template<typename T, T point, T... seq>
constexpr decltype(auto)
split_front(std::integer_sequence<T, seq...> _seq) {
    return slice<T, 0, point>(_seq);
}


template<typename T, T... seq1, T... seq2>
constexpr decltype(auto)
merge(std::integer_sequence<T, seq1...> _seq1,
      std::integer_sequence<T, seq2...> _seq2) {
    return std::integer_sequence<T, seq1..., seq2...>{};
}

template<typename T, T new_v, T... seq>
constexpr decltype(auto)
add_back(std::integer_sequence<T, seq...> _seq) {
    return std::integer_sequence<T, seq..., new_v>{};
}

template<typename T, T new_v, T... seq>
constexpr decltype(auto)
add_front(std::integer_sequence<T, seq...> _seq) {
    return std::integer_sequence<T, new_v, seq...>{};
}

template<typename T, T pos, T new_v, T... seq>
constexpr decltype(auto)
add_pos(std::integer_sequence<T, seq...> _seq) {
    return merge<T>(add_back<T, new_v>(split_front<T, pos>(_seq)),
                    split_back<T, pos>(_seq));
}

template<typename T, T... seq>
constexpr decltype(auto)
remove_back(std::integer_sequence<T, seq...> _seq) {
    return split_front<T, sizeof...(seq) - 1>(_seq);
}

template<typename T, T... seq>
constexpr decltype(auto)
remove_front(std::integer_sequence<T, seq...> _seq) {
    return split_back<T, 1>(_seq);
}

template<typename T, T pos, T... seq>
constexpr decltype(auto)
remove_pos(std::integer_sequence<T, seq...> _seq) {
    return merge<T>(split_front<T, pos>(_seq), split_back<T, pos + 1>(_seq));
}


template<typename T, T pos, T new_v, T... seq>
constexpr decltype(auto)
replace_pos(std::integer_sequence<T, seq...> _seq) {
    return add_pos<T, pos, new_v>(remove_pos<T, pos>(_seq));
}

template<uint32_t cond, typename T, T pos, T new_v, T... seq>
constexpr decltype(auto)
replace_if(std::integer_sequence<T, seq...> _seq) {
    if constexpr (cond) {
        return replace_pos<T, pos, new_v>(_seq);
    }
    else {
        return _seq;
    }
}

template<typename T, T... seq>
constexpr T
get_back(std::integer_sequence<T, seq...> _seq) {
    constexpr T ram_arr[sizeof...(seq)] = { seq... };
    return ram_arr[sizeof...(seq) - 1];
}

template<typename T, T... seq>
constexpr T
get_front(std::integer_sequence<T, seq...> _seq) {
    constexpr T ram_arr[sizeof...(seq)] = { seq... };
    return ram_arr[0];
}

template<typename T, T pos, T... seq>
constexpr T
get_pos(std::integer_sequence<T, seq...> _seq) {
    constexpr T ram_arr[sizeof...(seq)] = { seq... };
    return ram_arr[pos];
}

template<typename T, T... seq>
constexpr uint64_t
get_size(std::integer_sequence<T, seq...> _seq) {
    return sizeof...(seq);
}

template<uint32_t... seq1, uint32_t... seq2>
constexpr bool
equal_sequences(std::integer_sequence<uint32_t, seq1...> _seq1,
                std::integer_sequence<uint32_t, seq2...> _seq2) {

    static_assert(std::is_same<decltype(_seq1), decltype(_seq2)>::value);
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


static_assert(std::is_same<make_integer_range<uint32_t, 0, 3>,
                           std::make_integer_sequence<uint32_t, 3>>::value);

static_assert(std::is_same<make_integer_range<uint32_t, 0, 3>,
                           std::make_integer_sequence<uint32_t, 3>>::value);

static_assert(std::is_same<std::integer_sequence<uint32_t, 0, 1, 2, 3>,
                           std::integer_sequence<uint32_t, 0, 1, 2, 3>>::value);

static constexpr auto t0 = std::integer_sequence<uint32_t, 0, 1, 2, 3>{};
static_assert(get_front<uint32_t>(t0) == 0);
static_assert(get_back<uint32_t>(t0) == 3);
static_assert(get_pos<uint32_t, 1>(t0) == 1);


static constexpr auto t1 = add_front<uint32_t, 4>(t0);
static_assert(get_front<uint32_t>(t1) == 4);
static_assert(get_back<uint32_t>(t1) == 3);
static_assert(get_pos<uint32_t, 1>(t1) == 0);

static constexpr auto t2 = add_pos<uint32_t, 1, 32>(t1);
static_assert(get_front<uint32_t>(t2) == 4);
static_assert(get_back<uint32_t>(t2) == 3);
static_assert(get_pos<uint32_t, 1>(t2) == 32);


static constexpr auto t3 = add_pos<uint32_t, 0, 44>(t2);
static_assert(get_front<uint32_t>(t3) == 44);
static_assert(get_back<uint32_t>(t3) == 3);
static_assert(get_pos<uint32_t, 1>(t3) == 4);

static constexpr auto t4 = remove_pos<uint32_t, 1>(t3);
static_assert(get_front<uint32_t>(t4) == 44);
static_assert(get_back<uint32_t>(t4) == 3);
static_assert(get_pos<uint32_t, 1>(t4) == 32);

static constexpr auto t5 = remove_front<uint32_t>(t4);
static_assert(get_front<uint32_t>(t5) == 32);
static_assert(get_back<uint32_t>(t5) == 3);
static_assert(get_pos<uint32_t, 1>(t5) == 0);

static constexpr auto t6 = remove_back<uint32_t>(t5);
static_assert(get_front<uint32_t>(t6) == 32);
static_assert(get_back<uint32_t>(t6) == 2);
static_assert(get_pos<uint32_t, 1>(t6) == 0);

static constexpr auto t7 = add_back<uint32_t, 100>(t6);
static_assert(get_front<uint32_t>(t7) == 32);
static_assert(get_back<uint32_t>(t7) == 100);
static_assert(get_pos<uint32_t, 1>(t7) == 0);

static constexpr auto t8 = replace_pos<uint32_t, 1, 100>(t7);
static_assert(get_front<uint32_t>(t8) == 32);
static_assert(get_back<uint32_t>(t8) == 100);
static_assert(get_pos<uint32_t, 1>(t8) == 100);


#endif
