template<uint32_t p1,
         uint32_t p2,
         uint32_t cur,
         uint32_t end,
         uint32_t... pairs>
constexpr decltype(auto)
filter(std::integer_sequence<uint32_t, pairs...> _pairs) {
    if constexpr (cur < end) {
        // p2 == get_pos<uint32_t, cur + 1>(_pairs)
        //  uint32_t, cur + 1, p1
        // p2 == get_pos<uint32_t, cur>(_pairs)
        //  uint32_t, cur, p1
        // p1 == get_pos<uint32_t, cur + 1>(_pairs)
        //  uint32_t, cur + 1, p2
        // p1 == get_pos<uint32_t, cur>(_pairs)
        //  uint32_t, cur, p2
        return filter<p1, p2, cur + 2, end>(replace_if<
                                            p2 == get_pos<uint32_t, cur + 1>(
                                                      _pairs),
                                            uint32_t,
                                            cur + 1,
                                            p1>(
            replace_if<p2 == get_pos<uint32_t, cur>(_pairs), uint32_t, cur, p1>(
                replace_if<p1 == get_pos<uint32_t, cur + 1>(_pairs),
                           uint32_t,
                           cur + 1,
                           p2>(replace_if<p1 == get_pos<uint32_t, cur>(_pairs),
                                          uint32_t,
                                          cur,
                                          p2>(_pairs)))));
    }
    else {
        return _pairs;
    }
}

template<uint32_t cur, uint32_t end, uint32_t... pairs>
constexpr decltype(auto)
do_filter(std::integer_sequence<uint32_t, pairs...> _pairs) {
    constexpr uint32_t p1 = get_pos<uint32_t, cur>(_pairs);
    constexpr uint32_t p2 = get_pos<uint32_t, cur + 1>(_pairs);
    if constexpr (p1 > p2) {
        return replace_pos<uint32_t, cur + 1, p1>(
            replace_pos<uint32_t, cur, p2>(
                filter<p1, p2, cur + 2, end>(_pairs)));
    }
    else {
        return _pairs;
    }
}

template<uint32_t cur, uint32_t end, uint32_t... pairs>
constexpr decltype(auto)
bitonic_filter(std::integer_sequence<uint32_t, pairs...> _pairs) {
    if constexpr (cur < end) {
        return bitonic_filter<cur + 2, end>(do_filter<cur, end>(_pairs));
    }
    else {
        return _pairs;
    }
}
