template<uint32_t idx, uint32_t group_size>
constexpr decltype(auto)
make_group_kernel() {
    if constexpr (idx < group_size) {
        return merge<uint32_t>(make_group_kernel<idx + 1, group_size>(),
                               std::integer_sequence<uint32_t, group_size>{});
    }
    else {
        return std::make_integer_sequence<uint32_t, 0>{};
    }
}

template<uint32_t group_size>
constexpr decltype(auto)
make_group() {
    return make_group_kernel<0, group_size>();
}


template<uint32_t idx,
         uint32_t group_size,
         uint32_t p1,
         uint32_t p2,
         uint32_t group_start,
         uint32_t group_end,
         uint32_t... grouped_pairs>
constexpr uint32_t
check_group(std::integer_sequence<uint32_t, grouped_pairs...> _grouped_pairs) {
    if constexpr (group_start + idx >= group_end) {
        return 0;
    }
    else if constexpr (get_pos<uint32_t, group_start + idx>(_grouped_pairs) ==
                       p1) {
        return 1;
    }
    else if constexpr (get_pos<uint32_t, group_start + idx + 1>(
                           _grouped_pairs) == p1) {
        return 1;
    }
    else if constexpr (get_pos<uint32_t, group_start + idx>(_grouped_pairs) ==
                       p2) {
        return 1;
    }
    else if constexpr (get_pos<uint32_t, group_start + idx + 1>(
                           _grouped_pairs) == p2) {
        return 1;
    }
    else {
        return check_group<idx + 2, group_size, p1, p2, group_start, group_end>(
            _grouped_pairs);
    }
}

template<uint32_t idx,
         uint32_t group_size,
         uint32_t p1,
         uint32_t p2,
         uint32_t group_start,
         uint32_t group_end,
         uint32_t... grouped_pairs>
constexpr decltype(auto)
push_to_group(
    std::integer_sequence<uint32_t, grouped_pairs...> _grouped_pairs) {

    if constexpr (group_start == get_size<uint32_t>(_grouped_pairs)) {
        return merge<uint32_t>(
            _grouped_pairs,
            replace_pos<uint32_t, 0, p1>(
                replace_pos<uint32_t, 1, p2>(make_group<group_size>())));
    }

    else if constexpr (group_start + idx == group_end) {
    }

    else if constexpr (get_pos<uint32_t, group_start + idx>(_grouped_pairs) ==
                       group_size) {
        return replace_pos<uint32_t, group_start + idx + 1, p2>(
            replace_pos<uint32_t, group_start + idx, p1>(_grouped_pairs));
    }
    else {

        return push_to_group<idx + 2,
                             group_size,
                             p1,
                             p2,
                             group_start,
                             group_end>(_grouped_pairs);
    }
}

template<uint32_t group_idx,
         uint32_t group_size,
         uint32_t p1,
         uint32_t p2,
         uint32_t... grouped_pairs>
constexpr decltype(auto)
group_pair(std::integer_sequence<uint32_t, grouped_pairs...> _grouped_pairs) {

    constexpr uint32_t ngroups =
        get_size<uint32_t>(_grouped_pairs) / group_size;

    if constexpr (group_idx == (-1)) {

        return push_to_group<0, group_size, p1, p2, 0, group_size>(
            _grouped_pairs);
    }

    else if constexpr (check_group<0,
                                   group_size,
                                   p1,
                                   p2,
                                   group_size *(group_idx),
                                   group_size *(group_idx + 1)>(
                           _grouped_pairs)) {
        return push_to_group<0,
                             group_size,
                             p1,
                             p2,
                             group_size *(group_idx + 1),
                             group_size *(group_idx + 2)>(_grouped_pairs);
    }
    else {
        return group_pair<group_idx - 1, group_size, p1, p2>(_grouped_pairs);
    }
}

template<uint32_t cur,
         uint32_t group_size,
         uint32_t end,
         uint32_t... grouped_pairs,
         uint32_t... pairs>
constexpr decltype(auto)
group_pairs_kernel(
    std::integer_sequence<uint32_t, grouped_pairs...> _grouped_pairs,
    std::integer_sequence<uint32_t, pairs...>         _pairs) {
    if constexpr (cur < get_size<uint32_t>(_pairs)) {
        return group_pairs_kernel<cur + 2, group_size, end>(
            group_pair<(uint32_t)(
                           get_size<uint32_t>(_grouped_pairs) / group_size - 1),
                       group_size,
                       get_pos<uint32_t, cur>(_pairs),
                       get_pos<uint32_t, cur + 1>(_pairs)>(_grouped_pairs),
            _pairs);
    }
    else {
        return _grouped_pairs;
    }
}

template<uint32_t group_size, uint32_t... pairs>
constexpr decltype(auto)
group_pairs(std::integer_sequence<uint32_t, pairs...> _pairs) {

    return group_pairs_kernel<0, group_size, get_size<uint32_t>(_pairs)>(
        std::make_integer_sequence<uint32_t, 0>{},
        _pairs);
}
