#ifndef _TRANSFORMATIONS_H_
#define _TRANSFORMATIONS_H_

#include <util/constexpr_util.h>
#include <util/integer_range.h>
namespace vsort {
namespace transform {
namespace internal {


template<uint32_t n, uint32_t size, uint32_t... pairs>
struct permutation_transform_impl {
    static constexpr uint32_t rounded_n = next_p2(n);
    static constexpr uint32_t ngroups   = size / rounded_n;
    constexpr permutation_transform_impl() : arr() {
        constexpr uint32_t _pairs[size] = { static_cast<uint32_t>(pairs)... };

        static_assert(ngroups * rounded_n == size);
        for (uint32_t i = 0; i < ngroups; ++i) {
            for (uint32_t j = 0; j < rounded_n; ++j) {
                arr[i * rounded_n + j] = (rounded_n - 1) - j;
            }
        }

        uint64_t current_group = 0;
        uint32_t idx           = 0;
        for (uint32_t i = 0; i < ngroups; ++i) {
            while (idx < size) {
                if ((current_group & ((1UL) << _pairs[idx])) ||
                    (current_group & ((1UL) << _pairs[idx + 1]))) {
                    current_group = 0;
                    break;
                }
                current_group |= ((1UL) << _pairs[idx]);
                current_group |= ((1UL) << _pairs[idx + 1]);

                arr[i * rounded_n + ((rounded_n - 1) - _pairs[idx])] =
                    _pairs[idx + 1];
                arr[i * rounded_n + ((rounded_n - 1) - _pairs[idx + 1])] =
                    _pairs[idx];

                idx += 2;
            }
        }
    }
    uint32_t arr[size];
};


template<uint32_t n, uint32_t size, uint32_t... pairs>
struct group_transform_impl {

    constexpr group_transform_impl() : arr(), size_out() {
        constexpr uint32_t _pairs[size] = { static_cast<uint32_t>(pairs)... };

        uint32_t ngroups                    = 0;
        uint32_t group_indexes[next_p2(n)] = { 0 };
        uint32_t tmp_arr[next_p2(n) * (n + 1)]       = { 0 };

        for (uint32_t i = 0; i < size; i += 2) {
            const uint32_t x_i = _pairs[i];
            const uint32_t y_i = _pairs[i + 1];

            uint32_t idx = ngroups - 1;
            for (; idx != (-1); --idx) {
                uint32_t       j         = 0;
                const uint32_t group_idx = group_indexes[idx];
                for (; j < group_idx; j += 2) {
                    const uint32_t x_j = tmp_arr[n * idx + j];
                    const uint32_t y_j = tmp_arr[n * idx + j + 1];
                    if (x_i == x_j || x_i == y_j || y_i == x_j || y_i == y_j) {
                        break;
                    }
                }
                if (j != group_idx) {
                    break;
                }
            }
            if (++idx == ngroups) {
                ++ngroups;
            }
            const uint32_t ns_group_idx         = group_indexes[idx];
            tmp_arr[n * idx + ns_group_idx]     = x_i;
            tmp_arr[n * idx + ns_group_idx + 1] = y_i;
            group_indexes[idx] += 2;
        }
        uint32_t arr_idx = 0;
        for (uint32_t i = 0; i < ngroups; ++i) {
            for (uint32_t j = 0; j < group_indexes[i]; ++j) {
                arr[arr_idx] = tmp_arr[n * i + j];
                ++arr_idx;
            }
        }
        size_out = next_p2(n) * ngroups;
    }

    uint32_t size_out;
    uint32_t arr[size];
};

template<uint32_t n, uint32_t... pairs>
static constexpr uint32_t
get_network_size(std::integer_sequence<uint32_t, pairs...> _pairs) {
    group_transform_impl<n, get_size<uint32_t>(_pairs), pairs...> grouper =
        group_transform_impl<n, get_size<uint32_t>(_pairs), pairs...>();
    return grouper.size_out;
}

template<uint32_t n, uint32_t size, uint32_t... pairs>
struct unidirectional_transform_impl {

    constexpr unidirectional_transform_impl()
        : arr({ static_cast<uint32_t>(pairs)... }) {

        for (uint32_t i = 0; i < size; i += 2) {

            const uint32_t x_i = arr[i];
            const uint32_t y_i = arr[i + 1];
            if (x_i > y_i) {
                for (uint32_t j = i + 2; j < size; j += 2) {
                    const uint32_t x_j = arr[j];
                    const uint32_t y_j = arr[j + 1];

                    if (x_i == x_j) {
                        arr[j] = y_i;
                    }
                    if (x_i == y_j) {
                        arr[j + 1] = y_i;
                    }
                    if (y_i == x_j) {
                        arr[j] = x_i;
                    }
                    if (y_i == y_j) {
                        arr[j + 1] = x_i;
                    }
                }
                arr[i]     = y_i;
                arr[i + 1] = x_i;
            }
        }
    }
    uint32_t arr[size];
};


template<uint32_t n,
         uint32_t size,
         template<uint32_t _n, uint32_t _size, uint32_t... pairs>
         typename transform_impl>
struct transformer {

    template<uint32_t... pairs>
    static constexpr decltype(auto)
    return_transform() {
        return std::integer_sequence<uint32_t, pairs...>{};
    }

    template<uint32_t... pairs, uint32_t... seq>
    static constexpr decltype(auto)
    transform_kernel(std::integer_sequence<uint32_t, pairs...> _pairs,
                     std::integer_sequence<uint32_t, seq...>   _seq) {

        constexpr transform_impl<n, size, pairs...> _transform =
            transform_impl<n, size, pairs...>();
        return return_transform<_transform.arr[seq]...>();
    }


    template<uint32_t... pairs>
    static constexpr decltype(auto)
    transform(std::integer_sequence<uint32_t, pairs...> _pairs) {
        return transform_kernel<>(_pairs,
                                  std::make_integer_sequence<uint32_t, size>{});
    }
};


}  // namespace internal


template<uint32_t n, typename network>
struct group {
    using type =
        decltype(internal::transformer<
                 n,
                 get_size<uint32_t>(network{}),
                 internal::group_transform_impl>::transform(network{}));
};

template<uint32_t n,
         typename network,
         uint32_t size = get_size<uint32_t>(network{})>
struct permutation {
    using type = decltype(
        internal::transformer<n, size, internal::permutation_transform_impl>::
            transform(network{}));
};

template<uint32_t n, typename network>
struct unidirectional {
    using type = decltype(
        internal::transformer<
            n,
            get_size<uint32_t>(network{}),
            internal::unidirectional_transform_impl>::transform(network{}));
};

template<uint32_t n, typename network>
struct build {
    using type =
        typename permutation<n,
                             typename group<n, network>::type,
                             internal::get_network_size<n>(network{})>::type;
};

}  // namespace transform

}  // namespace vsort
#endif
