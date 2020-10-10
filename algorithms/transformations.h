#ifndef _TRANSFORMATIONS_H_
#define _TRANSFORMATIONS_H_

#include <util/integer_range.h>

namespace transform {
namespace internal {
namespace impl {

template<uint32_t n, uint32_t... pairs>
struct permutation_transform_impl {
    static constexpr uint32_t size    = sizeof...(pairs);
    static constexpr uint32_t ngroups = size / n;
    constexpr permutation_transform_impl() : arr() {
        constexpr uint32_t _pairs[size] = { static_cast<uint32_t>(pairs)... };

        for (uint32_t i = 0; i < ngroups; ++i) {
            for (uint32_t j = 0; j < n; j += 2) {
                uint32_t idx = i * n + j;

                arr[idx + ((n - 1) - _pairs[idx])]     = _pairs[idx + 1];
                arr[idx + ((n - 1) - _pairs[idx + 1])] = _pairs[idx];
            }
        }
    }
    uint32_t arr[size];
};

template<uint32_t n, uint32_t... pairs>
struct group_transform_impl {
    static constexpr uint32_t size = sizeof...(pairs);

    constexpr group_transform_impl() : arr() {
        constexpr uint32_t _pairs[size] = { static_cast<uint32_t>(pairs)... };

        uint32_t ngroups                             = 0;
        uint32_t group_indexes[(size + (n - 1)) / n] = { 0 };

        for (uint32_t i = 0; i < size; i += 2) {
            const uint32_t x_i = _pairs[i];
            const uint32_t y_i = _pairs[i + 1];

            uint32_t idx = ngroups - 1;
            for (; idx != (-1); --idx) {
                uint32_t       j         = 0;
                const uint32_t group_idx = group_indexes[idx];
                for (; j < group_idx; j += 2) {
                    const uint32_t x_j = arr[n * idx + j];
                    const uint32_t y_j = arr[n * idx + j + 1];
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
            const uint32_t ns_group_idx     = group_indexes[idx];
            arr[n * idx + ns_group_idx]     = x_i;
            arr[n * idx + ns_group_idx + 1] = y_i;
            group_indexes[idx] += 2;
        }
    }

    uint32_t arr[size];
};

template<uint32_t n, uint32_t... pairs>
struct unidirectional_transform_impl {
    static constexpr uint32_t size = sizeof...(pairs);

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

}  // namespace impl


template<uint32_t n,
         template<uint32_t _n, uint32_t... pairs>
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

        constexpr transform_impl<n, pairs...> _transform =
            transform_impl<n, pairs...>();
        return return_transform<_transform.arr[seq]...>();
    }


    template<uint32_t... pairs>
    static constexpr decltype(auto)
    transform(std::integer_sequence<uint32_t, pairs...> _pairs) {
        return transform_kernel<>(
            _pairs,
            std::make_integer_sequence<uint32_t, get_size<uint32_t>(_pairs)>{});
    }
};


}  // namespace internal

template<uint32_t n, typename network>
struct group {
    using type = decltype(
        internal::transformer<n, internal::impl::group_transform_impl>::
            transform(network{}));
};

template<uint32_t n, typename network>
struct permutation {
    using type = decltype(
        internal::transformer<n, internal::impl::permutation_transform_impl>::
            transform(network{}));
};

template<uint32_t n, typename network>
struct unidirectional {
    using type = decltype(
        internal::transformer<n,
                              internal::impl::unidirectional_transform_impl>::
            transform(network{}));
};

}  // namespace transform
#endif
