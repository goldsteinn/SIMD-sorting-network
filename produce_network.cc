#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>
#include <vector>

#include "util/constexpr_util.h"

uint32_t verbose = 0;

struct idx_pair {
    uint32_t x;
    uint32_t y;

    idx_pair(uint32_t _x, uint32_t _y) : x(_x), y(_y) {}

    char *
    to_string(char * buf) {
        sprintf(buf, "[%d,%d]", x, y);
        return buf;
    }

    void
    set(idx_pair & other) {
        x = other.x;
        y = other.y;
    }
};

bool
operator==(idx_pair & lhs, idx_pair & rhs) {
    return (lhs.x == rhs.x) && (lhs.y == rhs.y);
}


void
bn_merge(std::vector<idx_pair> & pairs,
         uint32_t                i,
         uint32_t                length_i,
         uint32_t                j,
         uint32_t                length_j) {
    if (length_i == 1 && length_j == 1) {
        pairs.push_back(idx_pair(i, j));
    }
    else if (length_i == 1 && length_j == 2) {
        pairs.push_back(idx_pair(i, j + 1));
        pairs.push_back(idx_pair(i, j));
    }
    else if (length_i == 2 && length_j == 1) {
        pairs.push_back(idx_pair(i, j));
        pairs.push_back(idx_pair(i + 1, j));
    }
    else {
        uint32_t i_mid = length_i / 2;
        uint32_t j_mid = (length_j + (!(length_i & 0x1))) / 2;
        bn_merge(pairs, i, i_mid, j, j_mid);
        bn_merge(pairs,
                 i + i_mid,
                 length_i - i_mid,
                 j + j_mid,
                 length_j - j_mid);
        bn_merge(pairs, i + i_mid, length_i - i_mid, j, j_mid);
    }
}

void
bn_split(std::vector<idx_pair> & pairs, uint32_t i, uint32_t length) {
    if (length >= 2) {
        uint32_t mid = length >> 1;
        bn_split(pairs, i, mid);
        bn_split(pairs, i + mid, length - mid);
        bn_merge(pairs, i, mid, i + mid, length - mid);
    }
}


void
bitonic_merge(std::vector<idx_pair> & pairs,
              uint32_t                lo,
              uint32_t                n,
              uint32_t                dir) {
    if (n > 1) {
        uint32_t m = next_p2(n) >> 1;
        for (uint32_t i = lo; i < (lo + n - (m)); ++i) {
            pairs.push_back(dir ? idx_pair(i, i + m) : idx_pair(i + m, i));
        }
        bitonic_merge(pairs, lo, m, dir);
        bitonic_merge(pairs, lo + m, n - m, dir);
    }
}

void
bitonic_sort(std::vector<idx_pair> & pairs,
             uint32_t                lo,
             uint32_t                n,
             uint32_t                dir) {
    if (n > 1) {
        uint32_t m = n >> 1;
        bitonic_sort(pairs, lo, m, !dir);
        bitonic_sort(pairs, lo + m, n - m, dir);
        bitonic_merge(pairs, lo, n, dir);
    }
}

void
filter_pairs(std::vector<idx_pair> & pairs) {
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        uint32_t x = pairs[i].x;
        uint32_t y = pairs[i].y;

        if (x > y) {
            for (uint32_t j = i + 1; j < pairs.size();) {
                uint32_t j_x = pairs[j].x;
                uint32_t j_y = pairs[j].y;

                if (x == j_x) {
                    pairs[j].x = y;
                }
                if (x == j_y) {
                    pairs[j].y = y;
                }

                if (y == j_x) {
                    pairs[j].x = x;
                }
                if (y == j_y) {
                    pairs[j].y = x;
                }
                ++j;
            }

            pairs[i].y = x;
            pairs[i].x = y;
        }
    }
}

void
group(std::vector<idx_pair> & pairs) {
    std::vector<std::vector<idx_pair>> grouped_pairs;
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        uint32_t x = pairs[i].x;
        uint32_t y = pairs[i].y;

        int32_t idx;
        for (idx = grouped_pairs.size() - 1; idx >= 0; --idx) {
            uint32_t j;
            for (j = 0; j < grouped_pairs[idx].size(); ++j) {
                if (grouped_pairs[idx][j].x == x ||
                    grouped_pairs[idx][j].y == y ||
                    grouped_pairs[idx][j].x == y ||
                    grouped_pairs[idx][j].y == x) {
                    break;
                }
            }
            if (j != grouped_pairs[idx].size()) {
                break;
            }
        }
        if (++idx == grouped_pairs.size()) {
            grouped_pairs.push_back(std::vector<idx_pair>());
        }

        grouped_pairs[idx].push_back(idx_pair(x, y));
    }

    uint32_t pidx = 0;
    for (uint32_t i = 0; i < grouped_pairs.size(); ++i) {
        for (uint32_t j = 0; j < grouped_pairs[i].size(); ++j) {
            pairs[pidx].set(grouped_pairs[i][j]);
            ++pidx;
        }
    }
}

void
vec_to_string(const char * const header, std::vector<idx_pair> & v) {
    char out_buf[16] = "";

    v[0].to_string(out_buf);
    fprintf(stderr, "%s -> [%s", header, out_buf);
    for (uint32_t i = 1; i < v.size(); ++i) {
        v[i].to_string(out_buf);
        fprintf(stderr, ", %s", out_buf);
    }
    fprintf(stderr, "]\n");
}

void
group_to_string(const char * const header, std::vector<idx_pair> & v) {
    char     out_buf[16] = "";
    uint64_t hits        = 0;
    uint32_t comma       = 0;
    uint32_t depth       = 1;
    fprintf(stderr, "Grouped %s Network\n", header);
    fprintf(stderr, "[");
    for (uint32_t i = 0; i < v.size(); ++i) {
        if ((hits & (1 << (v[i].x))) || (hits & (1 << (v[i].y)))) {
            hits = 0;
            fprintf(stderr, "\n ");
            comma = 0;
            ++depth;
        }
        hits |= (1 << (v[i].x));
        hits |= (1 << (v[i].y));

        if (comma) {
            comma = 0;
            fprintf(stderr, ", ");
        }
        fprintf(stderr, "%s", v[i].to_string(out_buf));
        comma = 1;
    }
    fprintf(stderr, "]");
    fprintf(stderr, "\n");
    fprintf(stderr, "Depth: %d\n", depth);
    fprintf(stderr, "\n");
    fprintf(stderr, "\n");
}


void
bitonic_pairs(uint32_t n) {
    std::vector<idx_pair> pairs;
    std::vector<idx_pair> raw;
    std::vector<idx_pair> filtered;

    bitonic_sort(pairs, 0, n, 1);
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        raw.push_back(pairs[i]);
    }

    filter_pairs(pairs);
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        filtered.push_back(pairs[i]);
    }

    group(pairs);

    group_to_string("Bitonic", pairs);
    if (verbose) {
        vec_to_string("raw      ", raw);
        vec_to_string("filtered ", filtered);
        vec_to_string("grouped  ", pairs);
    }
}


void
bosenelson_pairs(uint32_t n) {
    std::vector<idx_pair> pairs;
    std::vector<idx_pair> raw;

    bn_split(pairs, 0, n);
    for (uint32_t i = 0; i < pairs.size(); ++i) {
        raw.push_back(pairs[i]);
    }

    group(pairs);

    group_to_string("Bosenelson", pairs);
    if (verbose) {
        vec_to_string("raw      ", raw);
        vec_to_string("grouped  ", pairs);
    }
}


int
main(int argc, char ** argv) {
    assert(argc >= 3);
    uint32_t todo = argv[2][0] == 'A';
    verbose       = argc > 3;
    if (todo) {
        bosenelson_pairs(atoi(argv[1]));
    }
    else {
        bitonic_pairs(atoi(argv[1]));
    }
}
