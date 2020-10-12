#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <utility>
#include <vector>

#include "util/constexpr_util.h"

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
bitonic_pairs(uint32_t n, uint32_t EZ) {
    std::vector<idx_pair> pairs;
    bitonic_sort(pairs, 0, n, 1);
    if (EZ == 1) {
        filter_pairs(pairs);
    }
    if (EZ == 2) {
        filter_pairs(pairs);
        group(pairs);
    }
    if (EZ == 3) {
        filter_pairs(pairs);
        group(pairs);


        char out_buf[16] = "";

        uint64_t hits = 0;
        fprintf(stderr, "[");
        for (uint32_t i = 0; i < pairs.size(); ++i) {
            if ((hits & (1 << (pairs[i].x))) || (hits & (1 << (pairs[i].y)))) {
                hits = 0;
                fprintf(stderr, "\n");
            }
            hits |= (1 << (pairs[i].x));
            hits |= (1 << (pairs[i].y));
            fprintf(stderr, "%s, ", pairs[i].to_string(out_buf));
        }
        fprintf(stderr, "]");
        fprintf(stderr, "\n");
    }
    else {
        for (uint32_t i = 0; i < pairs.size(); ++i) {
            fprintf(stderr, ", %d, %d", pairs[i].x, pairs[i].y);
        }
        fprintf(stderr, "\n");
    }
}

int
main(int argc, char ** argv) {
    assert(argc >= 3);
    bitonic_pairs(atoi(argv[1]), atoi(argv[2]));
}
