#! /usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(
    description='Create vectorized implementation of sorting network')

parser.add_argument('-f',
                    '--network_file',
                    action='store',
                    default="",
                    help='File with network specifier')



args = parser.parse_args()

network_file = args.network_file
network_size = 0
network_alg = ""

network_algorithms = [
    "balanced", "batcher", "bitonic", "bosenelson", "bubble", "hibbard",
    "oddevenmerge", "oddeventrans"
]


if os.access(network_file, os.R_OK) is False:
    print("Error unable to access network file")
    exit(-1)
    
assert len(network_file.split("-")) == 2, "Error invalid filename"
network_alg = network_file.split("-")[0].split("/")
network_alg = network_alg[len(network_alg) - 1].upper()
network_size = int(network_file.split("-")[1])

if network_size < 2:
    print("Error invalid network size")
    exit(-1)


prefix = [
    "#ifndef _[NETWORK_ALG]_[NETWORK_SIZE]_IMPL_H_",
    "#define _[NETWORK_ALG]_[NETWORK_SIZE]_IMPL_H_", "",
    "#include <sort_base/vec_sort_incl.h>", "", "template<typename T>",
    "struct vsort<T, [NETWORK_SIZE]> {",
    "static constexpr uint32_t n = [NETWORK_SIZE];",
    "using v_type                = vec_t<T, n>;", "", "static void",
    "sort(T * arr) {", "v_type v = vec_load<T, n>(arr);"
]

postfix = ["vec_store<T, n>(arr, v);", "}", "};", "", "#endif"]

block = [
    "\nv = compare_exchange<T, n, \n// clang-format off\n[INDEX_PLACEMENT]\n// clang-format on\n>(v, [MERGE_MASK]);\n"
]

network_indexes = []
network_masks = []


def handle_replacements(line, indexes, mask):
    line = line.replace("[NETWORK_ALG]", network_alg)
    line = line.replace("[NETWORK_SIZE]", str(network_size))
    line = line.replace("[INDEX_PLACEMENT]", indexes)
    line = line.replace("[MERGE_MASK]", str(mask))
    return line


def format_index_placement(index_arr):
    assert len(index_arr) == network_size, "Error invalid index arr"

    index_str = str(index_arr[0])
    for i in range(1, network_size):
        index_str += ", "
        if i % 4 == 0:
            index_str += "\n"
        index_str += str(index_arr[i])
    return index_str


def print_arr(arr, indexes, mask):
    for line in arr:
        line = handle_replacements(line, indexes, mask)
        print(line)


def print_network():
    assert len(network_indexes) == len(
        network_masks), "Error mask and indexes misaligned"

    for i in range(0, len(network_indexes)):
        print_arr(block, format_index_placement(network_indexes[i]),
                  str(hex(network_masks[i])))


def process_file_line(line):
    line = line.replace("[[", "")
    line = line.replace("],\n", "")
    line = line.replace("]]\n", "")

    line = line.replace("[", "")
    line = line.replace("],", "|")

    line = line.split("|")
    return line


def create_index_pairs(line):
    index_pairs = []
    for index_pair in line:
        indexes = index_pair.split(",")
        assert len(indexes) == 2, "Error invalid index pair"

        index_pairs.append([int(indexes[0]), int(indexes[1])])

    return index_pairs


def create_index_arr(index_pairs):
    index_arr = []
    for i in range(0, network_size):
        index_arr.append((network_size - 1) - i)

    for index_pair in index_pairs:
        idx0 = index_pair[0]
        idx1 = index_pair[1]

        index_arr[(network_size - 1) - idx0] = idx1
        index_arr[(network_size - 1) - idx1] = idx0

    return index_arr


def create_index_mask(index_pairs):
    index_mask = 0
    for index_pair in index_pairs:
        idx0 = index_pair[0]
        index_mask |= (1 << idx0)

    return index_mask


def create_network():
    for line in open(network_file):
        line = process_file_line(line)
        index_pairs = create_index_pairs(line)
        index_arr = create_index_arr(index_pairs)
        index_mask = create_index_mask(index_pairs)
        network_indexes.append(index_arr)
        network_masks.append(index_mask)


create_network()
print_arr(prefix, "", "")
print_network()
print_arr(postfix, "", "")
