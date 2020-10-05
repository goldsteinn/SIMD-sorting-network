#! /usr/bin/env python3

import sys

if len(sys.argv) < 2:
    print("Need Size")
    exit(-1)
network_file = sys.argv[1]
glen = int(network_file.split("-")[1])


def print_tail():
    base = ["vec_store<T, n>(arr, v);", "}", "};"]

    for b in base:
        print(b)


def print_head():
    base = [
        "template<typename T>", "struct vsort<T, [LEN]> {",
        "static constexpr uint32_t n = [LEN];", "using v_type = vec_t<T, n>;",
        "", "static void", "sort(T * arr) {", "v_type v = vec_load<T, n>(arr);"
    ]

    for b in base:
        print(b.replace("[LEN]", str(glen)))


def print_arr(arr, mask):
    arr_str = str(arr[0])
    for i in range(1, glen):
        arr_str += ", "
        if i % 4 == 0:
            arr_str += "\n"
        arr_str += str(arr[i])
        if arr[i] < 10:
            arr_str += " "

    base = [
        "{", "// clang-format off", "v_type perm = vec_set<T, n>([ARR]);",
        "// clang-format on", "v_type cmp   = vec_perm<T, n>(perm, v);",
        "v_type s_min = vec_min<T, n>(v, cmp);",
        "v_type s_max = vec_max<T, n>(v, cmp);",
        "v           = vec_blend<T, n>(s_max, s_min, [MASK]);", "}"
    ]
    for b in base:
        print(b.replace("[ARR]", arr_str).replace("[MASK]", str(hex(mask))))


indexes = []
lnum = 0
for lines in open(network_file):
    indexes.append([])
    lines = lines.replace("\n", "")
    lines = lines.replace("[[", "[")
    lines = lines.replace("]]", "]")
    lines = lines.replace("[", "")
    lines = lines.replace("],", "|")
    lines = lines.replace("]", "")
    lines = lines.split("|")
    for p in lines:
        p = p.split(",")
        indexes[lnum].append([int(p[0]), int(p[1])])
        
    lnum += 1

print_head()
for idx_arr in indexes:
    indexes_out = []
    for i in range(0, glen):
        indexes_out.append((glen - 1) - i)
    mask = 0
    for idx in idx_arr:
        p1 = idx[0]
        p2 = idx[1]
        indexes_out[(glen - 1) - p1] = p2
        indexes_out[(glen - 1) - p2] = p1
        mask |= (1 << p1)

    print_arr(indexes_out, mask)
print_tail()
