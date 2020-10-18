#! /usr/bin/env python3

import os
import sys
import re


def err_assert(check, errstr):
    if check is False:
        print("Error: " + errstr)
        sys.exit(-1)


def open_file(fname, mode):
    err_assert(fname != "", "Empty file path")

    ACCESS = os.R_OK
    if "w" in mode or "a" in mode:
        ACCESS = os.W_OK
    err_assert(os.access(fname, ACCESS), "No access to {}".format(fname))

    fhandle = ""
    try:
        fhandle = open(fname, mode)
    except IOError:
        err_assert(False,
                   "Exception opening {} with mode {}".format(fname, mode))

    return fhandle


err_assert(len(sys.argv) > 1, "No file argument")
f = open_file(sys.argv[1], "r")


def cleanhtml(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext


tab = "    "


class Node():
    def __init__(self, title):
        self.title = title
        self.depth = 0
        self.N = 0
        self.ilist = ""
        self.graph = []
        self.padding = len(tab + "using network = std::integer_sequence<")
        self.padding_str = ""
        for i in range(0, self.padding):
            self.padding_str += " "

        self.struct_arr = [
            "/*", " * Sorting Network For N = [N], with Depth = [DEPTH]",
            " */", "template<>", "struct best_network<[N]> {",
            "[TAB]using network = std::integer_sequence<uint32_t,",
            "[PADDING]// clang-format off", "[ILIST]",
            "[PADDING]// clang-format on", "[PADDING]>;", "};\n"
        ]

    def add(self, line):
        self.graph.append(line)

    def to_string(self):
        out = ""
        for l in self.graph:
            out += l
        return out

    def as_class(self):
        self.parse()
        for lines in self.struct_arr:
            print(
                lines.replace("[ILIST]", self.ilist).replace(
                    "[N]",
                    str(self.N)).replace("[DEPTH]", str(self.depth)).replace(
                        "[PADDING]", self.padding_str).replace("[TAB]", tab))

    def show(self):
        print(
            "----------------------------------------------------------------------"
        )
        self.parse()
        print("N = {}, Depth = {}".format(self.N, self.depth))
        print(self.ilist)
        print(
            "----------------------------------------------------------------------"
        )

    def parse(self):
        self.parse_title()
        self.parse_graph()

    def parse_title(self):
        t = self.title.split()
        for i in range(0, len(t)):
            if t[i] == "for":
                self.N = int(t[i + 1])
            if t[i] == "CEs,":
                self.depth = int(t[i + 1])

    def parse_graph(self):
        s = self.to_string()
        s = s.replace("(", "").replace(")", "").replace("\n", "").replace(
            "[", "").replace("]", ",")
        s = s[0:len(s) - 1]
        s = s.split(",")
        out = "[PADDING]"
        for i in range(0, len(s)):
            if len(s[i]) == 1:
                out += " "
            if (i % 8) != 0:
                out += " "
            out += s[i]
            if i != len(s) - 1:
                out += ","
            if i != 0 and i != (len(s) - 1) and (i % 8) == 7:
                out += "\n[PADDING]"

        self.ilist = out


nodes = []

for lines in f:
    to_print = ""
    new_node = False
    if "sorting network for" in lines.lower() and "trivial" not in lines.lower(
    ):
        new_node = True
        to_print = lines
    elif "[(" in lines:
        to_print = lines
    elif "(]" in lines:
        to_print = lines
    else:
        continue

    to_print = cleanhtml(to_print)
    if new_node is True:
        nodes.append(Node(to_print))
    else:
        nodes[len(nodes) - 1].add(to_print)

for n in nodes:
    n.parse()

to_use = []
for i in range(0, len(nodes)):
    use = True
    if nodes[i].N < 4:
        use = False
    for j in range(0, len(nodes)):
        if nodes[i].N == nodes[j].N and nodes[i].depth > nodes[j].depth:
            use = False
    if use == True:
        to_use.append(nodes[i])

print("#ifndef _BEST_NETWORK_H_")
print("#define _BEST_NETWORK_H_")
print("\n")

print("/*")
print(" * Minimum Depth Known Sorting Networks For N = [4, 32]")
print(" * Taken from:")
print(
    " * http://users.telenet.be/bertdobbelaere/SorterHunter/sorting_networks.html"
)
print(" */")
print("")
print("#include <stdint.h>")
print("#include <utility>")
print("")
print("namespace vsort {")
print("namespace network {")
print("namespace internal {")
print("")

print("template<uint32_t n>")
print("struct best_network {")
print(tab + "// Default to Bitonic Sorting Network")
print(
    tab +
    "using network = typename network::internal::bitonic_network<n>::network;")
print("};\n")
for n in to_use:
    n.as_class()

print("}  // namespace internal")
print("}  // namespace network")
print("}  // namespace vsort")
print("")
print("#endif")
