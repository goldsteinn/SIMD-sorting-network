#! /usr/bin/env python3

import os
import sys
import signal

skip_to = 0
types = [
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t"
]
sizes = [1, 2, 4, 8, 1, 2, 4, 8]
max_b = [32, 64]
algorithms = ["best", "bitonic", "oddeven", "bosenelson", "batcher", "minimum"]
extra_flags = [
    "--clang-format 1231231 ", "--clang-format 1231231 -i",
    "--clang-format 1231231 -O space", "--clang-format 1231231 -O space -i",
    "--clang-format 1231231 -O uop", "--clang-format 1231231 -O uop -i",
    "--clang-format 1231231 -e", "--clang-format 1231231 --aligned",
    "--clang-format 1231231 -e --aligned", "--clang-format 1231231 -e -i",
    "--clang-format 1231231 --aligned -i",
    "--clang-format 1231231 -e --aligned -i"
]


def sig_exit(signum, empty):
    print("Exiting on Signal({})".format(str(signum)))
    sys.exit(-1)


signal.signal(signal.SIGINT, sig_exit)

for max_bytes in max_b:
    for i in range(0, len(types)):

        min_N = max(4, int(4 / sizes[i]))
        max_N = int(max_bytes / sizes[i]) + 1
        for n in range(min_N, max_N):
            if n < skip_to:
                continue
            for a in algorithms:
                if a == "minimum" and n >= 32:
                    continue
                for f in extra_flags:
                    os.system("rm -f export_tests/.tmp")
                    cmd = ""
                    if max_bytes == 32:
                        cmd = "./export2.py -N {} -T {} {} -c AVX512 --algorithm {} > export_tests/.tmp".format(
                            n, types[i], f, a)
                    else:
                        cmd = "./export2.py -N {} -T {} {} --algorithm {} > export_tests/.tmp".format(
                            n, types[i], f, a)

                    print("Running: {}".format(cmd))
                    os.system(cmd)

                    
                    true_N = 0
                    true_alg = ""
                    sort_impl = ""
                    for lines in open("export_tests/.tmp"):
                        if "Sort Size" in lines and ":" in lines:
                            tmp = lines.split()
                            true_N = int(tmp[len(tmp) - 1])
                        if "Network Generation Algorithm" in lines and ":" in lines:
                            tmp = lines.split()
                            true_alg = tmp[len(tmp) - 1]

                        sort_impl += lines


                    export_driver_impl = ""
                    for lines in open("export_driver.cc"):                            
                        export_driver_impl += lines

                    export_driver_impl = export_driver_impl.replace(
                        "[SORT_IMPL]", sort_impl)
                    export_driver_impl = export_driver_impl.replace(
                        "[TYPE]", types[i])
                    export_driver_impl = export_driver_impl.replace(
                        "[N]", str(n))

                    func_name = "{}_{}_{}".format(true_alg, str(true_N), types[i])
                    sname = "{}_{}_{}".format(a, str(n), types[i])
                    
                    export_driver_impl = export_driver_impl.replace(
                        "[SORT_NAME]", func_name)

                    fname = "export_tests/" + sname + ".cc"
                    f = open(fname, "w+")
                    f.write(export_driver_impl)
                    f.close()

                    if os.system(
                            "g++ -O3 -std=c++17 -march=native -mtune=native {} -o export_exe"
                            .format(fname)) != 0:
                        print("Error Building " + sname)
                        sys.exit(-1)
                    if os.system("./export_exe") != 0:
                        print("Error Running " + sname)
                        sys.exit(-1)
                    print("Success: " + sname)

                    os.system("rm -f export_exe")
