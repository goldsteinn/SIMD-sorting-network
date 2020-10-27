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

def sig_exit(signum, empty):
    print("Exiting on Signal({})".format(str(signum)))
    sys.exit(-1)


signal.signal(signal.SIGINT, sig_exit)


for max_bytes in max_b:
    for i in range(0, len(types)):
        if i < skip_to:
            continue
        min_N = max(4, int(16 / sizes[i]))
        max_N = int(max_bytes / sizes[i]) + 1
        for n in range(min_N, max_N):

            os.system("rm -f export_tests/.tmp")
            if max_bytes == 32:
                os.system(
                    "./export2.py -N {} -T {} -c AVX512 --algorithm bitonic > export_tests/.tmp".
                    format(n, types[i]))
            else:
                os.system(
                    "./export2.py -N {} -T {} --algorithm bitonic > export_tests/.tmp".
                    format(n, types[i]))

            sort_impl = ""
            for lines in open("export_tests/.tmp"):
                sort_impl += lines

            export_driver_impl = ""
            for lines in open("export_driver.cc"):
                export_driver_impl += lines

            export_driver_impl = export_driver_impl.replace(
                "[SORT_IMPL]", sort_impl)
            export_driver_impl = export_driver_impl.replace("[TYPE]", types[i])
            export_driver_impl = export_driver_impl.replace("[N]", str(n))

            sname = "{}_{}_{}".format("bitonic", str(n), types[i])
            export_driver_impl = export_driver_impl.replace("[SORT_NAME]", sname)

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
