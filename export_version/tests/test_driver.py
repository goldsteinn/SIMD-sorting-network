#! /usr/bin/env python3

import cpufeature
import os
import sys
import signal
from itertools import permutations

print("Driver To Test SIMD Sorting Network Exporter")
print("Default behavior is to run all possible configurations")
print(
    "To manually run a single test case; pass the test case flags as commandline arguments"
)
types = [
    "uint8_t", "uint16_t", "uint32_t", "uint64_t", "int8_t", "int16_t",
    "int32_t", "int64_t"
]
sizes = [1, 2, 4, 8, 1, 2, 4, 8]

# Check to see if CPU supports AVX512 tests
max_b = [32]
avx512_flags = ["AVX512f", "AVX512vl", "AVX512bw", "AVX512vbmi"]
add_64 = True
for f in avx512_flags:
    if cpufeature.CPUFeature[f] is False:
        add_64 = False
if add_64 is True:
    max_b.append(64)

manual_cmdline = ""
for i in range(1, len(sys.argv)):
    manual_cmdline += sys.argv[i]
    if i != len(sys.argv) - 1:
        manual_cmdline += " "

use_manual_cmdline = len(sys.argv) > 1

algorithms = ["best", "bitonic", "oddeven", "bosenelson", "batcher", "minimum"]

extra_flags_ops = ["-O uop", "-i", "--aligned", "-e"]

export_template_file = "export_template.cc"
exporter_exe = "../export.py"
test_dir = "export_tests"
tmp_file = ".tmp"


def sig_exit(signum, empty):
    print("Exiting on Signal({})".format(str(signum)))
    sys.exit(-1)


def arr_to_args(arr):
    out = ""
    for f in arr:
        out += "{} ".format(f)

    return out


def get_flags(foptions):
    p = []
    for i in range(1, len(foptions)):
        f_tuple = permutations(foptions, i)
        for f in f_tuple:
            p.append(arr_to_args(list(f)))

    return p


extra_flags = get_flags(extra_flags_ops)

signal.signal(signal.SIGINT, sig_exit)

for max_bytes in max_b:
    for i in range(0, len(types)):
        min_N = max(4, int(4 / sizes[i]))
        max_N = int(max_bytes / sizes[i]) + 1
        for n in range(min_N, max_N):
            for a in algorithms:
                if a == "minimum" and n >= 32:
                    continue
                for f in extra_flags:
                    tmp_file_full = "{}/{}".format(test_dir, tmp_file)
                    os.system("mkdir -p {}".format(test_dir))
                    os.system("rm -f " + tmp_file_full)

                    cmd_flags = ""
                    cmd = "./{} {} > " + tmp_file_full
                    if use_manual_cmdline is True:
                        cmd_flags = manual_cmdline
                    elif max_bytes == 32:
                        cmd_flags = "-N {} -T {} {} -c AVX512 --algorithm {}".format(
                            n, types[i], f, a)
                    else:
                        cmd_flags = "-N {} -T {} {} --algorithm {}".format(
                            n, types[i], f, a)

                    cmd = cmd.format(exporter_exe, cmd_flags)

                    running = "Running: {}".format(cmd_flags)
                    print(running, end="", flush=True)
                    os.system(cmd)

                    true_N = 0
                    true_alg = ""
                    true_T = ""
                    sort_impl = ""
                    for lines in open(tmp_file_full):
                        if "Sort Size" in lines and ":" in lines:
                            tmp = lines.split()
                            true_N = int(tmp[len(tmp) - 1])
                        if "Network Generation Algorithm" in lines and ":" in lines:
                            tmp = lines.split()
                            true_alg = tmp[len(tmp) - 1]
                        if "Underlying Sort Type" in lines and ":" in lines:
                            tmp = lines.split()
                            true_T = tmp[len(tmp) - 1]

                        sort_impl += lines

                    export_driver_impl = ""
                    for lines in open(export_template_file):
                        export_driver_impl += lines

                    export_driver_impl = export_driver_impl.replace(
                        "[SORT_IMPL]", sort_impl)
                    export_driver_impl = export_driver_impl.replace(
                        "[TYPE]", true_T)
                    export_driver_impl = export_driver_impl.replace(
                        "[N]", str(true_N))

                    func_name = "{}_{}_{}".format(true_alg, str(true_N),
                                                  true_T)
                    
                    sname = cmd_flags.replace("--", "-").replace(" -", "-").replace(" ", "_").replace("-", "_")
          
                    export_driver_impl = export_driver_impl.replace(
                        "[SORT_NAME]", func_name)

                    fname = test_dir + "/" + sname + ".cc"
                    f = open(fname, "w+")
                    f.write(export_driver_impl)
                    f.close()

                    if os.system(
                            "g++ -O3 -std=c++17 -march=native -mtune=native {} -o export_exe"
                            .format(fname)) != 0:
                        print("\nError Building " + sname)
                        sys.exit(-1)
                    if os.system("./export_exe") != 0:
                        print("\nError Running " + sname)
                        sys.exit(-1)
                    print("- Passed".rjust(88 - len(running)), flush=True)

                    os.system("rm -f export_exe")
                    if use_manual_cmdline is True:
                        sys.exit(0)
