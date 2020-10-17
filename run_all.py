#! /usr/bin/env python3

import os

types = ["uint8_t"]
sizes = [1]
builtin_status = [0, 1, 2]
simd_status = [2]

CXX = "g++"
INCLUDES = "-I."
CXXFLAGS = "-O3 -std=c++17 -march=native -mtune=native -mavx2"
stats_o = "{} {} {} timing/stats.cc -c"
driver_o = "{} {} {} -DTEST_TYPE={} -DTEST_N={} -DTEST_SIMD={} -DTEST_BUILTIN={} driver.cc -c"
exe = "{} {} {} driver.o stats.o -o driver"

for i in range(0, len(types)):
    for bstatus in builtin_status:
        for sstatus in simd_status:
            n_max = int(64 / sizes[i])
            n_min = int((8 / sizes[i]) + 1)
            for n in range(n_min, n_max + 1):
                os.system("rm -f *.o")
                cmd = stats_o.format(CXX, CXXFLAGS, INCLUDES)
                print(cmd)
                os.system(cmd)

                cmd = driver_o.format(CXX, CXXFLAGS, INCLUDES, types[i],
                                      str(n), str(sstatus), str(bstatus))
                print(cmd)
                os.system(cmd)

                cmd = exe.format(CXX, CXXFLAGS, INCLUDES)
                print(cmd)
                os.system(cmd)
                os.system("./driver")
