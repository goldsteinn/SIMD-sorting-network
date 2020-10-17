#! /usr/bin/env python3

import os
import time
import sys
import argparse
import subprocess
import signal

parser = argparse.ArgumentParser(
    description='Create vectorized implementation of sorting network')

parser.add_argument('-f',
                    '--file',
                    action='store',
                    default="",
                    help='File to output data to')
parser.add_argument('-s',
                    '--skip',
                    action='store_true',
                    default=True,
                    help='Set to skip ahead based on results already in file')
parser.add_argument('--verbose',
                    '-v',
                    action='count',
                    default=0,
                    help='Set verbosity')

args = parser.parse_args()
skip = args.skip
res_file = args.file
verbosity = args.verbose

types = ["uint8_t", "uint16_t", "uint32_t", "uint64_t"]
n_min = 2
algorithms = ["bitonic", "oddeven", "batcher", "balanced", "bosenelson"]
simds = [1, 2]
builtins = [0, 1, 2]


def sig_exit(signum, empty):
    print("Exiting on Signal({})".format(str(signum)))
    sys.exit(-1)


def get_type_size(T):
    if str(T) == "uint8_t":
        return 1
    if str(T) == "uint16_t":
        return 2
    if str(T) == "uint32_t":
        return 4
    if str(T) == "uint64_t":
        return 8
    else:
        err_assert(False, "Error: unknown type: {}".format(str(T)))


def get_simd_name(s):
    if str(s) == "1":
        return "AVX2"
    elif str(s) == "2":
        return "AVX512"
    else:
        err_assert(False, "Error: unknown simd value: {}".format(str(s)))


def get_builtin_name(b):
    if str(b) == "0":
        return "Builtin First"
    elif str(b) == "1":
        return "Builtin Fallback"
    elif str(b) == "2":
        return "Builtin None"
    else:
        err_assert(False, "Error: unknown builtin value: {}".format(str(b)))


def err_assert(check, errstr):
    if check is False:
        print(errstr)
        sys.exit(-1)


def trial_subprocess(cmd):
    if verbosity > 1:
        print("\tExecuting: " + cmd)
    sproc = subprocess.Popen(cmd,
                             shell=True,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    try:
        timeout = 180
        sproc.wait(timeout)
    except subprocess.TimeoutExpired:
        err_assert(
            False, "Error: timeout after {} seconds running: {}".format(
                str(timeout), cmd))

    err_assert(sproc.returncode == 0,
               "Error: error returned from: {}".format(cmd))

    stdout_data, stderr_data = sproc.communicate()
    stdout_data = stdout_data.decode("utf-8", "ignore")
    stderr_data = stderr_data.decode("utf-8", "ignore")

    if stdout_data != "" and stderr_data != "":
        print("Warning: both stdout and stderr contain data." + " " +
              "Concatenating ret = stderr + stdout")

    return stderr_data + stdout_data


class Trial():
    def __init__(self):
        self.T = ""
        self.N = ""
        self.algorithm = ""
        self.simd = ""
        self.builtin = ""

    def init(self, T, N, algorithm, simd, builtin):
        self.T = T
        self.N = N
        self.algorithm = algorithm
        self.simd = simd
        self.builtin = builtin

    def from_str(self, csv_line):
        err_assert(csv_line != "",
                   "Error: initializing trial from empty string")

        csv_data = csv_line.split(",")
        err_assert(
            len(csv_data) >= 5,
            "Error: initializing trial from invalid string: {}".format(
                csv_line))

        self.init(csv_data[0], csv_data[1], csv_data[2], csv_data[3],
                  csv_data[4])

    def to_string(self):
        ret_str = ""
        ret_str += "T = {}, ".format(str(self.T))
        ret_str += "N = {}, ".format(str(self.N))
        ret_str += "Algorithm = {}, ".format(str(self.algorithm))
        ret_str += "Simd = {}, ".format(str(get_simd_name(self.simd)))
        ret_str += "Builtin = {}".format(str(get_builtin_name(self.builtin)))
        return ret_str

    def get_field(self, field, field_arr):
        if field == "":
            return 0

        counter = 0
        for f in field_arr:
            if field == f:
                return counter
            counter += 1
        err_assert(
            False,
            "Error unknown field for trial: {}".format(self.to_string()))

    def get_T_idx(self):
        return self.get_field(str(self.T), types)

    def get_algorithm_idx(self):
        return self.get_field(str(self.algorithm), algorithms)

    def get_simd_idx(self):
        return self.get_field(str(self.simd), simds)

    def get_builtin_idx(self):
        return self.get_field(str(self.builtin), builtins)

    def get_N_idx(self):
        if str(self.N) == "" or int(self.N) == 0:
            return n_min
        err_assert(
            int(self.N) >= n_min,
            "Error: invalid N value: {}".format(str(self.N)))

        return int(self.N)

    def is_empty(self):
        if str(self.T) == "" or str(self.N) == "" or str(
                self.algorithm) == "" or str(self.simd) == "" or str(
                    self.builtin) == "":
            return True
        return False

    def build(self):
        CXX = "g++"
        INCLUDES = "-I."
        CXXFLAGS = "-O3 -std=c++17 -march=native -mtune=native -mavx2"

        BUILD_FLAGS = "{} {} {}".format(CXX, INCLUDES, CXXFLAGS)
        BUILD_FLAGS += " "

        TEST_FLAGS = "-DTEST_TYPE={}".format(str(self.T)) + " "
        TEST_FLAGS += "-DTEST_N={}".format(str(self.N)) + " "
        TEST_FLAGS += "-DTEST_NETWORK_ALGORITHM={}".format(str(
            self.algorithm)) + " "
        TEST_FLAGS += "-DTEST_SIMD={}".format(str(self.simd)) + " "
        TEST_FLAGS += "-DTEST_BUILTIN={}".format(str(self.builtin)) + " "

        stats_o = BUILD_FLAGS + "timing/stats.cc -c"
        driver_o = BUILD_FLAGS + TEST_FLAGS + "driver.cc -c"

        exe = BUILD_FLAGS + "driver.o stats.o -o driver"

        if verbosity > 1:
            print("\tBuild Step: " + stats_o)
        err_assert(os.system(stats_o) == 0, "Error building stats.o")

        if verbosity > 1:
            print("\tBuild Step: " + driver_o)

        start = float(time.monotonic())
        ret = os.system(driver_o)
        end = float(time.monotonic())
        err_assert(ret == 0, "Error building driver.o")

        if verbosity > 1:
            print("\tBuild Step: " + exe)
        err_assert(os.system(exe) == 0, "Error linking driver.o and stats.o")

        return float(end - start)

    def run(self):
        build_time = round(float(self.build()), 3)
        run_data = trial_subprocess("./driver")
        return str(build_time) + "," + run_data


def verify_access(fname, ACCESS):
    if fname == "":
        return False
    return os.access(fname, ACCESS)


def output_data(use_file, ow_file, data):
    if use_file is True:
        try:
            ow_file.write(data + "\n")
        except IOError:
            err_assert(
                False, "Error writing to file: {}\nLost line: {}".format(
                    res_file, data))
    else:
        print(data)


def get_last_trial():
    t = Trial()
    if skip is False:
        return t
    if verify_access(res_file, os.R_OK) is False:
        return t
    try:
        last_line = ""
        for csv_lines in open(res_file):
            if csv_lines != "":
                last_line = csv_lines
                if verbosity > 0:
                    t.from_str(last_line)
                    print("Skipping: " + t.to_string())

        t.from_str(last_line)
        return t
    except IOError:
        err_assert(False,
                   "Error while reading from input file: {}".format(res_file))


def runner():
    start_trial = get_last_trial()

    T_start = start_trial.get_T_idx()
    N_start = start_trial.get_N_idx()
    algorithm_start = start_trial.get_algorithm_idx()
    simd_start = start_trial.get_simd_idx()
    builtin_start = start_trial.get_builtin_idx()

    use_file = verify_access(res_file, os.W_OK)
    ow_file = ""
    try:
        if use_file is True:
            if skip is True:
                ow_file = open(res_file, "a+")
            else:
                ow_file = open(res_file, "w+")
    except IOError:
        err_assert(False, "Error: unable to open: {}".format(res_file))

    first = True
    if skip is True and start_trial.is_empty() is False:
        first = False

    for T_idx in range(T_start, len(types)):
        T_start = 0
        n_max = int(64 / get_type_size(types[T_idx]))
        for N_idx in range(N_start, n_max):
            N_start = 0
            for algorithm_idx in range(algorithm_start, len(algorithms)):
                algorithm_start = 0
                for simd_idx in range(simd_start, len(simds)):
                    simd_start = 0
                    for builtin_idx in range(builtin_start, len(builtins)):
                        if verbosity > 2:
                            print("Iter: [{}][{}][{}][{}][{}]".format(
                                T_idx, N_idx, algorithm_idx, simd_idx,
                                builtin_idx))
                        builtin_start = 0
                        test_type = types[T_idx]
                        test_n = N_idx
                        test_algorithm = algorithms[algorithm_idx]
                        test_simd = simds[simd_idx]
                        test_builtin = builtins[builtin_idx]

                        cur_trial = Trial()
                        cur_trial.init(test_type, test_n, test_algorithm,
                                       test_simd, test_builtin)

                        if verbosity > 0:
                            print("Running: " + cur_trial.to_string())

                        result = cur_trial.run()

                        if first is True:
                            first = False
                            header = "compile time(s)," + trial_subprocess(
                                "./driver --header")
                            output_data(use_file, ow_file, header)

                        output_data(use_file, ow_file, result)


signal.signal(signal.SIGINT, sig_exit)
runner()
