#! /usr/bin/env python3

import matplotlib.pyplot as plt
import argparse
import sys
import os


def err_assert(check, errstr):
    if check is False:
        print(errstr)
        sys.exit(-1)


parser = argparse.ArgumentParser(
    description='graphing program for network sort')
parser.add_argument("-v",
                    "--verbose",
                    action="count",
                    default=0,
                    help="increase output verbosity")
parser.add_argument('-f',
                    '--file',
                    action='store',
                    default="",
                    help='Data file')
parser.add_argument("-y",
                    "--y-axis",
                    action="store",
                    default="mean",
                    help=" field (Y axis)")
parser.add_argument("-x",
                    "--x-axis",
                    action="store",
                    default="test_n",
                    help="X axis field (N or type probably)")
parser.add_argument("-N",
                    "--N-constraints",
                    nargs="+",
                    type=str,
                    default=["16", "32"],
                    help="N range to graph (X axis)")
parser.add_argument(
    "-r",
    "--range",
    action="store_true",
    default=False,
    help="Set if N specifies a range rather than a set of points")
parser.add_argument("-l",
                    "--lines",
                    action="store_true",
                    default=False,
                    help="Set to include lines between points")

parser.add_argument("-T",
                    "--type-constraints",
                    nargs="+",
                    type=str,
                    default=["uint8_t", "uint16_t", "uint32_t", "uint64_t"],
                    help="Types to graph")
parser.add_argument(
    "-alg",
    "--algorithm-constraints",
    nargs="+",
    type=str,
    default=["bitonic", "oddeven", "batcher", "balanced", "bosenelson", "best"],
    help="Algorithms to graph")
parser.add_argument("-s",
                    "--simd-constraints",
                    nargs="+",
                    type=str,
                    default=["AVX2", "AVX512"],
                    help="SIMD sets to graph")
parser.add_argument("-b",
                    "--builtin-constraints",
                    nargs="+",
                    type=str,
                    default=["First", "Fallback", "None"],
                    help="Builtin methods to graph")

flags = parser.parse_args()
verbose = flags.verbose

data_file = flags.file

do_plot = flags.lines

y_axis = flags.y_axis
x_axis = flags.x_axis

N_as_range = flags.range

type_constraints = flags.type_constraints
_N_constraints = flags.N_constraints
algorithm_constraints = flags.algorithm_constraints
simd_constraints = flags.simd_constraints
builtin_constraints = flags.builtin_constraints

err_assert(len(_N_constraints) != 0, "Error: invalid N constraints")
err_assert(len(type_constraints) != 0, "Error: No type constraints")
err_assert(len(algorithm_constraints) != 0, "Error: No algorithm constaints")
err_assert(len(simd_constraints) != 0, "Error: No simd constraints")
err_assert(len(builtin_constraints) != 0, "Error: No builtin constraints")

N_constraints = []
if N_as_range is False:
    N_constraints = _N_constraints
else:
    for i in range(int(_N_constraints[0]), int(_N_constraints[1]) + 1):
        N_constraints.append(str(i))

constraints = ["type", "test_n", "algorithm", "simd", "builtin"]
constraint_indexes = []

y_axis_idx = 0

data_lines = []

graph_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
graph_markers = ['o', 'v', '^', '<', '>']
unique_graph_indicators = []
for gm in graph_markers:
    for gc in graph_colors:
        unique_graph_indicators.append([gc, gm])

y_max = float(0.0)
y_min = float(-1.0)


def verify_access(fname):
    if fname == "":
        return False
    return os.access(fname, os.R_OK)


err_assert(verify_access(data_file),
           "Error: unable to access {}".format(data_file))


def arr_idx(v, arr):
    counter = 0
    for a in arr:
        if v == a:
            return counter
        counter += 1
    err_assert(False,
               "Error: unable to find {} in {}".format(str(v), str(arr)))


def label_to_arr():
    if x_axis == "type":
        return type_constraints
    elif x_axis == "test_n":
        return N_constraints
    elif x_axis == "algorithm":
        return algorithm_constraints
    elif x_axis == "simd":
        return simd_constraints
    else:
        return builtin_constraints


def label_to_idx(v):
    return arr_idx(v, label_to_arr())


def get_simd_name(s):
    if str(s) == "1":
        return "AVX2"
    elif str(s) == "2":
        return "AVX512"
    else:
        err_assert(False, "Error: unknown simd value: {}".format(str(s)))


def get_builtin_name(b):
    if str(b) == "0":
        return "First"
    elif str(b) == "1":
        return "Fallback"
    elif str(b) == "2":
        return "None"
    else:
        err_assert(False, "Error: unknown builtin value: {}".format(str(b)))


def parse_csv_header(csv_hdr):
    err_assert(csv_hdr != "", "Error: empty csv header")
    csv_hdr = csv_hdr.split(",")
    for c in constraints:
        counter = 0
        for h in csv_hdr:
            if c == h:
                constraint_indexes.append(counter)
                break
            counter += 1
        err_assert(counter != len(csv_hdr),
                   "Error: unable to find all constraint field")

    counter = 0
    for h in csv_hdr:
        if h == y_axis:
            global y_axis_idx
            y_axis_idx = counter
            break
        counter += 1
    err_assert(counter != len(csv_hdr),
               "Error: unable to find stats field field")


def field_to_type(f):
    if f in type_constraints:
        return "type"
    elif f in N_constraints:
        return "test_n"
    elif f in algorithm_constraints:
        return "algorithm"
    elif f in simd_constraints:
        return "simd"
    elif f in builtin_constraints:
        return "builtin"
    else:
        err_assert(False,
                   "Error: unable to find type for field {}".format(str(f)))


def csv_get(field, csv_data):
    if field == y_axis:
        return csv_data[y_axis_idx]
    else:
        for i in range(0, len(constraints)):
            if field == constraints[i]:
                ret = csv_data[constraint_indexes[i]]
                if field == "simd":
                    ret = get_simd_name(ret)
                elif field == "builtin":
                    ret = get_builtin_name(ret)

                return ret

        err_assert(False, "Error: unable to find field {}".format(field))


def csv_data_to_fields(csv_data):
    fields = []
    for c in constraints:
        fields.append(csv_get(c, csv_data))
    return fields


class Point():
    def __init__(self, X, Y):
        global y_max
        if float(y_max) < float(Y):
            y_max = float(Y)

        global y_min
        if float(y_min) == float(-1.0) or float(y_min) > float(Y):
            y_min = float(Y)

        self.X = str(X)
        self.Y = float(Y)

    def to_string(self):
        return "[{},{}]".format(str(self.X), str(self.Y))


class Line():
    def __init__(self, fields, uid):
        self.fields = fields
        self.uid = uid
        self.points = []

    def to_string(self, complete):
        ret_str = ""
        for f in self.fields:
            if f != self.fields[0]:
                ret_str += ", "

            if f == "" or field_to_type(f) == x_axis:
                ret_str += "{} = {}".format(x_axis, "X-AXIS")
            else:
                ret_str += "{} = {}".format(field_to_type(f), f)
        if complete is True:
            ret_str += " -> ["
            for p in self.points:
                ret_str += p.to_string()
            ret_str += "]"
        return ret_str

    def match(self, csv_fields):
        err_assert(
            len(csv_fields) == len(self.fields), "error csv_field misaligned")
        for i in range(0, len(self.fields)):
            if field_to_type(self.fields[i]) == x_axis or self.fields[i] == "":
                continue
            if self.fields[i] != csv_fields[i]:
                return False
        return True

    def try_add(self, csv_line):
        err_assert(csv_line != "", "Error: empty line")

        csv_data = csv_line.split(",")
        err_assert(len(csv_data) > 5, "Error: line doesn't contain valid data")

        csv_fields = csv_data_to_fields(csv_data)

        if self.match(csv_fields) is False:
            return False

        data_x = csv_get(x_axis, csv_data)
        if data_x not in label_to_arr():
            return False

        data_y = csv_get(y_axis, csv_data)
        self.points.append(Point(data_x, data_y))

        return True


def to_len(arr):
    if field_to_type(arr[0]) == x_axis:
        return 1
    return len(arr)


def prep():
    counter = 0
    for T_idx in range(0, to_len(type_constraints)):
        for N_idx in range(0, to_len(N_constraints)):
            for algorithm_idx in range(0, to_len(algorithm_constraints)):
                for simd_idx in range(0, to_len(simd_constraints)):
                    for builtin_idx in range(0, to_len(builtin_constraints)):
                        err_assert(counter < len(unique_graph_indicators),
                                   "Error: to many lines requested")
                        data_lines.append(
                            Line([
                                type_constraints[T_idx], N_constraints[N_idx],
                                algorithm_constraints[algorithm_idx],
                                simd_constraints[simd_idx],
                                builtin_constraints[builtin_idx]
                            ], unique_graph_indicators[counter]))
                        counter += 1


def parse_csv():
    try:
        first = True
        for csv_lines in open(data_file):
            if first is True:
                first = False
                parse_csv_header(csv_lines)
                continue
            for lines in data_lines:
                if lines.try_add(csv_lines) is True:
                    break
    except IOError:
        err_assert(False, "Error: unable to read from {}".format(data_file))


def graph_data():
    x_max = 0
    x_min = int(sys.maxsize)
    for lines in data_lines:
        x_labels = []
        x_points = []
        y_points = []
        for points in lines.points:
            if points.X not in x_labels:
                x_labels.append(points.X)
            x_val = int(label_to_idx(points.X))
            if x_val < x_min:
                x_min = x_val
            if x_val > x_max:
                x_max = x_val

            x_points.append(x_val)
            y_points.append(points.Y)

        if do_plot:
            plt.scatter(x_points,
                        y_points,
                        linewidth=1,
                        color=lines.uid[0],
                        marker=lines.uid[1],
                        label=lines.to_string(False))
        else:
            plt.plot(x_points,
                     y_points,
                     linewidth=1,
                     color=lines.uid[0],
                     marker=lines.uid[1],
                     markersize=3,
                     linestyle='dashed',
                     label=lines.to_string(False))

        plt.xticks(x_points, x_labels)

    plt.legend(loc='upper left')
    plt.ylabel(y_axis)
    plt.xlabel(x_axis)

    plt.ylim(max(0, 0, y_min - 10.0), y_max + 10.0)
    plt.xlim(left=x_min - .5, right=x_max + .5)
    plt.show()


prep()
parse_csv()
graph_data()
