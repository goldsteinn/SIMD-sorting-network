#! /usr/bin/env python3

import subprocess
import argparse
import os

parser = argparse.ArgumentParser(description='Create implementation files')

parser.add_argument("-f",
                    "--network_file",
                    action='store',
                    default="",
                    help="File with network specifier (or directory containing specifier files)")

parser.add_argument("--nmin",
                    action='store',
                    default="2",
                    help="Min N if using f as directory")

parser.add_argument("--nmax",
                    action='store',
                    default="32",
                    help="Max N if using f as directory")

parser.add_argument("--new",
                    action='store_true',
                    default=False,
                    help="Overwrite existing header")
parser.add_argument("-i",
                    "--header_file",
                    action='store',
                    default="vec_sort.h",
                    help="set include file")

parser.add_argument("-d",
                    "--dest",
                    action='store',
                    default="implementation",
                    help="set destination of sort file")

args = parser.parse_args()
dest_dir = args.dest
header_file = args.header_file
network_file = args.network_file
new = args.new
n_min = int(args.nmin)
n_max = int(args.nmax)

if ".h" not in header_file:
    print("Error invalid header file")
    exit(-1)

if network_file == "":
    print("Error invalid network file")
    exit(-1)

if os.access(network_file,
             os.R_OK) is False and os.path.isdir(network_file) is False:
    print("Error unable to access network file")
    exit(-1)

algorithms = [
    'bosenelson', 'hibbard', 'batcher', 'bitonic', 'oddevenmerge', 'bubble',
    'oddeventrans', 'balanced'
]

header_content = []
header_content_default = [
    "#ifndef _VEC_SORT_H_", "#define _VEC_SORT_H_", "#endif"
]


def get_existing_header():
    if os.access(header_file, os.R_OK) is True and new is False:
        linecount = 0
        for lines in open(header_file):
            lines = lines.replace("\n", "")
            header_content.append(lines)
            linecount += 1

        if linecount >= 3:
            return

    for lines in header_content_default:
        header_content.append(lines)


def find_file(fname, cur_path, levels):
    if os.path.exists(fname) is True:
        return fname
    # its fine to check same dir a bunch of times given that depth is only 3
    other_dir = [".."]
    for f in os.listdir(cur_path):
        if fname in f:
            return cur_path + "/" + f
        if os.path.isdir(f):
            other_dir.append(f)
    if levels < 2:
        for d in other_dir:
            ret = find_file(fname, cur_path + "/" + d, levels + 1)
            if ret != "":
                return ret
    return ""


def get_impl_file():
    impl_file = network_file.split("/")
    impl_file = impl_file[len(impl_file) - 1]
    impl_file += "-impl.h"
    return dest_dir + "/" + impl_file


def get_implementation():
    create_impl_file = find_file("create_network_impl.py", ".", 0)

    if create_impl_file == "":
        print("Error unable to find create_impl_file")
        exit(-1)

    create_impl_cmd = "{} -f {}".format(create_impl_file, network_file)
    create_impl_process = subprocess.Popen(create_impl_cmd,
                                           shell=True,
                                           stdout=subprocess.PIPE,
                                           stderr=subprocess.PIPE)

    create_impl_process.wait(60)

    stdout_data, stderr_data = create_impl_process.communicate()
    stdout_data = stdout_data.decode("utf-8")
    stderr_data = stderr_data.decode("utf-8")

    ret = create_impl_process.returncode
    if ret != 0:
        print("Error running create_impl_process")
        print(stdout_data)
        exit(-1)

    return stdout_data


def write_data(implementation):
    os.system("mkdir -p {}".format(dest_dir))
    impl_file = get_impl_file()
    try:
        outfile = open(impl_file, "w+")
        outfile.write(implementation)
        outfile.close()
    except IOError:
        print("Error creating destination file")
        exit(-1)


def create_implementation():
    write_data(get_implementation())


def add_include():
    impl_file = get_impl_file()
    try:
        hf = open(header_file, "w+")

        new_header_content = []
        header_content_len = len(header_content)

        for i in range(0, header_content_len - 1):
            new_header_content.append(header_content[i])

        new_header_content.append("#include <" + impl_file + ">")

        new_header_content.append(header_content[header_content_len - 1])

        for lines in new_header_content:
            hf.write(lines + "\n")
        hf.close()
    except IOError:
        print("Error writing include file, dumping old content")
        print("-------------------------------------------------------")
        for lines in header_content:
            print(lines)
        print("-------------------------------------------------------")
        exit(-1)


network_file_list = []
if os.path.isdir(network_file) is True:
    if n_min < 2:
        print("Error invalid min")
        exit(-1)
    if n_max > 32:
        print("Error invalid max")
        exit(-1)

    for f in os.listdir(network_file):
        check = f.split("-")
        if len(check) != 2:
            continue
        if check[0] in algorithms and int(check[1]) >= n_min and int(
                check[1]) <= n_max:
            network_file_list.append(network_file + "/" + f)
else:
    network_file_list.append(network_file)

new_header_file = find_file(header_file, ".", 0)
if new_header_file != "":
    header_file = new_header_file

for nf in network_file_list:
    network_file = nf
    get_existing_header()
    create_implementation()
    add_include()
    header_content = []
    new = False
