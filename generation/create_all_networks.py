#! /usr/bin/env python3

import argparse
import os

parser = argparse.ArgumentParser(
    description='Create sorting network(s)')

parser.add_argument("-v", action='count', default=0)
parser.add_argument("--all",
                    action='store_true',
                    default=False,
                    help='Create network for all known algorithms 2 - 64')
parser.add_argument("-d",
                    "--dest",
                    default="networks",
                    help='Destination directory')
parser.add_argument('-n',
                    '--network_size',
                    action='store',
                    default="0",
                    help='Size of network to generate')
parser.add_argument('-a',
                    '--network_alg',
                    action='store',
                    default="",
                    help='Algorithm to use for network generation')
parser.add_argument('-g',
                    '--network_generator',
                    action='store',
                    default="create_networks.pl",
                    help='Algorithm to use for network generation')

args = parser.parse_args()

dest_dir = args.dest
do_all = args.all
network_size = int(args.network_size)
network_alg = args.network_alg
generator = args.network_generator
verbosity = args.v

n_max = 32
n_min = 2

algorithms = [
    'bosenelson', 'hibbard', 'batcher', 'bitonic', 'oddevenmerge', 'bubble',
    'oddeventrans', 'balanced'
]

if dest_dir == "":
    print("Error invalid dest")
    exit(-1)

if generator == "":
    print("Error invalid generator")
    exit(-1)

if do_all is False:
    if network_size < n_min or network_size > n_max:
        print("Error invalid size")
        exit(-1)

    if network_alg not in algorithms:
        print("Error invalid algorithm")
        exit(-1)


def find_dest(cur_dest, levels):
    test_path = cur_dest + "/" + dest_dir
    if os.path.isdir(test_path) is True:
        return test_path
    if levels == 1:
        return dest_dir
    return find_dest(cur_dest + "/..", levels + 1)


def mkdir(dirname):
    if os.system("mkdir -p {}".format(dirname)) != 0:
        print("Error making directory")
        exit(-1)


def execution_method():
    if os.access(generator, os.X_OK) is True:
        return "./"
    elif ".pl" in generator:
        return "perl "
    elif ".py" in generator:
        return "python3 "
    elif ".sh" in generator:
        return "bash "
    else:
        print("Unable to determine generator execution method")
        exit(-1)


def create_network(n, alg):
    out_file = dest_dir + "/" + alg
    mkdir(out_file)
    out_file += "/" + alg + "-" + str(n)

    executor = execution_method()

    cmd = "{}{} {} {} > {}".format(executor, generator, str(n), alg, out_file)
    if verbosity > 0:
        print(cmd)
    os.system(cmd)


dest_dir = find_dest(".", 0)
mkdir(dest_dir)

if do_all is True:
    for alg in algorithms:
        for n in range(n_min, n_max + 1):
            create_network(n, alg)
else:
    create_network(n, alg)
