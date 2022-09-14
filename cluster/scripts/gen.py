from ast import parse
import functools as ft
from scipy import stats
import random
import math
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import time
import subprocess
import os
import sys
import argparse 
from datetime import datetime
import tqdm
import pylab as plb
from leo import misc
import multiprocessing as mp
import shutil

time_suffix = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
script_abspath, script_name_w_ext = misc.get_script_abspath_n_name(__file__)
script_name = script_name_w_ext.split('.')[0]
script_dirpath = os.path.dirname(script_abspath)

def main():
    log = misc.get_logger(__name__)
    log.info(f"Started {script_abspath=}")

    log.info('Parse console args')
    parser = argparse.ArgumentParser(description="Generate test")

    parser.add_argument("-s", "--servers", type=int, default = 4, help="Servers number")
    parser.add_argument("-vm", "--virtualmachines", type=int, default = 16, help="Virtual machines number")
    parser.add_argument("-mnc", "--mincap", type=int, default = 0, help="Minimum server capacity")
    parser.add_argument("-mxc", "--maxcap", type=int, default = 4, help="Maximum server capacity")
    parser.add_argument("-o", "--output", type=str, default = "../tests", help="Output dirpath")

    args = parser.parse_args()
    args.output = args.output.rstrip('\\')
    log.info(f"Console args: {args}")

    log.info('Create output directory')
    out_dirpath = args.output + f'/test_{time_suffix}'
    if not os.path.isdir(out_dirpath):
        os.mkdir(out_dirpath)

    log.info(f'Generate servers')
    servers = []
    for i in range(args.servers):
        servers.append(random.randint(args.mincap, args.maxcap))

    log.info(f'Write servers.csv')
    with open(out_dirpath + '/servers.csv', 'w') as f:
        f.write('id,capacity\n')
        for id, cap in enumerate(servers):
            f.write(f'{id},{cap}\n')

    log.info(f'Generate virtual machines')
    vms = []
    for i in tqdm.tqdm(range(args.virtualmachines)):
        first_server = random.randint(0, args.servers - 1)
        second_server = random.randint(0, args.servers - 1)
        while second_server == first_server:
            second_server = random.randint(0, args.servers - 1)
        vms.append((first_server, second_server))
    
    log.info(f'Write vms.csv')
    with open(out_dirpath + '/vm.csv', 'w') as f:
        f.write('id,server1,server2\n')
        for id, (first_server, second_server) in enumerate(vms):
            f.write(f'{id},{first_server},{second_server}\n')


if __name__ == '__main__':
    main() 