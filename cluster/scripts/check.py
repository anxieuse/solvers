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
    parser = argparse.ArgumentParser(description="Check test results")

    parser.add_argument("-i", "--input", type=str, help="Input dirpath")
    parser.add_argument("-r", "--results", type=str, help="Results dirpath")

    args = parser.parse_args()
    args.input = args.input.rstrip('\\')

    out_dirpath = args.results
    if not os.path.isdir(out_dirpath):
        os.mkdir(out_dirpath)

    # Create logger
    log = misc.get_logger(__name__, log_fpath=out_dirpath + f'/{script_name}.log')
    log.info(f"Started {script_abspath=} with args: {args}")

    log.info(f'Load servers.csv')
    servers = pd.read_csv(args.input + '/servers.csv', index_col='id')

    log.info(f'Load vm.csv')
    virtual_machines = pd.read_csv(args.input + '/vm.csv', index_col='id')

    log.info(f'Load allocated.csv')
    allocated = pd.read_csv(args.results + '/allocated.csv')

    servers_num = servers.shape[0]
    virtual_machines_num = virtual_machines.shape[0]
    for id, row in allocated.iterrows():
      # check that server exists
      if row['server'] >= servers_num or row['server'] < 0:
        log.error(f"Virtual machine {id} is allocated to non-existing server {row['server']}")
      # check that vm exists
      if row['vm'] >= virtual_machines_num or row['vm'] < 0:
        log.error(f"Non-existing virtual machine {row['vm']} is allocated to server {row['server']}")
      # check that server has enough capacity
      numberOfVmAssignedToServer = allocated[allocated['server'] == row['server']].shape[0]
      if numberOfVmAssignedToServer > servers.loc[row['server']]['capacity']:
        log.error(f"Server {row['server']} has more virtual machines than its capacity")
      # check that vm is not allocated to more than one server
      if allocated[allocated['vm'] == row['vm']].shape[0] > 1:
        log.error(f"Virtual machine {row['vm']} is allocated to more than one server")

    # calc score
    allocated_vms = allocated.shape[0]
    log.info(f'Allocated {allocated_vms} virtual machines')


if __name__ == '__main__':
    main() 