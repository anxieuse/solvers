from ast import parse
from asyncio.log import logger
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

    log.info(f'Load edges.csv')
    edges = pd.read_csv(args.input + '/edges.csv', index_col='id')

    log.info(f'Load nodes.csv')
    nodes = pd.read_csv(args.input + '/nodes.csv', index_col='id')

    log.info(f'Load subgraph.csv') 
    subgraph = pd.read_csv(args.results + '/subgraph.csv')

    nodes_num = nodes.shape[0]
    edges_num = edges.shape[0]
    for id, row in subgraph.iterrows():
      # check that edge id is correct
      if row['edge'] >= edges_num or row['edge'] < 0:
        log.error(f"Non-existing edge {row['edge']} on line {id}")  

    # check that there's no duplicate edges
    if subgraph['edge'].duplicated().any():
        log.error(f"Duplicate edges in subgraph")

    # calculate total weight of subgraph
    edges_weight = 0
    nodes_set = set()
    for id, row in subgraph.iterrows():
      edges_weight += edges.iloc[row['edge']]['weight']
      nodes_set.add(edges.iloc[row['edge']]['node1'])
      nodes_set.add(edges.iloc[row['edge']]['node2'])

    nodes_weight = 0
    for node in nodes_set:
      nodes_weight += nodes.iloc[node]['weight']
    
    log.info(f'Edges weight: {edges_weight}')
    log.info(f'Nodes weight: {nodes_weight}')
    log.info(f'Total weight: {edges_weight + nodes_weight}')
      

if __name__ == '__main__':
    main() 