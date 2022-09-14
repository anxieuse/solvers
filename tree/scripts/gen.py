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
import rand_tree as rand_tree

time_suffix = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
script_abspath, script_name_w_ext = misc.get_script_abspath_n_name(__file__)
script_name = script_name_w_ext.split('.')[0]
script_dirpath = os.path.dirname(script_abspath)

def main():
    log = misc.get_logger(__name__)
    log.info(f"Started {script_abspath=}")

    log.info('Parse console args')
    parser = argparse.ArgumentParser(description="Generate test")

    parser.add_argument("-md", "--maxdepth", type=int, default = 4, help="Max depth of tree")
    parser.add_argument("-nc", "--nodechildren", type=int, default = 15, help="Maximum mber of children of each node")
    parser.add_argument("-bp", "--branchingprobability", type=float, default = 0.8, help="Probability of branching")
    parser.add_argument("-o", "--output", type=str, default = "../tests", help="Output dirpath")

    args = parser.parse_args()
    args.output = args.output.rstrip('\\')
    log.info(f"Console args: {args}")

    log.info('Create output directory')
    out_dirpath = args.output + f'/test_{time_suffix}'
    if not os.path.isdir(out_dirpath):
        os.mkdir(out_dirpath)

    log.info(f'Generate random tree of depth {args.maxdepth}')
    node_prefixes, edges = rand_tree.gen_random_tree(args.maxdepth, args.nodechildren, args.branchingprobability, out_dirpath + '/tree.png')

    log.info(f'Write edges.csv')
    with open(out_dirpath + '/edges.csv', 'w') as f:
        f.write('id,node1,node2,weight\n')
        for id, (u, v) in enumerate(edges):
            w = random.randint(1, 100)
            f.write(f'{id},{u},{v},{w}\n')
            edges[id] = (u, v, w)
    
    log.info(f'Write nodes.csv')
    nodes = []
    with open(out_dirpath + '/nodes.csv', 'w') as f:
        f.write('id,weight\n')
        for id in range(len(node_prefixes) + 1):
            w = -random.randint(1, 100)
            f.write(f'{id},{w}\n')
            nodes.append((id, w))

    if len(edges) <= 100:
        log.info(f'Write tree.png')
        g = nx.from_edgelist([(u, v) for u, v, w in edges])
        pos = nx.nx_pydot.graphviz_layout(g, prog="dot")
        # add node labels as (id, weight)
        node_labels = {node: f'({node}, {weight})' for node, weight in nodes}
        # add edge labels as (id, weight)
        edge_labels = {(u, v): f'({id}, {weight})' for id, (u, v, weight) in enumerate(edges)} 
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8, font_color='red')
        nx.draw_networkx_labels(g, pos, labels=node_labels, font_size=8, font_color='red') 
        nx.draw(g, pos)
        plt.savefig(out_dirpath + '/tree.png')

    log.info(f'Nodes: {len(nodes)}')
    log.info(f'Edges: {len(edges)}')


if __name__ == '__main__':
    main() 