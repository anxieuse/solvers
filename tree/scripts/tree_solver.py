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
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import csv

# Global variables for output
log_file = None
only_solver_time = 0
SolutionNotFound = False
solver_gap = 0
lower_bound = 0

# Set the solver parameters


def gen_ilpsolver_options(opt, solver, time_limit, mipgap=0.01):
    if solver == 'cplex':
        opt.options["threads"] = 1
        opt.options["timelimit"] = time_limit
        opt.options["mipgap"] = mipgap
    if solver == 'cbc':
        opt.options["ratio"] = mipgap
        opt.options["seconds"] = time_limit

# Run IP solver and check the solution


def run_ilpsolver(model, solver, time_lim, mip_gap, logger):
    global only_solver_time
    global SolutionNotFound
    global solver_gap
    global lower_bound
    # Init solver parameters
    #opt = SolverFactory("/home/huawei/Documents/Pyomo/cbc-linux64/cbc")
    opt = SolverFactory("/home/leo/var/cbc")
    gen_ilpsolver_options(opt, solver, time_lim, mip_gap)

    # Solve the model
    solverstart = time.time()
    results = opt.solve(model, tee=True, load_solutions=True)
    logger.info(f'Objective value: {model.objective.expr()}')
    # results = opt.solve(model, tee=
    # print(results)

    solvertime = time.time()-solverstart
    only_solver_time += solvertime
    print("Solver time: ", solvertime)
    print("Solver time: ", solvertime, file=log_file)
    print(f'Problem solved: {results.solver.termination_condition}')
    print(f'{results.solver.status=}')
    # and (results.solver.termination_condition == TerminationCondition.optimal):
    if (results.solver.status == SolverStatus.ok) or (results.solver.status == SolverStatus.aborted):
        if results.solver.termination_condition != TerminationCondition.optimal and results.solver.termination_condition != TerminationCondition.maxTimeLimit:
            print("ERROR: solver condition: ",
                  results.solver.termination_condition)
            SolutionNotFound = True
        gap = mip_gap
        if (len(results.solution) > 0):
            gap = results.solution(0).gap
        # model.solutions.load_from(results)
        objvalue = value(model.objective)
        # objvalue=value(instance.OBJ)
        if objvalue == None:
            print('objvalue = None')
            objvalue = 1
        if solver == 'cbc' and gap != None:
            gap *= objvalue / (1 + gap)
        if solver == 'cplex' and gap != None:
            lower_bound = objvalue - gap
        if gap == None:
            print('Obj = ', objvalue, file=log_file)
        else:
            if objvalue == 0:
                solver_gap = 0
            else:
                solver_gap = 100 * gap / objvalue
            print('Obj = ', objvalue, '; gap = ', gap,
                  '(', solver_gap, '%)', file=log_file)
    else:
        if (results.solver.termination_condition == TerminationCondition.infeasible):
            print("INFEASIBLE")
            # Do something when model is infeasible
        else:
            # Something else is wrong
            print("Solver Status: ",  results.solver.status)
        SolutionNotFound = True
        return False
    return True

# MILP model
# The goal is to find subgraph with maximum total nodes and edges weight


def MaxWeightSubgraph(g, solver, time_lim, mip_gap, logger):
    global only_solver_time

    model = ConcreteModel()
    model.name = "MaxWeightSubgraph"
    model.g = g
    model.n = len(g.nodes)
    model.m = len(g.edges)
    model.nodes = list(g.nodes)
    model.edges = list(g.edges)
    model.node_weight = {n: g.nodes[n]['weight'] for n in g.nodes}
    model.edge_weight = {(src, trg): g.edges[src, trg]['weight'] for src, trg in g.edges}

    # Variables
    model.nodes_bin = Var(model.nodes, domain=Binary)
    model.edges_bin = Var(model.edges, domain=Binary)

    # Constraints
    # def edge_constraint_rule(model, src, trg):
    #     return model.edges_bin[src, trg] >= model.nodes_bin[src] + model.nodes_bin[trg] - 1
    # model.edge_constraint = Constraint(model.edges, rule=edge_constraint_rule)

    def edge_constraint_rule_src(model, src, trg):
        return model.edges_bin[src, trg] <= model.nodes_bin[src]
    model.edge_constraint_src = Constraint(model.edges, rule=edge_constraint_rule_src)

    def edge_constraint_rule_trg(model, src, trg): 
        return model.edges_bin[src, trg] <= model.nodes_bin[trg]
    model.edge_constraint_trg = Constraint(model.edges, rule=edge_constraint_rule_trg)

    # Objective
    model.objective = Objective(expr=
        sum(model.node_weight[n] * model.nodes_bin[n] for n in model.nodes) + sum(model.edge_weight[src, trg] * model.edges_bin[src, trg] for src, trg in model.edges), sense=maximize)

    # Run solver
    run_ilpsolver(model, solver, time_lim, mip_gap, logger)

    # Get the solution
    solution = []
    for e in model.edges:
        if model.edges_bin[e].value == 1:
            solution.append(e)
    return solution


time_suffix = datetime.today().strftime("%d-%m-%Y_%H-%M-%S")
script_abspath, script_name_w_ext = misc.get_script_abspath_n_name(__file__)
script_name = script_name_w_ext.split('.')[0]


def main():
    # Parse console arguments
    parser = argparse.ArgumentParser(description="Tree solver")
    parser.add_argument("-i", "--input", type=str,
                        default="./", help="Input dirpath")
    parser.add_argument("-t", "--timelimit", type=int,
                        default=60, help="Output dirpath")
    args = parser.parse_args()
    args.input = args.input.rstrip('/')

    # Create output folder
    if not os.path.exists(args.input + f'/res'):
        os.makedirs(args.input + f'/res')
    out_dirpath = args.input + f'/res/tree_{time_suffix}'
    if not os.path.isdir(out_dirpath):
        os.mkdir(out_dirpath)

    # Create logger
    logger = misc.get_logger(__name__, log_fpath=out_dirpath + f'/{script_name}.log')
    logger.info(f"Started {script_abspath=} with args: {args}")

    logger.info(f'Init solver options')
    solver = "cbc"
    mip_gap = 0.00
    logger.info(f'solver={solver}, mip_gap={mip_gap}, time_limit={args.timelimit}')

    logger.info(f'Read nodes.csv')
    nodes_df = pd.read_csv(args.input + '/nodes.csv', sep=',', header=0)

    logger.info(f'Read edges.csv')
    edges_df = pd.read_csv(args.input + '/edges.csv', sep=',', header=0)

    G = nx.from_pandas_edgelist(edges_df, 'node1', 'node2', ['id', 'weight'])
    for i, row in nodes_df.iterrows():
        G.nodes[row['id']]['weight'] = row['weight']

    logger.info(f'Start solver')
    start = time.time()
    maxWeightEdges = MaxWeightSubgraph(G, solver, args.timelimit, mip_gap, logger)
    runtime = time.time() - start

    logger.info(f"Runtime: {runtime:.2f} sec")
    logger.info(f"Solver time: {only_solver_time:.2f} sec")

    if SolutionNotFound:
        logger.info(f"Solution not found")
    else:
        edges_total_weight = sum([G.edges[e]['weight'] for e in maxWeightEdges])
        adjacent_nodes = set()
        for e in maxWeightEdges:
            adjacent_nodes.add(e[0])
            adjacent_nodes.add(e[1])
        nodes_total_weight = sum([G.nodes[n]['weight'] for n in adjacent_nodes])
        logger.info(f"Result edges: {maxWeightEdges}")
        logger.info(f"Result nodes: {adjacent_nodes}")
        logger.info(f"Total nodes weight: {nodes_total_weight}")
        logger.info(f"Total edges weight: {edges_total_weight}")
        logger.info(f"Total weight: {nodes_total_weight + edges_total_weight}")

        logger.info(f"Write subgraph.csv")
        maxWeightEdgeIds = [G.edges[e]['id'] for e in maxWeightEdges]
        subgraph_df = pd.DataFrame(maxWeightEdgeIds, columns=['edge'])
        subgraph_df.to_csv(out_dirpath + f'/subgraph.csv', index=False)


if __name__ == '__main__':
    main()
