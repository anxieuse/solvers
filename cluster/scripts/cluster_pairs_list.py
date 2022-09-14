import functools as ft
from sqlite3 import connect
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


def run_ilpsolver(model, solver, time_lim, mip_gap):
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
    # results = opt.solve(model, tee=True, load_solutions=False)
    # results = opt.solve(model, tee=True)
    # model.pprint()
    # model.constraint.pprint()
    # # print results content
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
# The goal is to assign each vm to a cluster and maximize the number of assigned vm


def VM2Server(server_capacities, vm_to_server_pair,
              solver, time_lim, mip_gap,
              logger):
    global only_solver_time

    logger.info("VM2Server: server_capacities = %s, vm_to_server_pair = %s",
                server_capacities, vm_to_server_pair)

    model = ConcreteModel()
    model.server_capacities = server_capacities
    model.vm_to_server_pair = vm_to_server_pair
    
    model.num_vms = len(vm_to_server_pair)
    model.num_servers = len(server_capacities)

    model.server_ids = range(model.num_servers)
    model.server_indices = range(2)
    model.vm_ids = range(model.num_vms)

    logger.info(f'Create model: {model.num_servers} servers, {model.num_vms} vms')

    logger.info(f'Create variables')
    model.vm2server = Var(model.vm_ids, model.server_indices, domain=Binary)

    logger.info(f'Create objective')
    model.objective = Objective(
        expr=sum(model.vm2server[vm_id, server_idx] 
            for vm_id in model.vm_ids for server_idx in model.server_indices), sense=maximize)

    logger.info(f'Create constraints')

    def vm_one_serv_rule(model, vm_id):
        return sum(model.vm2server[vm_id, server_idx]
                   for server_idx in model.server_indices) <= 1

    model.vm2serverer = Constraint(model.vm_ids, rule=vm_one_serv_rule)

    # capacity rule for each server: number of assigned vm <= server capacity
    def server_cap_rule(model, serv_id):
        connected_vms = 0
        # lol = []
        for vm_id in model.vm_ids:
            if vm_to_server_pair[vm_id][0] == serv_id:
                serv_idx = 0
            elif vm_to_server_pair[vm_id][1] == serv_id:
                serv_idx = 1
            else:
                continue
            # lol.append(model.vm2server[vm_id, serv_idx])
            connected_vms += model.vm2server[vm_id, serv_idx]
        # connected_vms = sum(lol)
        print(f'server_cap_rule: serv_id = {serv_id}, connected_vms = {connected_vms}')
        return connected_vms <= model.server_capacities[serv_id]

    model.capacity = Constraint(model.server_ids, rule=server_cap_rule)

    logger.info(f'Run solver')
    run_ilpsolver(model, 'cbc', time_lim, mip_gap)

    vm_to_server = []
    for i in model.vm_ids:
        for j in model.server_indices:
            if value(model.vm2server[i, j]) == 1:
                server_id = vm_to_server_pair[i][j]
                vm_to_server.append((i, server_id))
                break

    return vm_to_server


time_suffix = datetime.today().strftime("%d-%m-%Y_%H-%M-%S")
script_abspath, script_name_w_ext = misc.get_script_abspath_n_name(__file__)
script_name = script_name_w_ext.split('.')[0]


def main():
    # Parse console arguments
    parser = argparse.ArgumentParser(description="Cluster solver")
    parser.add_argument("-i", "--input", type=str,
                        default="./", help="Input dirpath")
    parser.add_argument("-t", "--timelimit", type=int,
                        default=60, help="Output dirpath")
    args = parser.parse_args()
    args.input = args.input.rstrip('/')

    # Create output folder
    out_dirpath = args.input + f'/vm2server_{time_suffix}'
    if not os.path.isdir(out_dirpath):
        os.mkdir(out_dirpath)

    # Create logger
    logger = misc.get_logger(
        __name__, log_fpath=out_dirpath + f'/{script_name}.log')
    logger.info(f"Started {script_abspath=} with args: {args}")

    logger.info(f'Init solver options')
    solver = "cbc"
    mip_gap = 0.01
    logger.info(
        f'solver={solver}, mip_gap={mip_gap}, time_limit={args.timelimit}')

    logger.info(f'Read servers.csv')
    servers_df = pd.read_csv(args.input + '/servers.csv', sep=',', header=0)
    servers = servers_df['capacity'].tolist()
    logger.info(f'servers={servers}')

    logger.info(f'Read vm.csv')
    vms_df = pd.read_csv(args.input + '/vm.csv', sep=',', header=0)
    vms = list(zip(vms_df['server1'].tolist(), vms_df['server2'].tolist()))
    logger.info(f'vms={vms}')

    logger.info(f'Start solver')
    start = time.time()
    vm_to_server = VM2Server(servers, vms, solver,
                             args.timelimit, mip_gap, logger)
    runtime = time.time() - start

    logger.info(f"Runtime: {runtime:.2f} sec")
    logger.info(f"Solver time: {only_solver_time:.2f} sec")

    if SolutionNotFound:
        logger.info(f"Solution not found")
    else:
        logger.info(f"Result: {vm_to_server}")


if __name__ == '__main__':
    main()
