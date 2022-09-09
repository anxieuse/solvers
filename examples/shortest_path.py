import sys
import time
import datetime
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import networkx as nx
from collections import defaultdict

# Global variables for output
log_file = None
only_solver_time = 0
SolutionNotFound = False
solver_gap = 0
lower_bound = 0
    
# Set the solver parameters
def gen_ilpsolver_options(opt, solver, time_limit, mipgap=0.01):
    if solver =='cplex':
        opt.options["threads"]=1
        opt.options["timelimit"]=time_limit
        opt.options["mipgap"]=mipgap
    if solver =='cbc':
        opt.options["ratio"]=mipgap
        opt.options["seconds"]= time_limit

# Run IP solver and check the solution        
def run_ilpsolver(model, solver, time_lim, mip_gap):
    global only_solver_time
    global SolutionNotFound
    global solver_gap
    global lower_bound
    # Init solver parameters
    opt = SolverFactory("/home/huawei/Documents/Pyomo/cbc-linux64/cbc")
    gen_ilpsolver_options(opt, solver, time_lim, mip_gap)

    # Solve the model
    solverstart=time.time()
    results = opt.solve(model, tee=True)
    solvertime = time.time() - solverstart
    only_solver_time += solvertime
    print("Solver time: ", solvertime)
    print("Solver time: ", solvertime, file=log_file)
    if (results.solver.status == SolverStatus.ok) or (results.solver.status == SolverStatus.aborted): #and (results.solver.termination_condition == TerminationCondition.optimal):
        if results.solver.termination_condition != TerminationCondition.optimal and results.solver.termination_condition != TerminationCondition.maxTimeLimit:
            print ("ERROR: solver condition: ", results.solver.termination_condition)
            SolutionNotFound = True
        gap = mip_gap
        if (len(results.solution) > 0):
            gap = results.solution(0).gap
        model.solutions.load_from(results)
        objvalue = value(model.objective)
        #objvalue=value(instance.OBJ)
        if objvalue == None:
            print ('objvalue = None')
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
            print('Obj = ', objvalue, '; gap = ', gap, '(', solver_gap, '%)', file=log_file)
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
def shortest_path(digraph, source, target, solver, time_lim, mip_gap):
    global only_solver_time
    
    # Prepare dictionaries
    successors = defaultdict(set)
    predecessors = defaultdict(set)
    edges = {}
    edgeID = 0
    for e in digraph.edges.data('weight'):
        edges[edgeID] = e
        successors[e[0]] |= {edgeID}
        predecessors[e[1]] |= {edgeID}
        edgeID += 1
    print ('Edges:', edges)
    print ('Predecessors:', predecessors) 
        
    # Declare variables
    model = ConcreteModel()
    model.edge = Var(edges.keys(), within=Binary) #NonNegativeIntegers) #Reals) 
    print("Nodes =", digraph.number_of_nodes(), "Edges =", len(edges), file=log_file)

    # Declare constraints
    # Index can't be a tuple!    
    def FlowDifference(model, node):
        return sum(model.edge[eID] for eID in successors[node]) \
            - sum(model.edge[eID] for eID in predecessors[node])

    # Flow balance rule
    def FlowBalance_rule(model, node):
        if node == source:
            return FlowDifference(model, node) == 1
        elif node == target:
            return FlowDifference(model, node) == -1
        return FlowDifference(model, node) == 0
    model.FlowBalance = Constraint(digraph.nodes(), rule=FlowBalance_rule)

    # Objective
    def value_rule(model):
        return sum( model.edge[eID] * edges[eID][2] for eID in edges)
    model.objective = Objective( rule=value_rule, sense=minimize)

    # Solve
    if not run_ilpsolver(model, solver, time_lim, mip_gap):
        return None
    # Continue if Solver status is ok 

    # Get the results
    result = {}
    obj = 0
    for eID in edges:
        val = int(round(value(model.edge[eID])))
        if val > 0:
            e = edges[eID]
            obj += val * e[2]
            result[e[0]] = e[1]
    path = [source]
    while path[-1] in result:
        path.append(result[path[-1]])
    print('Path length = ', obj, 'Path:', path, file=log_file)
    return path

#    
# Main part
#   

starttime = time.time()

# Init options
solver = "cbc"
time_lim = 60
mip_gap = 0.01
log_file = open('shortes_path.log', "w+")

# Init a graph
G = nx.Graph()
G.add_edge(0,1,weight=1.0)
G.add_edge(1,2,weight=2.0)
G.add_edge(2,3,weight=3.0)
G.add_edge(3,5,weight=4.0)
G.add_edge(0,4,weight=5.0)
G.add_edge(4,5,weight=6.0)
G.add_edge(0,3,weight=7.0)
# Make a directed graph
H = nx.DiGraph(G)

# Find the shortest path
source = 0
target = 5
path = shortest_path(H, source, target, solver, time_lim, mip_gap)
all_time = time.time() - starttime
path_length = sum(H.edges[nd1,nd2]['weight'] for nd1, nd2 in zip(path[:-1], path[1:]))
print ("Shortest path:", path, 'Length =', path_length)
print ("Total time:", all_time, ', solver:', only_solver_time)

# Write info to a public log-file
with open("protocol.log", "at") as pf:
    print(datetime.datetime.now(), sys.argv[0], file=pf)
    if SolutionNotFound:
        print ("ERROR: solution is not found!", file=pf)
    print ("  Solver:", solver, ', Mipgap =', mip_gap, ', Time lim =', time_lim, file=pf)
    print ("  Shortest path:", path, 'Length =', path_length, file=pf)
    print ("  Time:", all_time, ', Solver time:', only_solver_time, file=pf)
