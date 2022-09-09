import sys
import time
import datetime
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import networkx as nx

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
    #opt = SolverFactory("/home/huawei/Documents/Pyomo/cbc-linux64/cbc")
    opt = SolverFactory("/home/huawei/Documents/Pyomo/cbc-linux64/cbc")
    gen_ilpsolver_options(opt, solver, time_lim, mip_gap)

    # Solve the model
    solverstart=time.time()
    results = opt.solve(model, tee=True)
    solvertime = time.time()-solverstart
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
def stable_set(graph, solver, time_lim, mip_gap):
    global only_solver_time
    
    edges = {}
    edgeID = 0
    for e in graph.edges():
        edges[edgeID] = e
        edgeID += 1
        
    # Declare variables
    model = ConcreteModel()
    model.node = Var(graph.nodes(), within=Binary) #NonNegativeIntegers) #Reals) 
    print("Nodes =", len(graph.nodes()), file=log_file)

    # Declare constraints
    def adj_rule(model, edge):
        return model.node[edges[edge][0]] + model.node[edges[edge][1]] <= 1
    # Index can't be a tuple    
    model.adj = Constraint(edges.keys(), rule=adj_rule)
    print("Edges =", len(edges), file=log_file)

    # Objective
    def value_rule(model):
        return sum( model.node[a] for a in graph.nodes() )
    model.objective = Objective( rule=value_rule, sense=maximize)

    # Solve
    if not run_ilpsolver(model, solver, time_lim, mip_gap):
        return None
    # Continue if Solver status is ok 

    # Get the results
    result = []
    obj = 0
    for nd in graph.nodes():
        val = int(round(value(model.node[nd])))
        if val > 0:
            obj += val
            result.append(nd)
    #print('Stable number = ', obj)
    print('Stable number = ', obj, file=log_file)
    return result

#    
# Main part
#   

starttime = time.time()

# Init options
solver = "cbc"
time_lim = 60
mip_gap = 0.01
log_file = open('stable.log', "w+")

# Init a graph
G = nx.Graph()
G.add_edge(0,1)
G.add_edge(1,2)
G.add_edge(2,3)
G.add_edge(0,3)
G.add_edge(0,4)
G.add_edge(4,5)

# Find a stable set
res = stable_set(G, solver, time_lim, mip_gap)
all_time = time.time() - starttime
print ("Stable number =", len(res), "Stable set:", res)
print ("Total time:", all_time, ', solver:', only_solver_time)

# Write info to a public log-file
with open("protocol.log", "at") as pf:
    print(datetime.datetime.now(), sys.argv[0], file=pf)
    if SolutionNotFound:
        print ("ERROR: solution is not found!", file=pf)
    print ("  Solver:", solver, ', Mipgap =', mip_gap, ', Time lim =', time_lim, file=pf)
    print ("  Snum =", len(res), ", set:", res, file=pf)
    print ("  Time:", all_time, ', Solver time:', only_solver_time, file=pf)
