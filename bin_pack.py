import sys
import time
import datetime
import pandas as pd
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition

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
    solverstart = time.time()
    #results = opt.solve(model, tee=True, load_solutions=False)
    results = opt.solve(model, tee=True)
    solvertime = time.time() - solverstart
    only_solver_time += solvertime
    #print("Solver time: ", solvertime)
    #print("Solver time: ", solvertime, file=log_file)
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
def bin_packing(item_sizes, bin_size, solver, time_lim, mip_gap):
    model = ConcreteModel()
    # Declare variables
    items_num = len(item_sizes)
    indexes = list(range(items_num))
    two_indexes = [(i,j) for i in indexes for j in indexes]
    model.y = Var(indexes, within=Binary) #NonNegativeIntegers) #Reals) 
    model.x = Var(two_indexes, within=Binary) #NonNegativeIntegers) #Reals) 

    # The sum of items can't be greater then bin_size
    def size_rule(model, bin):
        return sum (item_sizes[item] * model.x[(item, bin)] for item in indexes) <= bin_size * model.y[bin]
    model.size = Constraint(indexes, rule=size_rule)

    # Every item must be packed
    def item_rule(model, item):
        return sum (model.x[(item, bin)] for bin in indexes) == 1
    model.item = Constraint(indexes, rule=item_rule)

    # Objective
    def value_rule(model):
        return sum( model.y[bin] for bin in indexes)
    model.objective = Objective( rule=value_rule, sense=minimize)

    # Solve
    if not run_ilpsolver(model, solver, time_lim, mip_gap):
        return None
    # Continue if Solver status is ok 

    # Get the results
    result = {}
    obj = 0
    for bin in indexes:
        val = int(round(value(model.y[bin])))
        if val > 0:
            obj += val
            result[bin] = []
            for item in indexes:
                is_item = int(round(value(model.x[(item, bin)])))
                if is_item > 0:
                    result[bin].append(item)
    print('Bin number = ', obj, file=log_file)
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

# Input
item_sizes = [1,2,3,4,5,6,7,10,11,15,19]
bin_size = max(item_sizes) + 1

# Find a stable set
res = bin_packing(item_sizes, bin_size, solver, time_lim, mip_gap)
all_time = time.time() - starttime

print ("Bin size =", bin_size, ", Items sizes:", item_sizes)
print ("Bins number =", len(res))
print ("Bin packing:")
for j, bin_items in enumerate(res.values()):
    print (j, ": ", item_sizes[bin_items[0]], end='')
    for item in bin_items[1:]:
        print (" +", item_sizes[item], end='')
    print (" <=", bin_size)
        
print ("Total time:", all_time, ', solver:', only_solver_time)

# Write info to a public log-file
with open("protocol.log", "at") as pf:
    print(datetime.datetime.now(), sys.argv[0], file=pf)
    if SolutionNotFound:
        print ("ERROR: solution is not found!", file=pf)
    print ("  Solver:", solver, ', Mipgap =', mip_gap, ', Time lim =', time_lim, file=pf)
    print ("  Bnum =", len(res), ", packing:", res, file=pf)
    print ("  Time:", all_time, ', Solver time:', only_solver_time, file=pf)
