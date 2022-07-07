import datetime
import itertools
import math
from itertools import combinations

import gurobipy as gp
import numpy as np
import pandas as pd
import streamlit as st
from gurobipy import GRB

from utils import *


def idle_time(truck_i_sumation_time,
    truck_j_sumation_time,
    truck_i_tostock_time,
    truck_j_tostock_time,
    loader_i_to_j_stock_time,
    ):
    """
    Compute the idle time of an assignment of i,j trucks, taking into account
    different stocks and loaders for i,j trucks.
    idletime_(i,j) = to_stock_i+arrivaltime_ij+ timeshovel_i_to_j-to_stock_j
    change_arrival_(i,j)= time to wait at the entrance because j is assigned before i
        If idletime_(i,j) <0:  0 + change_arrival_(i,j)
        Else: idletime_(i,j): idletime_(i,j) + change_arrival_(i,j) 

    Args:
        truck_i(str): Truck i of a combination of (i,j)
        stock_i(str): Truck j of a combination of (i,j) trucks.
        loader_i(str): truck: time to get to the entrance (no long numbers).
        truck_j(str): truck: time to be at stock from ent. and get loaded.
        stock_j(str): stock:stock. Time for a loader to travel for stock to stock.
        loader_j(str)
    Returns:
        idle time: sum of changing time of arrivals (res)
                    +its time consequence with destinat.
    """
    #print(truck_i)
    time_arrival = 0
    if truck_i_sumation_time < truck_j_sumation_time:
        # consider movement of i until j gets to entrance
        time_arrival = truck_j_sumation_time-truck_i_sumation_time
    change_arrival = 0
    if truck_i_sumation_time >= truck_j_sumation_time:
        # if i joins facility even though j arrived first
        change_arrival = truck_i_sumation_time - truck_j_sumation_time
    idle_time_v = truck_i_tostock_time-time_arrival+loader_i_to_j_stock_time-truck_j_tostock_time
    if idle_time_v <0:
        return 0 + change_arrival
    else:
        return idle_time_v + change_arrival
def run_schedule(list_one,truck_times_with_recognizer, permut_list_two):
    """
    Gives the best sequence of assignments, (integer_1,integer_2), that represent the current 
    truck, stock and loader and its following assignment. This minimizes the overall 
    idle time of trucks.

    Args:
        list_one(list): Integer numbers that correspond to a (truck, stock, loader)
        truck_times_with_recognizer(dict): pairs of integers with its idle time, i.e., (1,2):20
        permut_list_two(list of lists): permutations of integers from list_two

    Returns:
        assignment (list of integers): Best sequence of assignments, i.e, [1,3,4,2] 
        minval(int): Total idle time of the best assignments 
    """ 
    minval = 1e9
    for x in permut_list_two:
        global trucks_
        # Adding one permutation from list_two 
        trucks_ = list_one+ list(x)
        truck_times = {k:v for k,v in truck_times_with_recognizer.items() if k[0] in trucks_ and k[1] in trucks_}
        # Create an empty model 
        m = gp.Model()
        # Create binary decision variables based on i,j truck combinations
        vars = m.addVars(truck_times.keys(), obj=truck_times, vtype=GRB.BINARY, name='x')
        # Create constraints; there must be one sequence going to a truck i
        # and one goind out from truck i
        m.addConstrs(vars.sum(c,'*') == 1 for c in trucks_)
        m.addConstrs(vars.sum('*',c) == 1 for c in trucks_)  
        # Clear logs
        m.Params.LogToConsole = 0
        m._vars = vars
        # Eliminate cycles
        m.Params.lazyConstraints = 1
        # Solving the model
        m.optimize(subtourelim)
        vals = m.getAttr('x', vars)
        selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)
        tour = subtour(selected) # Get the tour
        min_time = m.getObjective().getValue()
        if minval > min_time:
            minval = min_time
            assignment = tour
    return assignment, minval

    
# Part of solving the TSP
def subtour(edges):
    unvisited = trucks_[:]
    cycle = trucks_[:] # Dummy - guaranteed to be replaced
    while unvisited:  # true if list is non-empty
        thiscycle = []
        neighbors = unvisited
        while neighbors:
            current = neighbors[0]
            thiscycle.append(current)
            unvisited.remove(current)
            neighbors = [j for i, j in edges.select(current, '*')
                         if j in unvisited]
        if len(thiscycle) <= len(cycle):
            cycle = thiscycle # New shortest subtour
    return cycle

def subtourelim(model,where):
    if where == GRB.Callback.MIPSOL:
        # make a list of edges selected in the solution
        vals = model.cbGetSolution(model._vars)
        selected = gp.tuplelist((i, j) for i, j in model._vars.keys()
                             if vals[i, j] > 0.5)
        tour = subtour(selected)
        if len(tour) < len(trucks_):
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(gp.quicksum(model._vars[i,j] + model._vars[j,i] for i, j in combinations(tour, 2))
                         <= len(tour)-1)

