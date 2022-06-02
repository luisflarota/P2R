import datetime
import itertools
import math
from itertools import combinations

import gurobipy as gp
import numpy as np
import pandas as pd
import streamlit as st
from gurobipy import GRB

from back import *
from utils import *





def idle_time(i,j,truck_sumation_time,truck_tostock_time,shovel_sttost):
    """
    Compute the idle time of an assignment of i,j trucks.
    idletime_(i,j) = tostocktime_i+arrivaltime_ij+ timeshovel_j- tostocktime_j
    change_arrival_(i,j)= time to wait at the entrance because j is assigned before i
        If idletime_(i,j) <0:  0 + change_arrival 
        Else: idletime_(i,j): idletime_(i,j) + change_arrival 

    Args:
        i(str): Truck i of a combination of (i,j)
        j(str): Truck j of a combination of (i,j) trucks.
        truck_sumation_time(dict): truck: time to get to the entrance (no long numbers).
        truck_tostock_time(dict): truck: time to be at stock from ent. and get loaded.
        shovel_sttost(dict): stock:stock. Time for a loader to travel for stock to stock.
    Returns:
        idle time: sum of changing time of arrivals (res)
                    +its time consequence with destinat.
    """
    if i == 'dummy' or j == 'dummy':
        return 0
    else:
        time_arrival = 0
        if truck_sumation_time[i] < truck_sumation_time[j]:
            # consider movement of i until j gets to entrance
            time_arrival = truck_sumation_time[j]-truck_sumation_time[i]
        change_arrival = 0
        if truck_sumation_time[i] >= truck_sumation_time[j]:
            # if i joins facility even though j arrived first
            change_arrival = truck_sumation_time[i] - truck_sumation_time[j]
        idle_time_v = truck_tostock_time[i]-time_arrival+shovel_sttost[i,j]-truck_tostock_time[j] 
        if idle_time_v <0:
            return 0 + change_arrival
        else:
            return idle_time_v + change_arrival
# This is part of the objective function. Idle time part ends here

def scheduling(truck_times,perm_t_stm,prospect_schedules, trucks):
    """
    Gives the best sequence of assignments, (truck,stock), that minimizes the overall 
    idle time of trucks.

    Args:
        requirements(df): customer requirement (inner join with our stocks).
        materials_st(dict): materials:stock(s). Based on what was set in the frontend
        load_properties(dict): stocks, #eq, velo, payl, ctime for each type of loader
        nodes_file(csv): file from VGG Annotator
    Returns:
        selected_truck_stock (list of tuples): Best sequence of assignments of 
        (truck, stock)
        minval(int): Total idle time of the best assignments 
        shovelini(str): Initial position of a loader - "stock1"
    """ 
    global trucks_
    trucks_ = trucks
    # Create an empty model 
    m = gp.Model()
    # Create binary decision variables based on i,j truck combinations
    vars = m.addVars(truck_times.keys(), obj=truck_times, vtype=GRB.BINARY, name='x')
    # Create constraints; there must be one sequence going to a truck i
    # and one goind out from truck i
    m.addConstrs(vars.sum(c,'*') == 1 for c in trucks)
    m.addConstrs(vars.sum('*',c) == 1 for c in trucks)  
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
    # Updating the prospect_schedules
    prospect_schedules[tuple(perm_t_stm)] = [tour, m.getObjective().getValue()]
    
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

