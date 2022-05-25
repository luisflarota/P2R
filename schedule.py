import datetime
import itertools
import math
from itertools import combinations

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from back import *
from utils import *


def conv_custom_truck(ini_dict, muestra, materials):
    muestra_n = np.array(muestra)
    new_dict = {}
    new_dict['dummy']= [0,0,0]
    for t in ini_dict:
        if t in muestra_n[:,0]:
            st = [muestra_n[muestra_n[:,0] == t][0][1]]
        else:
            st = [st for st in materials if materials[st] == ini_dict[t][0]]
        new_dict[t]= st + ini_dict[t][1:]
    return new_dict


# This is part of the objective function. Idle time part here
def summation_if_arrival(i,j,truck_sumation_time):
    if truck_sumation_time[i] >= truck_sumation_time[j]:
        return 0
    else:
        return truck_sumation_time[j]-truck_sumation_time[i]
    
def summation_if_sum(i,j,truck_sumation_time):
    if truck_sumation_time[i] >= truck_sumation_time[j]:
        return truck_sumation_time[i] - truck_sumation_time[j]
    else:
        return 0
def time_(i,j,truck_sumation_time,truck_tostock_time,shovel_sttost):
    if i == 'dummy' or j == 'dummy':
        return 0
    else:
        #print(i,j,truck_tostock_time[i],summation_if_arrival(i,j,truck_sumation_time),
         #    shovel_sttost[i,j],truck_tostock_time[j])
        res = truck_tostock_time[i]-summation_if_arrival(i,j,truck_sumation_time)+shovel_sttost[i,j]-truck_tostock_time[j] 
        if res <0:
            return 0 + summation_if_sum(i,j,truck_sumation_time)
        else:
            return res + summation_if_sum(i,j,truck_sumation_time)
# This is part of the objective function. Idle time part ends here

def scheduling(requirements,materials):
    """
    This functions schedules based on what is required to where the requirement is. This
    will give you the best sequence of assignments that minimized the overall idle time 
    for trucks
    **input:(1) requirements(df)  : customer requirement (inner join with our stocks).
            (2) materials(dict)    : materials:stock. Based on what was set in the frontend
    **output
            (1) result (list of tuples) : Best sequence of assignments of (shovel, stock)
            (2) minval(int) : Total idle time of the best assignments 
            (3) shovelini(str)   : Initial position of a loader - "stock1"
    """    

    call_class = getNodes('new_nodes.csv')
    data = call_class.read_csv_2()
    graph = data[2]
    convert_kmh_msec = 0.28
    truck_velocity = 20 * convert_kmh_msec
    fixed = {'start': 'a0','entrance':'a5','stock1':'d2','stock2':'e1',
            'stock3':'f1','stock4':'g3','stock5':'h2','stock6':'k3',
            'stock7':'l5','stock8':'m2', 'scale':'c22', 'end':'c24'}
    loaders_times = {'loader':[10*0.28, 12, 30],
                'hopper': [5, 1],
                'excav':[3*0.28, 12, 20]}
    stocks = {k:v for k,v in fixed.items() if 'stock' in k}
    requirement = requirements
    requirement['Date']= requirement.apply(
                        lambda r : datetime.datetime.combine(r['Date'],r['Time']),1)
    requirement['Epoch'] = (requirement['Date']
            - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    requirement['Epoch'] = requirement['Epoch'] + 6*3600
    requirement = requirement.sort_values('Epoch')
    min_t= min(requirement['Epoch'])
    requirement['dect'] = np.diff([min_t]+list(requirement['Epoch']))
    for loader in np.unique(requirement['TypeLoader']):
        data_m = requirement[requirement['TypeLoader'] == loader]
        if loader == 'hopper':
            scheds[loader] = data_m['Truck']
        else:
            #print(loader)
            shovel_velocity = loaders_times[loader][0]
            shovel_payload = loaders_times[loader][1]
            shovel_cycletime = loaders_times[loader][2]
            scheds = {}
            cust_req = {x[2]:[x[3],x[10], x[6]]   for x in np.array(data_m)}
            global trucks
            trucks = list(cust_req.keys())
            n = len(trucks)
            dict_iterac = {truck:list(data_m[data_m['Truck'] == truck]['Dest'])[0] for truck in np.unique(data_m['Truck'])}
            keys, values = zip(*dict_iterac.items())
            permutations_dicts = [list(zip(keys, v)) for v in itertools.product(*values)]
            #print(permutations_dicts)
            for perm_d in permutations_dicts:
                new_ct = conv_custom_truck(cust_req, perm_d, materials)
                sorted_times = {key:new_ct[key][1] for i,key in  enumerate(new_ct)}
                sorted_times = dict(sorted(sorted_times.items(), key=lambda item: item[1]))
                new_ct = {key: new_ct[key]  for key in sorted_times}
                # List of stocks for the trucks coming
                stock_assigned = np.array(list(new_ct.values()), dtype=object)[:,0]
                # Index trucks based on stocks: {stok: truck, tonnage}
                #print(n)
                #duration for each surgery in minutes
                truck_destination = {k:v[0] for k,v in new_ct.items() if k != 'dummy'}
                truck_arrival_time = {k:v[1] for k,v in new_ct.items() if k != 'dummy'}
                truck_sumation_time ={k:sum(list(truck_arrival_time.values())[:(ind)]) for 
                                            ind,k in enumerate(new_ct) if k != 'dummy'}
                truck_tostock_time = {k: round(
                    float(Dijkstra(graph,fixed['entrance'],
                                fixed[new_ct[k][0]])[2])/truck_velocity) + int(
                    math.ceil(new_ct[k][2]/shovel_payload)*shovel_cycletime)
                                    for k,v in new_ct.items() if k != 'dummy'}
                truck_toload_time = {k: int(math.ceil(new_ct[k][2]/shovel_payload)*shovel_cycletime)
                                    for k,v in new_ct.items() if k != 'dummy'}
                set_stocks = list(set([v[0] for k,v in new_ct.items() if k != 'dummy']))
                perm = [p for p in itertools.product(set_stocks, repeat=2)]
                shovel_sttost = {(tr1,tr2): round(
                    float(Dijkstra(graph,fixed[new_ct[tr1][0]]
                                ,fixed[new_ct[tr2][0]])[2])/shovel_velocity + int(
                    math.ceil(new_ct[tr2][2]/shovel_payload)*shovel_cycletime)) for 
                                tr1 in trucks for tr2 in trucks if  tr1 != tr2 and 'dum' not in tr1 and 'dum' not in tr2}
                truck_times = {(tr1,tr2): time_(tr1,tr2,truck_sumation_time, truck_tostock_time,shovel_sttost) for tr1 in trucks for tr2 in trucks if  tr1 != tr2}
                #truck_times = {(tr1,tr2): time(tr1,tr2,truck_sumation_time) for tr1,tr2 in combinations(trucks,2)}
                m = gp.Model()
                # Variables: is ttuck 'i' adjacent to truck 'j' on the tour?
                vars = m.addVars(truck_times.keys(), obj=truck_times, vtype=GRB.BINARY, name='x')
                # Constraints: two edges incident to each ti tj


                m.addConstrs(vars.sum(c,'*') == 1 for c in trucks)
                m.addConstrs(vars.sum('*',c) == 1 for c in trucks)    

                #cons = m.addConstrs(vars.sum(c, '*') == 2 for c in trucks)
                m.Params.LogToConsole = 0

                m._vars = vars
                m.Params.lazyConstraints = 1
                m.optimize(subtourelim)

                vals = m.getAttr('x', vars)
                selected = gp.tuplelist((i, j) for i, j in vals.keys() if vals[i, j] > 0.5)

                tour = subtour(selected)
                #print(tour)
                scheds[tuple(perm_d)] = [tour, m.getObjective().getValue()]
                #print(scheds)
            minval = min(np.array(list(scheds.values()),dtype=object)[:,1])
            res = [[perm_d,v[0]] for k, v in scheds.items() if v[1]==minval]
            dict_loc = {x[0]:x[1] for x in res[0][0]}
            result = [(x,dict_loc[x]) for x in res[0][1]]
            shovelini = result[0][1]
            return result, minval, shovelini
def subtour(edges):
    unvisited = trucks[:]
    cycle = trucks[:] # Dummy - guaranteed to be replaced
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
        if len(tour) < len(trucks):
            # add subtour elimination constr. for every pair of cities in subtour
            model.cbLazy(gp.quicksum(model._vars[i,j] + model._vars[j,i] for i, j in combinations(tour, 2))
                         <= len(tour)-1)

#Welcome Kushi