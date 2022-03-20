import ast
import math
import string
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_csvfile(file,step_intmax):
    location_p = {'Start': [471, 3],
                'Entrance': [343, 58],
                'Stock1': [354, 287],
                'Stock2': [364, 386],
                'Stock3': [252, 422],
                'Stock4': [132, 169],
                'Scale': [290, 77],
                'Shovel': [109, 345]}
    set_of_stocks = [loc for loc in location_p if 'Stock' in loc]            
    raw_data= pd.read_csv(file)
    segment, points = filter_data(raw_data,'PeteLien.png')
    out = delete_doubles(segment)    
    letter = string.ascii_lowercase
    letter = list(letter)
    nodes = {}
    new_out =[]
    for let_i, segment in enumerate(np.unique(out[:,2])):
        let = letter[let_i]
        data_segm =out[out[:,2]==segment]
        for seg_i,c_data_segm in enumerate(data_segm):
            val_data_segm = list(c_data_segm[:2])
            if val_data_segm in nodes.values():
                x = [k for k,v in nodes.items() if v == val_data_segm]
                new_out.append(list(data_segm[seg_i])+[x[0]])
                continue
            nodes[let+str(seg_i)] = val_data_segm
            new_out.append(list(data_segm[seg_i])+[let+str(seg_i)])
    new_out = pd.DataFrame(new_out, columns = ['x','y','seg','node'])
    graph ={}
    for seg_n in np.unique(new_out['seg']):
        data_seg_out = np.array(new_out[new_out['seg']== seg_n])
        for i_dato in range(data_seg_out.shape[0]-1):
            bef_p = data_seg_out[i_dato][:2]
            node_bef = find_key(bef_p, nodes)
            aft_p = data_seg_out[i_dato+1][:2]
            node_aft = find_key(aft_p, nodes)
            name_node =node_bef+node_aft
            res_bef_aft  = aft_p-bef_p
            distance = np.sqrt(np.sum(np.square(res_bef_aft)))
            step = step_intmax
            if distance<step_intmax:
                step =int(distance/2)+1
            azimuth = math.atan2(res_bef_aft[1], res_bef_aft[0])
            points_i = [[bef_p[0] + st*math.cos(azimuth),bef_p[1] + st*math.sin(azimuth)]
                        for st in range(0, round(distance), step)]
            points_i.append(list(aft_p))
            distances_graph = [get_diference(bef_p,aft_p)]
            #print(node_bef, node_aft, len(points_i), node_bef+node_aft)
            dis_acum = 0
            for st in range(0, round(distance-step), step):
                node_na_before = name_node+str(st)
                if st == 0:
                    node_na_before = node_bef
                if node_na_before not in graph:
                    graph[node_na_before] = list()
                node_na_final  = name_node+str(st+step)
                x_new = bef_p[0] + (st+1)*math.cos(azimuth)
                y_new = bef_p[1] + (st+1)*math.sin(azimuth)
                if node_na_final not in nodes:
                    nodes[node_na_final] = [x_new, y_new]
                graph[node_na_before].append((node_na_final, step))
                dis_acum+=step
                if st+step>= round(distance-step):
                    node_na_before = name_node+str(st+step)
                    node_na_final = node_aft
                    if node_na_before not in graph:
                        graph[node_na_before] = list()
                    graph[node_na_before].append((node_na_final, distance-dis_acum))
    
    nodes_loc = {key_lp:node_lp for key_lp in location_p for node_lp in nodes if location_p[key_lp] == nodes[node_lp]}

    #new_out represents nodes
    return new_out, nodes, graph, nodes_loc, location_p, set_of_stocks,set_of_stocks
def shape_matrixmom_sec(dictionary, decision_time,p_customer, p_shovel, linewidth, N_Simulations, set_stocks, unassigned = [0,0,0]):
    change_stockpiles ={}
    matrix_for_customer = np.full((2**2**4,N_Simulations,2),None)
    matrix_for_shovel = np.empty(shape = (0,2), dtype=object)
    unassigned = [0,0,0]
    costumer_palette = np.full((2**2**4,N_Simulations,3),unassigned, dtype=float)
    costumer_linew = np.full((2**2**4,N_Simulations,),0)
    shovel_palette= np.empty(shape = (0,3), dtype=float)
    shovel_linew = np.empty(shape = (0,), dtype=float)
    stock_before = 0
    for index, key in enumerate(dictionary):
        customer_data = dictionary[key]
        to_entrance= [[None, None] for x in range(np.sum(decision_time[:index+1]))]+customer_data[0]
        to_stock = customer_data[1]
        to_end = customer_data[2]
        sho_to_sho = customer_data[3]
        if index == 0:
            none_shovel = [[None, None] for x in range(len(to_entrance+to_stock) - len(sho_to_sho))]
            sho_to_sho = none_shovel+sho_to_sho
            stock_before= len(to_stock)
        else:
            len_add_stock = stock_before - decision_time[index] +len(sho_to_sho) - len(to_stock)
            if len_add_stock> 0:
                to_stock = [[None, None] for x in range(len_add_stock)]+ to_stock
                stock_before= len(to_stock)
            else:
                len_add_stock = len_add_stock*-1
                sho_to_sho = sho_to_sho + [[None, None] for x in range(len_add_stock)]
                stock_before = len(to_stock)
        if index == len(dictionary)-1:
            large = len(to_entrance+to_stock+to_end)
        #change stockpiles
        
        matrix_for_customer[:,index][0:len(to_entrance+to_stock+to_end)] = np.array(to_entrance+to_stock+to_end)
        costumer_palette[:,index][0:len(to_entrance+to_stock+to_end)] = np.array([p_customer[
            index]for x in range(len(to_entrance+to_stock+to_end))])
        costumer_linew[:,index][0:len(to_entrance+to_stock+to_end)] = np.array([linewidth[
            index]for x in range(len(to_entrance+to_stock+to_end))])
        matrix_for_shovel = np.concatenate([matrix_for_shovel,np.array(sho_to_sho)])
        change_stockpiles[(set_stocks[index], index)] = len(matrix_for_shovel)
        shovel_palette= np.concatenate([shovel_palette,np.array([p_shovel[
            index]for x in range(len(sho_to_sho))])])
        shovel_linew= np.concatenate([shovel_linew,np.array([linewidth[
            index]for x in range(len(sho_to_sho))])])
    return matrix_for_customer, matrix_for_shovel, costumer_palette,costumer_linew,shovel_palette,shovel_linew,large,change_stockpiles


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def filter_data(data, filename):
    list_segments = []
    list_points=[]
    data_filter = data[data['filename']==filename]['region_shape_attributes']
    date_name = data[data['filename']==filename]['region_attributes']
    for dataf, datan in zip(enumerate(data_filter),enumerate(date_name)):
        index = dataf[0]
        data_r = dataf[1]
        dicti = ast.literal_eval(data_r)
        #print(dicti)
        if 'polyline' in dicti.values():
            x_points = dicti['all_points_x']
            y_points = dicti['all_points_y']
            list_segments+=[[x,y, index]for x,y in zip(x_points, y_points)]
        else:
            dict_n = ast.literal_eval(datan[1])
            x_a = dicti['cx']
            y_a= dicti['cy']
            name = dict_n['name']
            list_points.append([x_a, y_a, index,name])
    return np.array(list_segments),np.array(list_points)

def delete_doubles(array):
    new = array
    #print(array.shape)
    for index,coordinate in enumerate(array):
        new_array = np.array([np.linalg.norm(x) for x in array-coordinate])
        index_2   = np.argwhere((new_array>0)&(new_array<7))
        if len(index_2)>0:
            for ind_2 in index_2:
                new[index][0] =new[ind_2][0][0]
                new[index][1] =new[ind_2][0][1]
    return new    
def get_diference(a,b):
    a = np.array(a)
    b = np.array(b)
    delta = b-a
    distance= np.square(delta)
    return np.sqrt(np.sum(distance))
def find_key(val, diction):
    val = list(val)
    x = [k for k,v in diction.items() if v == val]
    return x[0]       
def calculateAP(path, graph):
    #path = ['S','A','B']
    #len = 3 - 1 = 2
    acumulatedPath = 0
    
    for position,item in enumerate(path):
        if position < len(path) - 1:
            for k,v in graph.items():
                if item == k:
                    expand = dict(v) #[('A',7),('B',2),('C',3)]
                    acumulatedPath += expand[path[position + 1]]
                    
    return acumulatedPath
def loadQueue(graph,start):
    queue = []
    for item in graph:
        if(item == start):
            queue.append([item,[item],0])
        else:
            queue.append([item,[],math.inf])
    
    return queue
def Dijkstra(graph,start,end):
    queue = loadQueue(graph,start)
    done = []
    
    count = 1
    
    while queue:
        
        head = queue[0]  #['S',[c],ca]
        rest = queue[1:] #[ [n,[c],ca], [n,[c],ca], [n,[c],ca], [n,[c],ca] ]
        
        if head[0] == end:

            break
            
        else:
            expand = [i[0] for i in graph[head[0]]]
            for position,(n,c,v) in enumerate(queue):
                if n in expand:
                    path = head[1] + [n] 
                    acumulatedPath = calculateAP(path, graph) #7
                    if acumulatedPath < v:
                        queue[position] = ([n,path,acumulatedPath])
            queue.remove(head)
            queue = sorted(queue,key=itemgetter(2))
            done.append(head)
            
            count += 1

    return head
