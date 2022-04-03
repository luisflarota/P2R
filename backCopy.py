import ast
import math
import sqlite3
import string
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def notinlist(list, element):
    if element in list:
        return list
    else:
        list.append(element)
    return list
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

class Dbreader:
    def __init__(self, connector,cursor,  csvfile):
        self.connector = connector
        self.csv = pd.read_csv(csvfile)
        self.interpolator = 20
        self.image = 'PeteLien.png'
        rows = cursor.execute("SELECT * FROM FixedLocations").fetchall()
        self.fixedloc = {row[1]:[row[3],row[4]] for row in rows}
        self.customer_req = pd.read_sql_query('SELECT * FROM CustomerReq INNER JOIN \
            FixedLocations ON FixedLocations.material = CustomerReq.typemat',self.connector)
        fixed_locations = pd.read_sql_query('SELECT * FROM FixedLocations',self.connector)
        self.setstocks = list(fixed_locations[fixed_locations\
            ['typelocation'] == 'stock']['name'])
        self.stockpileinfo = pd.read_sql_query('SELECT *\
             FROM StockpileInfo INNER JOIN\
            FixedLocations ON FixedLocations.id = StockpileInfo.stockid',self.connector)
        print(self.stockpileinfo)
        self.palette_piles = sns.color_palette("viridis",n_colors=len(self.customer_req)+1)
        self.palette_customer = sns.color_palette("husl",n_colors=len(self.customer_req))
        self.palette_shovel = sns.color_palette("coolwarm",n_colors=len(self.customer_req))
        self.from_w = ['start' for n_sim in range(len(self.customer_req))]
        self.decision_time = self.customer_req['timestamp']
        self.to_w = self.customer_req['name']
    def read_csvfile(self):
        self.csv_read = read_csvfile(self.csv,self.interpolator,self.fixedloc)
        return self.csv_read
    def location_piles(self):   
        self.loc_piles_info = {}
        for index, stock in enumerate(np.unique(self.stockpileinfo['name'])):
            info_stock = self.stockpileinfo[self.stockpileinfo['name']==stock]
            max_date = max(info_stock['timestamp'])
            filtered = info_stock[info_stock['timestamp'] == max_date]
            tonnage = filtered['tonnage']
            material = filtered['material']
            self.loc_piles_info[stock] = self.fixedloc[stock]+list(tonnage)+[self.palette_piles[index]] +[list(material)[0][:2]]
        return self.loc_piles_info

def read_csv_2(rawdata, location_p, file_picture_name):
    """
    This function deletes interpolation before building the graph. 
    Thus, it builds the graph with the main points. That way we decrease
    computational time when performing Djikstra A.
    **input: (1) rawdata(.csv)   : csvfile from VGG, contains the coordinates in pixels
             (2) location_p(dict)         : key=fixed locations and values=coordinates in pix
             (3) file_picture_name(.png)  : name of the png file (.png included). This is needed because the 
    csv file from VGG comes with it.

    **output:(0) new_out(df)             : plot direction of segments
             (1) nodes(dict)             : dict of nodes with xy coordinates
             (2) graph(dict)             : graph that contains routes (nodes and edges)
             (3) nodes_loc(dict)         : {node: name} of fixed location
             (4) location_p(dict)        : {name_fixed_location: coordinates}
             (5) set_of_stocks (list)    : name of the stocks
        """
    set_of_stocks = [loc for loc in location_p if 'stock' in loc]            
    segment, points = filter_data(rawdata,file_picture_name)
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
            if node_aft == node_bef:
                continue
            name_node =node_bef+node_aft
            if node_bef not in graph:
                graph[node_bef] = list()
            res_bef_aft  = aft_p-bef_p
            distance = round(np.sqrt(np.sum(np.square(res_bef_aft))),3)
            graph[node_bef].append((node_aft, distance))
    nodes_loc = {key_lp:node_lp for key_lp in location_p for node_lp in nodes if location_p[key_lp] == nodes[node_lp]}
    return new_out, nodes, graph, nodes_loc, location_p, set_of_stocks


def read_csvfile(raw_data,step_intmax, location_p):
    set_of_stocks = [loc for loc in location_p if 'stock' in loc]            
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
    return new_out, nodes, graph, nodes_loc, location_p, set_of_stocks

def input_opt(assignment,SHOVEL_NODE,decision_time,decision_time_sum,\
    fixed_location_b,from_w,entrance,graph,end, nodes,N_Simulations,set_of_stocks):
    assigned = assignment
    dict_discretize = dict()
    i=1
    for where, to  in zip(from_w,assigned):
        val_from_w = fixed_location_b[where]
        val_to_entrance = fixed_location_b[entrance]
        val_to_w = fixed_location_b[to]
        val_end = fixed_location_b[end]
        shortest_toentrace = Dijkstra(graph,val_from_w, val_to_entrance)
        shortest_path = Dijkstra(graph,val_to_entrance, val_to_w)
        shortest_end = Dijkstra(graph,val_to_w, val_end)
        shovel_path = Dijkstra(graph,SHOVEL_NODE, val_to_w)
        points_toentrance = np.array([nodes[val_shor] for val_shor in shortest_toentrace[1]]).tolist()
        points_customer_shortest =np.array([nodes[val_shor] for val_shor in shortest_path[1]]).tolist()
        points_customer_end =np.array([nodes[val_shor] for val_shor in shortest_end[1]]).tolist()
        points_shovel_shortest = np.array([nodes[val_shor] for val_shor in shovel_path[1]]).tolist()
        SHOVEL_NODE = val_to_w
        dict_discretize[where+str(i)] = [points_toentrance, points_customer_shortest,points_customer_end,points_shovel_shortest]
        i+=1
    delay = shape_matrixmom_delay(dict_discretize, decision_time, decision_time_sum, N_Simulations, set_of_stocks, unassigned = [0,0,0])
    return delay
def shape_matrixmom_delay(dictionary, decision_time,decision_time_sum, N_Simulations, set_stocks, unassigned = [0,0,0]):
    delay = 0
    stock_before = 0
    decisor_time_all = [(decision_time_sum[ind-1]-decision_time_sum[ind]) if (decision_time_sum[ind-1]-decision_time_sum[ind])>=0 and ind>0 \
        else 0 for ind,x in enumerate(decision_time_sum)]
    delay+= sum(decisor_time_all)
    print(delay)
    for index, key in enumerate(dictionary):
        customer_data = dictionary[key]
        #lastpointentrance
        last_point_entrance = customer_data[0][-1]
        last_p_x = last_point_entrance[0]
        last_p_y = last_point_entrance[1]
        to_entrance= [[None, None] for x in range(int(decision_time_sum[index]))]+customer_data[0]\
            +[[last_p_x,last_p_y] for i in range(decisor_time_all[index])]
        to_stock = customer_data[1]
        last_x_entr = to_stock[0][0]
        last_y_entr = to_stock[0][1]
        to_end = customer_data[2]
        sho_to_sho = customer_data[3]
        if len(sho_to_sho) ==1:
            last_x_sho = sho_to_sho[0][0]
            last_y_sho = sho_to_sho[0][1]
        else:
            last_x_sho = sho_to_sho[-1][0]
            last_y_sho = sho_to_sho[-1][1]
        if index == 0:
            none_shovel = [[None, None] for x in range(len(to_entrance+to_stock) - len(sho_to_sho))]
            sho_to_sho = none_shovel+sho_to_sho
            if index+1 == len(dictionary):
                end_shovel = [[last_x_sho, last_y_sho] for x in range(len(to_end))]
                sho_to_sho = sho_to_sho+end_shovel
            stock_before= len(to_stock)
        else:
            len_add_stock = stock_before - decision_time[index] +len(sho_to_sho) - len(to_stock)
            stock_before = len(to_stock)
            if len_add_stock> 0:
                delay += len_add_stock
                to_stock = [[last_x_entr, last_y_entr] for x in range(int(len_add_stock))]+ to_stock
                stock_before= len(to_stock)
            if index+1== len(dictionary):
                sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(len(to_end))]
            else:
                sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(int(len_add_stock*-1))]
        if index == len(dictionary)-1:
            large = len(to_entrance+to_stock+to_end)
        #change stockpiles      
    return delay  

def interpolate(all_nodes, set_nodes, time, velocity):
    points = []
    distance_vehicle = velocity * time
    extra_distance = 0
    for index, data in enumerate(set_nodes):
        if index+2 <= len(set_nodes):
            first_node = set_nodes[index]
            next_node = set_nodes[index+1]
            first_pos = np.array(all_nodes[first_node])
            if index == 0:
                points.append([first_pos[0],first_pos[1]])
            sec_pos = np.array(all_nodes[next_node])
            rest_pos  = sec_pos-first_pos
            distance = np.sqrt(np.sum(np.square(rest_pos)))
            azimuth = math.atan2(rest_pos[1], rest_pos[0])
            if distance+extra_distance >= distance_vehicle:
                int_loop = math.floor((distance+extra_distance)/distance_vehicle)
                diff_end = (distance+extra_distance) - distance_vehicle*int_loop
                for i in range(int_loop):
                    dist_added = distance_vehicle*(i+1) - extra_distance
                    x_d = round(first_pos[0] + (dist_added)*math.cos(azimuth),2)
                    y_d = round(first_pos[1] + (dist_added)*math.sin(azimuth),2)
                    points.append([x_d, y_d])
                extra_distance = 0
                extra_distance+=diff_end
            else:
                extra_distance+= distance
    return np.array(points)


def shape_matrixmom_sec(dictionary, decision_time,decision_time_sum,p_customer, 
                        N_Simulations, set_stocks,time_interval, unassigned = [0,0,0]):
    delay = 0
    change_stockpiles ={}
    matrix_for_customer = np.full((2**2**4,N_Simulations,2),None)
    matrix_for_shovel = np.empty(shape = (0,2), dtype=object)
    unassigned = [0,0,0]
    costumer_palette = np.full((2**2**4,N_Simulations,3),unassigned, dtype=float)
    #ADAPTING TIME TO TIME INTERVAL
    decision_time = [int(x/time_interval) for x in decision_time]
    decision_time_sum = [int(x/time_interval) for x in decision_time_sum]
    decisor_time_all = [(decision_time_sum[ind-1]-decision_time_sum[ind])\
         if (decision_time_sum[ind-1]-decision_time_sum[ind])>=0 and ind>0 \
        else 0 for ind,x in enumerate(decision_time_sum)]
    stock_before = 0
    delay+= sum(decisor_time_all)
    for index, key in enumerate(dictionary):
        customer_data = dictionary[key]
        pos = int(key[-2])
        #lastpointentrance
        last_point_entrance = customer_data[0][-1]
        last_p_x = last_point_entrance[0]
        last_p_y = last_point_entrance[1] 
        to_entrance= [[None, None] for x in range(int(decision_time_sum[index]))]+customer_data[0]\
            +[[last_p_x,last_p_y] for i in range(decisor_time_all[index])]
        to_stock = customer_data[1]
        last_x_entr = to_stock[0][0]
        last_y_entr = to_stock[0][1]
        to_end = customer_data[2]
        sho_to_sho = customer_data[3]
        if len(sho_to_sho) ==1:
            last_x_sho = sho_to_sho[0][0]
            last_y_sho = sho_to_sho[0][1]
        else:
            last_x_sho = sho_to_sho[-1][0]
            last_y_sho = sho_to_sho[-1][1]
        if index == 0:
            none_shovel = [[None, None] for x in range(len(to_entrance+to_stock) - len(sho_to_sho))]
            sho_to_sho = none_shovel+sho_to_sho
            if index+1 == len(dictionary):
                end_shovel = [[last_x_sho, last_y_sho] for x in range(len(to_end))]
                sho_to_sho = sho_to_sho+end_shovel
            stock_before= len(to_stock)
        else:
            len_add_stock = stock_before - decision_time[index] +len(sho_to_sho) - len(to_stock)
            stock_before = len(to_stock)
            if len_add_stock> 0:
                delay += len_add_stock
                to_stock = [[last_x_entr, last_y_entr] for x in range(int(len_add_stock))]+ to_stock
                stock_before= len(to_stock)
            if index+1== len(dictionary):
                sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(len(to_end))]
            else:
                sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(int(len_add_stock*-1))]
        if index == len(dictionary)-1:
            large = len(to_entrance+to_stock+to_end)
        #change stockpiles      
        matrix_for_customer[:,pos][0:len(to_entrance+to_stock+to_end)] = np.array(to_entrance+to_stock+to_end, dtype=object)
        costumer_palette[:,pos][0:len(to_entrance+to_stock+to_end)] = np.array([p_customer[
            index]for x in range(len(to_entrance+to_stock+to_end))])
        matrix_for_shovel = np.concatenate([matrix_for_shovel,np.array(sho_to_sho, dtype=object)])
        change_stockpiles[(set_stocks[index], index)] = len(to_entrance+to_stock)
    return matrix_for_customer, matrix_for_shovel, costumer_palette,large,change_stockpiles,delay


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def delete_doubles(array):
    new = array
    #print(array.shape)
    for index,coordinate in enumerate(new):
        new_array = np.array([np.linalg.norm(x) for x in new-coordinate])
        index_2   = np.argwhere((new_array>=0)&(new_array<7))
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
