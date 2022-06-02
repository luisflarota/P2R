import ast
import math
import string
from datetime import date, time
from io import BytesIO
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xlsxwriter


def return_stock_for_material(materials_stocks):
    """Return a dict with the form stocks:materials
    
    Args:
        materials_stocks(dict): stocks corresponding to each material

    Returns:
        stocks_material(dict): material corresponding to each stock
    """
    stocks_material = {}
    for k,v in materials_stocks.items():
        for st in v:
            stocks_material[st] = k
    return stocks_material

def return_loading_for_stocks(loading_properties):
    """Return a dict with the form loader:stocks
    
    Args:
        loading_properties(dict): data corresponding to each type of loader

    Returns:
        stock_for_load(dict): stocks corresponding to each type of loader
    """
    stock_for_load = {}
    for loader in loading_properties:
        stock_for_load[loader] = loading_properties[loader][0]
    return stock_for_load

def stock_load_truckmat(mat, st_mater, st_loading):
    """
    Returns the stocks where material for each truck can be found. Also,
    it returns the loaders for each stock

    Args:
        mat(str): material for each truck in the customer requirement
        st_mater(dict): materials corresponding to each stock
        st_loading(dict): type of loading equipment corresponding to each stock
    
    Returns
        [stocks], [load] : Combinations of stocks and type of loader for each
                            mat_truck
    """
    stocks = []
    load = []
    for k,v in st_mater.items():
        if v == mat:
            stocks.append(k)
            load.append(st_loading[k])
    return [stocks], [load]

def filter_data(data_node, filename_image):
    """ 
    Convert the dataframe converted from VGG Annotator to an array of 
    coordinates and segments corresponding to each point

    Args:
        data_node(df): dataframe from VGG that contains coordinates of points
        filename_image(png): Image within the data_node file
   
    Returns
        list_segments(array):  cooridinates and segment for each point
    """
    list_segments = []
    list_points=[]
    data_filter = data_node[
        data_node['filename']==filename_image]['region_shape_attributes']
    date_name = data_node[
        data_node['filename']==filename_image]['region_attributes']
    for dataf, datan in zip(enumerate(data_filter),enumerate(date_name)):
        index = dataf[0]
        data_r = dataf[1]
        dicti = ast.literal_eval(data_r)
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
    return np.array(list_segments)

def delete_doubles(array, max_dist=7):
    """ 
    Create intersections (convert two points in one) for points that are "max_dist" 
    pixels away from each other.
    When drawing, we are not able to set intersections, so this functions helps on that.
    Args:
        array(matrix): Matrix that contains the coordinates and the segment
                        for each specific point
        max_dist(integer): Join points if the distance between them is less
                        than 'max_dist'
   
    Returns
        new(matrix): Matrix with intersections
    """
    new = array
    for index,coordinate in enumerate(new):
        new_array = np.array([np.linalg.norm(x) for x in new-coordinate])
        index_2   = np.argwhere((new_array>=0)&(new_array<max_dist))
        if len(index_2)>0:
            for ind_2 in index_2:
                new[index][0] =new[ind_2][0][0]
                new[index][1] =new[ind_2][0][1]
    return new

def readnodesfile(nodes_file, image = 'PeteLien_bigger.png'):
    """ Convert the csv from VGG Annotator to a graph
    Args:
        nodes_file(csv): Output of VGG that contains coordinates of lines drawn
                        on the image
        image(png): Image where the lines in VGG were drawn
   
    Returns
        new_out(df): table that contains the name of each point (node), its
        coordinate and segment
        nodes(dict): node and its coordinate
        graph(dict): representation on how nodes are connected and distance(px)
        between them 
    """
    node_file =  pd.read_csv(nodes_file)
    # Coordinates and segment for each specific point 
    segment = filter_data(node_file,image)
    # Modigying segment to create intersections
    out = delete_doubles(segment)
    # Lowercase letters
    letter = list(string.ascii_lowercase)
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
    return new_out, nodes, graph

def add_index_typeloaders(req_in):
    """ Adds new column to recognize the loader in each truck's requirement
    Args:
        req_in(df): requirement processed for a customer that is in stocks
   
    Returns
        re_in(df): req_in with and additional column that recognizes the loader.
    """
    load_unique = np.unique(req_in['TypeLoader'])
    loaders = []
    for loader in load_unique:
        if loader not in loaders:
            loaders.append(loader)
    req_in['Load'] = np.array([loaders[0] for x in range(len(req_in))])
    req_in = req_in.sort_values('Epoch')
    return req_in
    # We need to implement a way to recognize excavator/hopper in order to retrieve
    # its data from load_properties

#---------------Dijkstra
#Start
def calculateAP(path, graph):
    acumulatedPath = 0
    for position,item in enumerate(path):
        if position < len(path) - 1:
            for k,v in graph.items():
                if item == k:
                    expand = dict(v) 
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
        head = queue[0] 
        rest = queue[1:] 
        if head[0] == end:
            break
        else:
            expand = [i[0] for i in graph[head[0]]]
            for position,(n,c,v) in enumerate(queue):
                if n in expand:
                    path = head[1] + [n] 
                    acumulatedPath = calculateAP(path, graph) 
                    if acumulatedPath < v:
                        queue[position] = ([n,path,acumulatedPath])
            queue.remove(head)
            queue = sorted(queue,key=itemgetter(2))
            done.append(head)          
            count += 1
    return head
#End
#---------------Dijkstra

def shape_matrixmom_delay_out(dictionary, decision_time,decision_time_sum):
    delay = 0
    decisor_time_all = [(decision_time_sum[ind-1]-decision_time_sum[ind]) if (decision_time_sum[ind-1]-decision_time_sum[ind])>=0 and ind>0
        else 0 for ind,x in enumerate(decision_time_sum)]
    delay+= sum(decisor_time_all)
    stock_before = 0
    for index, key in enumerate(dictionary):
        customer_data = dictionary[key]
        #lastpointentrance
        time_entrance =customer_data[0]
        time_entrance+= int(decision_time_sum[index]) +decisor_time_all[index]
        time_stock =customer_data[1] 
        time_end =customer_data[2]
        time_shovel = customer_data[3]
        if index == 0:
            none_shovel = time_entrance + time_stock- time_shovel
            time_shovel += none_shovel
            stock_before= time_stock
        else:
            len_add_stock = stock_before - decision_time[index] +time_shovel - time_stock
            stock_before = time_stock
            if len_add_stock> 0:
                delay += len_add_stock
    return delay 
def input_opt(assignment,SHOVEL_NODE,decision_time,decision_time_sum,
            fixed_location_b,from_w,entrance,graph,end,truck_velocity,
            truck_capacity,loader_payload,loader_cycletime, loader_velocity):
    assigned = assignment
    dict_discretize = dict()
    i=1
    for index_ind, datos in enumerate(from_w):
        where = from_w[index_ind]
        to_text = assigned[index_ind]
        to = to_text.split('_')[0]
        val_from_w = fixed_location_b[where]
        val_to_entrance = fixed_location_b[entrance]
        val_to_w = fixed_location_b[to]
        val_end = fixed_location_b[end]
        shortest_toentrace = Dijkstra(graph,val_from_w, val_to_entrance)
        time_to_entrance = int(math.ceil(shortest_toentrace[2]/truck_velocity))
        ###--------
        shortest_path = Dijkstra(graph,val_to_entrance, val_to_w)
        time_to_stock = int(math.ceil(shortest_path[2]/truck_velocity))
        time_to_load = int(math.ceil(truck_capacity[index_ind]/loader_payload))*int(loader_cycletime)
        time_to_stock += time_to_load
        ###--------
        shortest_end = Dijkstra(graph,val_to_w, val_end)
        time_to_end = int(math.ceil(shortest_end[2]/truck_velocity))
        ###--------
        shovel_path = Dijkstra(graph,SHOVEL_NODE, val_to_w)
        time_travel_shovel = int(math.ceil(shovel_path[2]/loader_velocity))
        time_travel_shovel += time_to_load
        SHOVEL_NODE = val_to_w
        dict_discretize[where+str(i)] = [time_to_entrance, time_to_stock,time_to_end,time_travel_shovel]
        i+=1
    delay = shape_matrixmom_delay_out(dict_discretize, decision_time, decision_time_sum)
    return delay

def interpolate(all_nodes, set_nodes, time, velocity):
    # TODO: Comment this 
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

def find_key(val, diction):
    val = list(val)
    x = [k for k,v in diction.items() if v == val]
    return x[0]       

def download_excelfile(materials):
    """Return bytes data in memory that contains the requirement form in an excel file
    
    Args:
        materials(list): materials produced in the site

    Returns:
        output(bytes): excel file converted into bytes to be downloaded later
    """

    today = date.today()
    year = today.year
    month = today.month
    day = today.day
    # Instanciate + create empty bytes object
    output = BytesIO()
    # Start writing the excel file in the bytes object
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    # Add a sheet
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'ID')
    worksheet.write('B1', 'Company Name')
    worksheet.write('C1', 'Truck ID')
    worksheet.write('D1', 'Material')
    worksheet.data_validation('$D$2:$D$8', {'validate': 'list',
                                  'source': list(set(materials))})

    worksheet.write('E1', 'Tonnage')
    worksheet.write('F1', 'Date')
    # Helps to input a correct date + date format
    worksheet.data_validation('$F$2:$F$20', {'validate': 'date',
                                    'criteria': 'between',
                                    'minimum': date(year, month, day),
                                    'maximum': date(year, 12, 31),
                                    'input_title': 'Enter date. Format:',
                                  'input_message': 'YY/MM//DD',
                                  'error_title': 'Input value not valid!',
                                  'error_message': 'Insert in a correct format or valid date'
                                    })
    worksheet.write('G1', 'Time')
    worksheet.data_validation('$G$2:$G$20', {'validate': 'time',
                                    'input_title': 'Enter time. Format:',
                                    'criteria': 'between',
                                  'minimum': time(9, 0,0),
                                  'maximum': time(17, 0,0),
                                  'input_message': 'hh:mm:ss',
                                  'error_title': 'Input value not valid!',
                                  'error_message': 'Insert in a correct format or valid time'
                                    })
    workbook.close()
    return output
