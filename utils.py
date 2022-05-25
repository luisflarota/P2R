import ast
import math
from datetime import date, time
from io import BytesIO
from operator import itemgetter
import matplotlib.pyplot as plt
import numpy as np
import xlsxwriter

def return_st_material(dict):
    new_mate = {}
    for k,v in dict.items():
        for st in v:
            new_mate[st] = k
    return new_mate

   
def return_ld_stock(for_loader,for_excavators,for_hopper):
    new_dict = {}
    for x in for_loader:
        new_dict[x] = 'loader'
    for x in for_excavators:
        new_dict[x] = 'excav'
    for x in for_hopper:
        new_dict[x] = 'hopper'
    return new_dict


def change_require(df, schedule):
    for comb in schedule:
        truck = comb[0]
        index = df[df['Truck'] == truck].index
        df.loc[index,'Dest'] = comb[1]
        #print(df[df['Truck'] == truck])
    return df

def get_diference(a,b):
    a = np.array(a)
    b = np.array(b)
    delta = b-a
    distance= np.square(delta)
    return np.sqrt(np.sum(distance))

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
    for index,coordinate in enumerate(new):
        new_array = np.array([np.linalg.norm(x) for x in new-coordinate])
        index_2   = np.argwhere((new_array>=0)&(new_array<7))
        if len(index_2)>0:
            for ind_2 in index_2:
                new[index][0] =new[ind_2][0][0]
                new[index][1] =new[ind_2][0][1]
    return new

def find_key(val, diction):
    val = list(val)
    x = [k for k,v in diction.items() if v == val]
    return x[0]       

def to_excel(materials):
    today = date.today()
    year = today.year
    month = today.month
    day = today.day
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'ID')
    worksheet.write('B1', 'Company Name')
    worksheet.write('C1', 'Truck ID')
    worksheet.write('D1', 'Material')
    worksheet.data_validation('$D$2:$D$8', {'validate': 'list',
                                  'source': list(set(materials))})

    worksheet.write('E1', 'Tonnage')
    worksheet.write('F1', 'Date') 
    #worksheet.write_comment('F1', 'Format: YY/MM/DD')
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
