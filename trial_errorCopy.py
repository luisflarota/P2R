import datetime
import math
import sqlite3
from itertools import count, permutations

import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from celluloid import Camera
from matplotlib import animation
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from simplejson import load

from backCopy import *

#from matplotlib.patches import Circle

np.random.seed(0)
plt.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\101114992\\Documents\\Research\\98Coding\\ffmpeg\\bin\\ffmpeg.exe"
#----------------------------
#TRASH FOR NOW
# shovel_payload = 10 #tons
# cicle_shovel = 20 #seconds
# truck_payload = 60 #tons
# time_load = (truck_payload/shovel_payload)*cicle_shovel
#----------------------------

#----------PLACES:
TIME_INTERVAL = 5
IMAGE_FILE_NAME ='PeteLien.png'
convert_met_sec = 0.28
truck_velocity = 20 * convert_met_sec#20km/h
loader_payload = 12 #tons #12 tons per bucket
loader_cycletime = 5 #40sec
loader_velocity = 8 * convert_met_sec#8km/h #938M
end ='scale'
entrance = 'entrance' 

N_Simulations = 3
#master
odb_y = True
fillodb = True
#schedule
schedule= False
title_ifsched = 'FIFO - '
if schedule:
    title_ifsched = 'Scheduled - '
if odb_y:
    connector = sqlite3.connect('P2Rplay.db')
    cursor = connector.cursor()
    dbreader = Dbreader(connector,cursor, 'last_nodeshovel_1.csv')
    master = dbreader.read_csvfile()
    nodes_master = master[0]
    nodes = master[1]
    graph = master[2]
    fixed_location_b = master[3]
    fixed_location = master[4]
    set_of_stocks = master[5]
    location_piles= dbreader.location_piles()
    SHOVEL_NODE = fixed_location_b['shovel']   
    from_w = dbreader.from_w
    decision_time_ini= np.array(dbreader.decision_time)
    decision_time = [decision_time_ini[0]]+list(decision_time_ini)
    decision_time = np.diff(decision_time)
    to_w = dbreader.to_w
    to_master = to_w
    palette_piles=dbreader.palette_piles
    palette_customer = dbreader.palette_customer
    palette_shovel = dbreader.palette_shovel
    N_Simulations = len(to_w)
    truck_capacity = list(dbreader.customer_req['tonnage'])
    time_taken = [int(math.ceil(dat/loader_payload)) for i,dat in enumerate(truck_capacity)]
    c_order = list(dbreader.customer_req['truck'])
    decision_time_sum = [sum(decision_time[:i+1]) for i in range(len(decision_time)) if i<len(decision_time)]
    dict_to_dec = {key+'_'+str(val):(val,val_s,tru,tcap) for \
        key,val,val_s,tru,tcap in zip(to_w,decision_time, decision_time_sum,c_order,truck_capacity)}
    to_master = to_w
    to_master_text = list(k for k in dict_to_dec)
    to_w_m = to_master_text

if not odb_y:
#---------------------------------------------------------------------------------------------
    

    STEP_INTERPOLATION = 20
    truck_capacity = [50 for x in range(N_Simulations)]
    time_taken = [int(math.ceil(dat/loader_payload)) for i,dat in enumerate(truck_capacity)]
    # print(time_taken)
    palette = sns.color_palette(None, 40)
    location_p = {'start': [613, 161],
                    'entrance': [343, 58],
                    'stock1': [354, 287],
                    'stock2': [364, 386],
                    'stock3': [252, 422],
                    'stock4': [132, 169],
                    'scale': [290, 77],
                    'shovel': [109, 345]}
    master = read_csv_2(pd.read_csv('last_nodeshovel_1.csv'), location_p, IMAGE_FILE_NAME)
    nodes_master = master[0]
    nodes = master[1]
    graph = master[2]
    fixed_location_b = master[3]
    fixed_location = master[4]
    set_of_stocks = master[5]
    palette_piles = sns.color_palette("viridis",n_colors=len(fixed_location)+1)
    location_piles = {key:fixed_location[key]+\
        [np.random.randint(truck_capacity[0]*(N_Simulations+1), truck_capacity[0]*(N_Simulations+2))]+[palette_piles[index]]\
        for index,key in enumerate(fixed_location) if 'tock' in key}
    
    SHOVEL_NODE = fixed_location_b['shovel']   

    #---------- NUMBER OF SIMULATIONS:
    ## N_SIMULATIONS  = len(self.customer_req)
    from_w = ['start' for n_sim in range(N_Simulations)]
    decision_time = [0] + [np.random.randint(3,20) for n_sim in range(N_Simulations-1)]
    decision_time_sum = [sum(decision_time[:i+1]) for i in range(len(decision_time)) if i<len(decision_time)]
    to_w =[set_of_stocks[np.random.randint(0, len(set_of_stocks))] for i in range(N_Simulations)]
    c_order =['C'+str(i) for i in range(N_Simulations)]
    
    dict_to_dec = {key+'_'+str(val):(val,val_s,tru,tcap) for \
        key,val,val_s,tru,tcap in zip(to_w,decision_time, decision_time_sum,c_order,truck_capacity)}
    to_master = to_w
    to_master_text = list(k for k in dict_to_dec)
    to_w_m = to_master_text
    
    palette_customer = sns.color_palette("husl",n_colors=N_Simulations)
    palette_shovel = sns.color_palette("coolwarm",n_colors=N_Simulations)

if schedule:
    permutation = [i for i in set([x for x in permutations(dict_to_dec)])]
    delay =[]
    for x in permutation:
        assignment=[y.split('_')[0] for y in x]
        decision_time = [dict_to_dec[dest][0] for dest in  x]
        decision_time_sum = [dict_to_dec[dest][1] for dest in x]
        truck_capacity = [dict_to_dec[dest][3] for dest in x]
        diff_sum = np.diff(np.array(decision_time_sum))
        decision_time = [0]+[x if x >0 else 0 for x in diff_sum]
        delay_each = input_opt(assignment, SHOVEL_NODE, decision_time, decision_time_sum,\
            fixed_location_b, from_w, entrance, graph, end,truck_velocity,\
                truck_capacity, loader_payload, loader_cycletime, loader_velocity)
        print(x, delay_each)
        delay.append(delay_each)
    min_d = min(delay)
    arg_min = np.argwhere(np.array(delay)==min_d)
    print(arg_min)
    #help with annotations
    to_w_m = permutation[arg_min[0][0]]
    print(to_w_m)
    #print(to_w_m)
    decision_time_sum = [dict_to_dec[tim][1] for tim in to_w_m]
    print(dict_to_dec)
    c_order = [dict_to_dec[tim][2] for tim in to_w_m]
    print(c_order)
    diff_sum = np.diff(np.array(decision_time_sum))
    decision_time = [0]+[x if x >0 else 0 for x in diff_sum]
    truck_capacity = [dict_to_dec[tim][3] for tim in to_w_m]
    to_w=[y.split('_')[0] for y in to_w_m]
fig, (ax, ax1) = plt.subplots(1,2,figsize=(15,7))

#circle_entrance = Circle(tuple(fixed_location['Entrance']), radius_entrance, color='b',fill=False, hatch= '+')
chart = ax.scatter([],[])
chart = ax1.scatter([],[])
I = cv2.imread('PeteLien.png')
ax.imshow(I)
ax1.imshow(I)

#GET COORDINATES
left,right = ax.get_xlim()
down,up = ax.get_ylim()

#---------- STARTING SIMULATION
dict_discretize = dict()
dict_delay_sec = dict()
#print(nodes_master)
print(to_w_m)
for index_ind, datos in enumerate(to_w_m):
    print('camion'+c_order[index_ind])
    where = from_w[index_ind]
    to_text = to_w_m[index_ind]
    to = to_text.split('_')[0]
    val_from_w = fixed_location_b[where]
    val_to_entrance = fixed_location_b[entrance]
    val_to_w = fixed_location_b[to]
    val_end = fixed_location_b[end]
    #path, nodes, time - entrance
    path_to_entrance = Dijkstra(graph,val_from_w, val_to_entrance)
    nodes_entrance = path_to_entrance[1]
    time_to_entrance = int(math.ceil(path_to_entrance[2]/truck_velocity))
    points_entrance = interpolate(nodes,nodes_entrance,TIME_INTERVAL, \
        truck_velocity).tolist()
    #print(time_to_entrance,points_entrance.shape)
    #path, nodes, time - stock
    path_to_stock = Dijkstra(graph,val_to_entrance, val_to_w)
    nodes_stock = path_to_stock[1]
    time_to_stock = int(math.ceil(path_to_stock[2]/truck_velocity))
    time_to_load_sec = int(math.ceil(truck_capacity[index_ind]/loader_payload))\
        *int(loader_cycletime)
    time_to_stock_sec = time_to_stock+time_to_load_sec     #use this for sec
    time_to_load = int(math.ceil(truck_capacity[index_ind]/loader_payload))\
        *int(loader_cycletime/TIME_INTERVAL)
    time_to_stock += time_to_load
    points_stock = interpolate(nodes,nodes_stock,TIME_INTERVAL, \
        truck_velocity)
    #print('original stock {}'.format(points_stock.shape))
    extra_load = np.array([list(points_stock[-1]) for x in range(time_to_load)])
    points_stock = np.concatenate((points_stock,extra_load)).tolist()
    #print(time_to_stock,points_stock.shape)
    #path, nodes, time - scale
    print('end'+str(to)+str(end))
    path_to_end =  Dijkstra(graph,val_to_w, val_end)
    nodes_end = path_to_end[1]
    time_to_end = int(math.ceil(path_to_end[2]/truck_velocity))
    points_end = interpolate(nodes,nodes_end,TIME_INTERVAL, \
        truck_velocity).tolist()
    #print(time_to_end,points_end.shape)
    #path, nodes, time - shovel
    shovel_path = Dijkstra(graph,SHOVEL_NODE, val_to_w)
    nodes_shovel = shovel_path[1]
    time_travel_shovel = int(math.ceil(shovel_path[2]/loader_velocity))
    time_travel_shovel_sec = time_travel_shovel+time_to_load_sec
    #print('ini shovel time: {}'.format(time_travel_shovel))
    time_travel_shovel += time_to_load
    points_shovel = interpolate(nodes,nodes_shovel,TIME_INTERVAL, \
        loader_velocity)

    if points_shovel.shape[0] == 0:
        points_shovel = np.array([nodes[nodes_shovel[0]]])
    extra_shovel = np.array([list(points_shovel[-1]) for x in range(time_to_load)])
    points_shovel = np.concatenate((points_shovel,extra_shovel))
    points_shovel = points_shovel.tolist()
    SHOVEL_NODE = val_to_w
    i= [index for index,dat in enumerate(to_master_text) if to_master_text[index]==to_text]
    dict_discretize[where+str(i)] = [points_entrance,points_stock,points_end,\
        points_shovel,time_travel_shovel]
    dict_delay_sec[where+str(i)] = [time_to_entrance, time_to_stock_sec,time_to_end,time_travel_shovel_sec,time_to_load_sec]

delay_master =  int(shape_matrixmom_delay(dict_delay_sec, decision_time,\
    decision_time_sum,odb=odb_y))

if odb_y and fillodb:
    fill_db(connector,cursor,dict_delay_sec, decision_time,decision_time_sum,\
        decision_time_ini[0],c_order, issched=schedule)
    
matrix_custo, matrix_sho,costumer_palette,large,set_stocks,total_delay =\
     shape_matrixmom_sec(dict_discretize, decision_time,\
            decision_time_sum,palette_customer,N_Simulations,to_w, TIME_INTERVAL)

#important to plot stockpiles
array_stockpiles = np.array(list(location_piles.values()),dtype=object)
plot_w = []
def ini():
    for segment in np.unique(nodes_master['seg']):
        sample = np.array(nodes_master[nodes_master['seg']==segment][['x','y']])
        for x1, x2 in zip(sample[:-1],sample[1:]):
            ax.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                arrowstyle='->', linestyle="--", lw=2, color='#ccffe7'))
            ax1.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                arrowstyle='->', linestyle="--", lw=2, color='#ccffe7'))    
    for index,  location in enumerate(fixed_location):
        coordinate =fixed_location[location]
        size = 200
        marker = 'x'
        if 'tock' not in location:
            color_f = 'r'
            if 'hovel' in location:
                color_f = 'k'
                marker = None
                size =10
                location = 'Loader Ini.'
            arrowprops=dict(arrowstyle='->', color=color_f, linewidth=1)
            ax.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=2, c= color_f)
            ax1.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=2, c= color_f)
            ax.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-60,coordinate[1]-25)\
                , c= color_f, arrowprops= arrowprops, backgroundcolor = 'w')
            ax1.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-60,coordinate[1]-25)\
                , c= color_f, arrowprops= arrowprops,backgroundcolor = 'w')
        # if location == 'Entrance':
        #     ax.add_patch(circle_entrance)
            #ax1.add_patch(circle_entrance)
    start_up = 50
    text_up = ['({})C{}-{}'.format(i+1,str(i+1),to_w[i]) for i in range(N_Simulations)]
    text_up = ', '.join(text_up)
    
    fig.suptitle(title_ifsched+'Idle Time: '+str(datetime.timedelta(seconds=delay_master))+' sec')
    ax.set_title("Customer's cycle")
    ax1.set_title("Loader's cycle")
    #ax.text(left,up-start_up, text_up)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.legend(loc='best',prop={'size': 15})
    plt.tight_layout()
    return chart

if odb_y:
    j_shovel = 0
    YESTERDAY_TIMESTAMP = decision_time_ini[0]
    YESTERDAY_dt = datetime.datetime.fromtimestamp(YESTERDAY_TIMESTAMP)
    STRING_YESTERDAY = YESTERDAY_dt.strftime("%Y-%m-%d %H:%M:%S")
else:
    j_shovel=0
    YESTERDAY_TIMESTAMP = datetime.datetime.now().timestamp() - 86400
    YESTERDAY_dt = datetime.datetime.fromtimestamp(YESTERDAY_TIMESTAMP)
    STRING_YESTERDAY = YESTERDAY_dt.strftime("%Y-%m-%d %H:%M:%S")

def animate(i):
    print(i)
    # print(costumer_palette[i])
    # print(shovel_palette[i])
    #try:
     #   if i< matrix_custo.shape[0]-1:
    #---------------------------%% THIS SHOULD NOT BE A FIXED PARAMETER!!!!!
    global plot_w
    global j_shovel
    [plot.remove() for plot in plot_w]
    y_text_move = 50
    
    if i in set_stocks.values():
        j_shovel+=1
        stock = [key for key,value in set_stocks.items() if value == i][0][0]
        location_piles[stock][2] = location_piles[stock][2] - truck_capacity[0]
        truck_capacity.remove(truck_capacity[0])
    size_stock = list(np.array(list(location_piles.values()),dtype=object)[:,2])
    text_stok = [key+'\n{:.0f} tons'.format(values[2]) for key,values in location_piles.items()]
    if odb_y:
        text_stok = [key+'\n{:.0f}t[{}]'.format(values[2], values[4]) for key,values in location_piles.items()]
    plot_w.append(ax.scatter(array_stockpiles[:,0],array_stockpiles[:,1], marker='o', s = size_stock, c= array_stockpiles[:,3]))
    plot_w.append([ax.text(location_piles[key][0]+10,location_piles[key][1]+y_text_move, \
        text_stok[index], c= location_piles[key][3], fontsize =12, backgroundcolor='white')\
            for index,key in enumerate(location_piles)])
    plot_w.append(ax1.scatter(array_stockpiles[:,0],array_stockpiles[:,1], marker='o', s = size_stock, c= array_stockpiles[:,3]))
    plot_w.append([ax1.text(location_piles[key][0]+10,location_piles[key][1]+y_text_move, \
        text_stok[index], c= location_piles[key][3], fontsize =12, backgroundcolor='white')\
            for index,key in enumerate(location_piles)])
    if i == large:
        exit()
    plot_w = []
    next_move = datetime.datetime.fromtimestamp(YESTERDAY_TIMESTAMP+i*TIME_INTERVAL)
    string_move = next_move.strftime("%Y-%m-%d %H:%M:%S")
    plot_w.append(ax.text(left+10,up-20, string_move))
    try:
        plot_w.append(ax.scatter(matrix_custo[i][:,0],matrix_custo[i][:,1], marker ='+', c=costumer_palette[i], s=100, linewidth=2))
        txt_up = 20
        #change color for customer to red
        active_sec = False
        if matrix_sho[i][0]!= None:
            plot_w.append(ax1.text(matrix_sho[i][0],matrix_sho[i][1],'L1'))
            if j_shovel+1<= len(to_w):
                label_s = 'Loading to:'+ to_w[j_shovel]
                active_sec = True
            else:
                label_s = 'DONE!'
                active_sec = False
            plot_w.append(ax1.text(left+10,up+20,label_s,c = 'k',\
                backgroundcolor='white'))
        for ind_i in range(len(matrix_custo[i])):
            coor_elec = matrix_custo[i][ind_i]
            if coor_elec[0] != None:
                destination = to_master_text[ind_i]
                # print(to_master_text)
                # print('--------------custocmer customer')
                # print(destination)
                ind_customer = [index for index, destin in enumerate(to_w_m) if destin == destination][0]
                customer = c_order[ind_i]
                label_plot = customer
                destination_text = destination.split('_')[0]
                label_text = customer+' to: {} ({})'.format(destination_text, str(ind_customer+1))
                color_costu = 'k'
                if odb_y:
                    label_plot = c_order[ind_i]
                    label_text = label_plot+' to: {} ({})'.format(destination_text, str(ind_customer+1))
                if active_sec:
                    if destination_text == to_w[j_shovel] and active_sec:
                        color_costu = '#FF0000'
                plot_w.append(ax.text(coor_elec[0],coor_elec[1], label_plot))
                plot_w.append(ax.text(left+10,up+txt_up, label_text, c = color_costu,backgroundcolor='white'))
                txt_up+=20
        plot_w.append(ax1.scatter(matrix_sho[i][0],matrix_sho[i][1], c='k',marker='^', linewidths=5))
       
        # if i>0 and shovel_linew[i]-shovel_linew[i-1]>0.1:
        #     j_shovel+=1
        #     plot_w.append(ax1.text(matrix_sho[i][0],matrix_sho[i][1], 'C'+str(j_shovel)))
    except:
        return chart

    return chart
        

ani = animation.FuncAnimation(fig, animate, init_func = ini,frames = 500, 
                                interval = 0.1, repeat=False ,cache_frame_data =True)
time_n = time.time()
ani.save(str(int(time_n))+'_sim_'+str(N_Simulations)+'_sched_'+str(schedule) +'_odb_'+\
        str(odb_y)+'.mp4')
plt.show()
#f = "C:/Users/101114992/Documents/Research/98Coding/animation.mp4" 
#writermp4 = animation.FFMpegWriter(fps=1000) 
#initial_time = time.time()
#print(initial_time)
#ani.save(f, writer=writermp4)
#end_time = time.time()
