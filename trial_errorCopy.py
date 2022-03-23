import sqlite3
from itertools import count
from tkinter import N

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
#from celluloid import Camera
from matplotlib import animation
from matplotlib.axes._axes import _log as matplotlib_axes_logger

from backCopy import *

#from matplotlib.patches import Circle


matplotlib_axes_logger.setLevel('ERROR')
plt.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\101114992\\Documents\\Research\\98Coding\\ffmpeg\\bin\\ffmpeg.exe"
#----------------------------
#TRASH FOR NOW
# shovel_payload = 10 #tons
# cicle_shovel = 20 #seconds
# truck_payload = 60 #tons
# time_load = (truck_payload/shovel_payload)*cicle_shovel
#----------------------------

#----------PLACES:


truck_velocity = 20 #km/h
loader_payload = 15 #tons
loader_velocity = 10 #km/h
end ='scale'
entrance = 'entrance'
#master
odb_y = True
j_shovel = 1
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
    decision_time = np.array(dbreader.decision_time)
    decision_time = [decision_time[0]]+list(decision_time)
    decision_time = np.diff(decision_time)
    to_w = dbreader.to_w
    palette_piles=dbreader.palette_piles
    palette_customer = dbreader.palette_customer
    palette_shovel = dbreader.palette_shovel
    N_Simulations = len(to_w)
    truck_capacity = list(dbreader.customer_req['tonnage'])
    trucks = dbreader.customer_req['truck']
if not odb_y:
#---------------------------------------------------------------------------------------------
    STEP_INTERPOLATION = 20
    N_Simulations = 4 
    truck_capacity = [50 for x in range(N_Simulations)]
    palette = sns.color_palette(None, 40)
    location_p = {'start': [450, 53],
                    'entrance': [343, 58],
                    'stock1': [354, 287],
                    'stock2': [364, 386],
                    'stock3': [252, 422],
                    'stock4': [132, 169],
                    'scale': [290, 77],
                    'shovel': [109, 345]}
    master = read_csvfile(pd.read_csv('last_nodeshovel_1.csv'), STEP_INTERPOLATION, location_p)
    nodes_master = master[0]
    nodes = master[1]
    graph = master[2]
    fixed_location_b = master[3]
    fixed_location = master[4]
    set_of_stocks = master[5]
    palette_piles = sns.color_palette("viridis",n_colors=len(fixed_location)+1)
    location_piles = {key:fixed_location[key]+\
        [np.random.randint(truck_capacity*(N_Simulations+1), truck_capacity*(N_Simulations+2))]+[palette_piles[index]]\
        for index,key in enumerate(fixed_location) if 'tock' in key}
    SHOVEL_NODE = fixed_location_b['shovel']   

    #---------- NUMBER OF SIMULATIONS:
    ## N_SIMULATIONS  = len(self.customer_req)
    from_w = ['start' for n_sim in range(N_Simulations)]
    decision_time = [np.random.randint(3,20) for n_sim in range(N_Simulations)]
    to_w =[set_of_stocks[np.random.randint(0, len(set_of_stocks))] for i in range(N_Simulations)]
    
    palette_customer = sns.color_palette("husl",n_colors=N_Simulations)
    palette_shovel = sns.color_palette("coolwarm",n_colors=N_Simulations)


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
i=1
for where, to  in zip(from_w,to_w):
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


matrix_custo, matrix_sho,\
    costumer_palette,shovel_palette,\
        large,set_stocks = shape_matrixmom_sec(dict_discretize, decision_time,palette_customer,
        palette_shovel,N_Simulations,to_w)

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
                location = 'Shovel Ini.'
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
    
    fig.suptitle('Allocation: '+text_up)
    ax.set_title('Customer')
    ax1.set_title('Loader/Shovel')
    #ax.text(left,up-start_up, text_up)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.legend(loc='best',prop={'size': 15})
    plt.tight_layout()
    return chart
truck_capacity 
j_shovel=1
if odb_y:
    j_shovel = 0
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
    print(np.array(list(location_piles.values()))[:,0])
    size_stock = list(np.array(list(location_piles.values()),dtype=object)[:,2])
    text_stok = [key+'\n{:.1f} tons'.format(values[2]) for key,values in location_piles.items()]
    if odb_y:
        text_stok = [key+'\n{:.1f}t[{}]'.format(values[2], values[4]) for key,values in location_piles.items()]
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
    try:
        plot_w.append(ax.scatter(matrix_custo[i][:,0],matrix_custo[i][:,1], marker ='+', c=costumer_palette[i], s=100, linewidth=2))
        actives = set()
        for ind_i in range(len(matrix_custo[i])):
            coor_elec = matrix_custo[i][ind_i]
            if coor_elec[0] != None:
                label = 'C'+str(ind_i+1)
                if odb_y:
                    label = trucks[ind_i]
                plot_w.append(ax.text(coor_elec[0],coor_elec[1], label))
                actives.add(label)
        plot_w.append(ax.text(left+10,up+20, 'Actives:'+str((actives)), c = 'k',\
            backgroundcolor='white'))
        plot_w.append(ax1.scatter(matrix_sho[i][0],matrix_sho[i][1], c='k',marker='^', linewidths=5))
        if matrix_sho[i][0]!= None:
            plot_w.append(ax1.text(matrix_sho[i][0],matrix_sho[i][1],'L1'))
            label_s = to_w[j_shovel]
            if j_shovel>ind_i:
                j_shovel = 'Done'
            
            plot_w.append(ax1.text(left+10,up+20,'Loading to:'+label_s,c = 'k',\
                backgroundcolor='white'))
        # if i>0 and shovel_linew[i]-shovel_linew[i-1]>0.1:
        #     j_shovel+=1
        #     plot_w.append(ax1.text(matrix_sho[i][0],matrix_sho[i][1], 'C'+str(j_shovel)))
    except:
        return chart

    return chart
        

ani = animation.FuncAnimation(fig, animate, init_func = ini,frames = 3000, interval = 500, repeat=False)
# ani.save('March22_P2Rss.mp4')
plt.show()
#f = "C:/Users/101114992/Documents/Research/98Coding/animation.mp4" 
#writermp4 = animation.FFMpegWriter(fps=1000) 
#initial_time = time.time()
#print(initial_time)
#ani.save(f, writer=writermp4)
#end_time = time.time()
