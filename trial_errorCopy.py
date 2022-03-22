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
STEP_INTERPOLATION = 20
N_Simulations = 1
truck_capacity = 50 #tons
truck_velocity = 20 #km/h
loader_payload = 15 #tons
loader_velocity = 10 #km/h

palette = sns.color_palette(None, 40)
master = read_csvfile('last_nodeshovel_1.csv', STEP_INTERPOLATION)
nodes_master = master[0]
nodes = master[1]
graph = master[2]
fixed_location_b = master[3]
fixed_location = master[4]
set_of_stocks = master[5]
print(set_of_stocks)
print(len(set_of_stocks))
palette_piles = sns.color_palette("viridis",n_colors=len(fixed_location)+1)
location_piles = {key:fixed_location[key]+\
    [np.random.randint(truck_capacity*(N_Simulations+1), truck_capacity*(N_Simulations+2))]+[palette_piles[index]]\
     for index,key in enumerate(fixed_location) if 'tock' in key}
print(location_piles)
SHOVEL_NODE = fixed_location_b['Shovel']   
end ='Scale'
entrance = 'Entrance'
#---------- NUMBER OF SIMULATIONS:
from_w = ['Start' for n_sim in range(N_Simulations)]
decision_time = [0]+[np.random.randint(15,40) for n_sim in range(N_Simulations-1)]
to_w =[set_of_stocks[np.random.randint(0, len(set_of_stocks))] for i in range(N_Simulations)]
j_shovel = 1
print(to_w)

fig, (ax, ax1) = plt.subplots(1,2,figsize=(15,7))
palette_customer = sns.color_palette("husl",n_colors=N_Simulations)
palette_shovel = sns.color_palette("coolwarm",n_colors=N_Simulations)
linewidth = [i for i in range(N_Simulations*2,0,-2)]
#---------- CUSTOMERS' TRUCK PAYLOAD/ THIS WILL DECIDE HOW FAST THE TRUCKS AND SHOVEL WILL LOAD



#getting location of entrance + plotting radius
radius_entrance = 50
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
    costumer_palette,costumer_linew,shovel_palette,\
        shovel_linew,large,set_stocks = shape_matrixmom_sec(dict_discretize, decision_time,palette_customer,
        palette_shovel, linewidth,N_Simulations,to_w)

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
                marker = '^'
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
    #ax.text(left,up-start_up, text_up)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.legend(loc='best',prop={'size': 15})
    plt.tight_layout()
    return chart

j_shovel=1
def animate(i):
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
        location_piles[stock][2] = location_piles[stock][2] - truck_capacity
    size_stock = list(np.array(list(location_piles.values()),dtype=object)[:,2])
    text_stok = [key+'\n{} tons'.format(values[2]) for key,values in location_piles.items()]
    plot_w.append(ax.scatter(array_stockpiles[:,0],array_stockpiles[:,1], marker='o', s = size_stock, c= array_stockpiles[:,3]))
    plot_w.append([ax.text(location_piles[key][0]+10,location_piles[key][1]+y_text_move, \
        text_stok[index], c= location_piles[key][3], fontsize =12, backgroundcolor='white')\
            for index,key in enumerate(location_piles)])
    plot_w.append(ax1.scatter(array_stockpiles[:,0],array_stockpiles[:,1], marker='o', s = size_stock, c= array_stockpiles[:,3]))
    plot_w.append([ax1.text(location_piles[key][0]+10,location_piles[key][1]+y_text_move, \
        text_stok[index], c= location_piles[key][3], fontsize =12, backgroundcolor='white')\
            for index,key in enumerate(location_piles)])
    if i == large+1:
        exit()
    plot_w = []
    try:
        plot_w.append(ax.scatter(matrix_custo[i][:,0],matrix_custo[i][:,1], marker ='+', c=costumer_palette[i], s=100, linewidth=2))
        for ind_i in range(len(matrix_custo[i])):
            coor_elec = matrix_custo[i][ind_i]
            if coor_elec[0] != None:
                plot_w.append(ax.text(coor_elec[0],coor_elec[1], 'C'+str(ind_i+1)))
        plot_w.append(ax1.scatter(matrix_sho[i][0],matrix_sho[i][1], c='k',marker='^', linewidths=5))
        if matrix_sho[i][0]!= None:
            plot_w.append(ax1.text(matrix_sho[i][0],matrix_sho[i][1],'C'+str(j_shovel)))
        # if i>0 and shovel_linew[i]-shovel_linew[i-1]>0.1:
        #     j_shovel+=1
        #     plot_w.append(ax1.text(matrix_sho[i][0],matrix_sho[i][1], 'C'+str(j_shovel)))
    except:
        return chart

    return chart
        

ani = animation.FuncAnimation(fig, animate, init_func = ini,frames = 2000, interval = 190, repeat=False)
#ani.save('Sunday.mp4')
plt.show()
#f = "C:/Users/101114992/Documents/Research/98Coding/animation.mp4" 
#writermp4 = animation.FFMpegWriter(fps=1000) 
#initial_time = time.time()
#print(initial_time)
#ani.save(f, writer=writermp4)
#end_time = time.time()
