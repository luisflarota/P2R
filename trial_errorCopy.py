from itertools import count

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
N_Simulations = 2
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
location_piles = {key:fixed_location[key]+\
    [np.random.randint(truck_capacity*(N_Simulations+1), 300)]+[palette[index+2]]\
     for index,key in enumerate(fixed_location) if 'tock' in key}
print(location_piles)
set_of_stocks = master[5]
SHOVEL_NODE = fixed_location_b['Shovel']   
end ='Scale'
entrance = 'Entrance'
#---------- NUMBER OF SIMULATIONS:
from_w = ['Start' for n_sim in range(N_Simulations)]
decision_time = [0]+[np.random.randint(15,40) for n_sim in range(N_Simulations-1)]
to_w =[set_of_stocks[np.random.randint(0, len(set_of_stocks))] for i in range(N_Simulations)]

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
I = cv2.imread('backg.png')
ax.imshow(I,cmap="gray")
ax1.imshow(I,cmap="gray")
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
        size = 500
        marker = 'x'
        if 'tock' not in location:
            ax.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=5, c= np.array(palette[index]))
            ax1.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=5, c= np.array(palette[index]))
            ax.text(coordinate[0]+10,coordinate[1]+10, location, c= np.array(palette[index]), fontsize =12, backgroundcolor='white')
            ax1.text(coordinate[0]+10,coordinate[1]+10, location, c= np.array(palette[index]), fontsize =12, backgroundcolor='white')
        # if location == 'Entrance':
        #     ax.add_patch(circle_entrance)
            #ax1.add_patch(circle_entrance)
    ax.text(100,100, to_w)
    ax1.text(100,100, to_w)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.legend(loc='best',prop={'size': 15})
    plt.tight_layout()
    return chart



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
print(array_stockpiles)



def animate(i):

     
    # print(costumer_palette[i])
    # print(shovel_palette[i])
    #try:
     #   if i< matrix_custo.shape[0]-1:
    #---------------------------%% THIS SHOULD NOT BE A FIXED PARAMETER!!!!!
    if i in set_stocks.values():
        stock = [key for key,value in set_stocks.items() if value == i][0][0]
        location_piles[stock][2] = location_piles[stock][2] - truck_capacity

    size_stock = list(np.array(list(location_piles.values()),dtype=object)[:,2])
    print(size_stock)
    text_stok = [key+' {} tons'.format(values[2]) for key,values in location_piles.items()]
    ax.scatter(array_stockpiles[:,0],array_stockpiles[:,1], marker='o', s = size_stock, c= array_stockpiles[:,3])
    [ax.text(location_piles[key][0]+10,location_piles[key][1]+10, \
        text_stok[index], c= location_piles[key][3], fontsize =12, backgroundcolor='white')\
            for index,key in enumerate(location_piles)]
    ax1.scatter(array_stockpiles[:,0],array_stockpiles[:,1], marker='o', s = size_stock, c= array_stockpiles[:,3])
    [ax1.text(location_piles[key][0]+10,location_piles[key][1]+10, \
        text_stok[index], c= location_piles[key][3], fontsize =12, backgroundcolor='white')\
            for index,key in enumerate(location_piles)]
        
    if i == large+1:
        exit()
    try:
        ax.scatter(matrix_custo[i][:,0],matrix_custo[i][:,1], c=costumer_palette[i], linewidths=costumer_linew[i])
    except:
        return chart
    
    try:
        ax1.scatter(matrix_sho[i][0],matrix_sho[i][1], c=shovel_palette[i], linewidths=shovel_linew[i])
    except:
        return chart        
    #except:
    #    return
    
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


