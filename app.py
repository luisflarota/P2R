import datetime
import math
import pathlib
import random
import sqlite3
import time
import uuid
from ast import Load
from cgitb import html
from ctypes import c_double
from io import BytesIO
from itertools import count, permutations
from re import L

import cv2
import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import scipy as sp
import seaborn as sns
import streamlit as st
import streamlit.components.v1 as components
#from celluloid import Camera
from matplotlib import animation
from matplotlib.artist import Artist
from matplotlib.axes._axes import _log as matplotlib_axes_logger
from pyrsistent import v
from simplejson import load

from back import *

# Sorting Algorithms.
np.random.seed(0)
plt.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\101114992\\Documents\\Research\\98Coding\\ffmpeg\\bin\\ffmpeg.exe"
matplotlib.use('TkAgg') 
matplotlib.rcParams['animation.embed_limit'] = 2**128
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

global j_shovels
j_shovel=0




move_txt_stock = 25
@st.experimental_singleton    
def ani(trigger =False):
    ani = animation.FuncAnimation(fig, animate, init_func = ini, frames = large,
                                                    interval = 40, repeat=False)
    
    return ani
@st.experimental_memo(suppress_st_warning=True)
def read_image():
    array_image = cv2.imread('Petegray.jpg')
    return array_image

@st.experimental_memo(suppress_st_warning=True)
def read_image():
    array_image = cv2.imread('Petegray.jpg')
    return array_image

def main():
    st.title('P2R')
    st.set_option('deprecation.showfileUploaderEncoding', False)

    st.sidebar.warning('Do you have a requirement form? If not:')
    df = to_excel()
    st.sidebar.download_button('Please, download our requirement form',\
        data = df.getvalue(),file_name= "req.xlsx")
    
    st.sidebar.warning('Now, bring your requirement here:')
    data_customer = st.sidebar.file_uploader('*Upload or drop your db file, maxsize = 200MB:', type = 'xlsx', on_change =True)
    #button:
    if st.sidebar.checkbox('Analize'):
        if data_customer:
            with Loading_file(data_customer) as data:
                connector = data.connector
                cursor = data.cursor
                dbreader = Dbreader(connector,cursor, 'last_nodeshovel_1.csv')
                master = dbreader.read_csvfile()
                global nodes_master
                global fixed_location
                global location_piles
                global truck_capacity
                global to_w
                global c_order
                global to_master_text
                global to_w_m
                global ax
                global ax1
                global fig
                global total_delay
                global chart
                global set_stocks
                global array_stockpiles
                global dict_text
                global chart_stocks 
                global dict_stocks
                global matrix_custo
                global matrix_sho
                global chart_2
                global text_stock
                global text_shovel
                global dict_text_ax_above
                global text_stock_above
                global text_time
                global YESTERDAY_TIMESTAMP
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
                YESTERDAY_TIMESTAMP = decision_time_ini[0]
                YESTERDAY_dt = datetime.datetime.fromtimestamp(YESTERDAY_TIMESTAMP)
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
                dict_to_dec = {key+'_'+str(val):(val,val_s) for key,val,val_s in zip(to_w,decision_time, decision_time_sum)}
                to_master = to_w
                to_master_text = list(k for k in dict_to_dec)
                to_w_m = to_master_text
                ###schedule or not:
                st.text('Customer Requirements')
                showreq = dbreader.creq
                showreq['date'] = pd.to_datetime(showreq['timestamp'], unit='s')    
                st.dataframe(showreq)
                st.text('Stockpile Information')
                showstock = dbreader.stockpileinfo[['id_c', 'name_s', 'material', 'tonnage', 'timestamp','stockid' ]]
                showstock['date'] = pd.to_datetime(showstock['timestamp'], unit='s')    
                st.dataframe(showstock)
                st.write('Velocity and payload')
                st.markdown('Truck velocity: {} km/h'.format(truck_velocity))
                st.markdown('Loader velocity: {} km/h'.format(loader_velocity))
                st.markdown('Cycle Time loader: {} km/h'.format(loader_cycletime))
                schedule = st.radio('Schedule?', ['Yes', 'No'])
                global title_ifsched
                title_ifsched = 'FIFO - '

                if schedule == 'Yes':
                    title_ifsched = 'Scheduled - '
                    permutation = [i for i in set([x for x in permutations(dict_to_dec)])]
                    delay =[]
                    for x in permutation:
                        assignment=[y.split('_')[0] for y in x]
                        decision_time = [dict_to_dec[dest][0] for dest in  x]
                        decision_time_sum = [dict_to_dec[dest][1] for dest in x]
                        diff_sum = np.diff(np.array(decision_time_sum))
                        decision_time = [0]+[x if x >0 else 0 for x in diff_sum]
                        delay_each = input_opt(assignment, SHOVEL_NODE, decision_time, decision_time_sum,\
                            fixed_location_b, from_w, entrance, graph, end,truck_velocity,\
                                truck_capacity, loader_payload, loader_cycletime, loader_velocity)
                        delay.append(delay_each)
                    min_d = min(delay)
                    arg_min = np.argwhere(np.array(delay)==min_d)
                    #help with annotations
                    to_w_m = permutation[arg_min[0][0]]
                    #print(to_w_m)
                    decision_time_sum = [dict_to_dec[tim][1] for tim in to_w_m]
                    diff_sum = np.diff(np.array(decision_time_sum))
                    decision_time = [0]+[x if x >0 else 0 for x in diff_sum]
                    to_w=[y.split('_')[0] for y in to_w_m]

                fig, (ax, ax1) = plt.subplots(1,2, figsize=(15,7))
                #circle_entrance = Circle(tuple(fixed_location['Entrance']), radius_entrance, color='b',fill=False, hatch= '+')
                chart = ax.scatter([],[], c = 'k', marker = '+',  s=100, linewidth=2)
                chart_2 = ax1.scatter([],[],c='k',marker='^', linewidths=5)
                text_stock =ax.text(0,0, '')
                text_shovel =ax1.text(0,0, '')
                #addtimea above left figure
                text_time = ax.text(0,0,'')
                text_stock_above = ax1.text(0,0, '',backgroundcolor='white')
                ##stocks
                dict_text_ax_above = {}
                for order in c_order:
                    dict_text_ax_above[order] = ax.text(0,0, '',backgroundcolor='white', c = 'k')
                chart_stocks = [ax.scatter([],[], marker = 'o', s = 1, c= 'k'),
                ax1.scatter([],[], marker = 'o', s = 1, c= 'k')]
                dict_text = {}
                for order in c_order:
                    dict_text[order] = ax.text(0,0, '')
                
                dict_stocks = {}
                for stk in location_piles:
                    dict_stocks[stk] = [ax.text(0,0, '',fontsize =10, backgroundcolor='white'),
                    ax1.text(0,0, '',fontsize =10, backgroundcolor='white')]
                # text2 = ax.text(0,0, '')
                #GET COORDINATES
                global left, up
                left,right = ax.get_xlim()
                down,up = ax.get_ylim()
                global customer_palette
                #---------- STARTING SIMULATION
                dict_discretize = dict()
                dict_delay_sec = dict()
                #print(nodes_master)
                for index_ind, datos in enumerate(to_w_m):
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
                global costumer_palette
                global delay_master
                global large
                delay_master =  int(shape_matrixmom_delay(dict_delay_sec, decision_time,decision_time_sum,odb=True))
                matrix_custo, matrix_sho,costumer_palette,large,set_stocks,total_delay =\
                    shape_matrixmom_sec(dict_discretize, decision_time,\
                            decision_time_sum,palette_customer,N_Simulations,to_w, TIME_INTERVAL)
                
                #important to plot stockpiles
                array_stockpiles = np.array(list(location_piles.values()),dtype=object)

                array_image = read_image()
                ax.imshow(array_image)
                ax1.imshow(array_image)
                if st.checkbox('Run Simulation', key = 's'):
                    st.success("Animating:")
                    anima = ani(trigger=schedule)
                    components.html(anima.to_jshtml(),height = 900,width=1800)
                if st.checkbox('Insights:'):
                    fill_db(connector,cursor,dict_delay_sec, decision_time,decision_time_sum,\
                            decision_time_ini[0],c_order, issched=schedule)
                    d_status = dbreader.customstatus
                    trucks = st.multiselect('Select truck:', np.unique(d_status['nametruck']))
                    for truck in trucks:
                        fig = go.Figure()
                        data = d_status[d_status['nametruck'] == truck]
                        sched_n = np.unique(data['issched'])
                        for iss in sched_n:
                            if iss == 0:
                                name = 'FIFO'
                            else:
                                name = 'Scheduled'
                            data_s =data[data['issched'] == iss]
                            fig.add_trace(go.Scatter(x=data_s['status'], \
                                y=data_s['duration'],name =name))

                        fig.update_layout(title=truck, xaxis=dict(
                                            title="Status"),
                                             yaxis=dict(
                                            title="Duration (sec)"
                                        ))
                        st.plotly_chart(fig)
                    st.write('Table Customer Status')
                    st.dataframe(d_status)

class Loading_file:
    def __init__(self, data):
        self.bytes = data.getvalue()
        self.data = None
        self.fp = None
    
    def __enter__(self):
        """Called when entering a `with` block."""
        self.fp = pathlib.Path(str(uuid.uuid4()))
        self.fp.write_bytes(self.bytes)
        self.connector = sqlite3.connect(str(self.fp))
        self.cursor = self.connector.cursor()
        # We return the current object
        return self  
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Run on `with` block exit."""
        # Here we just have to remove the temporary database file
        self.connector.close()
        self.fp.unlink()

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
    fig.suptitle(title_ifsched+'Idle Time: '+str(datetime.timedelta(seconds=delay_master))+' sec')
    ax.set_title("Customer's cycle")
    ax1.set_title("Loader's cycle")
    #ax.text(left,up-start_up, text_up)
    
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.legend(loc='best',prop={'size': 15})
    plt.tight_layout()
    return chart
global costumer_palette



def animate(i):
    print(i)
    global j_shovel
    txt_up = 20
    print(i)
    if i in set_stocks.values():
        j_shovel+=1
        stock = [key for key,value in set_stocks.items() if value == i][0][0]
        location_piles[stock][2] = location_piles[stock][2] - truck_capacity[0]
        truck_capacity.remove(truck_capacity[0])
    size_stock = np.array(list(location_piles.values()),dtype=object)[:,2]
    size_stock = size_stock.astype(float)
    #size_stock = np.reshape(size_stock, (len(size_stock),1))
    text_stok = [key+'\n{:.0f} tons'.format(values[2]) for key,values \
        in location_piles.items()]
    chart_stocks[0].set_offsets(array_stockpiles[:,:2])
    chart_stocks[1].set_offsets(array_stockpiles[:,:2])
    for ind,dat in enumerate(dict_stocks):
        dict_stocks[dat][0].set_text(text_stok[ind])
        dict_stocks[dat][0].set_position((array_stockpiles[:,:2][ind]+move_txt_stock))
        dict_stocks[dat][1].set_text(text_stok[ind])
        dict_stocks[dat][1].set_position((array_stockpiles[:,:2][ind]+move_txt_stock))
    chart_stocks[0].set_sizes(size_stock)
    chart_stocks[0].set_color(array_stockpiles[:,3])
    chart_stocks[1].set_sizes(size_stock)
    chart_stocks[1].set_color(array_stockpiles[:,3])
    matrix_chosen = matrix_custo[i][:,:2]
    matrix_shovel = matrix_sho[i]
    if j_shovel+1<= len(to_w):
        label_s = 'Loading to:'+ to_w[j_shovel]
        active_sec = True
    else:
        label_s = 'DONE!'
        active_sec = False
    text_stock_above.set_text(label_s)
    text_stock_above.set_position((left+10,up+20))
    for ind_i in range(len(matrix_chosen)):
        if matrix_chosen[ind_i][0] != None:
            destination = to_master_text[ind_i]
            ind_customer = [index for index, destin in enumerate(to_w_m) if destin == destination][0]
            customer = c_order[ind_i]
            destination_text = destination.split('_')[0]
            label_text = customer+' to: {} ({})'.format(destination_text, str(ind_customer+1))
            color_costu = 'k'
            if active_sec:
                if destination_text == to_w[j_shovel] and active_sec:
                    color_costu = '#FF0000'
            dict_text_ax_above[c_order[ind_i]].set_text(label_text)
            dict_text_ax_above[c_order[ind_i]].set_position((left+10,up+txt_up))
            dict_text_ax_above[c_order[ind_i]].set_color(color_costu)
            dict_text[c_order[ind_i]].set_text(customer)
            dict_text[c_order[ind_i]].set_position((matrix_chosen[ind_i]))
            txt_up+=20

    #         if ind_i ==1:
    #             text.set_text(c_order[ind_i])
    #             text.set_position((matrix_chosen[ind_i][0],matrix_chosen[ind_i][1]))
    #         else:
    #             text2.set_text(c_order[ind_i])
    #             text2.set_position((matrix_chosen[ind_i][0],matrix_chosen[ind_i][1]))
    # text.set_text(c_order)
    # text.set_position(matrix_chosen)
    next_move = datetime.datetime.fromtimestamp(YESTERDAY_TIMESTAMP+i*TIME_INTERVAL)
    string_move = next_move.strftime("%Y-%m-%d %H:%M:%S")
    text_time.set_text(string_move)
    text_time.set_position((left+10,up-20))
    chart.set_offsets(matrix_chosen)
    chart.set_color(costumer_palette[i])
    chart_2.set_offsets(matrix_shovel)

    if matrix_shovel[0] != None:
        text_shovel.set_text('L1')
        text_shovel.set_position((matrix_shovel))

    return (chart, chart_2, chart_stocks,dict_text, text_stock)
if __name__ == '__main__':
    main()

# components.html(anima.to_jshtml(), height=1000)
