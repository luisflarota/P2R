import copy
import datetime
import itertools
import math
import re
import time
from ast import Load
from io import BytesIO
from itertools import count, permutations
from re import L

import cv2
import gspread
import matplotlib
import matplotlib.animation as anim
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
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
from tenacity import before_log
from streamlit import caching
from back import *

plt.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\101114992\\Documents\\Research\\98Coding\\ffmpeg\\bin\\ffmpeg.exe"
matplotlib.use('Agg') 
matplotlib.rcParams['animation.embed_limit'] = 2**128

eastern = pytz.timezone('US/Mountain')
columns_valid = ['ID', 'Company Name', 'Truck ID', 'Material', 'Tonnage', 'Date', 'Time']


@st.experimental_memo
def return_fileuploader(data_customer):
    data = pd.read_excel(data_customer)
    connect = gspread.service_account_from_dict({
            "type": "service_account",
            "project_id": "p2rsheet",
            "private_key_id": "d5d63c83d13f24fa40ab2363eae6bc546b4ab3dd",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDavqh4H2MGmrze\nYVhKzV5wMBc+WHpRz2DtR+wkdYUOCrtGcPKoOpyfgcsYMGpYGJwM+rconl4LeiaB\nj5KjYRUj5CrAOoKIixICipWMiJazXwQkiu8+CmXCIhpTxsBbztKxPrsnVAp+ZPY0\n5GHE5dKCYD27Uwrm+P31OQiwNk5KEitIiaL3peuwZsjEa7mEjGqe6y1ptihLQAgY\n8OqcRiDmHzQuPlTlcn4HbbPV5C39rlvO/WU0eqqMu+F7mrxigVXZzzExUhw6kXpT\nmPE1StKZPNRX9z/OWgz9SLDKFZ5c1NuEF7v1T4StafJlOPyXYF0T9sPgCluLk+oO\nxhIXOkQLAgMBAAECggEAFfFajf48B/K9b84Hti08GXiW2aVffoBqjTLnSyWnd2f3\n2aeajT9Klyzvy26OjxAc61JlItkhaaOoZDEbo8b+gLoH2H5Quii/4ZW2+Gt6jpDB\nUMxya7X4TOKbOHyErvukIq0+ceyvcXa9me2vqbmSMIuTSwzCrbZxfJ1q3oj8DoLs\n43f/RDeXq6Luvk7PEPfaps618JnvmMdJylwQXHX+mbfHIxlqevVCBKn6t++DBw3Y\n2nbNeAtrVox1NFgLcwlVXP2WGjTE4DPrwlF0ipn8FzI8Hv43u8292lB7vhD8M2ih\nYrLYsXYVYGiJi17W236W3MtdnTkTTgRHu5AOJTjQWQKBgQD3JJKqluT7WYFmKvGJ\nKscKmcSKC5/8lAbIjqPJhTlEGDH13DdPkJ63L9y4QfS4mBA3VtdEOzOtLD1TYdPr\nbxyaW7UU1C2hxEHXomPwoUXHv7LVEnl1EBzNyC1D213eND7R17Dwjwh70FHxe2Zz\nwZg8OAJG9VY8WhGXf4tdJXEXqQKBgQDilYuIAeM66gcmrIytbC+1ulhyNnTq8xye\nL3PnuuJZrXsxZaNm5rOI+2DnC/JoEFY+KGZOFeVy6biu3sQQ5OCRAlNjayIw0G1F\nQ9+yLsR1exBPMaupSuYoRqL7IKZcIANHolU67O9rNSrGTinAfLveI7g36uq4t+oo\n+jTiNVD+kwKBgQDs5pTkitIiEbEVI1L2PhgflDgub2hDcA10kC52TIsRN/QkDZzD\nWwiY5ns38JlJnRHmSgr9L5agiAic9ehzBMYxPHk+5wh6ySqoLdSI476E87/TuOrO\nCMzjgN/K7Ot0xTX2ZkAIx8LFFHKH/Na/XTK1fqbIKAIqxdeZFjyb4/kdSQKBgQDM\nf6YAKZv5FzE/EWqiNstUnAupgUbCqoqApllYow4ZW/6c1ZvFiqAtGJwby2eLznrX\n/MRg41hD/3eEtF+G09tuZQf36cBhCCwm4Jxrh9QeJ+TPZQgGcigJ377HIm+jI+1x\n4KxF04Q+YSzq7661IJ66XcitByOzdaIsO64xH2erawKBgQDoWYNhFOtT2ImnJQjo\ntxVNT6ysJx6IO5/Ua7aMaMV0aJs8uA7eAP4c7MxxM3FUE388IGG/lAuLxVBw1szy\nM/pEeO/yMT34kEbWJfI6A5aLkF0e+W1MLE32Y830ncgNkS7TGoc4x2Sj6dEFy/sU\nYv2jaIEY1uQgQanntNqIr0gCGQ==\n-----END PRIVATE KEY-----\n",
            "client_email": "myp2rproject@p2rsheet.iam.gserviceaccount.com",
            "client_id": "118385622492695551129",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/myp2rproject%40p2rsheet.iam.gserviceaccount.com"
            })
    sh = connect.open('P2RData')

    connect_back = Cust_Reader(sh, data,'last_nodeshovel_1.csv')
    data_ani= 0
    if connect_back.requirement.shape[0]>0:
        data_ani = connect_back.get_data_animation()
        data_sched = connect_back.get_brute_force()
    return connect_back,data_ani,data_sched

    
def main():
    st.title('Pit 2 Road')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.warning('Do you have a requirement form? If not:')
    df = to_excel()
    st.sidebar.download_button('Please, download our requirement form',\
        data = df.getvalue(),file_name= "req.xlsx")
    data_customer = st.sidebar.file_uploader('*Upload or drop your db file, \
        maxsize = 200MB:', type = 'xlsx')
    if data_customer:
        print('NO!!!!!!')
        #try:
        connect_back,data_ani,data_sched = return_fileuploader(data_customer)
        datain = Animation_2(data_ani)
        data_sch = Animation_2(data_sched)
        analize = st.sidebar.checkbox('Insert Requirement')
        print('pasa por aqui')
        if analize and data_customer:
            popup_warning = st.sidebar.radio('Are you Sure?', ['No','Yes'], key ='radio1')
            if popup_warning == 'Yes':
                message = 'Your requirement is already in the database'
                data_insert = connect_back.requirement
                if data_insert.shape[0] > 0:
                    message = 'We have loaded: {} data points'.format(data_insert.shape[0])
                    data_modified = data_insert.copy()
                    col_ini = data_insert.columns.to_list()
                    data_modified['Destinat'] = connect_back.to_master
                    col_mod = col_ini[:4] + ['Destinat'] +col_ini[4:]
                    data_modified = data_modified[col_mod]
                st.warning(message)
                st.dataframe(data_modified)
                if st.checkbox('Show map:'):
                    fig = connect_back.get_map()
                    tab1, tab2 = connect_back.get_distances()
                    st.pyplot(fig)
                    col1,col2 = st.columns(2)
                    col1.dataframe(tab1)
                    col2.dataframe(tab2)
                if st.checkbox('Show animation'):
                    radio_sched = st.radio('Choose:', ['FIFO', 'ExhaustiveS.'])
                    if radio_sched == 'FIFO':
                        st.success('Running FIFO')
                        result_fifo = showani(datain)
                        components.html(result_fifo,height = 900,width=1800)
                    else:
                        st.success('Running Exhaustive Search')
                        result_exh= showani_2(data_sch)
                        components.html(result_exh,height = 900,width=1800)

                if st.checkbox('Draw Insights:'):
                    data_cus_ins = connect_sec_()
                    st.dataframe(data_cus_ins)
                    customer = st.selectbox('Select_customer', np.unique(\
                        data_cus_ins['status_customer']))
                    data_select = data_cus_ins[data_cus_ins['status_customer']== customer]
                    trucks = st.multiselect('Select_truck', np.unique(\
                        data_select['status_truck']))
                    for truck in trucks:
                            fig = go.Figure()
                            data = data_select[data_select['status_truck'] == truck]
                            sched_n = np.unique(data['status_is_sched'])
                            for iss in sched_n:
                                if iss == 0:
                                    name = 'FIFO'
                                else:
                                    name = 'Scheduled'
                                data_s =data[data['status_is_sched'] == iss]
                                fig.add_trace(go.Scatter(x=data_s['status_name'], \
                                    y=data_s['status_duration'],name =name))

                            fig.update_layout(title=truck, xaxis=dict(
                                                title="Status"),
                                                yaxis=dict(
                                                title="Duration (sec)"
                                            ))
                            st.plotly_chart(fig)
        # except:
        #     st.warning('Input already in db')



@st.cache
def connect_sec_():
    connect = gspread.service_account_from_dict({
            "type": "service_account",
            "project_id": "p2rsheet",
            "private_key_id": "d5d63c83d13f24fa40ab2363eae6bc546b4ab3dd",
            "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvwIBADANBgkqhkiG9w0BAQEFAASCBKkwggSlAgEAAoIBAQDavqh4H2MGmrze\nYVhKzV5wMBc+WHpRz2DtR+wkdYUOCrtGcPKoOpyfgcsYMGpYGJwM+rconl4LeiaB\nj5KjYRUj5CrAOoKIixICipWMiJazXwQkiu8+CmXCIhpTxsBbztKxPrsnVAp+ZPY0\n5GHE5dKCYD27Uwrm+P31OQiwNk5KEitIiaL3peuwZsjEa7mEjGqe6y1ptihLQAgY\n8OqcRiDmHzQuPlTlcn4HbbPV5C39rlvO/WU0eqqMu+F7mrxigVXZzzExUhw6kXpT\nmPE1StKZPNRX9z/OWgz9SLDKFZ5c1NuEF7v1T4StafJlOPyXYF0T9sPgCluLk+oO\nxhIXOkQLAgMBAAECggEAFfFajf48B/K9b84Hti08GXiW2aVffoBqjTLnSyWnd2f3\n2aeajT9Klyzvy26OjxAc61JlItkhaaOoZDEbo8b+gLoH2H5Quii/4ZW2+Gt6jpDB\nUMxya7X4TOKbOHyErvukIq0+ceyvcXa9me2vqbmSMIuTSwzCrbZxfJ1q3oj8DoLs\n43f/RDeXq6Luvk7PEPfaps618JnvmMdJylwQXHX+mbfHIxlqevVCBKn6t++DBw3Y\n2nbNeAtrVox1NFgLcwlVXP2WGjTE4DPrwlF0ipn8FzI8Hv43u8292lB7vhD8M2ih\nYrLYsXYVYGiJi17W236W3MtdnTkTTgRHu5AOJTjQWQKBgQD3JJKqluT7WYFmKvGJ\nKscKmcSKC5/8lAbIjqPJhTlEGDH13DdPkJ63L9y4QfS4mBA3VtdEOzOtLD1TYdPr\nbxyaW7UU1C2hxEHXomPwoUXHv7LVEnl1EBzNyC1D213eND7R17Dwjwh70FHxe2Zz\nwZg8OAJG9VY8WhGXf4tdJXEXqQKBgQDilYuIAeM66gcmrIytbC+1ulhyNnTq8xye\nL3PnuuJZrXsxZaNm5rOI+2DnC/JoEFY+KGZOFeVy6biu3sQQ5OCRAlNjayIw0G1F\nQ9+yLsR1exBPMaupSuYoRqL7IKZcIANHolU67O9rNSrGTinAfLveI7g36uq4t+oo\n+jTiNVD+kwKBgQDs5pTkitIiEbEVI1L2PhgflDgub2hDcA10kC52TIsRN/QkDZzD\nWwiY5ns38JlJnRHmSgr9L5agiAic9ehzBMYxPHk+5wh6ySqoLdSI476E87/TuOrO\nCMzjgN/K7Ot0xTX2ZkAIx8LFFHKH/Na/XTK1fqbIKAIqxdeZFjyb4/kdSQKBgQDM\nf6YAKZv5FzE/EWqiNstUnAupgUbCqoqApllYow4ZW/6c1ZvFiqAtGJwby2eLznrX\n/MRg41hD/3eEtF+G09tuZQf36cBhCCwm4Jxrh9QeJ+TPZQgGcigJ377HIm+jI+1x\n4KxF04Q+YSzq7661IJ66XcitByOzdaIsO64xH2erawKBgQDoWYNhFOtT2ImnJQjo\ntxVNT6ysJx6IO5/Ua7aMaMV0aJs8uA7eAP4c7MxxM3FUE388IGG/lAuLxVBw1szy\nM/pEeO/yMT34kEbWJfI6A5aLkF0e+W1MLE32Y830ncgNkS7TGoc4x2Sj6dEFy/sU\nYv2jaIEY1uQgQanntNqIr0gCGQ==\n-----END PRIVATE KEY-----\n",
            "client_email": "myp2rproject@p2rsheet.iam.gserviceaccount.com",
            "client_id": "118385622492695551129",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/myp2rproject%40p2rsheet.iam.gserviceaccount.com"
            })
    sh = connect.open('P2RData')
    data_cust = sh.worksheet('CustomerStatus').get_all_records()
    data_cust_df = pd.DataFrame.from_dict(data_cust)
    return data_cust_df


@st.experimental_memo
def showani_2(_anima_2):
    ani_var_2 = animation.FuncAnimation(_anima_2.fig, _anima_2.ani, 
            init_func = _anima_2.ini,frames = _anima_2.large-1,interval = 40,
            repeat=False)
    outcome_2 = ani_var_2.to_jshtml()
    return outcome_2
@st.experimental_memo
def showani(_anima_mast):
    ani_var = animation.FuncAnimation(_anima_mast.fig, _anima_mast.ani, 
            init_func = _anima_mast.ini,frames = _anima_mast.large-1,interval = 40,
            repeat=False)
    outcome = ani_var.to_jshtml()
    return outcome

class Animation_2(object):
    def __init__(self, mom_sec):
        self.m_customer = mom_sec[0][0]
        self.m_shovel = mom_sec[0][1]
        self.costumer_palette = mom_sec[0][2]
        self.large = mom_sec[0][3] 
        self.change_stock = mom_sec[0][4]
        print(self.change_stock)
        self.new_out = mom_sec[1]
        self.c_order= mom_sec[2]
        self.fixed_location = mom_sec[3]
        self.location_piles= mom_sec[4]
        self.delay = mom_sec[5]
        self.truck_capacity = mom_sec[6]
        self.to_master_text= mom_sec[7]
        self.to_w, self.to_w_m,self.min_dectime = mom_sec[8], mom_sec[9],mom_sec[10]
        self.interpolator=mom_sec[11]
        self.array_stockpiles = np.array(list(self.location_piles.values()),dtype=object)
        self.image = cv2.imread('Petegray.jpg')
        self.move_txt_stock = 25
        self.j_shovel = 0
        self.fig = plt.figure(figsize = (15,7))
        self.ax = self.fig.add_subplot(1,2,1)
        self.ax1 = self.fig.add_subplot(1,2,2)
        self.chart = self.ax.scatter([],[], c = 'k', marker = '+',  s=100, linewidth=2)
        self.chart_2 = self.ax1.scatter([],[],c='k',marker='^', linewidths=5)
        self.text_stock =self.ax.text(0,0, '')
        self.text_shovel =self.ax1.text(0,0, '')
        
        
        #addtimea above left figure
        self.text_time = self.ax.text(0,0,'')
        self.text_stock_above = self.ax1.text(0,0, '',backgroundcolor='white')
        ##stocks
        self.dict_text_ax_above = {}
        for order in self.c_order:
            self.dict_text_ax_above[order] = self.ax.text(0,0, '',backgroundcolor='white',\
                 c = 'k')
        self.chart_stocks = [self.ax.scatter([],[], marker = 'o', s = 1, c= 'k'),
        self.ax1.scatter([],[], marker = 'o', s = 1, c= 'k')]
        self.dict_text = {}
        for order in self.c_order:
            self.dict_text[order] = self.ax.text(0,0, '')
        
        self.dict_stocks = {}
        for stk in self.location_piles:
            self.dict_stocks[stk] = [self.ax.text(0,0, '',fontsize =10, \
                backgroundcolor='white'),self.ax1.text(0,0, '',fontsize =10, \
                    backgroundcolor='white')]
        # text2 = ax.text(0,0, '')
        #GET COORDINATES
        self.left,self.right = self.ax.get_xlim()
        self.down,self.up = self.ax.get_ylim()
    def ini(self):
        #circle_entrance = Circle(tuple(fixed_location['Entrance']), radius_entrance, color='b',fill=False, hatch= '+')
        self.ax.imshow(self.image)
        self.ax1.imshow(self.image)
        for segment in np.unique(self.new_out['seg']):
            sample = np.array(self.new_out[self.new_out['seg']==segment][['x','y']])
            for x1, x2 in zip(sample[:-1],sample[1:]):
                self.ax.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                    arrowstyle='->', linestyle="--", lw=2, color='#ccffe7'))
                self.ax1.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                    arrowstyle='->', linestyle="--", lw=2, color='#ccffe7'))    
        for index,  location in enumerate(self.fixed_location):
            coordinate =self.fixed_location[location]
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
                self.ax.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=2, c= color_f)
                self.ax1.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=2, c= color_f)
                self.ax.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-60,coordinate[1]-25)\
                    , c= color_f, arrowprops= arrowprops, backgroundcolor = 'w')
                self.ax1.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-60,coordinate[1]-25)\
                    , c= color_f, arrowprops= arrowprops,backgroundcolor = 'w')
            # if location == 'Entrance':
            #     ax.add_patch(circle_entrance)
                #ax1.add_patch(circle_entrance)
        self.fig.suptitle('Idle Time: '+str(datetime.timedelta(seconds=self.delay))+' sec')
        self.ax.set_title("Customer's cycle")
        self.ax1.set_title("Loader's cycle")
        #ax.text(left,up-start_up, text_up)
        
        plt.gca().set_aspect('equal', adjustable='box')
        #plt.legend(loc='best',prop={'size': 15})
        plt.tight_layout()
    
    def ani(self, i):
        txt_up = 20
        print(i)
        if i in self.change_stock.values():
            self.j_shovel+=1
            stock = [key for key,value in self.change_stock.items() if value == i][0][0]
            self.location_piles[stock][2] = self.location_piles[stock][2] - self.truck_capacity[0]
            self.truck_capacity.remove(self.truck_capacity[0])
        size_stock = np.array(list(self.location_piles.values()),dtype=object)[:,2]
        size_stock = size_stock.astype(float)
        #size_stock = np.reshape(size_stock, (len(size_stock),1))
        text_stok = [key+'\n{:.0f} tons'.format(values[2]) for key,values \
            in self.location_piles.items()]
        self.chart_stocks[0].set_offsets(self.array_stockpiles[:,:2])
        self.chart_stocks[1].set_offsets(self.array_stockpiles[:,:2])
        for ind,dat in enumerate(self.dict_stocks):
            self.dict_stocks[dat][0].set_text(text_stok[ind])
            self.dict_stocks[dat][0].set_position((self.array_stockpiles[:,:2]\
                [ind]+self.move_txt_stock))
            self.dict_stocks[dat][1].set_text(text_stok[ind])
            self.dict_stocks[dat][1].set_position((self.array_stockpiles[:,:2]\
                [ind]+self.move_txt_stock))
        self.chart_stocks[0].set_sizes(size_stock)
        self.chart_stocks[0].set_color(self.array_stockpiles[:,3])
        self.chart_stocks[1].set_sizes(size_stock)
        self.chart_stocks[1].set_color(self.array_stockpiles[:,3])
        try: 
            matrix_chosen = self.m_customer[i][:,:2]
            matrix_shovel = self.m_shovel[i]
            if self.j_shovel+1<= len(self.to_w):
                label_s = 'Loading to:'+ self.to_w[self.j_shovel]
                active_sec = True
            else:
                label_s = 'DONE!'
                active_sec = False
            self.text_stock_above.set_text(label_s)
            self.text_stock_above.set_position((self.left+10,self.up+20))
            for ind_i in range(len(matrix_chosen)):
                if matrix_chosen[ind_i][0] != None:
                    destination = self.to_master_text[ind_i]
                    ind_customer = [index for index, destin in enumerate(self.to_w_m) if destin == destination][0]
                    customer = self.c_order[ind_i]
                    destination_text = destination.split('_')[0]
                    label_text = customer+' to: {} ({})'.format(destination_text, str(ind_customer+1))
                    color_costu = 'k'
                    if active_sec:
                        if destination_text == self.to_w[self.j_shovel] and active_sec:
                            color_costu = '#FF0000'
                    self.dict_text_ax_above[self.c_order[ind_i]].set_text(label_text)
                    self.dict_text_ax_above[self.c_order[ind_i]].set_position((\
                        self.left+10,self.up+txt_up))
                    self.dict_text_ax_above[self.c_order[ind_i]].set_color(color_costu)
                    self.dict_text[self.c_order[ind_i]].set_text(customer)
                    self.dict_text[self.c_order[ind_i]].set_position((matrix_chosen[ind_i]))
                    txt_up+=20

            #         if ind_i ==1:
            #             text.set_text(c_order[ind_i])
            #             text.set_position((matrix_chosen[ind_i][0],matrix_chosen[ind_i][1]))
            #         else:
            #             text2.set_text(c_order[ind_i])
            #             text2.set_position((matrix_chosen[ind_i][0],matrix_chosen[ind_i][1]))
            # text.set_text(c_order)
            # text.set_position(matrix_chosen)
            next_move = datetime.datetime.fromtimestamp(self.min_dectime+i*self.interpolator)
            string_move = next_move.strftime("%Y-%m-%d %H:%M:%S")
            self.text_time.set_text(string_move)
            self.text_time.set_position((self.left+10,self.up-20))
            self.chart.set_offsets(matrix_chosen)
            self.chart.set_color(self.costumer_palette[i])
            self.chart_2.set_offsets(matrix_shovel)

            if matrix_shovel[0] != None:
                self.text_shovel.set_text('L1')
                self.text_shovel.set_position((matrix_shovel))
        except:
            return
        return (self.chart, self.chart_2, self.chart_stocks,self.dict_text,self.text_stock)


if __name__ == '__main__':
    main()
