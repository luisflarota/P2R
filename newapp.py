import copy
import datetime
import itertools
import math
import re
import time
from ast import Load
from io import BytesIO
from itertools import count, permutations
from re import A, L
from dateutil.relativedelta import relativedelta
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
import copy
from simplejson import load
from tenacity import before_log
from streamlit import caching
from back import *

plt.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\101114992\\Documents\\Research\\98Coding\\ffmpeg\\bin\\ffmpeg.exe"
matplotlib.use('Agg') 
matplotlib.rcParams['animation.embed_limit'] = 2**128

eastern = pytz.timezone('US/Mountain')
columns_valid = ['ID', 'Company Name', 'Truck ID', 'Material', 'Tonnage', 'Date', 'Time']

fixed = {'start': 'a0','entrance':'a5','stock1':'d2','stock2':'e1',
            'stock3':'f1','stock4':'g3','stock5':'h2','stock6':'k3',
            'stock7':'l5','stock8':'m2', 'scale':'c22', 'end':'c24'}
stocks = {k:v for k,v in fixed.items() if 'stock' in k}
def main(): 
    st.title('Pit 2 Road')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    st.sidebar.warning('Do you have a requirement form? If not:')
    df = to_excel()
    st.sidebar.download_button('Please, download our requirement form',
        data = df.getvalue(),file_name= "req.xlsx")
    data_customer = st.sidebar.file_uploader('*Upload or drop your db file, maxsize = 200MB:', type = 'xlsx')
    if st.checkbox('Show map with routes'):
        hola = give_image('new_nodes.csv')
        st.pyplot(hola)
    if st.checkbox('Select types of shovels for stocks'):
        list_stocks = [k for k in stocks]
        col1,col2,col3 = st.columns(3)
        for_excavators = col1.multiselect('For Excavators', list_stocks)
        if len(for_excavators)>0:
            exc_number = col1.number_input('#Excavators', min_value=1, max_value=3,step=1, key='ex_n')
            #exc_velocity = col1.number_input('#Velocity (km/h)', min_value=0, max_value=10,step=1, key='ex_v')
            exc_payload = col1.number_input('#Payload (tons)', min_value=1, max_value=100,step=1, key='ex_p')
            exc_cycletime = col1.number_input('#CycleTime (sec)', min_value=1, max_value=100,step=1, key='ex_c')
        for_loader = col2.multiselect('For Loaders', [fruit for fruit in list_stocks if fruit not in for_excavators])
        if len(for_loader)>0:
            load_number = col2.number_input('#Loader', min_value=1, max_value=3,step=1, key='lo_n')
            load_velocity = col2.number_input('#Velocity (km/h)', min_value=0, max_value=10,step=1, key='lo_v')
            load_payload = col2.number_input('#Payload (tons)', min_value=1, max_value=100,step=1, key='lo_p')
            load_cycletime = col2.number_input('#CycleTime (sec)', min_value=1, max_value=100,step=1, key='lo_c')
        for_hopper = col3.multiselect('For Hoppers', [fruit for fruit in list_stocks if fruit not in for_loader and fruit not in for_excavators])
        hopper_tons = col3.number_input('Productivity (tons/sec)', min_value=1, max_value=100,step=1, key='lo_p')
def give_image(file):
    hola = getNodes(file)
    info_nodes = hola.read_csv_2()
    nodes = info_nodes[1]
    new_out = info_nodes[0]
    coordinates = np.array(list(nodes.values()))
    reading_ima = cv2.imread(hola.image)
    fig = plt.figure()
    plt.imshow(reading_ima)
    plt.gca().set_aspect('equal', adjustable='box')
    #plt.scatter(coordinates[:,0],coordinates[:,1], s=2)
    for key in nodes:
        plt.text(nodes[key][0],nodes[key][1], key, fontsize=3)
    for key,values in fixed.items():
        #plt.text(nodes[values][0],nodes[values][1]+30, key, fontsize=8, color='blue')
        plt.scatter(nodes[values][0],nodes[values][1], s=10,c='b')
        plt.annotate(key,xy=(nodes[values][0],nodes[values][1]), xytext=(nodes[values][0]+40,nodes[values][1]+35), 
                        arrowprops=dict(arrowstyle='->', lw=1, color='blue'),fontsize=7) 
    for segment in np.unique(new_out['seg']):
        sample = np.array(new_out[new_out['seg']==segment][['x','y']])
        for x1, x2 in zip(sample[:-1],sample[1:]):
            plt.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                arrowstyle='->', linestyle="--", lw=1, color='#ccffe7')) 
    return fig
    st.pyplot(fig)

if __name__ == '__main__':
    main()