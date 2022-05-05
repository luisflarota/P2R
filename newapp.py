from asyncio import LimitOverrunError
import copy
import datetime
from email.policy import default
import itertools
import math
import re
import time
from ast import Load
from io import BytesIO
from itertools import count, permutations
from re import A, L
from cv2 import FONT_HERSHEY_COMPLEX
from dateutil.relativedelta import relativedelta
import cv2
import gspread
import matplotlib
import matplotlib.animation as anim
from matplotlib.ft2font import LOAD_LINEAR_DESIGN
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pytz
import scipy as sp
import seaborn as sns
from sqlalchemy import LABEL_STYLE_DISAMBIGUATE_ONLY
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
list_stocks = [k for k in stocks]


def return_st_material(dict):
    new_mate = {}
    for k,v in dict.items():
        for st in v:
            new_mate[st] = k
    return new_mate
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
   #st.pyplot(fig)
   
def return_ld_stock(for_loader,for_excavators,for_hopper):
    new_dict = {}
    for x in for_loader:
        new_dict[x] = 'loader'
    for x in for_excavators:
        new_dict[x] = 'excav'
    for x in for_hopper:
        new_dict[x] = 'hopper'
    return new_dict

@st.experimental_memo
def return_fileuploader(datarequired, datamissed):
    data = datarequired
    datamissed = datamissed
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
    connect_back = Cust_Reader(sh, data,datamissed,'new_nodes.csv')
    data_ani= 0
    if connect_back.requirement.shape[0]>0:
        data_ani = connect_back.get_data_animation()
        data_sched = connect_back.get_brute_force()
        matrix_custo = data_ani[0][0]
        matrix_sho = data_ani[0][1]
    return connect_back,data_ani,data_sched,matrix_custo,matrix_sho

def main(): 
    st.title('Pit 2 Road')
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if st.checkbox('Show map with routes'):
        hola = give_image('new_nodes.csv')
        st.pyplot(hola)
    select_materials = st.checkbox('Select materials and stockpiles')
    if select_materials:
        sand_gravel = st.multiselect('Sand&Gravel', list_stocks, default=['stock1','stock4'])
        crushed_stone = st.multiselect('CrushedStone', [fruit for fruit in list_stocks if fruit not in sand_gravel],default=['stock2'])
        decorat_rock = st.multiselect('Decorative Rock', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone],default=['stock3'])
        lime  = st.multiselect('Lime', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone and fruit not in decorat_rock],default=['stock5','stock6'])
        calcium = st.multiselect('Calcium', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone and fruit not in decorat_rock and fruit not in lime],default=['stock7'])
        pre_concrete = st.multiselect('Precast Concrete', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone and fruit not in decorat_rock and fruit not in lime and fruit not in calcium])
    select_shovels = st.checkbox('Select types of shovels for stocks')
    if select_shovels:
        col1,col2,col3 = st.columns(3)
        for_excavators = col1.multiselect('For Excavators', list_stocks, default = ['stock2', 'stock3'])
        if len(for_excavators)>0:
            exc_number = col1.number_input('#Excavators', min_value=1, max_value=3,step=1, key='ex_n', value = 1)
            exc_velocity = col1.number_input('#Velocity (km/h)', min_value=0, max_value=10,step=1, key='ex_v', value = 5)
            exc_payload = col1.number_input('#Payload (tons)', min_value=1, max_value=100,step=1, key='ex_p', value = 20)
            exc_cycletime = col1.number_input('#CycleTime (sec)', min_value=1, max_value=100,step=1, key='ex_c', value = 15)
        for_loader = col2.multiselect('For Loaders', [fruit for fruit in list_stocks if fruit not in for_excavators],default = ['stock1', 'stock4','stock5','stock6'])
        if len(for_loader)>0:
            load_number = col2.number_input('#Loader', min_value=1, max_value=3,step=1, key='lo_n', value=1)
            load_velocity = col2.number_input('#Velocity (km/h)', min_value=0, max_value=10,step=1, key='lo_v', value = 10)
            load_payload = col2.number_input('#Payload (tons)', min_value=1, max_value=100,step=1, key='lo_p', value = 12)
            load_cycletime = col2.number_input('#CycleTime (sec)', min_value=1, max_value=100,step=1, key='lo_c', value= 20)
        for_hopper = col3.multiselect('For Hoppers', [fruit for fruit in list_stocks if fruit not in for_loader and fruit not in for_excavators],default = ['stock7', 'stock8'])
        hopper_tons = col3.number_input('Productivity (tons/sec)', min_value=1, max_value=100,step=1, key='lo_p')

    if select_materials and select_shovels:
        radio = st.radio('Save properties:', ['No', 'Yes'], key = 'radiolast')
        if radio =='Yes':
            materials_in = {'sandgravel':sand_gravel,'crushedstone':crushed_stone,
            'decorock':decorat_rock,'lime':lime,'calcium':calcium,'preconcrete':pre_concrete}
            #st.write(materials_out)
            loader_s = [for_loader, load_velocity, load_payload,load_cycletime]
            exc_s = [for_excavators, exc_velocity, exc_payload,exc_cycletime]
            hopper_s = [for_hopper, hopper_tons]
            materials_out = return_st_material(materials_in)
            new_new = {}
            for k,v in materials_out.items():
                if v not in new_new:
                    new_new[v] = list()
                new_new[v].append(k)
            #st.write(materials_out)
            loading_stocks = return_ld_stock(for_loader,for_excavators,for_hopper)
            #st.write(materials_in)
            st.success('Do you have a requirement form? If not:')
            df = to_excel(list(materials_in.keys()))
            st.download_button('Please, download our requirement form',
                data = df.getvalue(),file_name= "req.xlsx")
            data_customer = st.file_uploader('*Upload or drop your db file, maxsize = 200MB:', type = 'xlsx')
            if data_customer:
                data = pd.read_excel(data_customer)
                material = np.unique(data['Material'])
                no_material = [mat for mat in material if mat not in list(materials_out.values())]
                yes_material = [mat for mat in material if mat in list(materials_out.values())]
                if len(no_material)>=1:
                    st.warning('We do not have this set of materials:{}'.format(no_material))
                    
                if len(no_material) != len(material):
                    st.success('We can process the following requirement:')
                    notrequired = data[data['Material'].isin(no_material)]
                    require = data[data['Material'].isin(yes_material)]
                    new_require = [list(x[:4]) + union(x[3], materials_out, loading_stocks)[0] + 
                    union(x[3], materials_out, loading_stocks)[1] + list(x[4:]) for x in np.array(require)]
                    new_require = pd.DataFrame(np.array(new_require),columns =['Id','Company','Truck','Rock', 
                    'Dest','TypeLoader', 'Tonnage','Date','Time'])
                    st.dataframe(new_require)
                    if st.checkbox('Start scheduling:'):
                        scheduling(new_require,new_new)
                        #return_fileuploader(require,notrequired)
def union(mat, mat_out, loaders):
    stocks = []
    load = []
    for k,v in mat_out.items():
        if v == mat:
            stocks.append(k)
            load.append(loaders[k])
    print(stocks+load)
    return [stocks], list(set(load))

if __name__ == '__main__':
    main()