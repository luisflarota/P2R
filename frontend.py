# !/usr/bin/env python 3.8

import gspread
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import streamlit as st
from dateutil.relativedelta import relativedelta

from anima import *
from backe import *
from schedule import *
from utils import *

# In case we need to plot a video
# plt.rcParams['animation.ffmpeg_path'] = r"C:\\Users\\101114992\\Documents\\Research\\98Coding\\ffmpeg\\bin\\ffmpeg.exe"
#matplotlib.rcParams['animation.embed_limit'] = 2**128

eastern = pytz.timezone('US/Mountain')

fixed = {'start': 'a0','entrance':'a5','stock1':'d2','stock2':'e1',
            'stock3':'f1','stock4':'g3','stock5':'h2','stock6':'k3',
            'stock7':'l5','stock8':'m2', 'scale':'c22', 'end':'c25'}
colors_f = {'start': '#f6fa00','entrance':'#f6fa00','stock1':'b','stock2':'b',
                    'stock3':'b','stock4':'b','stock5':'b','stock6':'b',
                    'stock7':'b','stock8':'b', 'scale':'b', 'end':'b'}
list_stocks  = [stock for stock in fixed if 'stock' in stock]

#Freeze
@st.experimental_memo
def connect_gsheet(datarequired, datamissed,first_st, schedule):
    """
    Connects to the Google Sheet (P2RData) to retrieve + upload data. It also
    connects to the class Cust_Reader to mix info from P2RData and the req.
    Args:
        datarequired(df): What we have in inventory from the customer req.
        datamissed(df): What we do not have in inventory from the customer req.
        first_st(str): First stock assigned to the loader.
        schedule (list of tuples): Best combination of truck, stock.

    Returns:
        data_ani(dict): Bunch of variables to be used to animate
        matrix_custo(matrix): Matrix of # * n of trucks where each element is a position. This behaves
            based on how trucks are scheduled
        matrix_sho(dict): Matrix with positions of shovels eeach 'interval' seconds'
    """
    # This will authenticate and authorize the app 
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
    # Connect to the P2RData google sheet
    sh = connect.open('P2RData')
    connect_back = Customer_Reader(sh, datarequired,datamissed,first_st, schedule)
    data_ani= 0
    #print(connect_back.requirement)
    if connect_back.requirement.shape[0]>0:
        data_ani = connect_back.get_data_animation()
        matrix_custo = data_ani[0][0]
        matrix_sho = data_ani[0][1]
    return data_ani,matrix_custo,matrix_sho


def main():
    """ Front-end part starts here."""
    st.title('Pit 2 Road')
    # Message to set a sort of dispatcher for day
    st.info('Set the resources for the day') 
    materials_stocks_cb = st.checkbox('Group materials and stockpiles')
    if materials_stocks_cb:
        # TODO: One change should not modify the other multiselect options that come
        # after
        adding_stocks = [] # Helps to avoid repetition in the upcoming multiselect boxes
        stocks_sandg = st.multiselect('Sand&Gravel', list_stocks) # Stocks_4_sand&gravel
        adding_stocks +=stocks_sandg # Adds previous selection and avoid future repetition
        stocks_crusheds = st.multiselect('CrushedStone', [ # Stocks for crushed stone
            stock for stock in list_stocks if stock not in adding_stocks
            ]) 
        adding_stocks +=stocks_crusheds
        stocks_decrock = st.multiselect('Decorative Rock', [ # Stocks for decrock
            stock for stock in list_stocks if stock not in adding_stocks
            ])
        adding_stocks +=stocks_decrock
        stocks_lime  = st.multiselect('Lime', [ # Stocks for lime
            stock for stock in list_stocks if stock not in adding_stocks
            ])
        adding_stocks +=stocks_lime
        stocks_calc = st.multiselect('Calcium', [ # Stocks for calcium
            stock for stock in list_stocks if stock not in adding_stocks
            ])
        adding_stocks +=stocks_calc
        stocks_prec = st.multiselect('Precast Concrete', [ # Stocks for preconcrete
              stock for stock in list_stocks if stock not in adding_stocks
              ]) 
        adding_stocks +=stocks_prec
    
    loading_properties_cb = st.checkbox('Select properties of the loader') 
    if loading_properties_cb:
        # TODO: One change should not modify the other multiselect options that come
        # after
        loading_properties = {} # Save properties of the loading equipment
        col1,col2,col3 = st.columns(3) #Split the view in 3 columns
        adding_stocks_load = [] # Help to avoid repetition in stocks for loaders
        # Select stocks for excavator
        stocks_excavator = col1.multiselect('Stocks for Excavators', list_stocks)
        adding_stocks_load += stocks_excavator
        if len(stocks_excavator):
            # Number of excavators and their velocity, payload and cycle time
            excavator_numb= col1.number_input('#Excavators', 
                min_value=1, max_value=3, step=1, key='ex_n', value=1)
            excavator_velo= col1.number_input('Velocity (km/h)', 
                min_value=0, max_value=10,step=1, key='ex_v', value=5)
            excavator_load = col1.number_input('Payload (tons)', 
                min_value=1, max_value=100,step=1, key='ex_p', value=20)
            excavator_ctim = col1.number_input('CycleTime (sec)', 
                min_value=1, max_value=100,step=1, key='ex_c', value=15)
            # Saving properties of the excavator in the loading_properties(dict)
            loading_properties['excavator'] = list()
            loading_properties['excavator']+=[
                stocks_excavator,
                excavator_numb,
                excavator_velo,
                excavator_load,
                excavator_ctim
                ]
        # Select stocks for loaders 
        stocks_loader = col2.multiselect('Stocks for Loaders', 
            [stock for stock in list_stocks if stock not in adding_stocks_load])
        adding_stocks_load += stocks_loader
        if len(stocks_loader):
             # Number of loaders and their velocity, payload and cycle time
            loader_numb = col2.number_input('#Loaders',
                min_value=1, max_value=3,step=1, key='lo_n', value=1)
            loader_velo = col2.number_input('Velocity (km/h)', 
                min_value=0, max_value=10,step=1, key='lo_v', value=10)
            loader_load = col2.number_input('Payload (tons)', 
                min_value=1, max_value=100,step=1, key='lo_p', value=12)
            loader_ctime = col2.number_input('CycleTime (sec)',
                min_value=1, max_value=100,step=1, key='lo_c', value=20)
            # Saving properties of the loader in the loading_properties(dict)
            loading_properties['loader'] = list()
            loading_properties['loader']+=[
                stocks_loader,
                loader_numb,
                loader_velo,
                loader_load,
                loader_ctime
                ]
        # Select stocks for hoppers 
        stocks_hopper = col3.multiselect('Stocks for Hoppers', 
            [stock for stock in list_stocks if stock not in adding_stocks_load])
        if len(stocks_hopper):
            # Set productivity of hoppers
            hopper_load = col3.number_input('Productivity (tons/sec)',
            min_value=1, max_value=100,step=1, key='lo_p')
            # Saving properties of the hopper in the loading_properties(dict)
            loading_properties['hopper'] = list()
            # Dict has 1 to index velo_laod_ctime from previous loaders
            loading_properties['hopper']+=[
                stocks_hopper,
                1,
                hopper_load
                ]
    if materials_stocks_cb and loading_properties_cb:
        radio_saveprop = st.radio('Save properties:', ['Yes', 'No']
            , key='radiosaveprop', index=1)
        if radio_saveprop =='Yes':
            materials_stocks = {'sandgravel':stocks_sandg,
                            'crushedstone':stocks_crusheds,
                            'decorock':stocks_decrock,
                            'lime':stocks_lime,
                            'calcium':stocks_calc,
                            'preconcrete':stocks_prec}
            st.success('Do you have a requirement form? If not:')
            # Requirement form in a df
            require_form = download_excelfile(list(materials_stocks.keys())) 
            st.download_button('Please, download our requirement form',
                data = require_form.getvalue(),file_name= "req_form.xlsx")
            # Requirement form filled
            data_customer_form = st.file_uploader(
                '*Upload or drop your excel file',type = 'xlsx')
            if data_customer_form:
                # Convert the excel file in a dataframe
                data_customer = pd.read_excel(data_customer_form)
                conn_customer = ConnectCustomer(data_customer,loading_properties, 
                                                    materials_stocks)
                if len(conn_customer.unavailable_materials)>=1:
                    st.warning('We do not have this set of materials:{}'.format(conn_customer.unavailable_materials))
                if len(conn_customer.unavailable_materials)!=len(conn_customer.materials_customer_req):
                    st.success('We can process the following requirement:')
                    st.dataframe(conn_customer.customer_req_available)
                    if st.checkbox('Start scheduling:'):
                        # Returns the (trucks,stock), min_idletime, first_stock
                        schedule, minval, first_stock = conn_customer.scheduling()
                        # TODO: How to comunicate the best schedule
                        st.success('Schedule')
                        st.markdown(schedule)
                        st.write('Idle time **{}** secs'.format(minval))
                        # Choose one destination for truck based on the schedule
                        last_data_customer = conn_customer.modify_req_schedule_dest(schedule)
                        st.dataframe(last_data_customer)
                        data_ani,matr_custo,matrix_shov = connect_gsheet(
                            last_data_customer, data_customer_out,first_stock, schedule
                            )
                        datain = Animation_2(data_ani)
                        if st.checkbox('Show animation_2'):
                            format = 'MMMD,YYYY, hh:mm:ss'  # format output
                            st.write(datetime.datetime.fromtimestamp(datain.min_dectime).strftime('%c'))
                            start_date = datetime.datetime.fromtimestamp(datain.min_dectime) #  I need some range in the past
                            end_date = start_date+relativedelta(seconds=datain.large)
                            max_days = end_date-start_date
                            slider = st.slider('Select date', min_value=start_date, value= end_date, max_value=end_date,step=datetime.timedelta(seconds=1), format=format,key = 'a3')
                            if 'a3' not in st.session_state:
                                st.session_state.a1 =  True
                                st.session_state.a2 =  True
                            dif = (slider-start_date).seconds
                            #print(dif)
                            #try:
                            plt_slider = datain.ini_2()
                            to_master_text = datain.to_master_text
                            to_w_m = datain.to_w_m
                            left,right = datain.ax.get_xlim()
                            down,up = datain.ax.get_ylim()
                            c_order = datain.c_order
                            change_stock = datain.change_stock
                            change_stock_play = dict((k, v) for k, v in change_stock.items() if  v<=dif)
                            #print(change_stock_play)
                            j_shovel = len(change_stock_play)
                            #print(j_shovel)
                            location_piles = datain.location_piles
                            loc_pil_plot = {k:v for k,v in datain.fixed_location.items() if 'stock' in k}
                            truck_capacity = datain.truck_capacity
                            to_w = datain.to_w
                            matrix_chosen = matr_custo[dif]
                            txt_up = 20
                            for key,val in change_stock_play.items():
                                print(location_piles)
                                print('location piles: {}'.format(location_piles[key[0]][2]))
                                print(truck_capacity[0])
                                location_piles[key[0]][1] = location_piles[key[0]][1] - truck_capacity[0]
                                truck_capacity.remove(truck_capacity[0])
                            text_stok = [key+'\n{:.0f}t[{}]'.format(values[1], values[3]) for key,values in location_piles.items()]
                            for id, loc_pil in enumerate(loc_pil_plot):

                                plt.scatter(loc_pil_plot[loc_pil][0],loc_pil_plot[loc_pil][1], c = 'b')
                                plt.text(loc_pil_plot[loc_pil][0],loc_pil_plot[loc_pil][1]+50, text_stok[id],
                                backgroundcolor = 'white')
                            active_sec = True
                            label_s = to_w_m[0]
                            if j_shovel>=1 and j_shovel < len(to_w_m):
                                label_s = to_w_m[j_shovel]
                                active_sec = True
                            elif j_shovel==len(to_w_m):
                                label_s = 'DONE!'
                                active_sec = False
                            print(matrix_chosen)
                            for ind_i in range(len(matrix_chosen)):
                                if matrix_chosen[ind_i][0] != None:
                                    destination = to_master_text[ind_i]
                                    ind_customer = [index for index, destin in enumerate(to_w_m) if destin == destination][0]
                                    customer = c_order[ind_i]
                                    destination_text = destination.split('_')[0]
                                    label_text = customer+' to: {} ({})'.format(destination_text, str(ind_customer+1))
                                    color_costu = 'k'
                                    customer = datain.c_order[ind_i]
                                    color_costu = 'k'
                                    if active_sec:
                                        if destination == to_w_m[j_shovel] and active_sec:
                                            color_costu = '#FF0000'
                                    plt.text(left+10,up+txt_up,label_text, backgroundcolor = 'white',fontsize =8, color = color_costu)
                                    plt.text(matrix_chosen[ind_i][0],matrix_chosen[ind_i][1], customer,color = color_costu)
                                    plt.scatter(matrix_chosen[ind_i][0],matrix_chosen[ind_i][1],marker='+', s=100,linewidth=2,color = color_costu)
                                    txt_up+=25
                            #plt.scatter(matrix_chosen[:,0],matrix_chosen[:,1], )
                            plt.scatter(matrix_shov[dif][0],matrix_shov[dif][1])
                            plt.text(matrix_shov[dif][0],matrix_shov[dif][1], 'L1-'+label_s[0]+label_s[5], color ='k',backgroundcolor = '#FFFF00')
                            plt.text(400,600, label_s,backgroundcolor = 'white')
                            plt.tight_layout()
                            st.pyplot(plt_slider)
                    #except:
                     #   st.warning('Try another date/ Select stocks for loaders correctly')
    # else:
    #     select_materials = st.checkbox('Select materials and stockpiles')
    #     if select_materials:
    #         sand_gravel = st.multiselect('Sand&Gravel', list_stocks, default=['stock1','stock4'])
    #         crushed_stone = st.multiselect('CrushedStone', [fruit for fruit in list_stocks if fruit not in sand_gravel],default=['stock2'])
    #         decorat_rock = st.multiselect('Decorative Rock', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone],default=['stock3'])
    #         lime  = st.multiselect('Lime', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone and fruit not in decorat_rock],default=['stock5','stock6'])
    #         calcium = st.multiselect('Calcium', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone and fruit not in decorat_rock and fruit not in lime],default=['stock7'])
    #         pre_concrete = st.multiselect('Precast Concrete', [fruit for fruit in list_stocks if fruit not in sand_gravel and fruit not in crushed_stone and fruit not in decorat_rock and fruit not in lime and fruit not in calcium])
    #     select_shovels = st.checkbox('Select types of shovels for stocks')
    #     if select_shovels:
    #         col1,col2,col3 = st.columns(3)
    #         for_excavators = col1.multiselect('For Excavators', list_stocks)
    #         if len(for_excavators)>0:
    #             exc_number = col1.number_input('#Excavators', min_value=1, max_value=3,step=1, key='ex_n', value = 1)
    #             exc_velocity = col1.number_input('#Velocity (km/h)', min_value=0, max_value=10,step=1, key='ex_v', value = 5)
    #             exc_payload = col1.number_input('#Payload (tons)', min_value=1, max_value=100,step=1, key='ex_p', value = 20)
    #             exc_cycletime = col1.number_input('#CycleTime (sec)', min_value=1, max_value=100,step=1, key='ex_c', value = 15)
    #         for_loader = col2.multiselect('For Loaders', [fruit for fruit in list_stocks if fruit not in for_excavators],default = ['stock1', 'stock4','stock5','stock6'])
    #         if len(for_loader)>0:
    #             load_number = col2.number_input('#Loader', min_value=1, max_value=3,step=1, key='lo_n', value=1)
    #             load_velocity = col2.number_input('#Velocity (km/h)', min_value=0, max_value=10,step=1, key='lo_v', value = 10)
    #             load_payload = col2.number_input('#Payload (tons)', min_value=1, max_value=100,step=1, key='lo_p', value = 12)
    #             load_cycletime = col2.number_input('#CycleTime (sec)', min_value=1, max_value=100,step=1, key='lo_c', value= 20)
    #         for_hopper = col3.multiselect('For Hoppers', [fruit for fruit in list_stocks if fruit not in for_loader and fruit not in for_excavators])
    #         hopper_tons = col3.number_input('Productivity (tons/sec)', min_value=1, max_value=100,step=1, key='lo_p')

    #     if select_materials and select_shovels:
    #         radio = st.radio('Save properties:', ['No', 'Yes'], key = 'radiolast')
    #         if radio =='Yes':
    #             materials_in = {'sandgravel':sand_gravel,'crushedstone':crushed_stone,
    #             'decorock':decorat_rock,'lime':lime,'calcium':calcium,'preconcrete':pre_concrete}
    #             #st.write(materials_out)
    #             #loader_s = [for_loader, load_velocity, load_payload,load_cycletime]
    #             #exc_s = [for_excavators, exc_velocity, exc_payload,exc_cycletime]
    #             #hopper_s = [for_hopper, hopper_tons]

    #             materials_out = return_st_material(materials_in)

    #             new_new = {}
    #             for k,v in materials_out.items():
    #                 if v not in new_new:
    #                     new_new[v] = list()
    #                 new_new[v].append(k)
    #             #st.write(materials_out)
    #             loading_stocks = return_ld_stock(for_loader,for_excavators,for_hopper)
    #             #st.write(materials_in)
    #             st.success('Do you have a requirement form? If not:')
    #             df = to_excel(list(materials_in.keys()))
    #             st.download_button('Please, download our requirement form',
    #                 data = df.getvalue(),file_name= "req.xlsx")
    #             data_customer = st.file_uploader('*Upload or drop your db file, maxsize = 200MB:', type = 'xlsx')
    #             if data_customer:
    #                 data = pd.read_excel(data_customer)
    #                 material = np.unique(data['Material'])
    #                 no_material = [mat for mat in material if mat not in list(materials_out.values())]
    #                 yes_material = [mat for mat in material if mat in list(materials_out.values())]
    #                 if len(no_material)>=1:
    #                     st.warning('We do not have this set of materials:{}'.format(no_material))

    #                 if len(no_material) != len(material):
    #                     st.success('We can process the following requirement:')
    #                     notrequired = data[data['Material'].isin(no_material)]
    #                     require = data[data['Material'].isin(yes_material)]
    #                     new_require = [list(x[:4]) + union(x[3], materials_out, loading_stocks)[0] +
    #                     union(x[3], materials_out, loading_stocks)[1] + list(x[4:]) for x in np.array(require)]
    #                     new_require = pd.DataFrame(np.array(new_require),columns =['Id','Company','Truck','Rock',
    #                     'Dest','TypeLoader', 'Tonnage','Date','Time'])
    #                     st.dataframe(new_require)
    #                     if st.checkbox('Start scheduling:'):
    #                         scheduling(new_require,new_new)

    #                         #return_fileuploader(require,notrequired)



if __name__ == '__main__':
    main()
