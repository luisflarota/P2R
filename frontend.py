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
from backend import *
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
            st.info('Do you have a requirement form? If not:')
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
                    # TODO: Check if trucks are duplicated 
                if len(conn_customer.unavailable_materials)!=len(conn_customer.materials_customer_req):
                    st.info('We can process the following requirement:')
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
                        st.subheader("Customer Schedule with assigned stocks")
                        st.dataframe(last_data_customer)
                        # TODO: Have a checkbox that ask the user if they want to connect
                        # and update the Google Sheet 
                        # Connecting to the P2RData Google Sheet
                        # conn_customer.connect_gsheet()

                        if st.checkbox('Show animation_2'):
                            print(conn_customer.get_data_animation(first_stock,last_data_customer))
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
