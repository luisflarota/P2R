import ast
import datetime
import itertools
import math
import string
from datetime import date, time
from io import BytesIO

import cv2
from cv2 import add
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import xlsxwriter
from PIL import Image

from utils import *


class getNodes:
    def __init__(self, nodes_file):
        self.node_file = pd.read_csv(nodes_file)
        self.image = 'Pete_big.jpg'
    def read_csv_2(self):
        """
        This function deletes interpolation before building the graph. 
        Thus, graph gets built with main points. That way we decrease
        computational time when performing Djikstra A.
        **input: (1) rawdata(.csv)            : csvfile from VGG, contains the coordinates in pixels
                (2) location_p(dict)         : key=fixed locations and values=coordinates in pix
                (3) file_picture_name(.png)  : name of the png file (.png included). This is needed because the 
        csv file from VGG comes with it.

        **output:(0) new_out(df)             : plot direction of segments
                (1) nodes(dict)             : dict of nodes with xy coordinates
                (2) graph(dict)             : graph that contains routes (nodes and edges)
                (3) nodes_loc(dict)         : {name: node} of fixed location
                (4) location_p(dict)        : {name_fixed_location: coordinates}
                (5) set_of_stocks (list)    : name of the stocks
            """         
        segment, points = filter_data(self.node_file,'PeteLien_bigger.png')
        out = delete_doubles(segment)
        letter = string.ascii_lowercase
        letter = list(letter)
        nodes = {}
        new_out =[]
        for let_i, segment in enumerate(np.unique(out[:,2])):
            let = letter[let_i]
            data_segm =out[out[:,2]==segment]
            for seg_i,c_data_segm in enumerate(data_segm):
                val_data_segm = list(c_data_segm[:2])
                if val_data_segm in nodes.values():
                    x = [k for k,v in nodes.items() if v == val_data_segm]
                    new_out.append(list(data_segm[seg_i])+[x[0]])
                    continue
                nodes[let+str(seg_i)] = val_data_segm
                new_out.append(list(data_segm[seg_i])+[let+str(seg_i)])
        new_out = pd.DataFrame(new_out, columns = ['x','y','seg','node'])
        graph ={}
        for seg_n in np.unique(new_out['seg']):
            data_seg_out = np.array(new_out[new_out['seg']== seg_n])
            for i_dato in range(data_seg_out.shape[0]-1):
                bef_p = data_seg_out[i_dato][:2]
                node_bef = find_key(bef_p, nodes)
                aft_p = data_seg_out[i_dato+1][:2]
                node_aft = find_key(aft_p, nodes)
                if node_aft == node_bef:
                    continue
                name_node =node_bef+node_aft
                if node_bef not in graph:
                    graph[node_bef] = list()
                res_bef_aft  = aft_p-bef_p
                distance = round(np.sqrt(np.sum(np.square(res_bef_aft))),3)
                graph[node_bef].append((node_aft, distance))
        # nodes_loc = {key_lp:node_lp for key_lp in self.fixedloc for
        #      node_lp in nodes if self.fixedloc[key_lp] == nodes[node_lp]}
        return new_out, nodes, graph#, nodes_loc
    

class Cust_Reader:
    def __init__(self, connect_v,cust_req,nodescsv, interval = 5):
        self.nodescsv = pd.read_csv(nodescsv)
        self.cust_req = cust_req
        self.interpolator = interval
        self.sh = connect_v
        self.entrance ='entrance'
        self.end = 'scale'
        self.add_sched = 0
        convert_met_sec = 0.28
        self.truck_velocity = 20 * convert_met_sec#20km/h
        self.loader_payload = 12 
        self.loader_cycletime = 5
        self.loader_velocity = 8 * convert_met_sec#8km
        self.fixedloc_ini = self.sh.worksheet('FixedLoc').get_all_records()
        self.custereq_ini = self.sh.worksheet('CustomerReq').get_all_records()
        self.stockpile_ini = self.sh.worksheet('StockInfo').get_all_records()
        self.customer_status = self.sh.worksheet('CustomerStatus').get_all_records()
        self.table_customstatus = self.info_customer_status()
        self.table_stockpile_ini = self.info_stockpile_ini()
        self.table_shovel = pd.DataFrame.from_dict(self.sh.worksheet('Shovel').get_all_records())
        self.table_billing = pd.DataFrame.from_dict(self.sh.worksheet('Billing').get_all_records())
        self.sched = 0
        self.max_id_cust = 0
        self.max_id_stockinfo = 0 
        self.max_id_shovel = 0
        self.max_billing = 0
        self.stock_tonnage = {}
        if self.table_shovel.shape[0]>0:
            self.max_id_shovel = max(self.table_shovel['shovel_id'])
        if self.table_billing.shape[0]>0:
            self.max_billing = max(self.table_billing['billing_id'])
        if self.table_customstatus.shape[0]>0:
            self.max_id_cust = max(self.table_customstatus['status_id'])
        if self.table_stockpile_ini.shape[0]>0:
            stocks_ini = np.unique(self.table_stockpile_ini['stock_name'])
            for st_ini in stocks_ini:
                data_sel = self.table_stockpile_ini[self.table_stockpile_ini['stock_name']== st_ini]
                max_time_sel = max(data_sel['stock_times'])
                tonnage = float(data_sel[data_sel['stock_times']==max_time_sel]['stock_tonnage'])
                self.stock_tonnage[st_ini] = tonnage
            self.max_id_stockinfo = max(self.table_stockpile_ini['stock_id'])
        
        self.image = 'Pete_big.jpg'
        # position of each fixed location
        self.fixedloc = {row['fixed_name']:[row['fixed_x'],row['fixed_y']] for row in self.fixedloc_ini}
        self.stock_id = {row['fixed_name']: [row['fixed_id'],row['stock_cost_ton']] for row in self.fixedloc_ini if 'stock' in row['fixed_name']}
        # list of stocks
        self.table_fixedloc = self.info_fixed_loc()
        self.setstocks = list(self.table_fixedloc[self.table_fixedloc['fixed_type'] == 'stock']['fixed_name'])
        # location of stockpiles and material
        self.table_stockpile = self.info_stockpile_ini()
        self.stockpileinfo = pd.merge(self.table_stockpile,self.table_fixedloc,
        left_on='stock_stockid',right_on='fixed_id')[self.table_stockpile.columns.to_list()+
                ['fixed_mat','fixed_x', 'fixed_y']]
        # info piles in data structure 
        self.palette_piles = sns.color_palette("viridis",n_colors=len(self.setstocks)+1)
        self.loc_piles = self.location_piles()
         
        #datafinal customer
        self.requirement = self.get_requirement()
        if self.requirement.shape[0]>0:
            self.palette_customer = sns.color_palette("husl",n_colors=len(self.requirement))
            self.palette_shovel = sns.color_palette("coolwarm",n_colors=len(self.requirement))
            self.from_w = ['start' for n_sim in range(len(self.requirement))]
            #print(self.requirement)
            self.decision_time = self.requirement['customer_timest']
            
            #min
            self.min_dectime = min(self.decision_time)
            self.decision_time = np.diff([self.min_dectime]+list(self.decision_time))
            self.decision_time_sum = [sum(self.decision_time[:i+1]) for i in
                range(len(self.decision_time)) if i<len(self.decision_time)]
            self.requirem_combined = pd.merge(self.requirement,self.table_fixedloc,
            left_on='customer_mat',right_on='fixed_mat')
            self.requirem_combined = self.requirem_combined.sort_values('customer_timest')
            #print(self.requirem_combined)
            self.to_w = self.requirem_combined['fixed_name']
            self.to_master = self.to_w
            #print(self.to_master)
            self.new_out, self.nodes, self.graph, self.nodes_loc = self.read_csv_2()
            self.shovelnode = self.nodes_loc['shovel']
            self.truck_capacity = list(self.requirement['customer_tonnage'])
            self.c_order = list(self.requirement['customer_truck'])
            self.truck_ton = {self.c_order[i]: self.truck_capacity[i] for i in 
                range(len(self.c_order))}

            self.dict_to_dec ={key+'_'+str(val):(val,val_s,tru,tcap) for 
                                key,val,val_s,tru,tcap in zip(self.to_w,self.decision_time, 
                                    self.decision_time_sum
                                    ,self.c_order,self.truck_capacity)}
            #print(self.dict_to_dec)
            self.to_master_text = list(k for k in self.dict_to_dec)
            self.to_w_m = self.to_master_text
            self.shovelnode = self.nodes_loc['shovel']
    
    def get_brute_force(self):
        permutation = [x for x in itertools.permutations(self.dict_to_dec)]
        delay =[]
        for x in permutation:
            assignment=[y.split('_')[0] for y in x]
            decision_time = [self.dict_to_dec[dest][0] for dest in  x]
            decision_time_sum = [self.dict_to_dec[dest][1] for dest in x]
            truck_capacity = [self.dict_to_dec[dest][3] for dest in x]
            diff_sum = np.diff(np.array(decision_time_sum))
            decision_time = [0]+[x if x >0 else 0 for x in diff_sum]
            delay_each = input_opt(
                assignment, self.shovelnode, decision_time, decision_time_sum,
                self.nodes_loc, self.from_w, self.entrance, self.graph, self.end,self.truck_velocity,
                truck_capacity, self.loader_payload, self.loader_cycletime, self.loader_velocity)
            delay.append(delay_each)
        min_d = min(delay)
        arg_min = np.argwhere(np.array(delay)==min_d)
        #help with annotations
        self.to_w_m = permutation[arg_min[0][0]]
        #print(to_w_m)
        self.decision_time_sum = [self.dict_to_dec[tim][1] for tim in self.to_w_m]
        #self.c_order = [self.dict_to_dec[tim][2] for tim in self.to_w_m]
        diff_sum = np.diff(np.array(decision_time_sum))
        self.decision_time = [0]+[x if x >0 else 0 for x in diff_sum]
        self.truck_capacity = [self.dict_to_dec[tim][3] for tim in self.to_w_m]
        self.to_w=[y.split('_')[0] for y in self.to_w_m]
        self.sched = 1
        self.add_sched = 1
        return self.get_data_animation()
    def info_fixed_loc(self):
        ##fixedloc
        table_fixedloc = pd.DataFrame.from_dict(self.fixedloc_ini)
        return table_fixedloc

    def info_custo_db(self):
        table_custereq_ini = pd.DataFrame.from_dict(self.custereq_ini)
        return table_custereq_ini

    def info_stockpile_ini(self):
        table_stockpile = pd.DataFrame.from_dict(self.stockpile_ini)
        return table_stockpile
    def info_customer_status(self):
        table_customerstatus = pd.DataFrame.from_dict(self.customer_status)
        return table_customerstatus

    def get_requirement(self):
        self.cust_req['Date'] = self.cust_req.apply(
                        lambda r : datetime.datetime.combine(r['Date'],r['Time']),1)
        self.cust_req['Epoch'] = (self.cust_req['Date']
                - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        self.cust_req['Epoch'] = self.cust_req['Epoch'] + 6*3600
        data_insert = self.cust_req[['ID', 'Company Name', 'Truck ID'
            , 'Material', 'Tonnage', 'Epoch']]
        columns_change =['customer_id','customer_name','customer_truck'
            ,'customer_mat','customer_tonnage','customer_timest']
        data_insert.columns = columns_change

        data_from_db_customer = self.info_custo_db()
        data_insert = pd.merge(data_insert,data_from_db_customer, 
            on =['customer_name',
                    'customer_truck',
                    'customer_tonnage',
                    'customer_timest'],indicator=True,
                         how='outer').query('_merge=="left_only"').drop('_merge', axis=1)
        if data_insert.shape[0]>0: 
            col_sim = data_insert.columns.to_list()
            data_insert = data_insert[col_sim[:6]].sort_values('customer_timest')
            data_insert.columns = columns_change
            self.sh.worksheet('CustomerReq').append_rows(
                data_insert.values.tolist(),value_input_option="USER_ENTERED"
            )
        return data_insert
    
    def read_csv_2(self):
        """
        This function deletes interpolation before building the graph. 
        Thus, graph gets built with main points. That way we decrease
        computational time when performing Djikstra A.
        **input: (1) rawdata(.csv)            : csvfile from VGG, contains the coordinates in pixels
                (2) location_p(dict)         : key=fixed locations and values=coordinates in pix
                (3) file_picture_name(.png)  : name of the png file (.png included). This is needed because the 
        csv file from VGG comes with it.

        **output:(0) new_out(df)             : plot direction of segments
                (1) nodes(dict)             : dict of nodes with xy coordinates
                (2) graph(dict)             : graph that contains routes (nodes and edges)
                (3) nodes_loc(dict)         : {name: node} of fixed location
                (4) location_p(dict)        : {name_fixed_location: coordinates}
                (5) set_of_stocks (list)    : name of the stocks
            """         
        segment, points = filter_data(self.nodescsv,'PeteLien_bigger.png')
        out = delete_doubles(segment)
        letter = string.ascii_lowercase
        letter = list(letter)
        nodes = {}
        new_out =[]
        for let_i, segment in enumerate(np.unique(out[:,2])):
            let = letter[let_i]
            data_segm =out[out[:,2]==segment]
            for seg_i,c_data_segm in enumerate(data_segm):
                val_data_segm = list(c_data_segm[:2])
                if val_data_segm in nodes.values():
                    x = [k for k,v in nodes.items() if v == val_data_segm]
                    new_out.append(list(data_segm[seg_i])+[x[0]])
                    continue
                nodes[let+str(seg_i)] = val_data_segm
                new_out.append(list(data_segm[seg_i])+[let+str(seg_i)])
        new_out = pd.DataFrame(new_out, columns = ['x','y','seg','node'])
        graph ={}
        for seg_n in np.unique(new_out['seg']):
            data_seg_out = np.array(new_out[new_out['seg']== seg_n])
            for i_dato in range(data_seg_out.shape[0]-1):
                bef_p = data_seg_out[i_dato][:2]
                node_bef = find_key(bef_p, nodes)
                aft_p = data_seg_out[i_dato+1][:2]
                node_aft = find_key(aft_p, nodes)
                if node_aft == node_bef:
                    continue
                name_node =node_bef+node_aft
                if node_bef not in graph:
                    graph[node_bef] = list()
                res_bef_aft  = aft_p-bef_p
                distance = round(np.sqrt(np.sum(np.square(res_bef_aft))),3)
                graph[node_bef].append((node_aft, distance))
        nodes_loc = {key_lp:node_lp for key_lp in self.fixedloc for
             node_lp in nodes if self.fixedloc[key_lp] == nodes[node_lp]}
        return new_out, nodes, graph, nodes_loc
    
    def get_map(self):
        fig, ax = plt.subplots()
        ax.imshow(cv2.imread(self.image))
        for segment in np.unique(self.new_out['seg']):
            sample = np.array(self.new_out[self.new_out['seg']==segment][['x','y']])
            for x1, x2 in zip(sample[:-1],sample[1:]):
                ax.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                    arrowstyle='->', linestyle="--", lw=1, color='#ccffe7')) 
        for index,  location in enumerate(self.fixedloc):
            coordinate =self.fixedloc[location]
            size = 20
            marker = 'x'
            if 'tock' not in location:
                color_f = 'r'
                if 'hovel' in location:
                    color_f = 'k'
                    marker = None
                    size =4
                    location = 'Loader Ini.'
            arrowprops=dict(arrowstyle='->', color=color_f, linewidth=1)
            ax.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=2, c= color_f)
            ax.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0],coordinate[1]-25), c= color_f,fontsize =6)

            # if location == 'Entrance':
            #     ax.add_patch(circle_entrance)
                #ax1.add_patch(circle_entrance)
        plt.suptitle('Map guide + distances + directions of routes')
        ax.set_title("Customer's cycle")
        plt.tight_layout()  
        return fig
    def get_distances(self):
        list_a = []
        list_b = []
        start = 'start'
        entrance = 'entrance'
        end = 'scale'
        path = Dijkstra(self.graph, self.nodes_loc[start],self.nodes_loc[entrance])
        list_a.append([start, entrance, round(path[2])])
        for ind, st in enumerate(self.setstocks):
            path = Dijkstra(self.graph, self.nodes_loc[entrance],self.nodes_loc[st])
            path1 = Dijkstra(self.graph, self.nodes_loc[st],self.nodes_loc[end])
            list_a.append([entrance, st, round(path[2])])
            list_a.append([st, end, round(path1[2])])
        permutations = list(itertools.permutations(self.setstocks,2))
        for permu in permutations:
            bef = permu[0]
            aft = permu[1]
            path = Dijkstra(self.graph, self.nodes_loc[bef],self.nodes_loc[aft])
            list_b.append([bef, aft, round(path[2])])
        list_a = pd.DataFrame(list_a, columns = ['Ini_Point', 'End_Point','Dist.(px.)'])
        list_b = pd.DataFrame(list_b, columns = ['Ini_Point', 'End_Point','Dist.(px.)'])
        return list_a, list_b

    def location_piles(self):   
        loc_piles_info = {}
        for index, stock in enumerate(np.unique(self.stockpileinfo['stock_name'])):
            info_stock = self.stockpileinfo[self.stockpileinfo['stock_name']==stock]
            max_date = max(info_stock['stock_times'])
            filtered = info_stock[info_stock['stock_times'] == max_date]
            tonnage = filtered['stock_tonnage']
            material = filtered['fixed_mat']
            loc_piles_info[stock] = self.fixedloc[stock]+list(tonnage)+[self.palette_piles[index]] +[list(material)[0][:2]]
        return loc_piles_info
    
    def get_data_animation(self):
        self.dict_discretize = dict()
        self.dict_delay_sec = dict()
        SHOVEL_NODE = self.shovelnode
        if self.requirement.shape[0]:
            #print(self.to_w_m)
            for index_ind, datos in enumerate(self.to_w_m):
                where = self.from_w[index_ind]
                to_text = self.to_w_m[index_ind]
                to = to_text.split('_')[0]
                val_from_w = self.nodes_loc[where]
                val_to_entrance = self.nodes_loc[self.entrance]
                val_to_w = self.nodes_loc[to]
                val_end = self.nodes_loc[self.end]
                #path, nodes, time - entrance
                path_to_entrance = Dijkstra(self.graph,val_from_w, val_to_entrance)
                nodes_entrance = path_to_entrance[1]
                time_to_entrance = int(math.ceil(path_to_entrance[2]/self.truck_velocity))
                points_entrance = interpolate(self.nodes,nodes_entrance,self.interpolator, 
                    self.truck_velocity).tolist()
                #print(time_to_entrance,points_entrance.shape)
                #path, nodes, time - stock
                path_to_stock = Dijkstra(self.graph,val_to_entrance, val_to_w)
                nodes_stock = path_to_stock[1]
                time_to_stock = int(math.ceil(path_to_stock[2]/self.truck_velocity))
                time_to_load_sec = int(math.ceil(self.truck_capacity[index_ind]/self.loader_payload))*int(self.loader_cycletime)
                time_to_stock_sec = time_to_stock+time_to_load_sec     #use this for sec
                time_to_load = int(math.ceil(self.truck_capacity[index_ind]/self.loader_payload))*int(self.loader_cycletime/self.interpolator)
                time_to_stock += time_to_load
                points_stock = interpolate(self.nodes,nodes_stock,self.interpolator, 
                    self.truck_velocity)
                #print('original stock {}'.format(points_stock.shape))
                extra_load = np.array([list(points_stock[-1]) for x in range(time_to_load)])
                points_stock = np.concatenate((points_stock,extra_load)).tolist()
                #print(time_to_stock,points_stock.shape)
                #path, nodes, time - scale
                path_to_end =  Dijkstra(self.graph,val_to_w, val_end)
                nodes_end = path_to_end[1]
                time_to_end = int(math.ceil(path_to_end[2]/self.truck_velocity))
                points_end = interpolate(self.nodes,nodes_end,self.interpolator, 
                    self.truck_velocity).tolist()
                #print(time_to_end,points_end.shape)
                #path, nodes, time - shovel
                shovel_path = Dijkstra(self.graph,SHOVEL_NODE, val_to_w)
                nodes_shovel = shovel_path[1]
                time_travel_shovel = int(math.ceil(shovel_path[2]/self.loader_velocity))
                time_travel_shovel_sec = time_travel_shovel+time_to_load_sec
                #print('ini shovel time: {}'.format(time_travel_shovel))
                time_travel_shovel += time_to_load
                points_shovel = interpolate(self.nodes,nodes_shovel,self.interpolator, 
                    self.loader_velocity)

                if points_shovel.shape[0] == 0:
                    points_shovel = np.array([self.nodes[nodes_shovel[0]]])
                extra_shovel = np.array([list(points_shovel[-1]) for x in range(time_to_load)])
                points_shovel = np.concatenate((points_shovel,extra_shovel))
                points_shovel = points_shovel.tolist()
                SHOVEL_NODE= val_to_w
                i= [index for index,dat in enumerate(self.to_master_text) if 
                    self.to_master_text[index]==to_text]
                self.dict_discretize[where+str(i)] = [points_entrance,points_stock,points_end,
                    points_shovel,time_travel_shovel]
                self.dict_delay_sec[where+str(i)] = [time_to_entrance, time_to_stock_sec,time_to_end,time_travel_shovel_sec,time_to_load_sec]
            delay_master = self.shape_matrixmom_delay()
            return self.shape_matrixmom_sec(),self.new_out, self.c_order, self.fixedloc,self.loc_piles, delay_master, self.truck_capacity,self.to_master_text,self.to_w, self.to_w_m,self.min_dectime,self.interpolator
        else:
            return None
    def add_data_shovel(self, datarows):
        self.sh.worksheet('Shovel').append_rows(
                datarows
                    ,value_input_option="USER_ENTERED")
    def add_data_billing(self, datarows):
        self.sh.worksheet('Billing').append_rows(
                datarows,value_input_option="USER_ENTERED"
            )
    def add_data_stock(self, datarows):
        
        self.sh.worksheet('StockInfo').append_rows(
                datarows,value_input_option="USER_ENTERED"
            )
    def add_data(self,datarows):
        self.sh.worksheet('CustomerStatus').append_rows(
                datarows,value_input_option="USER_ENTERED"
            )
    def shape_matrixmom_delay(self, add_db = True):
        delay = 0
        customer = np.unique(self.requirement['customer_name'])[0]
        decisor_time_all = [(self.decision_time_sum[ind-1]-self.decision_time_sum[ind]) 
            if (self.decision_time_sum[ind-1]-self.decision_time_sum[ind])>=0 and ind>0 
            else 0 for ind,x in enumerate(self.decision_time_sum)]
        delay+= sum(decisor_time_all)
        stock_before = 0
        idle_p_truck = 0

        list_stock = []
        list_custstatus = []
        list_shovel = []
        list_bill = []
        for index, key in enumerate(self.dict_delay_sec):
            ido_key = int(key[-2])
            truck = self.c_order[index]
            truck_db = self.c_order[ido_key]
            customer_data = self.dict_delay_sec[key]
            stock = self.to_master[ido_key]
            #lastpointentrance
            time_entrance =customer_data[0]
            first_time_tot = self.min_dectime +int(self.decision_time_sum[index])
            end_time_entrance = first_time_tot+time_entrance
            time_entrance+= int(self.decision_time_sum[index]) +decisor_time_all[index]
            time_stock =customer_data[1] 
            time_end =customer_data[2]
            time_shovel = customer_data[3]
            time_shov_in = time_shovel
            time_to_load = customer_data[4]
            if index == 0:
                none_shovel = time_entrance + time_stock- time_shovel
                time_shovel += none_shovel
                stock_before= time_stock
            else:
                len_add_stock = stock_before - int(self.decision_time[index]) +time_shovel - time_stock
                stock_before = time_stock
                if len_add_stock> 0:
                    delay += len_add_stock
                    idle_p_truck +=len_add_stock
            if add_db:
                fin_ton  = self.stock_tonnage[stock] -self.truck_ton[truck_db]
                list_custstatus.append([self.max_id_cust,customer,truck_db, stock,int(first_time_tot),int(end_time_entrance),
                    int(end_time_entrance-first_time_tot),'toentrance',0, int(self.sched)])
                # self.add_data(self.max_id_cust,customer,truck_db, stock,first_time_tot,end_time_entrance,end_time_entrance\
                #     -first_time_tot,'toentrance',0, self.sched)
                idle_p_truck += decisor_time_all[index]
                init_time_idle = end_time_entrance
                end_time_idle = init_time_idle+idle_p_truck
                self.max_id_cust+=1
                list_custstatus.append([self.max_id_cust,customer,truck_db, stock,int(init_time_idle),int(end_time_idle),int(idle_p_truck),
                    'idle',0, int(self.sched)])
                # self.add_data(self.max_id_cust,customer,truck_db, stock,init_time_idle,end_time_idle,int(idle_p_truck),\
                #     'idle',0, self.sched)
                self.max_id_cust+=1
                init_time_stock = end_time_idle
                end_time_stock =init_time_stock + time_stock -time_to_load
                list_shovel.append([self.max_id_shovel,'L1',stock,truck_db, int(end_time_stock-time_shov_in),int(end_time_stock),
                    int(time_shov_in),'tostock', int(self.sched)])
                # self.add_data_shovel(self.max_id_shovel,'L1',stock,truck_db, end_time_stock-time_shov_in,end_time_stock,\
                #     time_shov_in,'tostock', self.sched)
                list_custstatus.append([self.max_id_cust,customer,truck_db,stock, int(init_time_stock),int(end_time_stock),
                    int(time_stock -time_to_load),'tostock',0, int(self.sched)])
                # self.add_data(self.max_id_cust,customer,truck_db,stock, init_time_stock,end_time_stock,\
                #     time_stock -time_to_load,'tostock',0, self.sched)
                self.max_id_cust+=1
                init_time_load = end_time_stock
                end_time_load =init_time_load + time_to_load
                self.max_id_shovel+=1
                list_shovel.append([self.max_id_shovel,'L1',stock,truck_db, int(init_time_load),int(end_time_load),
                    int(time_to_load),'loading', int(self.sched)])
                # self.add_data_shovel(self.max_id_shovel,'L1',stock,truck_db, init_time_load,end_time_load,\
                #     time_to_load,'loading', self.sched)
                self.max_id_shovel+=1
                list_custstatus.append([self.max_id_cust,customer,truck_db,stock, int(init_time_load),int(end_time_load),
                    int(time_to_load),'loading',0, int(self.sched)])
                # self.add_data(self.max_id_cust,customer,truck_db,stock, init_time_load,end_time_load,\
                #     time_to_load,'loading',0, self.sched)
                self.max_id_cust+=1
                init_time_end = end_time_load
                end_time_end = init_time_end+time_end
                list_custstatus.append([self.max_id_cust,customer,truck_db,stock, int(init_time_end),int(end_time_end),
                    int(time_end),'toscale',0, int(self.sched)])
                # self.add_data(self.max_id_cust,customer,truck_db,stock, init_time_end,end_time_end,\
                #     time_end,'toscale',0, self.sched)
                if self.add_sched == 0:
                    self.max_id_stockinfo+=1
                    list_stock.append([self.max_id_stockinfo, stock, int(self.stock_id[stock][0]),
                        int(fin_ton), int(end_time_end)])
                    # self.add_data_stock(self.max_id_stockinfo, stock, self.stock_id[stock][0],\
                    #     fin_ton, end_time_end)
                price = int(self.stock_id[stock][1])
                self.stock_tonnage[stock] = fin_ton
                self.max_id_cust+=1
                list_custstatus.append([self.max_id_cust,customer,truck_db, stock,int(end_time_end),int(end_time_end+5),
                    5,'outside',0, int(self.sched)])
                # self.add_data(self.max_id_cust,customer,truck_db, stock,end_time_end,end_time_end+5,\
                #     5,'outside',0, self.sched)
                self.max_billing+=1
                list_bill.append([self.max_billing,customer,truck_db, self.truck_ton[truck_db],
                    int(self.truck_ton[truck_db]*price), int(end_time_end+5),'sent', int(self.sched)])
                list_bill.append([self.max_billing,customer,truck_db, self.truck_ton[truck_db],
                    int(self.truck_ton[truck_db]*price), int(end_time_end+10),'paid', int(self.sched)])
                #self.add_data_billing(self.max_billing,customer,truck_db, self.truck_ton[truck_db],100, end_time_end+5, self.sched)
        self.add_data(list_custstatus)
        self.add_data_billing(list_bill)
        self.add_data_shovel(list_shovel)
        self.add_data_stock(list_stock)
        return int(delay)  

    def shape_matrixmom_sec_newmet(self):
        N_Simulations = len(self.c_order)
        change_stockpiles ={}
        matrix_for_customer = np.full((2**2**4,N_Simulations+1,2),None)
        matrix_for_shovel = np.empty(shape = (0,3), dtype=object)
        unassigned = [0,0,0]
        costumer_palette = np.full((2**2**4,N_Simulations+1,3),unassigned, dtype=float)
        #ADAPTING TIME TO TIME INTERVAL
        decision_time = [int(x/self.interpolator) for x in self.decision_time]
        decision_time_sum = [int(x/self.interpolator) for x in self.decision_time_sum]
        decisor_time_all = [(decision_time_sum[ind-1]-decision_time_sum[ind])
            if (decision_time_sum[ind-1]-decision_time_sum[ind])>=0 and ind>0 
            else 0 for ind,x in enumerate(decision_time_sum)]
        stock_before = 0
        #print(self.dict_discretize)
        for index, key in enumerate(self.dict_discretize):
            customer_data = self.dict_discretize[key]
            pos = int(key[-2])+1
            #lastpointentrance

            last_point_entrance = customer_data[0][-1]
            last_p_x = last_point_entrance[0]
            last_p_y = last_point_entrance[1] 
            to_entrance= [[None, None] for x in range(int(decision_time_sum[index]))]+customer_data[0]+[[last_p_x,last_p_y] for i in range(decisor_time_all[index])]
            to_stock = customer_data[1]
            last_x_entr = to_stock[0][0]
            last_y_entr = to_stock[0][1]
            to_end = customer_data[2]
            sho_to_sho = customer_data[3]
            if len(sho_to_sho) ==1:
                last_x_sho = sho_to_sho[0][0]
                last_y_sho = sho_to_sho[0][1]
            else:
                last_x_sho = sho_to_sho[-1][0]
                last_y_sho = sho_to_sho[-1][1]
            if index == 0:
                none_shovel = [[None, None] for x in range(len(to_entrance+to_stock) - len(sho_to_sho))]
                sho_to_sho = none_shovel+sho_to_sho
                if index+1 == len(self.dict_discretize):
                    end_shovel = [[last_x_sho, last_y_sho] for x in range(len(to_end))]
                    sho_to_sho = sho_to_sho+end_shovel
                stock_before= len(to_stock)
            else:
                len_add_stock = stock_before - decision_time[index] +len(sho_to_sho) - len(to_stock)
                stock_before = len(to_stock)
                if len_add_stock> 0:
                    to_stock = [[last_x_entr, last_y_entr] for x in range(int(len_add_stock))]+ to_stock
                    stock_before= len(to_stock)
                if index+1== len(self.dict_discretize):
                    sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(len(to_end))]
                else:
                    sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(int(len_add_stock*-1))]
            
            if index == len(self.dict_discretize)-1:
                large = len(to_entrance+to_stock+to_end)
                arange_time = np.arange(self.min_dectime, self.min_dectime+large)
                matrix_for_customer[:,0][0:large] = np.array(arange_time, dtype=object)
            len_add = len(to_entrance+to_stock+to_end)
            #change stockpiles  
            # print('len_add {}'.format(len_add))
            # print(len(to_entrance))
            # print(len(to_stock))
            # print(len(to_end))
            # print(np.array(to_entrance+to_stock+to_end, dtype=object).shape[0])
            matrix_for_customer[:,pos][0:len_add] = np.array(to_entrance+to_stock+to_end, dtype=object)
            costumer_palette[:,pos][0:len_add] = np.array([self.palette_customer[
                index]for x in range(len_add)])
            len_add_sho = len(sho_to_sho)
            np_add_sho = np.arange(self.min_dectime, self.min_dectime+len_add_sho).reshape(len_add_sho,1)
            matrix_for_shovel = np.concatenate([matrix_for_shovel,np.hstack((np_add_sho, np.array(sho_to_sho, dtype=object)))])
            change_stockpiles[(self.to_w[index], index)] = [self.min_dectime+len(to_entrance+to_stock)-1, len(to_entrance+to_stock)]
        return matrix_for_customer, matrix_for_shovel, costumer_palette,large,change_stockpiles
    def shape_matrixmom_sec(self):
        N_Simulations = len(self.c_order)
        change_stockpiles ={}
        matrix_for_customer = np.full((2**2**4,N_Simulations,2),None)
        matrix_for_shovel = np.empty(shape = (0,2), dtype=object)
        unassigned = [0,0,0]
        costumer_palette = np.full((2**2**4,N_Simulations,3),unassigned, dtype=float)
        #ADAPTING TIME TO TIME INTERVAL
        decision_time = [int(x/self.interpolator) for x in self.decision_time]
        decision_time_sum = [int(x/self.interpolator) for x in self.decision_time_sum]
        decisor_time_all = [(decision_time_sum[ind-1]-decision_time_sum[ind])
            if (decision_time_sum[ind-1]-decision_time_sum[ind])>=0 and ind>0 
            else 0 for ind,x in enumerate(decision_time_sum)]
        stock_before = 0
        #print(self.dict_discretize)
        for index, key in enumerate(self.dict_discretize):
            customer_data = self.dict_discretize[key]
            pos = int(key[-2])
            #lastpointentrance

            last_point_entrance = customer_data[0][-1]
            last_p_x = last_point_entrance[0]
            last_p_y = last_point_entrance[1] 
            to_entrance= [[None, None] for x in range(int(decision_time_sum[index]))]+customer_data[0]+[[last_p_x,last_p_y] for i in range(decisor_time_all[index])]
            to_stock = customer_data[1]
            last_x_entr = to_stock[0][0]
            last_y_entr = to_stock[0][1]
            to_end = customer_data[2]
            sho_to_sho = customer_data[3]
            if len(sho_to_sho) ==1:
                last_x_sho = sho_to_sho[0][0]
                last_y_sho = sho_to_sho[0][1]
            else:
                last_x_sho = sho_to_sho[-1][0]
                last_y_sho = sho_to_sho[-1][1]
            if index == 0:
                none_shovel = [[None, None] for x in range(len(to_entrance+to_stock) - len(sho_to_sho))]
                sho_to_sho = none_shovel+sho_to_sho
                if index+1 == len(self.dict_discretize):
                    end_shovel = [[last_x_sho, last_y_sho] for x in range(len(to_end))]
                    sho_to_sho = sho_to_sho+end_shovel
                stock_before= len(to_stock)
            else:
                len_add_stock = stock_before - decision_time[index] +len(sho_to_sho) - len(to_stock)
                stock_before = len(to_stock)
                if len_add_stock> 0:
                    to_stock = [[last_x_entr, last_y_entr] for x in range(int(len_add_stock))]+ to_stock
                    stock_before= len(to_stock)
                if index+1== len(self.dict_discretize):
                    sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(len(to_end))]
                else:
                    sho_to_sho = sho_to_sho + [[last_x_sho, last_y_sho] for x in range(int(len_add_stock*-1))]
            if index == len(self.dict_discretize)-1:
                large = len(to_entrance+to_stock+to_end)
            #change stockpiles      
            matrix_for_customer[:,pos][0:len(to_entrance+to_stock+to_end)] = np.array(to_entrance+to_stock+to_end, dtype=object)
            costumer_palette[:,pos][0:len(to_entrance+to_stock+to_end)] = np.array([self.palette_customer[
                index]for x in range(len(to_entrance+to_stock+to_end))])
            matrix_for_shovel = np.concatenate([matrix_for_shovel,np.array(sho_to_sho, dtype=object)])
            change_stockpiles[(self.to_w[index], index)] = len(to_entrance+to_stock)
        return matrix_for_customer, matrix_for_shovel, costumer_palette,large,change_stockpiles


def to_excel():
    today = date.today()
    year = today.year
    month = today.month
    day = today.day
    output = BytesIO()
    workbook = xlsxwriter.Workbook(output, {'in_memory': True})
    worksheet = workbook.add_worksheet()

    worksheet.write('A1', 'ID')
    worksheet.write('B1', 'Company Name')
    worksheet.write('C1', 'Truck ID')
    worksheet.write('D1', 'Material')
    worksheet.data_validation('$D$2:$D$8', {'validate': 'list',
                                  'source': ['lime', 'crushedstone',
                                       'gravel','carbonate']})

    worksheet.write('E1', 'Tonnage')
    worksheet.write('F1', 'Date') 
    #worksheet.write_comment('F1', 'Format: YY/MM/DD')
    worksheet.data_validation('$F$2:$F$20', {'validate': 'date',
                                    'criteria': 'between',
                                    'minimum': date(year, month, day),
                                    'maximum': date(year, 12, 31),
                                    'input_title': 'Enter date. Format:',
                                  'input_message': 'YY/MM//DD',
                                  'error_title': 'Input value not valid!',
                                  'error_message': 'Insert in a correct format or valid date'
                                    })
    worksheet.write('G1', 'Time')
    worksheet.data_validation('$G$2:$G$20', {'validate': 'time',
                                    'input_title': 'Enter time. Format:',
                                    'criteria': 'between',
                                  'minimum': time(9, 0,0),
                                  'maximum': time(17, 0,0),
                                  'input_message': 'hh:mm:ss',
                                  'error_title': 'Input value not valid!',
                                  'error_message': 'Insert in a correct format or valid time'
                                    })
    workbook.close()
    return output
