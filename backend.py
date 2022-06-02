import datetime
import itertools
import math
import string

import cv2
import gspread
from matplotlib import scale
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from back import *
from schedule import *
from utils import *


class ConnectCustomer:
    def __init__(self, customer_req, loading_properties,
                materials_stocks,image_animation = 'Pete_big.jpg',
                nodescsv ='new_nodes.csv', interval = 5):
        self.customer_req = customer_req
        self.materials_stocks = materials_stocks
        self.image_animation = image_animation
        self.loading_properties = loading_properties
        # Image that comes from  
        self.nodescsv = nodescsv
        self.interval = interval
        # Tonnage of each stock
        self.value_stocks = 5000
        # Fixed location: node
        self.fixedloc = {
            'start': 'a0','entrance':'a5','stock1':'d2','stock2':'e1',
            'stock3':'f1','stock4':'g3','stock5':'h2','stock6':'k3',
            'stock7':'l5','stock8':'m2', 'scale':'c22', 'end':'c25'
            }
        # Fixed location: color (Animation purpose)
        self.colors_f = {'start': '#f6fa00', 'entrance':'#f6fa00', 'stock1':'b',
                    'stock2':'b','stock3':'b','stock4':'b', 'stock5':'b',
                    'stock6':'b','stock7':'b','stock8':'b', 'scale':'b',
                     'end':'b'
                     }
        self.setstocks = {k:v for k,v in self.fixedloc.items() if 'stock' in k}
        self.stock_tonnage = {stock:self.value_stocks for stock in self.setstocks}
        #self.list_stocks = [k for k in self.fixedloc]
        self.start = 'start'
        self.entrance ='entrance'
        self.scale = 'scale'
        self.end = 'end'
        # Convert km/h to m/s
        convert_meters_sec = 0.28
        self.truck_velocity = 20 * convert_meters_sec  
        self.new_out, self.nodes, self.graph = readnodesfile(nodescsv)
        # We can create this palette based on materials
        self.palette_piles = sns.color_palette(
            "viridis",n_colors=len(self.setstocks)+1
            )
        self.availability_customer_req()
        # truck: summation time (initial)
    def availability_customer_req(self):
        """ 
        Split the customer requirement based on what is available and
        what is not.
        """
        # Return the material corresponding to each stock
        stocks_materials = return_stock_for_material(self.materials_stocks)
        # Return the stocks corresponding to each loader
        loadingeq_stocks = return_loading_for_stocks(self.loading_properties)
        # Loader to each specific stock
        stocks_loadingequip = {
            val:k for k,v in loadingeq_stocks.items() for val in v
            }
        # Materials required for the customer
        self.materials_customer_req = np.unique(self.customer_req['Material'])
        # Materials required and in stocks
        self.available_materials = [
            m for m in self.materials_customer_req if m in stocks_materials.values()
            ]
        # Materials required and NOT in stocks
        self.unavailable_materials = [
            m for m in self.materials_customer_req if m not in stocks_materials.values()
            ]
        # Filter customer requirement with what is currently in stocks
        self.customer_req_available = self.customer_req[
            self.customer_req['Material'].isin(self.available_materials)]
        # Filter customer requirement with what is NOT in stocks
        self.customer_req_unavailable = self.customer_req[
            self.customer_req['Material'].isin(self.unavailable_materials)]
        self.customer_req_available = [
                list(x[:4])
                +stock_load_truckmat(x[3], stocks_materials, stocks_loadingequip)[0] 
                +stock_load_truckmat(x[3], stocks_materials, stocks_loadingequip)[1]
                +list(x[4:]) for x in np.array(self.customer_req_available)]
        self.customer_req_available = pd.DataFrame(np.array(self.customer_req_available),
            columns =['Id','Company','Truck','Rock','Dest',
            'TypeLoader','Tonnage','Date','Time'])
        self.add_epoch_column()
        self.trucks = [truck for truck in self.customer_req_available['Truck']]

    def connect_gsheet(self):
        """Connects to the Google Sheet (P2RData) to retrieve + upload data."""        
        self.connect = gspread.service_account_from_dict({
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
        self.sh = self.connect.open('P2RData')
        # connect_back = Customer_Reader(sh, datarequired,datamissed,first_st, schedule)
        # data_ani= 0
        # #print(connect_back.requirement)
        # if connect_back.requirement.shape[0]>0:
        #     data_ani = connect_back.get_data_animation()
        #     matrix_custo = data_ani[0][0]
        #     matrix_sho = data_ani[0][1]
        # return data_ani,matrix_custo,matrix_sho

    def get_data_animation(self,first_stock, last_data_customer):
        """ 
        Returns the matrix that contains coordinates based on how trucks have moved
        for self.interval seconds.
        Args:
            first_stock(str): Destination for the first truck assigned
            schedule(list of tuples): Best truck:stock assignment
        """
        dict_discretize = dict()
        # TODO: Think if we need self.start as the string or its node?

        previous_loader_node = self.fixedloc[first_stock]
        start_node = self.fixedloc[self.start]
        entrance_node = self.fixedloc[self.entrance]
        scale_node = self.fixedloc[self.scale]
        end_node = self.fixedloc[self.end]
        # From start to entrance
        path_to_entrance = Dijkstra(self.graph, start_node, entrance_node)
        nodes_to_entrance = path_to_entrance[1]
        time_to_entrance = int(math.ceil(path_to_entrance[2]/self.truck_velocity))
        points_to_entrance = interpolate(self.nodes,nodes_to_entrance,self.interval, 
                    self.truck_velocity)
        # Previous time to stock
        previous_truck_tstock = 0
        for index, truck in enumerate(np.array(last_data_customer['Truck'])):
            data_for_truck = last_data_customer[last_data_customer['Truck'] == truck]
            stock = str(data_for_truck['Dest'])
            truck_tonnage = str(data_for_truck['Tonnage'])
            # TODO: Change Loader information based on what is assigned to each truck
            loader_velocity = self.loading_properties['loader'][2]
            loader_payload =  self.loading_properties['loader'][3]
            loader_cycletime =  self.loading_properties['loader'][4]
            stock_node = self.fixedloc[stock]
            # TODO: Add more points for entrance based on time of arrival
            # Modify start to entrance based on arrival time
            if index>0:
                previous_truck = np.array(last_data_customer['Truck'])[index-1]
                previous_truck_arrtime = last_data_customer[
                    last_data_customer['Truck'] == previous_truck]['ArrTime']
                current_truck_arrtime = data_for_truck['ArrTime']
                if current_truck_arrtime < previous_truck_arrtime:
                    waiting_step = math.ceil(
                        (previous_truck_arrtime - current_truck_arrtime)/self.interval)
                    points_waiting = np.array([list(points_to_entrance[-1]) for x in range(waiting_step)])
                    points_to_entrance = np.concatenate((points_to_entrance,points_waiting))
            points_to_entrance = points_to_entrance.tolist()
            # From entrance to stock
            path_to_stock = Dijkstra(self.graph,entrance_node, stock_node)
            nodes_to_stock = path_to_stock[1]
            time_to_stock = int(math.ceil(path_to_stock[2]/self.truck_velocity))
            time_to_load = int(math.ceil(truck_tonnage/loader_payload))*int(loader_cycletime)
            time_to_stock += time_to_load
            points_to_stock = interpolate(self.nodes,nodes_to_stock,self.interval, 
                    self.truck_velocity)
            points_loading = np.array([list(points_to_stock[-1]) for x in range(time_to_load/self.interval)])
            points_to_stock = np.concatenate((points_to_stock,points_loading)).tolist()

            # From stock to scale
            path_to_scale = Dijkstra(self.graph,stock_node, scale_node)
            nodes_to_scale = path_to_scale[1]
            time_to_scale = int(math.ceil(path_to_scale[2]/self.truck_velocity))
            points_to_scale = interpolate(self.nodes,nodes_to_scale,self.interval, 
                    self.truck_velocity)
            time_scaling = 30
            points_scaling = np.array([list(points_to_scale[-1]) for x in range(time_scaling/self.interval)])
            points_to_scale = np.concatenate((points_to_scale,points_scaling)).tolist()

            # From stock to stock (loader)
            path_loader = Dijkstra(self.graph,previous_loader_node, stock_node)
            nodes_loader = path_loader[1]
            time_loader = int(math.ceil(path_loader[2]/loader_velocity)) + time_to_load
            points_loader =  interpolate(self.nodes,nodes_loader,self.interval, 
                    loader_velocity)
            if points_loader.shape[0] == 0:
                points_loader = np.array(points_to_stock[-1])
            points_loader = np.concatenate((points_loader,points_loading)).tolist() 
            previous_loader_node = stock_node

            # Modify path to entrance based on idle time
            queue_step = 0
            if index>0:
                queue_time = previous_truck_tstock + time_loader - time_to_stock
                if queue_time > 0:
                    queue_step = queue_time/self.interval
            points_queue = np.array([list(points_to_entrance[-1]) for x in range(queue_step)])
            points_to_entrance = np.concatenate((points_to_entrance,points_queue))
            # Saving time to stock of the previous truck
            previous_truck_tstock = time_to_stock
        # From scale to end
        path_to_end = Dijkstra(self.graph, scale_node, end_node)
        nodes_to_end = path_to_end[1]
        time_to_end = int(math.ceil(path_to_end[2]/self.truck_velocity))
        points_to_end = interpolate(self.nodes,nodes_to_end,self.interval, 
                    self.truck_velocity).tolist()
        # TODO: Build the matrix for trucks and shovels 

    def add_epoch_column(self):
        """ Modify date (yy-mm-dd.. format) to epoch, which is a number.
        Returns
            data_insert(df): Modified customer requirement.
        """
        self.customer_req_available['Date'] = self.customer_req_available.apply(
                        lambda r : datetime.datetime.combine(r['Date'],r['Time']),1)
        self.customer_req_available['Epoch'] = (self.customer_req_available['Date']
                - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        self.customer_req_available['Epoch'] = self.customer_req_available['Epoch'] + 6*3600

    def scheduling(self):
        """
        Gives the best sequence of assignments, (truck,stock), that minimizes the 
        overall idle time of trucks.

        Returns:
            selected_truck_stock (list of tuples): Best sequence of assignments of 
            (truck, stock)
            minval(int): Total idle time of the best assignments 
            shovelini(str): Initial position of a loader - "stock1"
        """

        min_t= min(self.customer_req_available['Epoch'])
        self.customer_req_available = add_index_typeloaders(self.customer_req_available)
        # Converting epochtime to integers from 1-100
        self.customer_req_available['ArrTime'] = np.array([
            time_-min_t for time_ in np.array(self.customer_req_available['Epoch'])])
        # Save brute force when comparing stocks for materials
        # Schedule (truck,stock): [best_assignment, minimumtime]
        prospect_schedules = {}
        for loader in np.unique(self.customer_req_available['Load']):
            customer_req_selected = self.customer_req_available[
                self.customer_req_available['Load'] == loader]
            # TODO: This might be avoided when creating another binary variable
            # in gurobi
            if loader == 'hopper':
                prospect_schedules[loader] = customer_req_selected['Truck']
            else:
                loader_velocity = self.loading_properties[loader][2]
                loader_payload =  self.loading_properties[loader][3]
                loader_cycletime =  self.loading_properties[loader][4]
                # Saves truck: material, time_arrival (1-100), tonnage
                cinfo  = {
                    x[2]:[x[3],x[11], x[6]] for x in np.array(customer_req_selected)}
                # Truck: [Destinations]. This is important for permutations_dicts
                truck_destinations = {truck:list(
                        customer_req_selected[customer_req_selected['Truck'] == truck]['Dest'])[0]
                        for truck in np.unique(customer_req_selected['Truck'])}
                keys_truck, values_destination = zip(*truck_destinations.items())
                # Permutations of Truck and Destinations(This is being looped below)
                permut_trucks_stocks = [list(zip(keys_truck, v)) 
                                        for v in itertools.product(*values_destination)]
                
                for perm_t_st in permut_trucks_stocks:
                    # Adds dummy variable. The values for the dummy will be [0,0,0]
                    customer_req_truckinfo = self.convert_cust_req(
                                            cinfo, perm_t_st)

                    # Sorts customer_req_truckinfo based on arrival time (sanity check)
                    sorted_times = {key:customer_req_truckinfo[key][1]
                                 for i,key in enumerate(customer_req_truckinfo)}
                    sorted_times = dict(sorted(sorted_times.items(), key=lambda item: item[1]))
                    customer_req_truckinfo = {
                        key: customer_req_truckinfo[key] for key in customer_req_truckinfo}
                    # Truck:summation of arrival time
                    truck_sumation_time ={k:customer_req_truckinfo[k][1] for 
                                                ind,k in enumerate(customer_req_truckinfo) if k != 'dummy'}
                    # Truck:time to stock from entrance + time to be loaded
                    truck_tostock_time = {k: round(
                        float(Dijkstra(self.graph,self.fixedloc['entrance'],self.fixedloc[customer_req_truckinfo[k][0]])[2])
                                    /self.truck_velocity)  # Dist/veloc = time to travel
                                    +int(math.ceil(customer_req_truckinfo[k][2]/loader_payload)*loader_cycletime)  # Time to load
                                    for k,v in customer_req_truckinfo.items() if k != 'dummy'}
                    # Time for loader to travel from stock of truck tr1 to stock of tr2.
                    loader_stock_time = {(tr1,tr2): round(
                        float(Dijkstra(self.graph, self.fixedloc[customer_req_truckinfo[tr1][0]], self.fixedloc[customer_req_truckinfo[tr2][0]])[2])
                        /loader_velocity #dist/veloc = time to travel
                        +int(math.ceil(customer_req_truckinfo[tr2][2]/loader_payload)*loader_cycletime))  # Tload
                        for tr1 in self.trucks for tr2 in self.trucks 
                            if tr1 != tr2 and 'dum' not in tr1 and 'dum' not in tr2}
                    # Compute the idle time based on an assignment of 2 trucks. These combina-
                    # tions fit in the way that TSP is solved. 
                    # TODO: We can also run the total time  and run a comparison between idle time vs total time (includes idle time)
                    truck_times = {(tr1,tr2): idle_time(
                            tr1,tr2,truck_sumation_time, truck_tostock_time,loader_stock_time)
                            for tr1 in self.trucks for tr2 in self.trucks if  tr1 != tr2}
                    # Updating prospect_schedules
                    scheduling(truck_times,perm_t_st,prospect_schedules, self.trucks)
                # Selecting the minimum idle time
                minimum_idle_time = min(np.array(list(prospect_schedules.values()),dtype=object)[:,1])
                # Selecting the permutation and best sequence that had the minimum idle time 
                best_permutation = [[perm_t_st,v[0]] for k, v in prospect_schedules.items() if v[1]==minimum_idle_time]
                # Truck: destination from permutation but not the best sequence
                initial_truck_destination = {x[0]:x[1] for x in best_permutation[0][0]}
                # Best sequence of (Truck,destination)
                schedule = [(x,initial_truck_destination[x]) for x in best_permutation[0][1]]
                initial_position_loader = schedule[0][1]
                return schedule, minimum_idle_time, initial_position_loader   

    def modify_req_schedule_dest(self, schedule):
        """ Modify the customer requirement based on the best assignments (schedule).
        It will delete multiple stocks for each truck and choose one based on sched.
        Args:
            schedule(list of tuples): pairs of truck, stock within the best assignments
    
        Returns
            requirement_in(df):  customer req. processed for a customer with one
            destination
        """
        empty_list = []
        for comb in schedule:
            truck = comb[0]
            dataf_truck = self.customer_req_available[
                self.customer_req_available['Truck']==truck]
            dataf_truck['Dest'] = comb[1]
            empty_list.append(dataf_truck)
        new_df = pd.concat(empty_list)
        # TODO: Delete multiple loaders based on the schedule
        return new_df


    def convert_cust_req(self, cust_req_dict, permut):
        """
        This will add a dummy node to the variable customer_req_truckinfo to perform the Asymmetric TSP,
        and return a dictionary similar to customer_req_truckinfo, but +dummy 
        (beginning).

        Args:
            cust_req_dict(dict): customer requirement with the form of truck: [mat, time
                                of arrival (compared to previous one), and tonnage]
            permut(list of tuples): (material,stock)

        Returns:
            new_cust_req_dict(dict): new customer requirement with the form of truck: [dest,
                                time of arrival (compared to previous one), and tonnage].
                                This includes the dummy variable at the beginning
                                with [0,0,0].
        """

        # Convert the list of tuples to array for better indexing
        permut_array = np.array(permut)
        new_cust_req_dict = {}
        # Add the dummy key with its values
        new_cust_req_dict['dummy']= [0,0,0]
        for truck in cust_req_dict:
            if truck in permut_array[:,0]:
                st = [permut_array[permut_array[:,0] == truck][0][1]]
            else:
                st = [st for st in self.materials_stocks if self.materials_stocks[st] == cust_req_dict[truck][0]]
            new_cust_req_dict[truck]= st + cust_req_dict[truck][1:]

            # Note: The commented code was there for some reasons, but I tested without it
            # and it gives me a correct result. It is left behing just in case.
        return new_cust_req_dict
