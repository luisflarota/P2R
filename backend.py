import datetime
import itertools
import math
import string

import cv2
import gspread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib import scale

from schedule import *
from utils import *


#TODO: See if we need an interval time different than 1 
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
            'stock7':'l5','stock8':'m2', 'scale':'c22', 'end':'c24'
            }
        # Fixed location: color (Animation purpose)
        self.colors_f = {'start': '#f6fa00', 'entrance':'#f6fa00', 'stock1':'b',
                    'stock2':'b','stock3':'b','stock4':'b', 'stock5':'b',
                    'stock6':'b','stock7':'b','stock8':'b', 'scale':'b',
                     'end':'b'
                     }
        # Stocks: node
        self.setstocks = {k:v for k,v in self.fixedloc.items() if 'stock' in k}
        self.stock_tonnage = {stock:self.value_stocks for stock in self.setstocks}
        self.start = 'start'
        self.entrance ='entrance'
        self.scale = 'scale'
        self.end = 'end'
        # Convert km/h to m/s
        convert_meters_sec = 0.28
        self.truck_velocity = 20 * convert_meters_sec
        # Getting the nodes (node: location) and the graph (node: its path)
        self.nodes, self.graph = readnodesfile(nodescsv)
        # TODO: We can create this palette based on materials
        self.palette_piles = sns.color_palette(
            "viridis",n_colors=len(self.setstocks)+1
            )
        # Getting the available/unavailable materials and the customer requirements based on them.
        self.availability_customer_req()
        
    def availability_customer_req(self):
        """ 
        Split the customer requirement based on what is available and
        what is not. It creates 4 attributes, available materials, unavailable materials,
        customer requirement available and customer requirement unavailable.
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
        # Creating a matrix for the customer req. available that has the necessary columns
        self.customer_req_available = [
                list(x[:4])
                +stock_load_truckmat(x[3], stocks_materials, stocks_loadingequip)[0] 
                +stock_load_truckmat(x[3], stocks_materials, stocks_loadingequip)[1]
                +list(x[4:]) for x in np.array(self.customer_req_available)]
        self.customer_req_available = pd.DataFrame(np.array(self.customer_req_available),
            columns =['Id','Company','Truck','Rock','Dest',
            'TypeLoader','Tonnage','Date','Time'])
        # Adding an epoch time to the customer_req_available
        self.add_epoch_column()
        # Add dummy to trucks so the order of the list does not get fixed
        self.trucks = ['dummy']+ [truck for truck in self.customer_req_available['Truck']]

    # TODO: Connecting and updating the google sheet. Either one or multiple methods
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
            last_data_customer(df): Customer requirement available after scheduling

        Returns:
            matrix_trucks_coordinates(matrix): Matrix that contains the coordinates for
                each truck. Columns represent each truck's positions and rows represent
                the truck's position at different times.
            TODO: Add more type of loaders and the amount of different loaders 
            matrix_loaders_coordinates(matrix): Matrix that contains the coordinates for
                the loader. Columns represent the loader's positions and rows represent 
                the loader's position at different times. 
        """
        
        # Create an empty matrix for trucks' coordinates
        matrix_trucks_coordinates = np.full((2**2**4,last_data_customer.shape[0],2),None)
        # Create an empty matrix for loaders' coordinates
        matrix_loaders_coordinates = np.empty(shape = (0,2), dtype=object)
        # TODO: Think if we need self.start as the string or its node?
        previous_loader_node = self.fixedloc[first_stock]
        start_node = self.fixedloc[self.start]
        entrance_node = self.fixedloc[self.entrance]
        scale_node = self.fixedloc[self.scale]
        # Time that truck takes to weigh itself
        time_scaling = 30
        end_node = self.fixedloc[self.end]
        
        # From start to entrance
        path_to_entrance = Dijkstra(self.graph, start_node, entrance_node)
        nodes_to_entrance = path_to_entrance[1]
        time_to_entrance = int(math.ceil(path_to_entrance[2]/self.truck_velocity))
        points_to_entrance = interpolate(self.nodes,nodes_to_entrance,self.interval, 
                    self.truck_velocity).tolist()
        # From scale to end
        path_to_end = Dijkstra(self.graph, scale_node, end_node)
        nodes_to_end = path_to_end[1]
        time_to_end = int(math.ceil(path_to_end[2]/self.truck_velocity))
        points_to_end = interpolate(self.nodes,nodes_to_end,self.interval, 
                    self.truck_velocity).tolist()

        # Previous time to stock
        previous_truck_tstock = 0

        for index, truck in enumerate(np.array(last_data_customer['Truck'])):
            # Get the row of the dataframe that belongs to current truck
            data_for_truck = last_data_customer[last_data_customer['Truck'] == truck]
            stock = np.array(data_for_truck['Dest'])[0]
            truck_tonnage = float(data_for_truck['Tonnage'])
            # TODO: Change Loader information based on what is assigned to each truck
            loader_velocity = self.loading_properties['loader'][2]
            loader_payload =  self.loading_properties['loader'][3]
            loader_cycletime =  self.loading_properties['loader'][4]
            stock_node = self.fixedloc[stock]
            # TODO: Add more points for entrance based on time of arrival
            # Modify start to entrance based on arrival time
            if index>0:
                previous_truck = np.array(last_data_customer['Truck'])[index-1]
                previous_truck_arrtime = np.array(last_data_customer[
                    last_data_customer['Truck'] == previous_truck]['ArrTime'])[0]
                current_truck_arrtime = np.array(data_for_truck['ArrTime'])[0]
                # Adding None points based on the time of arrival so we do not show the current truck
                # in the animation until it join the start point.
                points_to_entrance = [[None, None] for x in range(current_truck_arrtime)] + points_to_entrance
                # If current truck is assigned after the previous one but showed up before it. Then, we need
                # to add a waiting time.
                if current_truck_arrtime < previous_truck_arrtime:
                    waiting_step = math.ceil(
                        (previous_truck_arrtime - current_truck_arrtime)/self.interval)
                    # Duplicating the last point to entrance based on waiting step
                    points_waiting = np.array([list(points_to_entrance[-1]) for x in range(waiting_step)])
                    points_to_entrance = np.concatenate((points_to_entrance,points_waiting)).tolist()

            # From entrance to stock
            path_to_stock = Dijkstra(self.graph,entrance_node, stock_node)
            nodes_to_stock = path_to_stock[1]
            time_to_stock = int(math.ceil(path_to_stock[2]/self.truck_velocity))
            time_to_load = int(math.ceil(truck_tonnage/loader_payload))*int(loader_cycletime)
            time_to_stock += time_to_load
            points_to_stock = interpolate(self.nodes,nodes_to_stock,self.interval, 
                    self.truck_velocity)
            points_loading = np.array([list(points_to_stock[-1]) for x in range(round(time_to_load/self.interval))])
            points_to_stock = np.concatenate((points_to_stock,points_loading)).tolist()

            # From stock to scale
            path_to_scale = Dijkstra(self.graph,stock_node, scale_node)
            nodes_to_scale = path_to_scale[1]
            time_to_scale = int(math.ceil(path_to_scale[2]/self.truck_velocity))
            points_to_scale = interpolate(self.nodes,nodes_to_scale,self.interval, 
                    self.truck_velocity)
            points_scaling = np.array([list(points_to_scale[-1]) for x in range(round(time_scaling/self.interval))])
            points_to_scale = np.concatenate((points_to_scale,points_scaling)).tolist()

            # From stock to stock (loader)
            path_loader = Dijkstra(self.graph,previous_loader_node, stock_node)
            nodes_loader = path_loader[1]
            time_loader = int(math.ceil(path_loader[2]/loader_velocity)) + time_to_load
            points_loader =  interpolate(self.nodes,nodes_loader,self.interval, 
                    loader_velocity)
            # If it is the first assignment
            if index == 0:
                # Duplicating the stock's position for the loader based on the time that the first truck will
                #  take to get to the first stock
                initial_loader_points = [points_to_stock[-1] for x in range(len(points_to_entrance+points_to_stock))]
                points_loader = np.array(initial_loader_points)
            # If the loader is assigned to the same stock as the previous assignment
            elif points_loader.shape[0] == 0:
                points_loader = np.array([points_to_stock[-1]])
            points_loader = np.concatenate((points_loader,points_loading)) 
            # Update previous loader node to use for the next assignment
            previous_loader_node = stock_node
            # Modify path to entrance based on idle time
            if index>0:
                # Queue time for the current truck at the entrance
                queue_time = previous_truck_tstock + time_loader - time_to_stock
                if queue_time > 0:
                    # Number of times to duplicate the last position to entrance
                    queue_step = round(queue_time/self.interval)
                    points_queue = np.array([list(points_to_entrance[-1]) for x in range(queue_step)])
                    points_to_entrance = np.concatenate((points_to_entrance,points_queue)).tolist()
                else:
                    # If loader needs to wait because the time to go to stock for the current
                    # truck is more than (the time to go to stock for previous truck +
                    # time for loading + time for the loader to move from prev to current truck).
                    waiting_points_loader = [points_loader[-1] for i in range(queue_time*-1)]
                    # Updating the coordinates of the loader to wait at the stock for the 
                    # current truck
                    points_loader = np.concatenate((points_loader,waiting_points_loader)) 
            # Saving time to stock of the previous truck
            previous_truck_tstock = time_to_stock 
            # Assigning all coordinates of the trucks to one variable
            all_points_truck = points_to_entrance + points_to_stock + points_to_scale + points_to_end
            # Saving all coordinates of the trucks in the matrix
            matrix_trucks_coordinates[:,index][0:len(all_points_truck)] = np.array(all_points_truck)
            # Updating the loader's matrix based on its positions
            matrix_loaders_coordinates = np.concatenate([matrix_loaders_coordinates,np.array(points_loader, dtype=object)])
        return matrix_trucks_coordinates, matrix_loaders_coordinates

    def add_epoch_column(self):
        """ Modify date (yy-mm-dd..) format to epoch, which is a number."""
        self.customer_req_available['Date'] = self.customer_req_available.apply(
                        lambda r : datetime.datetime.combine(r['Date'],r['Time']),1)
        self.customer_req_available['Epoch'] = (self.customer_req_available['Date']
                - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
        self.customer_req_available['Epoch'] = self.customer_req_available['Epoch'] + 6*3600

    def request_truck_info(self):
        """
        Returns a dictionary that contains:
        key: truck | values: [[stock, loader], summation_time (compared to first one), capacity of truck].
        In addition, it adds the dummy variable
        """
        truck_info = {}
        for x in np.array(self.customer_req_available):
            dest_loader = []
            for index, dest in enumerate(x[4]):
                # Adds stock, laoder for the specific truck to a list
                dest_loader.append([dest, x[5][index]])
            # Adds the [stock, loader] list and summation time  + capacity to truck_info
            truck_info[x[2]]=[dest_loader,x[10], x[6]]
        # Adds the dummy variable
        truck_info['dummy'] = [[[0,0]],0,0]
        return truck_info

    def scheduling(self):
        """
        Gives the best sequence of assignments, (truck,stock), that minimizes the 
        overall idle time of trucks.

        Returns:
            schedule(list of tuples): Best sequence of assignments (truck, stock)
            minimum_idle_time(int): Total idle time of the best assignment
            initial_position_loader(str): Initial position of a loader
        """
        # Getting the minimum epoch to substract the others' epochs
        min_t= min(self.customer_req_available['Epoch'])
        # Converting epochtime to integers from 1-100
        self.customer_req_available['ArrTime'] = np.array([
            time_-min_t for time_ in np.array(self.customer_req_available['Epoch'])])
        self.customer_req_available = self.customer_req_available.sort_values('Epoch')
        # Get trucks: [[stock, loader], summation_time(compared to first), capacity]
        truck_info = self.request_truck_info()
        # Saving posible pairs of (truck, stock, loader) with their idle time
        truck_times = {}
        # Setting an integer to unique (truck,stock,loader) items
        truck_times_recognizer = {}
        # Helps to chose an integer
        i = 1
        for tr1 in self.trucks:
            # Loop through stocks assigned to the first truck in the pair
            numbers_stocks_tr1 = len(truck_info[tr1][0])
            for n_stock in range(numbers_stocks_tr1):
                for tr2 in self.trucks:
                    # Loop through stocks assigned to the second truck in the pair
                    numbers_stocks_tr2 = len(truck_info[tr2][0])
                    for n_stock_2 in range(numbers_stocks_tr2):
                        if tr1 != tr2:
                            # No idle time if dummy is the first truck
                            if tr1 == 'dummy':
                                truck_times[((tr1,0,0),
                                            (tr2,truck_info[tr2][0][n_stock_2][0],truck_info[tr2][0][n_stock_2][1]))] = 0
                                if ('dummy',0,0) not in truck_times_recognizer.values():
                                    truck_times_recognizer[999] = ('dummy',0,0)
                                    i +=1
                            # No idle time if dummy is the second truck
                            elif tr2 == 'dummy':
                                truck_times[((tr1,truck_info[tr1][0][n_stock][0],truck_info[tr1][0][n_stock][1]),
                                            (tr2,0,0))] = 0
                                if ('dummy',0,0) not in truck_times_recognizer.values():
                                    truck_times_recognizer[999] = ('dummy',0,0)
                                    i +=1
                            else:
                                stock_from = truck_info[tr1][0][n_stock][0]
                                loader_from = truck_info[tr1][0][n_stock][1]
                                stock_to = truck_info[tr2][0][n_stock_2][0]
                                loader_to = truck_info[tr2][0][n_stock_2][1]
                                truck_i_sumation_time = truck_info[tr1][1]
                                truck_j_sumation_time = truck_info[tr2][1]
                                truck_i_tostock_time =  round(
                                    float(Dijkstra(self.graph,self.fixedloc['entrance'],self.fixedloc[stock_from])[2])
                                    /self.truck_velocity)+int(math.ceil(truck_info[tr1][2]/self.loading_properties[loader_from][3])*self.loading_properties[loader_from][4])
                                truck_j_tostock_time = round(
                                    float(Dijkstra(self.graph,self.fixedloc['entrance'],self.fixedloc[stock_to])[2])
                                    /self.truck_velocity)+int(math.ceil(truck_info[tr2][2]/self.loading_properties[loader_to][3])*self.loading_properties[loader_to][4])
                                loader_i_to_j_stock_time = 0
                                # if same loader is assigned to both trucks but different stocks, otherwise it would be idle time 0
                                # TODO: Make sure that the logic does what it is supposed to when animating.
                                if loader_from == loader_to and stock_from != stock_to:
                                    loader_i_to_j_stock_time = round(
                                                float(Dijkstra(self.graph, self.fixedloc[stock_from], self.fixedloc[stock_to])[2])/self.loading_properties[loader_from][3]
                                                +int(math.ceil(truck_info[tr2][2]/self.loading_properties[loader_to][3])*self.loading_properties[loader_to][4]))  # Tload
                                truck_times[((tr1,stock_from,loader_from),
                                            (tr2,stock_to,loader_to))] = idle_time(truck_i_sumation_time,
                                                                                    truck_j_sumation_time,
                                                                                    truck_i_tostock_time,
                                                                                    truck_j_tostock_time,
                                                                                    loader_i_to_j_stock_time)
                                # Saving unique values from tr1 into truck_times_recognizer
                                if (tr1,stock_from,loader_from) not in truck_times_recognizer.values():
                                    truck_times_recognizer[i] =  (tr1,stock_from,loader_from)
                                    i+=1
                                # Saving unique values from tr2 into truck_times_recognizer
                                if  (tr2,stock_to,loader_to) not in truck_times_recognizer.values():
                                    truck_times_recognizer[i] =   (tr2,stock_to,loader_to)
                                    i+=1
        # Helps to index the trucks with their respective integers, i.e. S2: [2,3]               
        truck_recognizer = {}
        # Save integer that has one stock and one type of loader
        list_one = []
        # Save integer that has either >=two stocks or loaders 
        list_two = []
        # Build the truck_recognizer
        for recognizer in truck_times_recognizer:
            truck_stock_loader = truck_times_recognizer[recognizer]
            truck_r = truck_stock_loader[0]
            if truck_r not in truck_recognizer:
                truck_recognizer[truck_r] = list()
            truck_recognizer[truck_r].append(recognizer)
        # Build the list_one and list_two
        for truck_r in truck_recognizer:
            recognizers = truck_recognizer[truck_r]
            if len(recognizers)>1:
                list_two.append(recognizers)
            else:
                list_one.append(recognizers[0])
        # Changing pairs of (truck, stock, loader) for the respective integers.
        truck_times_with_recognizer = {}
        for key in truck_times:
            from_old_key = key[0]
            to_old_key = key[1]
            from_key = [key for key in truck_times_recognizer if truck_times_recognizer[key] == from_old_key]
            to_key = [key for key in truck_times_recognizer if truck_times_recognizer[key] == to_old_key]
            truck_times_with_recognizer[(from_key[0]), (to_key[0])] = truck_times[key]
        # Make sure that the first integer contains the dummy variable
        list_one = sorted(list_one,reverse=True)    
        # TODO: We can also run the total time and make a comparison between idle time vs total cycle time
        # Getting permutations from trucks within list_two to be added to list_one
        permut_list_two = [s for s in itertools.product(*list_two)]
        schedule, minimum_idle_time  = run_schedule(list_one,truck_times_with_recognizer, permut_list_two)
        return schedule, minimum_idle_time,truck_times_recognizer 

    def modify_req_schedule_dest(self, schedule):
        """ Modify the customer requirement based on the best assignments (schedule).
        It will delete multiple stocks for each truck and choose one and sort the table based
        on the schedule.
        Args:
            schedule(list of tuples): Pairs of (truck, stock) within the best assignment
    
        Returns
            new_df(df): Customer requirement with one stock for the destinations and also
            sorted based on the schedule
        """
        new_truck_info = []
        for comb in schedule:
            truck = comb[0]
            dataf_truck = self.customer_req_available[
                self.customer_req_available['Truck']==truck]
            # Changing multiple stocks to one stock
            dataf_truck['Dest'] = comb[1]
            dataf_truck['TypeLoader'] = comb[2]
            # Adding each row of a dataframe to a list
            new_truck_info.append(dataf_truck)
        new_df = pd.concat(new_truck_info)
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
        return new_cust_req_dict
