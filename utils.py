import ast

import numpy as np


def filter_data(data, filename):
    list_segments = []
    list_points=[]
    data_filter = data[data['filename']==filename]['region_shape_attributes']
    date_name = data[data['filename']==filename]['region_attributes']
    for dataf, datan in zip(enumerate(data_filter),enumerate(date_name)):
        index = dataf[0]
        data_r = dataf[1]
        dicti = ast.literal_eval(data_r)
        #print(dicti)
        if 'polyline' in dicti.values():
            x_points = dicti['all_points_x']
            y_points = dicti['all_points_y']
            list_segments+=[[x,y, index]for x,y in zip(x_points, y_points)]
        else:
            dict_n = ast.literal_eval(datan[1])
            x_a = dicti['cx']
            y_a= dicti['cy']
            name = dict_n['name']
            list_points.append([x_a, y_a, index,name])
    return np.array(list_segments),np.array(list_points)

def delete_doubles(array):
    new = array
    #print(array.shape)
    for index,coordinate in enumerate(new):
        new_array = np.array([np.linalg.norm(x) for x in new-coordinate])
        index_2   = np.argwhere((new_array>=0)&(new_array<7))
        if len(index_2)>0:
            for ind_2 in index_2:
                new[index][0] =new[ind_2][0][0]
                new[index][1] =new[ind_2][0][1]
    return new

def find_key(val, diction):
    val = list(val)
    x = [k for k,v in diction.items() if v == val]
    return x[0]       
