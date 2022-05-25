import numpy as np
import cv2
import matplotlib.pyplot as plt
import datetime
class Animation_2(object):
    def __init__(self, mom_sec):
        self.m_customer = mom_sec[0][0]
        self.m_shovel = mom_sec[0][1]
        self.costumer_palette = mom_sec[0][2]
        self.large = mom_sec[0][3] 
        self.change_stock = mom_sec[0][4]
        #print(self.change_stock)
        self.new_out = mom_sec[1]
        self.c_order= mom_sec[2]
        self.fixed_location = {stock:list(np.array(self.new_out[
            self.new_out['node']==mom_sec[3][stock]][['x','y']])[0]) for stock in mom_sec[3]}
        print(self.fixed_location)
        self.location_piles= mom_sec[4]
        self.delay = mom_sec[5]
        self.truck_capacity = mom_sec[6]
        self.to_master_text= mom_sec[7]
        self.to_w, self.to_w_m,self.min_dectime = mom_sec[8], mom_sec[9],mom_sec[10]
        self.interpolator=mom_sec[11]
        self.array_stockpiles = np.array(list(self.location_piles.values()),dtype=object)
        self.image = cv2.imread('Pete_big.png')
        self.move_txt_stock = 25
        self.j_shovel = 0
        self.fig = plt.figure()
        
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
            self.dict_text_ax_above[order] = self.ax.text(0,0, '',backgroundcolor='white',
                 c = 'k')
        self.chart_stocks = [self.ax.scatter([],[], marker = 'o', s = 1, c= 'k'),
        self.ax1.scatter([],[], marker = 'o', s = 1, c= 'k')]
        self.dict_text = {}
        for order in self.c_order:
            self.dict_text[order] = self.ax.text(0,0, '')
        
        self.dict_stocks = {}
        for stk in self.location_piles:
            self.dict_stocks[stk] = [self.ax.text(0,0, '',fontsize =10, 
                backgroundcolor='white'),self.ax1.text(0,0, '',fontsize =10, 
                    backgroundcolor='white')]
        # text2 = ax.text(0,0, '')
        #GET COORDINATES
        self.left,self.right = self.ax.get_xlim()
        self.down,self.up = self.ax.get_ylim()

    def ini_2(self):
        self.ax2 = plt.figure()
        plt.imshow(self.image)
        for segment in np.unique(self.new_out['seg']):
            sample = np.array(self.new_out[self.new_out['seg']==segment][['x','y']])
            for x1, x2 in zip(sample[:-1],sample[1:]):
                plt.annotate('',xy=(x2[0],x2[1]), xytext=(x1[0],x1[1]), arrowprops=dict(
                    arrowstyle='->', linestyle="--", lw=2, color='#ccffe7'))  
        for index,  location in enumerate(self.fixed_location):
            coordinate =self.fixed_location[location]
            size = 200
            marker = 'x'
            if 'tock' not in location:
                color_f = 'b'
                if 'hovel' in location:
                    color_f = 'k'
                    marker = None
                    size =10
                    location = 'Loader Ini.'
                arrowprops=dict(arrowstyle='->', color=color_f, linewidth=1)
                #plt.scatter(coordinate[0],coordinate[1], label = location, marker=marker, s = size, linewidths=2, c= color_f) 
                plt.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-90,coordinate[1]-30)
                    , c= color_f, arrowprops= arrowprops, backgroundcolor = 'w')
            # if location == 'Entrance':
            #     ax.add_patch(circle_entrance)
                #ax1.add_patch(circle_entrance)
        
        plt.suptitle('Idle Time: '+str(datetime.timedelta(seconds=self.delay))+' sec')
        #ax.text(left,up-start_up, text_up)
        
        plt.gca().set_aspect('equal', adjustable='box')
        return plt
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
                self.ax.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-60,coordinate[1]-25)
                    , c= color_f, arrowprops= arrowprops, backgroundcolor = 'w')
                self.ax1.annotate(location,xy = (coordinate[0],coordinate[1]), xytext = (coordinate[0]-60,coordinate[1]-25)
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
        print(i)
        txt_up = 20
        #print(i)
        if i in self.change_stock.values():
            self.j_shovel+=1
            stock = [key for key,value in self.change_stock.items() if value == i][0][0]
            self.location_piles[stock][2] = self.location_piles[stock][2] - self.truck_capacity[0]
            self.truck_capacity.remove(self.truck_capacity[0])
        size_stock = np.array(list(self.location_piles.values()),dtype=object)[:,2]
        size_stock = size_stock.astype(float)
        #size_stock = np.reshape(size_stock, (len(size_stock),1))
        text_stok = [key+'\n{:.0f} tons'.format(values[2]) for key,values 
            in self.location_piles.items()]
        self.chart_stocks[0].set_offsets(self.array_stockpiles[:,:2])
        self.chart_stocks[1].set_offsets(self.array_stockpiles[:,:2])
        for ind,dat in enumerate(self.dict_stocks):
            self.dict_stocks[dat][0].set_text(text_stok[ind])
            self.dict_stocks[dat][0].set_position((self.array_stockpiles[:,:2]
                [ind]+self.move_txt_stock))
            self.dict_stocks[dat][1].set_text(text_stok[ind])
            self.dict_stocks[dat][1].set_position((self.array_stockpiles[:,:2]
                [ind]+self.move_txt_stock))
        #self.chart_stocks[0].set_sizes(size_stock)
        self.chart_stocks[0].set_color(self.array_stockpiles[:,3])
        #self.chart_stocks[1].set_sizes(size_stock)
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
            print('x'*100)
            print(matrix_chosen.shape)
            for ind_i in range(len(matrix_chosen)):
                if matrix_chosen[ind_i][0] != None:
                    destination = self.to_master_text[ind_i]
                    ind_customer = [index for index, destin in enumerate(self.to_w_m) if destin == destination][0]
                    customer = self.c_order[ind_i]
                    destination_text = destination.split('_')[0]
                    label_text = customer+' to: {} ({})'.format(destination_text, str(ind_customer+1))
                    color_costu = 'k'
                    if active_sec:
                        if destination == self.to_w_m[self.j_shovel] and active_sec:
                            color_costu = '#FF0000'
                    self.dict_text_ax_above[self.c_order[ind_i]].set_text(label_text)
                    self.dict_text_ax_above[self.c_order[ind_i]].set_position((self.left+10,self.up+txt_up))
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
