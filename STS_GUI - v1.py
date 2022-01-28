import numpy as np
import seaborn as sns
from tkinter import Frame, Button, Label, Entry, Checkbutton, IntVar, Scale
from tkinter.ttk import Notebook
from tkinter import LEFT, TOP, X, FLAT, RAISED, RIGHT, SUNKEN, GROOVE, RIDGE
import matplotlib
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
import math
from mpl_toolkits.axes_grid1 import make_axes_locatable

from tkinter import *
import matplotlib
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import tkinter as tk
from PIL import Image

matplotlib.use('TkAgg')


class CustomToolbar(NavigationToolbar2Tk):
    def save_Figure(self):
        filename = fd.asksaveasfilename(initialdir="/", title="Select file", filetypes=(
            ('png files', '*.png'), ("jpeg files", "*.jpg"), ("all files", "*.*")))
        plt.savefig(filename)

    def __init__(self, canvas_, parent_):
        self.toolitems = (
            ('Home', 'Reset original view', 'home', 'home'),
            ('Back', 'Back to previous view', 'back', 'back'),
            ('Forward', 'Forward to next view', 'forward', 'forward'),
            ('Pan', 'Pan axes with left mouse, zoom with right', 'move', 'pan'),
            ('Zoom', 'Zoom to rectangle', 'zoom_to_rect', 'zoom'),
            ('save_fig', 'Save Figure', 'Filesave', 'save_Figure')
        )
        NavigationToolbar2Tk.__init__(self, canvas_, parent_)


class interface(Frame):

    def __init__(self):
        super().__init__()

        self.initUI()
        self.mainloop()

    def initUI(self):

        #self.current_palette = sns.cubehelix_palette(len(sets), start=2, rot=0, dark=.2, light=.7, reverse=True)
        #self.sets = sets
        self.coordinate = (0,0)
        self.voltage = 0
        self.color_min = -1000
        self.rectangle = False
        self.unclicked = True
        self.color_max = 1000
        self.coords_to_iv = {}
        self.z_to_xyf = {}

        outer_options_frame = Frame(self.master)
        outer_options_frame.pack(side=LEFT, expand=True, fill='both')

        self.fig = plt.figure()
        self.fig.patch.set_alpha(1)

        self.ax1 = self.fig.add_subplot(2, 2, 1)
        self.ax2 = self.fig.add_subplot(2, 2, 2)
        self.ax3 = self.fig.add_subplot(2, 2, 3)
        self.ax4 = self.fig.add_subplot(2, 2, 4)
        self.ax2.grid()
        self.ax4.grid()
        self.ax1.title.set_text('dIdV Image at V = ' + str(self.voltage))
        self.ax2.title.set_text('IV Plot at (x,y) = ' + str(self.coordinate))
        self.ax2.set_ylabel('Current (nA)')
        self.ax3.title.set_text('Normalized DIdV Image at V = ' + str(self.voltage))
        self.ax4.title.set_text('dIdV Plot at (x,y) = ' + str(self.coordinate))
        self.ax4.set_xlabel('Voltage (V)')
        self.ax4.set_ylabel('dIdv (nA/V)')



        self.ax1.format_coord = lambda x, y:  format(x, '1.4f') + ', ' + format(y, '1.4f')

        self.fig.canvas.mpl_connect('button_press_event', self.click_on_image)
        self.fig.canvas.mpl_connect('button_release_event', self.click_on_image)

        self.options_frame = Frame(outer_options_frame)
        self.options_frame.pack(side=LEFT)

        '''
        Creates the frame for the figures to go in, then packs the existing figure into it
        '''

        self.canvas_frame = Frame(outer_options_frame, bg='')
        self.canvas_frame.pack(side=LEFT, expand=True, fill='both')
        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.toolbar = CustomToolbar(self.canvas, self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=LEFT, expand=True, fill='both')
        #self.ax1.bind("<Button 1>",self.click_on_image)

        '''
        Creates open file button
        '''

        open_file_frame = Frame(self.options_frame)
        open_file_frame.pack( expand=True, fill = 'x')
        self.open_file_button = Button(open_file_frame, text='Open', command = self.select_file)
        self.open_file_button.pack()

        '''
        Creates the data type box
        '''

        data_type_frame = Frame(self.options_frame)
        data_type_frame.pack(expand=True, fill='x')
        self.data_type_label = Label(data_type_frame, text='Data Source')
        self.data_type_label.pack(expand=True, fill='x')
        self.data_type_box = Entry(data_type_frame)
        self.data_type_box.pack(side=LEFT, fill='x', expand=True)
        '''
        Creates the Coordinate Box
        '''

        coord_frame = Frame(self.options_frame)
        coord_frame.pack(expand = True, fill = 'x')
        coord_label = Label(coord_frame, text='Coordinate')
        coord_label.pack(expand=True, fill = 'x')
        self.coord_box = Entry(coord_frame)
        self.coord_box.pack(side=LEFT, expand=True, fill='x')
        self.coord_box.insert(0,'(0,0)')

        colorbar_lim_frame = Frame(self.options_frame)
        colorbar_lim_frame.pack(expand = True, fill = 'x')
        colorbar_label = Label(colorbar_lim_frame, text='Colorbar Scale')
        colorbar_label.pack(expand = True, fill = 'x')
        colorbar_lower_label = Label(colorbar_lim_frame, text='Lower Limit:')
        colorbar_lower_label.pack(expand=True, fill='x', side=LEFT)
        self.colorbar_lower_box = Entry(colorbar_lim_frame, width=4)
        self.colorbar_lower_box.pack(side=LEFT, expand=True, fill='x')
        colorbar_upper_label = Label(colorbar_lim_frame, text='Upper Limit:')
        colorbar_upper_label.pack(side=LEFT, expand=True, fill='x')
        self.colorbar_higher_box = Entry(colorbar_lim_frame, width=4)
        self.colorbar_higher_box.pack(side=RIGHT, expand=True, fill='x')
        self.colorbar_higher_box.insert(0,'1')
        self.colorbar_lower_box.insert(0, '0')

        def close(self):
            self.master.destroy()
            self.master.quit()

        self.master.protocol("WM_DELETE_WINDOW", lambda: close(self))

    def apply(self):
        self.set_voltage()
        self.set_coord()
        self.dfdz_image_plot()

    def set_voltage(self):
        old_volt = self.voltage
        v_slider = self.volt_slider.get()
        self.voltage = float(self.nearest_voltage(v_slider))

        if old_volt != self.voltage:
            self.dfdz_image_plot()

        self.ax1.title.set_text('dIdV Image at V=' + str(self.voltage))

    def nearest_voltage(self, voltage):
        nearest_d = 9999
        closest = 0
        for v in self.z_to_xyf.keys():
            dist = abs(voltage - v)
            if dist < nearest_d:
                nearest_d = dist
                closest = v

        return closest

    def set_coord(self):
        old_coord = self.coordinate
        new_coord_str = self.coord_box.get()
        new_coord = self.str_to_coord(new_coord_str)
        if new_coord != old_coord:
            self.coordinate = new_coord
            if self.data_type != 'ORNL':
                self.plot_IV()
            self.plot_didv()

    def str_to_coord(self, coord_string):

        no_parenths = coord_string[1:len(coord_string)-1]
        str_vals = no_parenths.split(',')
        x = float(str_vals[0])
        y = float(str_vals[1])
        coord = (x,y)
        return coord

    def coord_to_str(self, x,y):
        x = "{0:.4f}".format(x)
        y = "{0:.4f}".format(y)
        coord_str = '(' + x + ',' + y + ')'
        return coord_str

    def select_file(self):
        filetypes = ( ('text files', '*.txt'),('All files', '*.*') )
        filename = fd.askopenfilename(title='Open a file',initialdir='/',filetypes=filetypes)
        #showinfo(title='Selected File',message=filename)
        self.initialize(filename)
        self.plot_didv()
        if self.data_type != 'ORNL':
            self.plot_IV()
        self.dfdz_image_plot()

        '''
        Creates the voltage box
        '''

        self.volt_frame = Frame(self.options_frame)
        self.volt_frame.pack(expand=True, fill='x')
        volt_label = Label(self.volt_frame, text='Voltage')
        volt_label.pack(expand=True, fill='x')

        self.volt_min = min(self.z_to_xyf.keys())
        self.volt_max = max(self.z_to_xyf.keys())
        self.volt_step = round((self.volt_max - self.volt_min)/(len(self.z_to_xyf.keys())-1),7)
        self.volt_slider = Scale(self.volt_frame, from_=self.volt_min, to_=self.volt_max, orient=HORIZONTAL,
                                 resolution=self.volt_step)
        self.volt_slider.pack(side=LEFT, expand=True, fill='x')

        apply_button_frame = Frame(self.options_frame)
        apply_button_frame.pack()

        self.apply_button = Button(apply_button_frame, text='Apply', command=self.apply)
        self.apply_button.grid(row=0, column=0)

    def get_data(self, filepath):

        """
        Opens the file at the location specified and converts it to a raw string. This will always
        need to be parsed later. It is necessary to include the .txt file extension in order for it
        to open properly.
        """

        with open(filepath) as raw_txt:
            raw = raw_txt.readlines()
        return raw

    def parse_data(self, raw):

        """
        Parses the data into a usable format. Depending on the source of this data this may or may not work.
        This specific parser is designed for use with our lab's STS data. If this parser does not work, create a
        new one (do not change this one) after this function with the name parse_XXXXX_data() with XXXXX being
        source of the data. This also creates a dictionary, coords_to_iv which maps a coordinate to its corresponding z
        and f values.

        :param raw: This parameter is intended to be the result of the previous function, get_data(), raw is in this case
        a list of strings where each row is tab delimited and each row terminates in a \n

        """
        str_data = raw[4:]
        x_data = []
        y_data = []
        z_data = []
        f_data = []
        for row in str_data:
            x_str, y_str, z_str, f_str = row.strip('\n').split('\t')
            x_val = float(x_str)
            y_val = float(y_str)
            x_data.append(x_val)
            y_data.append(y_val)
            z_val = float(z_str)
            f_val = float(f_str)
            z_data.append(z_val)
            f_data.append(f_val)
            coord_str = self.coord_to_str(x_val, y_val)
            if coord_str not in self.coords_to_iv.keys():
                self.coords_to_iv[coord_str] = [[z_val], [f_val], [], []]
            else:
                self.coords_to_iv[coord_str][0].append(z_val)
                self.coords_to_iv[coord_str][1].append(f_val)

        for coord_str in self.coords_to_iv.keys():
            coord = self.str_to_coord(coord_str)
            x = coord[0]
            y = coord[1]
            didv_vals, didv_norm_vals = self.tp_der_approx_at_point(x,y)
            self.coords_to_iv[coord_str][2] = didv_vals
            self.coords_to_iv[coord_str][3] = didv_norm_vals
        self.coord_step = x_data[1] - x_data[0]
        data = [x_data, y_data, z_data, f_data]
        self.voltage = z_data[0]
        return data

    def parse_ORNL_data(self,raw):

        """
        Parses the data into a usable format. Depending on the source of this data this may or may not work.
        This specific parser is designed for use with our lab's STS data. If this parser does not work, create a
        new one (do not change this one) after this function with the name parse_XXXXX_data() with XXXXX being
        source of the data. This also creates a dictionary, coords_to_iv which maps a coordinate to its corresponding z
        and f values.

        :param raw: This parameter is intended to be the result of the previous function, get_data(), raw is in this case
        a list of strings where each row is tab delimited and each row terminates in a \n

        """
        str_data = raw[4:]
        x_data = []
        y_data = []
        z_data = []
        didv_data = []
        for row in str_data:
            x_str, y_str, z_str, didv_str = row.strip('\n').split('\t')
            x_val = float(x_str)*10**9
            y_val = float(y_str)*10**9
            x_data.append(x_val)
            y_data.append(y_val)
            z_val = float(z_str)
            didv_val = float(didv_str)
            z_data.append(z_val)
            didv_data.append(didv_val)
            coord_str = self.coord_to_str(x_val, y_val)
            if coord_str not in self.coords_to_iv.keys():
                self.coords_to_iv[coord_str] = [[z_val], [didv_val], [], []]
            else:
                self.coords_to_iv[coord_str][0].append(z_val)
                self.coords_to_iv[coord_str][1].append(didv_val)

        for coord_str in self.coords_to_iv.keys():
            coord = self.str_to_coord(coord_str)
            x = coord[0]
            y = coord[1]
            didv_vals, didv_norm_vals = self.tp_der_approx_at_point(x,y)
            self.coords_to_iv[coord_str][2] = didv_vals
            self.coords_to_iv[coord_str][3] = didv_norm_vals
        self.coord_step = x_data[1] - x_data[0]
        data = [x_data, y_data, z_data, didv_data]
        self.voltage = z_data[0]
        return data

    def initialize(self, filepath):
        raw = self.get_data(filepath)
        self.data_type = self.data_type_box.get()
        if self.data_type == 'ORNL':
            data = self.parse_ORNL_data(raw)
        else:
            data = self.parse_data(raw)
        self.seperate_voltages(data)

    def get_dIdV(self, x, y, z):
        """
        Gets the dI/dV value for a given voltage value (z) at a specific point (x,y)
        :param x: the x value associated with the point of interest
        :param y: the y value associated with the point of interest
        :param z: the desired voltage value
        :return: the dI/dV value at the location and voltage
        """
        dfdz_data = self.coords_to_iv[self.coord_to_str(x,y)][2]
        dfdz_norm_data = self.coords_to_iv[self.coord_to_str(x,y)][3]
        z_data = self.coords_to_iv[self.coord_to_str(x,y)][0]
        ind = z_data.index(z)
        dfdz = dfdz_data[ind]
        dfdz_norm = dfdz_norm_data[ind]

        return [dfdz, dfdz_norm]

    def seperate_voltages(self, data):

        x_data, y_data, z_data, f_data = data
        z_last = z_data[0]
        first_ind = 0

        for z in z_data:
            if z_last != z:
                ind = z_data.index(z)
                x_vals = x_data[first_ind:ind]
                y_vals = y_data[first_ind:ind]
                didv_vals = []
                didv_norm_vals = []
                for i in range(0, len(x_vals)):
                    x = x_vals[i]
                    y = y_vals[i]
                    didv, didv_norm = self.get_dIdV(x,y,z)
                    didv_vals.append(didv)
                    didv_norm_vals.append(didv_norm)
                dataset = [x_vals, y_vals, didv_vals, didv_norm_vals]
                self.z_to_xyf[z_last] = dataset
                first_ind = ind

            z_last = z


        return

    def tp_der_approx_at_point(self, x, y):  # two point derivative approximation at a single point across all voltages
        z_data, f_data, trash1, trash2 = self.coords_to_iv[self.coord_to_str(x, y)]
        if min(z_data) <= 0:
            shift = -min(z_data) + .5
        else:
            shift = 0

        if self.data_type == 'ORNL':
            dfdz_data = f_data
            dfdz_norm_data = f_data
        else:
            dfdz_data = []
            dfdz_norm_data = []
            for i in range(len(z_data)):
                if i == 0:  # forward difference for the first point
                    z1 = z_data[i]
                    f1 = f_data[i]
                    f2 = f_data[i + 1]
                    z2 = z_data[i + 1]
                elif i == len(z_data) - 1:  # backward difference for the last point
                    z1 = z_data[i]
                    f1 = f_data[i]
                    f2 = f_data[i - 1]
                    z2 = z_data[i - 1]
                else:  # centered difference for all of the other points
                    z1 = z_data[i - 1]
                    f1 = f_data[i - 1]
                    f2 = f_data[i + 1]
                    z2 = z_data[i + 1]
                dfdz = (f2 - f1) / (z2 - z1)
                dfdz_norm = dfdz/(z_data[i]*f_data[i]+shift)
                dfdz_norm_data.append(dfdz_norm)
                dfdz_data.append(dfdz)
        return [dfdz_data, dfdz_norm_data]

    def dfdz_image_plot(self):
        self.ax1.cla()
        self.ax3.cla()
        try:
            self.cb1.remove()
            self.cb3.remove()
        except:
            pass
        volt = self.voltage
        x_data, y_data, didz_data, didz_norm_data = self.z_to_xyf[volt]
        x = x_data[0:int(len(x_data)**.5)]
        y = x

        didz_ar = np.array(didz_data)
        didz_grid = didz_ar.reshape(int(len(x_data)**.5), int(len(x_data)**.5))



        didz_norm_ar = np.array(didz_norm_data)
        didz_norm_grid = didz_norm_ar.reshape(int(len(x_data) ** .5), int(len(x_data) ** .5))
        if self.color_max == 1000:
            im3 = self.ax3.pcolormesh(x, y, didz_norm_grid, shading='nearest', picker=True)
            im1 = self.ax1.pcolormesh(x, y, didz_grid, shading='nearest', picker=True)
            self.color_max = 100
        else:
            self.color_max = float(self.colorbar_higher_box.get())
            self.color_min = float(self.colorbar_lower_box.get())
            im3 = self.ax3.pcolormesh(x, y, didz_norm_grid, shading='nearest', picker=True, vmin=self.color_min, vmax=self.color_max)
            im1 = self.ax1.pcolormesh(x, y, didz_grid, shading='nearest', picker=True, vmin=self.color_min, vmax=self.color_max)
        self.ax1.title.set_text('dIdV Image at V = ' + str(self.voltage))
        self.cb1 = self.fig.colorbar(im1, ax=self.ax1, orientation='vertical')
        self.ax3.title.set_text('dIdV Normalized Image at V = ' + str(self.voltage))
        self.cb3 = self.fig.colorbar(im3, ax=self.ax3, orientation='vertical')
        self.canvas.draw()

    def plot_IV(self):
        if self.rectangle == False:
            self.ax2.cla()
            self.ax2.grid()
            self.ax2.title.set_text('IV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
            self.ax2.set_ylabel('Current (nA)')
        x,y = self.coordinate
        z_data, f_data, didv_vals, didv_norm_vals = self.coords_to_iv[self.coord_to_str(x,y)]
        self.ax2.plot(z_data, f_data, c='#1f77b4')
        if self.rectangle == False:
            self.canvas.draw()

    def sort_by_position(self):
        with open('sorted_by_pos.csv', 'w') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',')
            for coord_str in self.coords_to_iv.keys():
                x,y = self.str_to_coord(coord_str)
                z_data, f_data = self.coords_to_iv[coord_str]
                for i in range(len(z_data)):
                    z_val = z_data[i]
                    f_val = f_data[i]
                    line = [x, y, z_val, f_val]
                    datawriter.writerow(line)

    def scatter_IV(self):
        raw_data = self.get_data()
        x_data, y_data, z_data, f_data = self.parse_ORNL_data(raw_data)
        plt.scatter(z_data, f_data, c='b')
        plt.grid()
        plt.xlabel('Voltage')
        plt.ylabel('Current')
        plt.show()

    def plot_didv(self):
        if self.rectangle == False:
            self.ax4.cla()
            self.ax4.grid()
            self.ax4.title.set_text('dIdV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
            self.ax4.set_xlabel('Voltage (V)')
            self.ax4.set_ylabel('dIdv (nA/V)')
        x,y = self.coordinate
        z_data, f_data, didv_vals, didv_norm_vals = self.coords_to_iv[self.coord_to_str(x,y)]
        self.ax4.plot(z_data, didv_vals, c='#1f77b4')

        if self.rectangle == False:
            self.canvas.draw()

    def nearest_coord(self, c):
        x = c[0]
        y = c[1]
        closest = (0,0)
        nearest_d = 9999999
        for key in self.coords_to_iv.keys():
            coord = self.str_to_coord(key)
            x2 = coord[0]
            y2 = coord[1]
            dist = ((x2 - x) ** 2 + (y2 - y) ** 2) ** .5
            if dist < nearest_d:
                nearest_d = dist
                closest = coord
        return closest

    def click_on_image(self, event):
        if event.name == 'button_press_event':
            self.last_click = event
        if event.name == 'button_release_event':
            self.last_release = event

        if self.unclicked == True:
            xdata = event.xdata
            ydata = event.ydata
            click_point = (xdata, ydata)
            image_point = self.nearest_coord(click_point)
            # print(image_point)
            self.coordinate = image_point
            self.coord_box.delete(0, END)
            self.coord_box.insert(0, self.coord_to_str(image_point[0], image_point[1]))
            if self.data_type != 'ORNL':
                self.plot_IV()
            self.plot_didv()
            self.unclicked = False
        else:
            click_coord = (self.last_click.xdata, self.last_click.ydata)
            release_coord = (self.last_release.xdata, self.last_release.ydata)
            print(click_coord, release_coord)
            if click_coord != release_coord and event.name != 'button_press_event':
                self.ax2.cla()
                self.ax2.grid()
                self.ax2.title.set_text(
                    'IV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
                self.ax2.set_ylabel('Current (nA)')
                self.ax4.cla()
                self.ax4.grid()
                self.ax4.title.set_text(
                    'dIdV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
                self.ax4.set_xlabel('Voltage (V)')
                self.ax4.set_ylabel('dIdv (nA/V)')
                self.rectangle = True
                nearest_to_click = self.nearest_coord(click_coord)
                nearest_to_release = self.nearest_coord(release_coord)
                x_bounds = [nearest_to_click[0], nearest_to_release[0]]
                y_bounds = [nearest_to_click[1], nearest_to_release[1]]
                x_min = min(x_bounds)
                x_max = max(x_bounds)
                y_min = min(y_bounds)
                num_vals = int((x_max-x_min)/self.coord_step)

                for i in range(num_vals):
                    for j in range(num_vals):
                        x = x_min + i*self.coord_step
                        y = y_min + j*self.coord_step
                        nearest = self.nearest_coord((x, y))
                        self.coordinate = nearest
                        if self.data_type != 'ORNL':
                            self.plot_IV()
                        self.plot_didv()
                self.canvas.draw()
                self.rectangle = False

            else:

                xdata = event.xdata
                ydata = event.ydata
                click_point = (xdata, ydata)
                image_point = self.nearest_coord(click_point)
                self.coordinate = image_point
                self.coord_box.delete(0,END)
                self.coord_box.insert(0, self.coord_to_str(image_point[0], image_point[1]))
                if self.data_type != 'ORNL':
                    self.plot_IV()
                self.plot_didv()

if __name__ == "__main__":
    interface()




