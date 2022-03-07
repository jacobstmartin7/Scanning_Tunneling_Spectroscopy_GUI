
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
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta

from tkinter import *
import matplotlib
import csv
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import tkinter as tk
from PIL import Image
from decimal import Decimal, ROUND_UP
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

        # self.current_palette = sns.cubehelix_palette(len(sets), start=2, rot=0, dark=.2, light=.7, reverse=True)
        # self.sets = sets
        self.coordinate = (0, 0)
        self.voltage = 0
        self.color_min = -1000
        self.rectangle = False
        self.unclicked = True
        self.color_max = 1000
        self.coords_to_iv = {}
        self.coords_to_lines = {}
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

        self.ax1.format_coord = lambda x, y: format(x, '1.4f') + ', ' + format(y, '1.4f')

        self.fig.canvas.mpl_connect('button_press_event', self.click_on_image)
        self.fig.canvas.mpl_connect('button_release_event', self.click_on_image)

        self.options_frame = Frame(outer_options_frame)
        self.options_frame.pack(side=LEFT)

        '''
        Creates the frame for the figures to go in, then packs the existing figure into it
        '''
        self.tabs = Notebook(outer_options_frame)
        self.tabs.pack(side=LEFT, expand=True, fill='both')

        self.canvas_frame = Frame(self.tabs, bg='')
        self.canvas_frame.pack(side=LEFT, expand=True, fill='both')
        self.tabs.add(self.canvas_frame)

        self.canvas = FigureCanvasTkAgg(self.fig, self.canvas_frame)
        self.toolbar = CustomToolbar(self.canvas, self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=LEFT, expand=True, fill='both')

        self.tab2_frame = Frame(self.tabs)
        self.tab2_frame.pack(side=LEFT, expand=True, fill='both')
        self.tabs.add(self.tab2_frame)

        '''
        Creates open file button
        '''

        open_file_frame = Frame(self.options_frame)
        open_file_frame.pack(expand=True, fill='x')
        self.open_file_button = Button(open_file_frame, text='Open', command=self.select_file)
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
        coord_frame.pack(expand=True, fill='x')
        coord_label = Label(coord_frame, text='Coordinate')
        coord_label.pack(expand=True, fill='x')
        self.coord_box = Entry(coord_frame)
        self.coord_box.pack(side=LEFT, expand=True, fill='x')
        self.coord_box.insert(0, '(0,0)')

        colorbar_lim_frame = Frame(self.options_frame)
        colorbar_lim_frame.pack(expand=True, fill='x')
        colorbar_label = Label(colorbar_lim_frame, text='Colorbar Scale')
        colorbar_label.pack(expand=True, fill='x')
        colorbar_lower_label = Label(colorbar_lim_frame, text='Lower Limit:')
        colorbar_lower_label.pack(expand=True, fill='x', side=LEFT)
        self.colorbar_lower_box = Entry(colorbar_lim_frame, width=4)
        self.colorbar_lower_box.pack(side=LEFT, expand=True, fill='x')
        colorbar_upper_label = Label(colorbar_lim_frame, text='Upper Limit:')
        colorbar_upper_label.pack(side=LEFT, expand=True, fill='x')
        self.colorbar_higher_box = Entry(colorbar_lim_frame, width=4)
        self.colorbar_higher_box.pack(side=RIGHT, expand=True, fill='x')
        self.colorbar_higher_box.insert(0, '1')
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

        no_parenths = coord_string[1:len(coord_string) - 1]
        str_vals = no_parenths.split(',')
        x = float(str_vals[0])
        y = float(str_vals[1])
        coord = (x, y)
        return coord

    def coord_to_str(self, x, y):
        x = Decimal(str(x)).quantize(Decimal('.0001'), rounding=ROUND_UP)
        y = Decimal(str(y)).quantize(Decimal('.0001'), rounding=ROUND_UP)
        coord_str = '(' + str(x) + ',' + str(y) + ')'
        return coord_str

    def select_file(self):
        filetypes = (('text files', '*.txt'), ('All files', '*.*'))
        filename = fd.askopenfilename(title='Open a file', initialdir='/', filetypes=filetypes)
        # showinfo(title='Selected File',message=filename)
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
        self.volt_step = round((self.volt_max - self.volt_min) / (len(self.z_to_xyf.keys()) - 1), 7)
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
        largest_f = -1000
        smallest_f = 1000
        for row in str_data:
            x_str, y_str, z_str, f_str = row.strip('\n').split('\t')
            x_val = float(x_str)
            y_val = float(y_str)
            x_data.append(x_val)
            y_data.append(y_val)
            z_val = float(z_str)
            f_val = float(f_str)
            if f_val < smallest_f:
                smallest_f = f_val
            if f_val > largest_f:
                largest_f = f_val
            z_data.append(z_val)
            f_data.append(f_val)
            coord = (x_val,y_val)
            if coord not in self.coords_to_iv.keys():
                self.coords_to_iv[coord] = [[z_val], [f_val], [], []]
            else:
                self.coords_to_iv[coord][0].append(z_val)
                self.coords_to_iv[coord][1].append(f_val)
        self.i_max = largest_f
        self.i_min = smallest_f
        largest_didv = -1000
        smallest_didv = 1000
        for coord in self.coords_to_iv.keys():
            x = coord[0]
            y = coord[1]
            didv_vals, didv_norm_vals = self.tp_der_approx_at_point(x, y)
            didv_max = max(didv_vals)
            didv_min = min(didv_vals)
            if didv_max > largest_didv:
                largest_didv = didv_max
            if didv_min < smallest_didv:
                smallest_didv = didv_min
            self.coords_to_iv[coord][2] = didv_vals
            self.coords_to_iv[coord][3] = didv_norm_vals
        self.didv_max = largest_didv
        self.didv_min = smallest_didv
        self.coord_step = x_data[1] - x_data[0]
        data = [x_data, y_data, z_data, f_data]
        self.voltage = z_data[0]
        return data

    def init_lines(self):
        len_1d = len(self.one_dim_coords)
        self.line_dataset = np.zeros((len_1d, len_1d, 2, self.z_len + 1, 2), dtype='float32')

    def generate_lines(self):
        print(self.coords_to_iv.keys())
        for coord in self.coords:
            z_vals, f_vals, didv_vals, didv_norm_vals = self.coords_to_iv[coord]

            for k in range(len(z_vals)):
                z = z_vals[k]
                f = f_vals[k]
                didv = didv_vals[k]
                self.line_dataset[i][j][0][k][0] = z
                self.line_dataset[i][j][1][k][0] = z
                self.line_dataset[i][j][0][k][1] = f
                self.line_dataset[i][j][1][k][1] = didv

    def parse_ORNL_data(self, raw):

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
        largest_didv = -1000
        smallest_didv = 1000
        coords = []
        for row in str_data:
            x_str, y_str, z_str, didv_str = row.strip('\n').split('\t')
            x_val = float(x_str)*10**9
            y_val = float(y_str)*10**9
            coord = (x_val, y_val)
            coords.append(coord)
            x_data.append(x_val)
            y_data.append(y_val)
            z_val = float(z_str)
            didv_val = float(didv_str)
            if didv_val < smallest_didv:
                smallest_didv = didv_val
            if didv_val > largest_didv:
                largest_didv = didv_val
            z_data.append(z_val)
            didv_data.append(didv_val)
            if coord not in self.coords_to_iv.keys():
                self.coords_to_iv[coord] = [[z_val], [didv_val], [didv_val], [didv_val]]
            else:
                self.coords_to_iv[coord][0].append(z_val)
                self.coords_to_iv[coord][1].append(didv_val)
                self.coords_to_iv[coord][2].append(didv_val)
                self.coords_to_iv[coord][3].append( didv_val)  # this should be changed once I figure out normalization for ORNL data
        self.didv_max = largest_didv
        self.didv_min = smallest_didv
        self.coord_step = x_data[1] - x_data[0]
        data = [x_data, y_data, z_data, didv_data]
        self.voltage = z_data[0]
        self.coords = coords
        return data

    def initialize(self, filepath):
        raw = self.get_data(filepath)
        self.data_type = self.data_type_box.get()
        start = timer()
        if self.data_type == 'ORNL':
            data = self.parse_ORNL_data(raw)
        else:
            data = self.parse_data(raw)
        end = timer()
        print('Time to Parse: ', timedelta(seconds=end - start))
        start = timer()
        self.seperate_voltages(data)
        end = timer()
        self.init_lines()
        print('Time to seperate_voltages: ', timedelta(seconds=end - start))
        start = timer()
        self.generate_lines()
        end = timer()
        print('Time to Generate Lines: ', timedelta(seconds=end - start))

    def get_dIdV(self, coord, z):

        z_data, f_data, dfdz_data, dfdz_norm_data = self.coords_to_iv[coord]
        ind = z_data.index(z)
        dfdz = dfdz_data[ind]
        dfdz_norm = dfdz_norm_data[ind]

        return [dfdz, dfdz_norm]

    def seperate_voltages(self, data):

        x_data, y_data, z_data, f_data = data
        self.z_len = len(z_data)
        z_last = z_data[0]
        first_ind = 0
        first_run = True

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
                    coord = (x,y)
                    didv, didv_norm = self.get_dIdV(coord, z)
                    didv_vals.append(didv)
                    didv_norm_vals.append(didv_norm)
                dataset = [x_vals, y_vals, didv_vals, didv_norm_vals]
                self.z_to_xyf[z_last] = dataset
                first_ind = ind
                if first_run == True:
                    self.one_dim_coords = x_data[0:int(len(x_vals) ** .5)]
                    first_run = False

            z_last = z

        self.z_len = len(self.z_to_xyf.keys())
        self.coords = self.coords[0:len(self.coords)//self.z_len]
        return

    def tp_der_approx_at_point(self, x, y):  # two point derivative approximation at a single point across all voltages
        z_data, f_data, trash1, trash2 = self.coords_to_iv[(x,y)]
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
                dfdz_norm = dfdz / (z_data[i] * f_data[i] + shift)
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
        x = self.one_dim_coords
        y = self.one_dim_coords

        didz_ar = np.array(didz_data)
        didz_grid = didz_ar.reshape(int(len(x_data) ** .5), int(len(x_data) ** .5))

        didz_norm_ar = np.array(didz_norm_data)
        didz_norm_grid = didz_norm_ar.reshape(int(len(x_data) ** .5), int(len(x_data) ** .5))
        if self.color_max == 1000:
            self.im3 = self.ax3.pcolormesh(x, y, didz_norm_grid, shading='nearest', picker=True)
            self.im1 = self.ax1.pcolormesh(x, y, didz_grid, shading='nearest', picker=True)
            self.color_max = 100
        else:
            self.color_max = float(self.colorbar_higher_box.get())
            self.color_min = float(self.colorbar_lower_box.get())
            self.im3 = self.ax3.pcolormesh(x, y, didz_norm_grid, shading='nearest', picker=True, vmin=self.color_min,
                                           vmax=self.color_max)
            self.im1 = self.ax1.pcolormesh(x, y, didz_grid, shading='nearest', picker=True, vmin=self.color_min,
                                           vmax=self.color_max)
        self.ax1.title.set_text('dIdV Image at V = ' + str(self.voltage))
        self.cb1 = self.fig.colorbar(self.im1, ax=self.ax1, orientation='vertical')
        self.ax3.title.set_text('dIdV Normalized Image at V = ' + str(self.voltage))
        self.cb3 = self.fig.colorbar(self.im3, ax=self.ax3, orientation='vertical')
        self.canvas.draw()

    def plot_IV(self):
        if self.rectangle == False:
            self.ax2.cla()
            self.ax2.grid()
            self.ax2.title.set_text('IV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
            self.ax2.set_ylabel('Current (nA)')
        z_data, f_data, didv_vals, didv_norm_vals = self.coords_to_iv[self.coordinate]
        self.ax2.plot(z_data, f_data, c='#1f77b4')
        if self.rectangle == False:
            self.canvas.draw()

    def sort_by_position(self):
        with open('sorted_by_pos.csv', 'w') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',')
            for coord in self.coords_to_iv.keys():
                x, y = coord
                z_data, f_data = self.coords_to_iv[coord]
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
        z_data, f_data, didv_vals, didv_norm_vals = self.coords_to_iv[self.coordinate]
        self.ax4.plot(z_data, didv_vals, c='#1f77b4')

        if self.rectangle == False:
            self.canvas.draw()

    def nearest_coord(self, c):
        x = c[0]
        y = c[1]
        closest = (0, 0)
        nearest_d = 9999999
        for coord in self.coords_to_iv.keys():
            x2 = coord[0]
            y2 = coord[1]
            dist = ((x2 - x) ** 2 + (y2 - y) ** 2) ** .5
            if dist < nearest_d:
                nearest_d = dist
                closest = coord
        return closest

    def reset_ax(self, ax):

        if ax == self.ax2:
            self.ax2.cla()
            self.ax2.grid()
            self.ax2.title.set_text(
                'IV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
            self.ax2.set_ylabel('Current (nA)')

        if ax == self.ax4:
            self.ax4.cla()
            self.ax4.grid()
            self.ax4.title.set_text(
                'dIdV Plot at (x,y) = ' + self.coord_to_str(self.coordinate[0], self.coordinate[1]))
            self.ax4.set_xlabel('Voltage (V)')
            self.ax4.set_ylabel('dIdv (nA/V)')

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
            if click_coord != release_coord and event.name != 'button_press_event':
                self.reset_ax(self.ax2)
                self.reset_ax(self.ax4)
                self.rectangle = True
                nearest_to_click = self.nearest_coord(click_coord)
                nearest_to_release = self.nearest_coord(release_coord)
                x_bounds = [nearest_to_click[0], nearest_to_release[0]]
                y_bounds = [nearest_to_click[1], nearest_to_release[1]]
                x_min = min(x_bounds)
                x_max = max(x_bounds)
                y_min = min(y_bounds)
                y_max = max(y_bounds)
                x_vals = np.arange(self.one_dim_coords.index(x_min), self.one_dim_coords.index(x_max))
                y_vals = np.arange(self.one_dim_coords.index(y_min), self.one_dim_coords.index(y_max))
                ixgrid = np.ix_(x_vals, y_vals)
                line_data = self.line_dataset[ixgrid]

                iv_lines = []
                didv_lines = []
                for row in line_data:
                    for cell in row:
                        ivline, didv_line = cell
                        iv_lines.append(ivline)
                        didv_lines.append(didv_line)
                iv_plot = matplotlib.collections.LineCollection(iv_lines)
                didv_plot = matplotlib.collections.LineCollection(didv_lines)
                self.ax4.add_collection(didv_plot)
                self.ax4.set_xlim(self.volt_min, self.volt_max)
                if self.data_type != 'ORNL':
                    self.ax2.set_xlim(self.volt_min, self.volt_max)
                    self.ax2.set_ylim(self.i_min, self.i_max)
                    self.ax2.add_collection(iv_plot)
                self.ax4.set_ylim(self.didv_min, self.didv_max)
                self.canvas.draw()
                self.rectangle = False

            if click_coord == release_coord:

                xdata = event.xdata
                ydata = event.ydata
                click_point = (xdata, ydata)
                image_point = self.nearest_coord(click_point)
                self.coordinate = image_point
                self.coord_box.delete(0, END)
                self.coord_box.insert(0, self.coord_to_str(image_point[0], image_point[1]))
                if self.data_type != 'ORNL':
                    self.plot_IV()
                self.plot_didv()


if __name__ == "__main__":
    interface()

