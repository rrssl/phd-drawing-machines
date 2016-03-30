# -*- coding: utf-8 -*-
"""
Spirograph plotting classes.

@author: Robin Roussel
"""

from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from discreteslider import DiscreteSlider
from curves import Hypotrochoid

class SpiroPlot:
    points_per_turn = 50

    def __init__(self, show_spiro=True):
        self.R_min_val = 1
        self.R_max_val = 15 # included
        self.R_vals = np.arange(self.R_min_val, self.R_max_val + 1, 1)
        self.R = 8
#        self.R = 5.0051

        self.r_min_val = 1
        self.r_max_val = 10 # included
        self.r_vals = np.arange(self.r_min_val, self.r_max_val + 1, 1)
        self.r = 5

        self.d_min_val = 0.01
        self.d_max_val = 10. # included
        self.d = 0.5

        self.fig, self.ax = plt.subplots()
#        fig = plt.figure()
        plt.subplots_adjust(bottom=0.25)
#        ax = fig.add_subplot(111)

        axcolor = 'lightgoldenrodyellow'
        self.sliderax_R = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg=axcolor)
        self.sliderax_r = plt.axes([0.15, 0.1, 0.65, 0.03], axisbg=axcolor)
        self.sliderax_d = plt.axes([0.15, 0.15, 0.65, 0.03], axisbg=axcolor)

        self.s_R = DiscreteSlider(self.sliderax_R, 'R', self.R_min_val,
                                  self.R_max_val, allowed_vals=self.R_vals,
                                  valinit=self.R)
#        self.s_R = Slider(self.sliderax_R, 'R', self.R_min_val, self.R_max_val,
#                          valinit=self.R)
        self.s_r = DiscreteSlider(self.sliderax_r, 'r', self.r_min_val,
                                  self.r_max_val, allowed_vals=self.r_vals,
                                  valinit=self.r)
        self.s_d = Slider(self.sliderax_d, 'd', self.d_min_val, self.d_max_val,
                          valinit=self.d)

        self.s_R.on_changed(self.update)
        self.s_r.on_changed(self.update)
        self.s_d.on_changed(self.update)

        self.show_spiro = show_spiro

        self.draw()

    def draw(self):
        R = self.R
        r = self.r
        d = self.d
#        N = Fraction(int(R), int(r)).denominator # /!\ R & r are numpy floats
        N = Fraction.from_float(R/r).limit_denominator(100).denominator
        theta_range = np.linspace(0., N * 2 * np.pi,
                                  N * SpiroPlot.points_per_turn)
        ax = self.ax

        hypo = Hypotrochoid(theta_range, R, r, d)
        if self.show_spiro:
            out_gear = plt.Circle((0,0), R, color='r', fill=False)
            int_gear = plt.Circle((R - r,0), r, color='g', fill=False)
            hole = plt.Circle((R - r + d,0), r / 20, color='g', fill=False)

        ax.cla()
        ax.set_aspect('equal')
        ax.grid(True)

        if self.show_spiro:
            ax.add_artist(out_gear)
            ax.add_artist(int_gear)
            ax.add_artist(hole)
        ax.plot(hypo.getX(), hypo.getY())

        dim = max(R - r + d, R) + 1
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)

        self.fig.canvas.draw()
#        self.fig.canvas.draw_idle()

    def update(self, value):
        self.R = self.s_R.val
        self.r = self.s_r.val
        self.d = self.s_d.val
        self.draw()

    def show(self):
        plt.show()

class SpiroGridPlot():
    def __init__(self):
        self.plot_increasing_ratios = True
        self.draw_grid()        

    def draw_grid(self):
        num_R_vals = 7
        num_d_vals = 5

        # Compute combinations.
        combi = Hypotrochoid.get_param_combinations(num_R_vals, num_d_vals)
        if self.plot_increasing_ratios:
            ratios = (combi[:,1] / combi[:,0]) + 1e-6 * combi[:,2]
            sorted_indices = np.argsort(ratios)
            combi = combi[sorted_indices, :]

        # Draw curves.
        fig, frame = plt.subplots()

        for i, c in enumerate(combi):
            position = np.array([i % num_d_vals, i // num_d_vals]) * [5, -3]
            if c[1] / c[0] != combi[i - 1, 1] / combi[i - 1, 0]:
                frame.text(-6, position[1] - 0.3, 
                           r'$\frac{{{:.0f}}}{{{:.0f}}}$'.format(c[1], c[0]), 
                           fontsize=20)            
            self.draw_grid_cell(c, position)
        
        # Adjust figure size. Don't forget 'forward=True' to change window size
        # as well.
        base_size = fig.get_size_inches()
        fig.set_size_inches((base_size[0], 3 * base_size[1]),
                            forward=True)
                            
        plt.subplots_adjust(bottom=0.02, top = 0.98)
        frame.set_ylim(-3 * (combi.shape[0] // num_d_vals), 3)
        frame.set_aspect('equal')
        frame.axis('off')

    def draw_grid_cell(self, parameters, position):
        R, r, d = parameters
        N = r   # r is already the smallest integer denominator.
        theta_range = np.linspace(0., N * 2 * np.pi,
                                  N * SpiroPlot.points_per_turn)

        hypo = Hypotrochoid(theta_range, R, r, d)
        
        curve = np.array([hypo.getX(), hypo.getY()])
        curve = curve / abs(curve).max()
        curve = curve + position.reshape(2, 1)

        plt.plot(curve[0], curve[1], 'b-')

    def show(self):
        plt.show()