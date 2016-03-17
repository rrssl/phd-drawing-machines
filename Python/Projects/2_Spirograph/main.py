# -*- coding: utf-8 -*-
"""
Project 2: Spirograph

Author: Robin Roussel
"""

from fractions import Fraction
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from discreteslider import DiscreteSlider

class Curve:
    def __init__(self, samples):
        self.samples = samples
    def getX(self):
        pass
    def getY(self):
        pass
    def getRho(self):
        pass
    def getTheta(self):
        pass
    
class Hypotrochoid(Curve):
    def __init__(self, samples, R, r, d):
        super().__init__(samples)
        self.ext_gear_radius = R
        self.int_gear_radius = r
        self.hole_dist = d
        
    def getX(self):
        R = self.ext_gear_radius
        r = self.int_gear_radius
        d = self.hole_dist
        s = self.samples
        X = (R - r) * np.cos(s) + d * np.cos(s * (R - r) / r)
        
        return X
        
    def getY(self):
        R = self.ext_gear_radius
        r = self.int_gear_radius
        d = self.hole_dist
        s = self.samples
        Y = (R - r) * np.sin(s) - d * np.sin(s * (R - r) / r)
        
        return Y
        
    def getRho(self):
        pass
    def getTheta(self):
        pass

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
             
    def draw_grid(self):
        num_R_vals = 8
        num_d_vals = 5
        
        # Compute combinations.
        height = 0
        combi = np.empty((0, 3))
        for R in range(1, num_R_vals + 1):
            for r in range(1, R):
                if Fraction(R,r).denominator == r: # Avoid repeating patterns
                    height += 1
                    for d in np.linspace(0, r, num_d_vals + 1):
                        if d != 0.:
                            combi = np.vstack([combi, np.array([R, r, d])])
        
        # Draw curves.
        plt.close(plt.gcf())
        self.fig, axes = plt.subplots(height, num_d_vals)
        idx = 0                            
        for c in combi:
            self.R, self.r, self.d = c
            self.ax = axes.flat[idx]
            idx += 1
            self.draw_grid_cell()
        self.fig.canvas.draw()
        
        plt.subplots_adjust(bottom=0.02, top = 0.98)
        # Adjust figure size. Don't forget 'forward=True' to change window size
        # as well.
        base_size = self.fig.get_size_inches()
        self.fig.set_size_inches((base_size[0], 3 * base_size[1]), 
                                  forward=True)
                                  
    def draw_grid_cell(self):
        R = self.R
        r = self.r
        d = self.d
        N = Fraction.from_float(R/r).limit_denominator(100).denominator
        theta_range = np.linspace(0., N * 2 * np.pi, 
                                  N * SpiroPlot.points_per_turn)
        ax = self.ax
        
        hypo = Hypotrochoid(theta_range, R, r, d)
        
        ax.cla()
        ax.set_aspect('equal')
        ax.axis('off')

        ax.plot(hypo.getX(), hypo.getY())
        
        dim = R - r + d + 1
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)

if __name__ == "__main__":    
    plt.ioff()
    sp = SpiroPlot(show_spiro=True)
#    sp.draw_grid()
    sp.show()