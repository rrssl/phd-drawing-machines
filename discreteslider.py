# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:39:30 2016

@author: Robin R. from a script by an anonymous author, cf. source link below.

https://stackoverflow.com/questions/23703105/discrete-slider-in-matplotlib-widget
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

class ChangingPlot:
    def __init__(self):
        min_val = 1
        step = 1
        max_val = 10
        x = np.arange(min_val, max_val + step, step)

        self.fig, self.ax = plt.subplots()
        self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03], 
                                          axisbg='yellow')
        self.slider = DiscreteSlider(self.sliderax, 'Value', min_val, max_val,
                                     allowed_vals=x, valinit=x[0])

        self.slider.on_changed(self.update)

        self.ax.plot(x, x, 'ro')
        self.dot, = self.ax.plot(x[0], x[0], 'bo', markersize=18)

    def update(self, value):
        self.dot.set_data([[value],[value]])
        self.fig.canvas.draw()

    def show(self):
        plt.show()

class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """
        Identical to Slider.__init__, except for the new keyword 'allowed_vals'.
        This keyword specifies the allowed positions of the slider
        """
        self.allowed_vals = kwargs.pop('allowed_vals', None)
        self.previous_val = kwargs['valinit']
        Slider.__init__(self, *args, **kwargs)
        if self.allowed_vals == None:
            self.allowed_vals = [self.valmin,self.valmax]

    def set_val(self, val):
        discrete_val = self.allowed_vals[abs(val - self.allowed_vals).argmin()]
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        if self.drawon: 
            self.ax.figure.canvas.draw()
        self.val = discrete_val
        if self.previous_val != discrete_val:
            self.previous_val = discrete_val
            if not self.eventson: 
                return
            for cid, func in self.observers.items():
                func(discrete_val)

if __name__ == "__main__":  
    p = ChangingPlot()
    p.show()