# -*- coding: utf-8 -*-
"""
Specialization of matplotlib's Slider widget to take discrete values.

@author: Robin Roussel & anonymous author
Source:
    stackoverflow.com/questions/23703105/discrete-slider-in-matplotlib-widget
"""
from matplotlib.widgets import Slider
from numpy import float_


class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps.

    Parameters are identical to Slider.__init__, except for the new keyword
    'allowed_vals', which specifies the allowed positions of the slider.
    """

    def __init__(self, *args, **kwargs):
        self.allowed_vals = kwargs.pop('allowed_vals', None)
        if kwargs.get('valfmt') is None:
            kwargs['valfmt'] = '%.0f'
        self.previous_val = kwargs['valinit']
        Slider.__init__(self, *args, **kwargs)
        if self.allowed_vals is None:
            self.allowed_vals = [self.valmin, self.valmax]

    def set_val(self, val):
        # If 'valmin' or 'valmax' is given as an int, it will be passed as an
        # int when the user reaches an extremity of the slider, causing an
        # error with 'val - self.allowed_vals'.
        val = float_(val)

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
