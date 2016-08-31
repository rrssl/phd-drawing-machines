# -*- coding: utf-8 -*-
"""
Control pane (with sliders)

@author: Robin Roussel
"""
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.widgets import Slider

from discreteslider import DiscreteSlider


class ControlPane:
    """Control pane.

    Constructor parameters:
     -- figure:
         matplotlib figure.
     -- param_data:
         sequence of (id, dict) pairs, where 'id' is whatever identifier you
         want to give to the callback to know who called it, and 'dict'
         contains the keys 'valmin', 'valmax', 'valinit', and 'label'.
     -- update_func:
         callable object with arguments (id, value).
     -- subplot_spec:
         SubplotSpec defining the space in which to draw the control pane; if
         not specified, room will be made at the bottom of the figure.

    If objects of type int are given for the dictionary values, a discrete
    slider will be used.
    """

    def __init__(self, figure, param_data, update_func, subplot_spec=None):
        self.fig = figure
        self.param_data = param_data
        self.update = update_func
        self.subspec = subplot_spec

        self.draw()

    def _create_update_func(self, id_):
        return lambda val: self.update(id_, val)

    def draw(self):
        """Draw the control pane."""
        N = len(self.param_data)
        if self.subspec is None:
            self.fig.subplots_adjust(bottom=0.1 + 0.05*N)
        else:
            gs = GridSpecFromSubplotSpec(N, 1, subplot_spec=self.subspec)

        self.sliders = []
        for i, (id_, args) in enumerate(self.param_data):
            if self.subspec is None:
                ax = self.fig.add_axes([0.15, 0.05 * (N-i), 0.65, 0.03])
            else:
                ax = self.fig.add_subplot(gs[i, :])
            if type(args.get('valmin')) == int:
                vals = range(args.get('valmin'), args.get('valmax') + 1)
                slider = DiscreteSlider(ax, allowed_vals=vals, **args)
            else:
                slider = Slider(ax, **args)
            slider.on_changed(self._create_update_func(id_))
            self.sliders.append(slider)
