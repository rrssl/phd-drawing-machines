#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Forward simulation of a ball-kicking toy.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import context
from controlpane import ControlPane
from mecha import Kicker
from mechaplot import mechaplot_factory


class KickPlot:
    """Example of Kicker controller."""

    def __init__(self, init_data):
        self.init_data = init_data

        self.mecha = Kicker(
            *[d.get('valinit') for _, d in init_data])
        self.crv = self.mecha.get_curve()

        self.init_draw()

    def init_draw(self):
        """Initialize the figure."""
        self.fig = plt.figure(figsize=(16,8))
        gs = GridSpec(1, 6)
        self.ax = self.fig.add_subplot(gs[:3])
        self.ax.set_aspect('equal')
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        plt.subplots_adjust(left=.05, wspace=0., hspace=1.)

        self.mecha_plot = mechaplot_factory(self.mecha, self.ax)

        self.crv_plot = self.ax.plot([], [], lw=2, alpha=.8)[0]

        bounds = []
        for i in range(len(self.mecha.props)):
            bounds.append(self.mecha.get_prop_bounds(i))
#            if i > 2:
#                # Account for slider imprecision wrt bounds.
#                bounds[-1] = bounds[-1][0] + 1e-3, bounds[-1][1] - 1e-3
        self.control_pane = ControlPane(self.fig, self.init_data, self.update,
                                        subplot_spec=gs[4:], bounds=bounds)

        self.redraw()

    def update(self, pid, val):
        """Update the figure."""
        success = self.mecha.update_prop(pid, val)
        if success:
            for i in range(len(self.mecha.props)):
                bounds = self.mecha.get_prop_bounds(i)
#                if i > 1:
#                    # Account for slider imprecision wrt bounds.
#                    bounds = bounds[0] + 1e-3, bounds[1] - 1e-3
                # Slider id is the same as parameter id.
                self.control_pane.set_bounds(i, bounds)
            self.crv = self.mecha.get_curve()
            self.redraw()
        else:
            print("Val", val, "with bounds", self.mecha.get_prop_bounds(pid))

    def redraw(self):
        """Redraw the canvas."""
        self.crv_plot.set_data(*self.crv)
        self.mecha_plot.redraw()


def main():
    plt.ioff()

    param_data = (
        (0,                     # Radius of gear 1.
         {'valmin': 1.,
          'valmax': 10.,
          'valinit': 4.,
          'label': "Gear 1 radius"}),
        (1,                     # Radius of gear 2.
         {'valmin': 1.,
          'valmax': 10.,
          'valinit': 2.,
          'label': "Gear 2 radius"}),
        (2,                     # X-coord of gear 1.
         {'valmin': 0.,
          'valmax': 10.,
          'valinit': 8.,
          'label': "Gear 2 X"}),
        (3,                     # Y-coord of gear 2.
         {'valmin': 0.,
          'valmax': 10.,
          'valinit': 1.,
          'label': "Gear 2 Y"}),
        (4,                     # Length of arm 1.
         {'valmin': 0.,
          'valmax': 10.,
          'valinit': 6.,
          'label': "Arm 1 length"}),
        (5,                     # Length of arm 2.
         {'valmin': 0.,
          'valmax': 10.,
          'valinit': 5.,
          'label': "Arm 2 length"}),
        (6,                     # Distance of end_effector from arm junction.
         {'valmin': 0.,
          'valmax': 10.,
          'valinit': 2.,
          'label': "End-effector distance"})
        )

    KickPlot(param_data)

    plt.show()


if __name__ == "__main__":
    main()
