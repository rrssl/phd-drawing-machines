#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation and control of the "Hoot-Nanny Magic Designer".

@author: Robin Roussel
"""
import math
import matplotlib.pyplot as plt

from controlpane import ControlPane
from mecha import HootNanny
from mechaplot import mechaplot_factory


class CDMPlot:
    """Simulation of the Cycloid Drawing Machine with the 'simple setup'."""

    def __init__(self, init_data):
        self.init_data = init_data

        self.mecha = HootNanny(
            *[d.get('valinit') for _, d in init_data])
        self.crv = self.mecha.get_curve(nb=2**10)

        self.init_draw()

    def init_draw(self):
        """Initialize the figure."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

        self.mecha_plot = mechaplot_factory(self.mecha, self.ax)

        self.crv_plot = self.ax.plot([], [], alpha=.8)[0]

        bounds = None
        bounds = []
        for i in range(len(self.mecha.props)):
            bounds.append(self.mecha.get_prop_bounds(i))
            if i > 2:
                # Account for slider imprecision wrt bounds.
                bounds[-1] = bounds[-1][0] + 1e-3, bounds[-1][1] - 1e-3
        self.control_pane = ControlPane(self.fig, self.init_data, self.update,
                                        bounds=bounds)

        self.redraw()

    def update(self, pid, val):
        """Update the figure."""
        success = self.mecha.update_prop(pid, val)
        if success:
            for i in range(len(self.mecha.props)):
                bounds = self.mecha.get_prop_bounds(i)
                if i > 2:
                    # Account for slider imprecision wrt bounds.
                    bounds = bounds[0] + 1e-3, bounds[1] - 1e-3
                # Slider id is the same as parameter id.
                self.control_pane.set_bounds(i, bounds)
            self.crv = self.mecha.get_curve(nb=2**10)
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
        (0,                     # Radius of the turntable.
         {'valmin': 1,
          'valmax': 25,
          'valinit': 10,
          'label': "Turntable radius"}),
        (1,                     # Radius of gear 1.
         {'valmin': 1,
          'valmax': 20,
          'valinit': 4,
          'label': "Gear 1 radius"}),
        (2,                     # Radius of gear 2.
         {'valmin': 1,
          'valmax': 20,
          'valinit': 2,
          'label': "Gear 2 radius"}),
        (3,                     # Polar angle between gears.
         {'valmin': 0.,
          'valmax': math.pi,
          'valinit': math.pi / 3,
          'label': "Gear 2 angle"}),
        (4,                     # Distance from gear 1 center to pivot.
         {'valmin': 0.,
          'valmax': 20.,
          'valinit': 2.5,
          'label': "Pivot 1 radius"}),
        (5,                     # Distance from gear 2 center to pivot.
         {'valmin': 0.,
          'valmax': 20.,
          'valinit': 1.5,
          'label': "Pivot 2 radius"}),
        (6,                     # Length of arm 1.
         {'valmin': 0.,
          'valmax': 40.,
          'valinit': 10.,
          'label': "Arm 1 length"}),
        (7,                     # Length of arm 2.
         {'valmin': 0.,
          'valmax': 40.,
          'valinit': 8.,
          'label': "Arm 2 length"})
        )

    CDMPlot(param_data)

    plt.show()


if __name__ == "__main__":
    main()
