#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation and display of the Cycloid Drawing Machine.

@author: Robin Roussel
"""
import math
import matplotlib.pyplot as plt

from controlpane import ControlPane
from mecha import SingleGearFixedFulcrumCDM
from mechaplot import mechaplot_factory


class CDMPlot:
    """Simulation of the Cycloid Drawing Machine with the 'simple setup'."""

    def __init__(self, init_data):
        self.init_data = init_data

        self.mecha = SingleGearFixedFulcrumCDM(
            *[d.get('valinit') for _, d in init_data])
        self.crv = self.mecha.get_curve()

        self.init_draw()

    def init_draw(self):
        """Initialize the figure."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

        self.mecha_plot = mechaplot_factory(self.mecha, self.ax)

        self.crv_plot = self.ax.plot([], [])[0]

        bounds = []
        for i in range(len(self.mecha.props)):
            bounds.append(self.mecha.get_prop_bounds(i))
            if i > 1:
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
                if i > 1:
                    # Account for slider imprecision wrt bounds.
                    bounds = bounds[0] + 1e-3, bounds[1] - 1e-3
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
        (0,                     # Radius of the turntable.
         {'valmin': 1,
          'valmax': 25,
          'valinit': 13,
          'label': "Turntable radius"}),
        (1,                     # Radius of the gear.
         {'valmin': 1,
          'valmax': 20,
          'valinit': 9,
          'label': "Gear radius"}),
        (2,                     # Distance from origin to fulcrum.
         {'valmin': 5.,
          'valmax': 30.,
          'valinit': 15,
          'label': "Fulcrum dist"}),
        (3,                     # Polar angle of the gear center.
         {'valmin': 0.,
          'valmax': math.pi,
          'valinit': 2 * math.pi / 3,
          'label': "Fulcrum-gear angle"}),
        (4,                     # Distance from fulcrum to penholder.
         {'valmin': 0.,
          'valmax': 40.,
          'valinit': 12,
          'label': "Fulcrum-penholder dist"}),
        (5,                     # Distance from gear center to slider.
         {'valmin': 0.,
          'valmax': 15.,
          'valinit': 6,
          'label': "Gear-slider dist"})
        )

    CDMPlot(param_data)

    plt.show()


if __name__ == "__main__":
    main()
