# -*- coding: utf-8 -*-
"""
Simulation and control of the elliptic spirograph.

@author: Robin Roussel
"""

import matplotlib.pyplot as plt

import context
from controlpane import ControlPane
from mecha import EllipticSpirograph
from mechaplot import mechaplot_factory


class EllipticSpiroPlot:
    """Simulation of the Spirograph."""

    def __init__(self, init_data):
        self.init_data = init_data

        self.mecha = EllipticSpirograph(
            *[d.get('valinit') for _, d in init_data])
        self.crv = self.mecha.get_curve()

        self.init_draw()

    def init_draw(self):
        """Initialize the figure."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')

        self.mecha_plot = mechaplot_factory(self.mecha, self.ax)

        self.crv_plot = self.ax.plot([], [], alpha=.8)[0]

        bounds = []
        for i in range(len(self.mecha.props)):
            bounds.append(self.mecha.get_prop_bounds(i))
#            if i > 1:
#                # Account for slider imprecision wrt bounds.
#                bounds[-1] = bounds[-1][0] + 1e-3, bounds[-1][1] - 1e-3
        self.control_pane = ControlPane(self.fig, self.init_data, self.update,
                                        bounds=bounds)

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
        (0,
         {'valmin': 1,
          'valmax': 15,
          'valinit': 8,
          'label': "Outer gear radius"}),
        (1,
         {'valmin': 1,
          'valmax': 15,
          'valinit': 5,
          'label': "Inner gear equiv. radius"}),
        (2,
         {'valmin': 0.,
          'valmax': 1.,
          'valinit': .2,
          'label': "Squared eccentricity"}),
        (3,
         {'valmin': 0.,
          'valmax': 15,
          'valinit': 2.,
          'label': "Hole distance"})
        )

    EllipticSpiroPlot(param_data)

    plt.show()


if __name__ == "__main__":
    main()
