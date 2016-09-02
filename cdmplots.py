#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation and display of the Cycloid Drawing Machine.

@author: Robin Roussel
"""
import math
import numpy as np
import matplotlib.collections as col
import matplotlib.patches as pat
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

from controlpane import ControlPane
from mecha import SingleGearFixedFulcrumCDM


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
        self.fig, self.canvas = plt.subplots()
        self.canvas.set_aspect('equal')

        patches = [
            # Gears
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            # Fulcrum
            pat.Circle((0., 0.), 1., color='red', alpha=0.7),
            # Slider
            pat.Circle((0., 0.), 1., color='green', alpha=0.7),
            # Connecting rod
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=0.7),
            # Penholder
            pat.Circle((0., 0.), 1., color='lightblue', alpha=0.7)]
        self.shapes = patches
        self.collection = self.canvas.add_collection(
            col.PatchCollection(patches, match_original=True))

        self.crv_plot = self.canvas.plot([], [])[0]

        bounds = []
        for i in range(len(self.mecha.props)):
            bounds.append(self.mecha.get_prop_bounds(i))
        self.control_pane = ControlPane(self.fig, self.init_data, self.update,
                                        bounds=bounds)

        self.redraw()

    def update(self, pid, val):
        """Update the figure."""
        success = self.mecha.update_prop(pid, val)
        if success:
            for i in range(len(self.mecha.props)):
                bounds = self.mecha.get_prop_bounds(i)
                # Slider id is the same as parameter id.
                self.control_pane.set_bounds(i, bounds)

            self.crv = self.mecha.get_curve()
            self.redraw()

    def redraw(self):
        """Redraw the canvas."""
        R_t, R_g, d_f, theta_g, d_p, d_s = self.mecha.props
        C_g = (R_t + R_g) * np.array([math.cos(theta_g),
                                      math.sin(theta_g)])
        C_f = np.array([d_f, 0.])

        # Turntable
        self.shapes[0].radius = R_t
        self.shapes[1].radius = R_t * 0.1
        # Gear
        self.shapes[2].center = C_g
        self.shapes[2].radius = R_g
        self.shapes[3].center = C_g
        self.shapes[3].radius = R_g * 0.1
        # Fulcrum
        self.shapes[4].center = C_f
        # Slider
        slider_pos = C_g + (d_s, 0.)
        self.shapes[5].center = slider_pos
        # Connecting rod
        rod_vect = slider_pos - C_f
        rod_length = np.linalg.norm(rod_vect)
        rod_angle = math.atan2(rod_vect[1], rod_vect[0])
        rod_thickness = 0.2
        rectangle_offset = ((rod_thickness / 2) *
                            np.array([ math.sin(rod_angle),
                                      -math.cos(rod_angle)]))
        self.shapes[6].xy = C_f + rectangle_offset
        self.shapes[6].set_width(rod_length*1.5)
        self.shapes[6].set_height(rod_thickness)
        rot = Affine2D().rotate_around(C_f[0], C_f[1], rod_angle)
#        self.shapes[6].get_transform().get_affine().rotate_around(C_f[0], C_f[1], rod_angle)
        self.shapes[6].set_transform(rot)
        # Penholder
        penholder_pos = C_f + rod_vect * d_p / rod_length
        self.shapes[7].center = penholder_pos

        self.collection.set_paths(self.shapes)

        self.crv_plot.set_data(*self.crv)

        self.canvas.set_xlim(1.1*min(C_g[0] - R_g, -R_t), 1.1*max(d_f, R_t))
        self.canvas.set_ylim(-1.1*R_t, 1.1*max(C_g[1] + R_g, R_t))


def main():
    plt.ioff()

    param_data = (
        (0,                     # Radius of the turntable.
         {'valmin': 1,
          'valmax': 25,
          'valinit': 23,
          'label': "Turntable radius"}),
        (1,                     # Radius of the gear.
         {'valmin': 1,
          'valmax': 20,
          'valinit': 17,
          'label': "Gear radius"}),
        (2,                     # Distance from origin to fulcrum.
         {'valmin': 5.,
          'valmax': 30.,
          'valinit': 24,
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
          'valinit': 10,
          'label': "Gear-slider dist"})
        )

    CDMPlot(param_data)

    plt.show()


if __name__ == "__main__":
    main()
