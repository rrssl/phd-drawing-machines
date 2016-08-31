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


class SingleGearFixedFulcrum:
    """Simulation of the Cycloid Drawing Machine with the 'simple setup'."""
    points_per_cycle = 100

    def __init__(self, init_data):
        self.init_data = init_data
        self.reset(**{d.get('label'):d.get('valinit') for key, d in init_data})
        self.init_draw()

    def reset(self, turntable_radius, gear_radius, fulcrum_dist,
              fulcrum_gear_angle, fulcrum_penholder_dist, gear_slider_dist):
        """Reset the parameters."""
        self.R_t = turntable_radius
        self.R_g = gear_radius
        self.d_f = fulcrum_dist
        self.theta_g = fulcrum_gear_angle
        self.d_p = fulcrum_penholder_dist
        self.d_s = gear_slider_dist
        # Gear center
        self.C_g = (self.R_t + self.R_g) * np.array([math.cos(self.theta_g),
                                                     math.sin(self.theta_g)])
        # Fulcrum center
        self.C_f = np.array([self.d_f, 0.])

    def init_draw(self):
        """Initialize the figure."""
        self.fig, self.canvas = plt.subplots()
        self.canvas.set_aspect('equal')

        self.init_canvas()

        self.control_pane = ControlPane(self.fig, self.init_data, self.update)

    def init_canvas(self):
        """Initialize the canvas elements."""
        patches = [
            # Gears
            pat.Circle((0., 0.), self.R_t, color='grey', alpha=0.7),
            pat.Circle((0., 0.), self.R_t * 0.1, color='grey', alpha=0.7),
            pat.Circle(self.C_g, self.R_g, color='grey', alpha=0.7),
            pat.Circle(self.C_g, self.R_g * 0.1, color='grey', alpha=0.7),
            # Fulcrum
            pat.Circle(self.C_f, 0.3, color='red', alpha=0.7)]
        # Slider
        slider_pos = self.C_g + (self.d_s, 0.)
        patches.append(pat.Circle(slider_pos, 0.3, color='green', alpha=0.7))
        # Connecting rod
        rod_vect = slider_pos - self.C_f
        rod_length = np.linalg.norm(rod_vect)
        rod_angle = math.atan2(rod_vect[1], rod_vect[0])
        rod_thickness = 0.2
        rectangle_offset = ((rod_thickness / 2) *
                            np.array([ math.sin(rod_angle),
                                      -math.cos(rod_angle)]))
        patches.append(pat.Rectangle(
            self.C_f + rectangle_offset,
            width=rod_length*1.5,
            height=rod_thickness,
            angle=rod_angle*180/math.pi,
            color='grey', alpha=0.7))
        patches[-1].init_angle = rod_angle
        # Penholder
        penholder_pos = self.C_f + rod_vect * self.d_p / rod_length
        patches.append(
            pat.Circle(penholder_pos, 0.3, color='lightblue', alpha=0.7))

        self.shapes = patches
        self.collection = self.canvas.add_collection(
            col.PatchCollection(patches, match_original=True))

        # Tip: calling plot() at the end will call autoscale_view()
        curve = self.simulate_cycle()
        self.curve = self.canvas.plot(curve[0], curve[1])[0]

    def update(self, pid, val):
        """Update the figure."""
        setattr(self, pid, val)
        self.reset(self.R_t, self.R_g, self.d_f, self.theta_g , self.d_p,
                   self.d_s)
        self.redraw_canvas()

#    def update_param_bounds(self, param_changed):
#        """Recompute the parameter bounds after an update."""

    def simulate_cycle(self):
        """Simulate a complete cycle of the mechanism."""
        N_cycles = self.R_t
        # Parameter range
        t = np.linspace(0, N_cycles * 2 * math.pi,
                        N_cycles * SingleGearFixedFulcrum.points_per_cycle)
        # Slider curve
        curve = (self.d_s * np.vstack([np.cos(t), np.sin(t)]) +
                 self.C_g.reshape((2, 1)))
        # Connecting rod vector
        curve -= self.C_f.reshape((2, 1))
        # Penholder curve
        curve *= self.d_p / np.linalg.norm(curve, axis=0)
        curve += self.C_f.reshape((2, 1))
        # Space rotation
        ratio = self.R_g / self.R_t
        cos = np.cos(t * ratio)
        sin = np.sin(t * ratio)
        rot = np.array([[cos, -sin], [sin, cos]])
        curve = np.einsum('ijk,jk->ik', rot, curve)

        return curve

    def redraw_canvas(self):
        """Redraw the canvas."""
        # Turntable
        self.shapes[0].radius = self.R_t
        self.shapes[1].radius = self.R_t * 0.1
        # Gear
        self.shapes[2].center = self.C_g
        self.shapes[2].radius = self.R_g
        self.shapes[3].center = self.C_g
        self.shapes[3].radius = self.R_g * 0.1
        # Fulcrum
        self.shapes[4].center = self.C_f
        # Slider
        slider_pos = self.C_g + (self.d_s, 0.)
        self.shapes[5].center = slider_pos
        # Connecting rod
        rod_vect = slider_pos - self.C_f
        rod_length = np.linalg.norm(rod_vect)
        rod_angle = math.atan2(rod_vect[1], rod_vect[0])
        rod_thickness = 0.2
        rectangle_offset = ((rod_thickness / 2) *
                            np.array([ math.sin(rod_angle),
                                      -math.cos(rod_angle)]))
        self.shapes[6].xy = self.C_f + rectangle_offset
        self.shapes[6].set_width(rod_length*1.5)
        self.shapes[6].set_height(rod_thickness)
        rot = Affine2D().rotate_around(self.C_f[0], self.C_f[1],
                                       rod_angle - self.shapes[6].init_angle)
#        self.shapes[6].get_transform().get_affine().rotate_around(self.C_f[0], self.C_f[1], rod_angle)
        self.shapes[6].set_transform(rot)
        # Penholder
        penholder_pos = self.C_f + rod_vect * self.d_p / rod_length
        self.shapes[7].center = penholder_pos

        self.collection.set_paths(self.shapes)

        # Tip: calling plot() at the end will call autoscale_view()
        curve = self.simulate_cycle()
        self.curve.set_data(curve[0], curve[1])

#        self.canvas.relim()
#        self.canvas.autoscale_view(True, True, True)
#        self.canvas.redraw_in_frame()
#        self.fig.canvas.update()


def main():
    plt.ioff()

    param_data = (
        ('R_t',                     # Radius of the turntable.
         {'valmin': 1,
          'valmax': 25,
          'valinit': 23,
          'label': "turntable_radius"}),
        ('R_g',                     # Radius of the gear.
         {'valmin': 1,
          'valmax': 20,
          'valinit': 17,
          'label': "gear_radius"}),
        ('d_f',                     # Distance from origin to fulcrum.
         {'valmin': 5.,
          'valmax': 30.,
          'valinit': 24,
          'label': "fulcrum_dist"}),
        ('theta_g',                 # Polar angle of the gear center.
         {'valmin': 0.,
          'valmax': math.pi,
          'valinit': 2 * math.pi / 3,
          'label': "fulcrum_gear_angle"}),
        ('d_p',                     # Distance from fulcrum to penholder.
         {'valmin': 0.,
          'valmax': 20.,
          'valinit': 12,
          'label': "fulcrum_penholder_dist"}),
        ('d_s',                     # Distance from gear center to slider.
         {'valmin': 0.,
          'valmax': 15.,
          'valinit': 10,
          'label': "gear_slider_dist"})
        )

    SingleGearFixedFulcrum(param_data)

    plt.show()


if __name__ == "__main__":
    main()
