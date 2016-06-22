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
from matplotlib.widgets import Slider

from discreteslider import DiscreteSlider


class SingleGearFixedFulcrum:
    """Simulation of the Cycloid Drawing Machine with the 'simple setup'."""
    points_per_cycle = 100
    
    def __init__(self, *args, **kwargs):
        self.reset(*args, **kwargs)
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
    
    def get_param_data(self):
        """Get a sequence of the parameter bounds and default values."""
        parameters = (
            ('R_t', {'valmin': 1,       # Radius of the turntable.
                     'valmax': 20,
                     'valinit': 5}),
            ('R_g', {'valmin': 1,       # Radius of the gear.
                     'valmax': 20,
                     'valinit': 3}),
            ('d_f', {'valmin': 5.,      # Distance from origin to fulcrum.
                     'valmax': 30.,
                     'valinit': 7.}),
            ('theta_g', {'valmin': 0.,    # Polar angle of the gear center.
                         'valmax': 2 * math.pi,
                         'valinit': 2 * math.pi / 3}),
            ('d_p', {'valmin': 0.,      # Distance from fulcrum to penholder.
                     'valmax': 20.,
                     'valinit': 6.}),
            ('d_s', {'valmin': 0.,      # Distance from gear center to slider.
                     'valmax': 3.,
                     'valinit': 2.})
            )

        return parameters
    
    def init_draw(self):
        """Initialize the figure."""
        self.fig, self.canvas = plt.subplots()
        self.canvas.set_aspect('equal')
        
        self.init_canvas()
        self.draw_control_pane()

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
    
    def create_update_func(self, name):
        return lambda val: self.update(name, val)

    def draw_control_pane(self):
        """Draw the control pane."""
        params = self.get_param_data()
        self.fig.subplots_adjust(bottom=0.1 + 0.05 * len(params))

        self.sliders = []
        for i, (name, args) in enumerate(params):
            ax = plt.axes([0.15, 0.05 * len(params) - 0.05 * i, 0.65, 0.03])
            if type(args['valmin']) == int:
                vals = range(args['valmin'], args['valmax'] + 1)
                slider = DiscreteSlider(ax, name, allowed_vals=vals, **args)
            else:
                slider = Slider(ax, name, **args)
            slider.on_changed(self.create_update_func(name))
            self.sliders.append(slider)

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
        
    def update(self, param, val):
        """Update the figure."""
        setattr(self, param, val)
        self.reset(self.R_t, self.R_g, self.d_f, self.theta_g , self.d_p,
                   self.d_s)
        self.redraw_canvas()


def main():
    plt.ioff()

    params = {
        'turntable_radius': 5, 
        'gear_radius': 3, 
        'fulcrum_dist': 7, 
        'fulcrum_gear_angle': 2 * math.pi / 3, 
        'fulcrum_penholder_dist': 6, 
        'gear_slider_dist': 2}
    SingleGearFixedFulcrum(**params)
#    cdm.draw()
    
    plt.show()


if __name__ == "__main__":
    main()
