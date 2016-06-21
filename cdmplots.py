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


class SingleGearFixedFulcrum:
    """Simulation of the Cycloid Drawing Machine with the 'simple setup'."""
    points_per_cycle = 100
    
    def __init__(self, turntable_radius, gear_radius, fulcrum_dist, 
                 fulcrum_gear_angle, fulcrum_penholder_dist, gear_slider_dist):
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
    
    def draw(self):
        """Draw the curve and the machine."""
        curve = self.simulate_cycle()
        
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
        # Penholder
        penholder_pos = self.C_f + rod_vect * self.d_p / rod_length
        patches.append(
            pat.Circle(penholder_pos, 0.3, color='lightblue', alpha=0.7))

        _, ax = plt.subplots()
        ax.set_aspect('equal')
        
        patches = col.PatchCollection(patches, match_original=True)
        ax.add_collection(patches)
        
        # Tip: calling plot() at the end will call autoscale_view()
        ax.plot(curve[0], curve[1])
        


def main():
    plt.ioff()

    params = {
        'turntable_radius': 5, 
        'gear_radius': 3, 
        'fulcrum_dist': 7, 
        'fulcrum_gear_angle': 2 * math.pi / 3, 
        'fulcrum_penholder_dist': 6, 
        'gear_slider_dist': 2}
    cdm = SingleGearFixedFulcrum(**params)
    cdm.draw()
    
    plt.show()


if __name__ == "__main__":
    main()
