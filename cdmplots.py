# -*- coding: utf-8 -*-
"""
Simulation and display of the Cycloid Drawing Machine.

@author: Robin Roussel
"""
import math
import numpy as np
import matplotlib.lines as line
import matplotlib.patches as pat
import matplotlib.pyplot as plt


class SingleGearFixedFulcrum:
    """Simulation of the Cycloid Drawing Machine with the 'simple setup'."""
    points_per_cycle = 100
    
    def __init__(self):
        pass

def get_curve(canvas_pos, canvas_rad, pinion_pos, pinion_rad, tracer_dist):
    points_per_cycle = 100
    N = canvas_rad
    t = np.linspace(0, N * 2 * math.pi, N * points_per_cycle)
    
    curve = (pinion_pos.reshape((2, 1)) + 
             pinion_rad * np.vstack([np.cos(t), np.sin(t)]))
    curve *= tracer_dist / np.linalg.norm(curve, axis=0)
    
    cos = np.cos(t * pinion_rad / canvas_rad)
    sin = np.sin(t * pinion_rad / canvas_rad)
    rot = np.array([[cos, -sin], [sin, cos]])
    curve = (np.einsum('ijk,jk->ik', rot, curve - canvas_pos.reshape((2, 1))) + 
             canvas_pos.reshape((2, 1)))

    return curve

def main():
    gear_1_center = np.array([11., 9.])
    gear_1_radius = 7
    gear_2_center = np.array([18., 2.])
    gear_2_radius = 3
    tracer_dist = 12.
    curve = get_curve(gear_1_center, gear_1_radius, gear_2_center,
                      gear_2_radius, tracer_dist)
    
    gear_1 = pat.Circle(gear_1_center, gear_1_radius, color='grey', alpha=0.7)
    gear_2 = pat.Circle(gear_2_center, gear_2_radius, color='grey', alpha=0.7)
    slider_pos = gear_2_center + (gear_2_radius, 0.)
    bar = line.Line2D((0., slider_pos[0] * 1.1), (0., slider_pos[1] * 1.1), 
                      linewidth=5, color='grey', alpha=0.7)
    tracer_pos = slider_pos * tracer_dist / np.linalg.norm(slider_pos)
    tracer = pat.Circle(tracer_pos, 0.3, color='red', alpha=0.7)
    origin = pat.Circle((0., 0.), 0.3, color='grey', alpha=0.7)
    slider = pat.Circle(slider_pos, 0.3, color='green', alpha=0.7)
    
    plt.ioff()
    _, ax = plt.subplots()
    ax.set_aspect('equal')
    
    ax.add_artist(gear_1)
    ax.add_artist(gear_2)
    ax.add_artist(bar)
    ax.add_artist(tracer)
    ax.add_artist(origin)
    ax.add_artist(slider)
    plt.plot(curve[0], curve[1])
    ax.set_xlim(-1., 23.)
    ax.set_ylim(-3., 21.)

    plt.show()

if __name__ == "__main__":
    main()
