# -*- coding: utf-8 -*-
"""
Simple rendering of mechanisms.

@author: Robin Roussel
"""
import math
import matplotlib.patches as pat
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
import numpy as np

import mecha


def mechaplot_factory(mechanism, ax):
    if type(mechanism) is mecha.BaseSpirograph:
        return None
    elif type(mechanism) is mecha.EllipticSpirograph:
        return None
    elif type(mechanism) is mecha.SingleGearFixedFulcrumCDM:
        return SingleGearFixedFulcrumCDM(mechanism, ax)


class SingleGearFixedFulcrumCDM:

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax

        self.shapes = [
            # Gears
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=0.7),
            # Fulcrum
            pat.Circle((0., 0.), 1., color='pink', alpha=0.7),
            # Slider
            pat.Circle((0., 0.), 1., color='lightgreen', alpha=0.7),
            # Connecting rod
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=0.7),
            # Penholder
            pat.Circle((0., 0.), 1., color='lightblue', alpha=0.7)]

        self.collection = self.ax.add_collection(
            PatchCollection(self.shapes, match_original=True))

    def redraw(self):
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
#        self.shapes[6].get_transform().get_affine().rotate_around(
#            C_f[0], C_f[1], rod_angle)
        self.shapes[6].set_transform(rot)
        # Penholder
        penholder_pos = C_f + rod_vect * d_p / rod_length
        self.shapes[7].center = penholder_pos

        self.collection.set_paths(self.shapes)

        self.ax.set_xlim(1.1*min(C_g[0] - R_g, -R_t), 1.1*max(d_f, R_t))
        self.ax.set_ylim(-1.1*R_t, 1.1*max(C_g[1] + R_g, R_t))
