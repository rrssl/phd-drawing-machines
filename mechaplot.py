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
        return BaseSpirograph(mechanism, ax)
    elif type(mechanism) is mecha.EllipticSpirograph:
        return EllipticSpirograph(mechanism, ax)
    elif type(mechanism) is mecha.SingleGearFixedFulcrumCDM:
        return SingleGearFixedFulcrumCDM(mechanism, ax)


class BaseSpirograph:

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax

        self.shapes = [
            # Outer shape of the static ring
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=.8),
            # Inner shape of the static ring.
            pat.Circle((0., 0.), 0., color='white', alpha=1.),
            # Shape of the rolling gear.
            pat.Circle((0., 0.), 0., color='green', alpha=.6),
            pat.Circle((0., 0.), 0., edgecolor='green', facecolor='w',
                       alpha=1.)
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[:-2], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[-2:], match_original=True))

    def redraw(self):
        R, r, d = self.mecha.props

        # Static ring
        self.shapes[0].xy = np.array([-1.1 * R, -1.1 * R])
        self.shapes[0].set_width(R * 2.2)
        self.shapes[0].set_height(R * 2.2)
        self.shapes[1].radius = R
        # Rolling gear.
        self.shapes[2].center = np.array([R - r, 0.])
        self.shapes[2].radius = r
        self.shapes[3].center = np.array([R - r + d, 0.])
        self.shapes[3].radius = r * 0.1

        self.bg_coll.set_paths(self.shapes[:-2])
        self.fg_coll.set_paths(self.shapes[-2:])
        self.fg_coll.set_zorder(1)

        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*R, 1.1*R)


class EllipticSpirograph:

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax

        self.shapes = [
            # Outer shape of the static ring
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=.8),
            # Inner shape of the static ring.
            pat.Circle((0., 0.), 0., color='white', alpha=1.),
            # Shape of the rolling gear.
            pat.Ellipse((0., 0.), 0., 0., color='green', alpha=.6),
            pat.Circle((0., 0.), 0., edgecolor='green', facecolor='w',
                       alpha=1.)
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[:-2], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[-2:], match_original=True))

    def redraw(self):
        R, req, e2, d = self.mecha.props
        a = self.mecha._simulator.roul.m_obj.a

        # Static ring
        self.shapes[0].xy = np.array([-1.1 * R, -1.1 * R])
        self.shapes[0].set_width(R * 2.2)
        self.shapes[0].set_height(R * 2.2)
        self.shapes[1].radius = R
        # Rolling gear.
        self.shapes[2].center = np.array([R - a, 0.])
        self.shapes[2].width = 2 * a
        self.shapes[2].height = 2 * a * math.sqrt(1 - e2)
        self.shapes[3].center = np.array([R - a + d, 0.])
        self.shapes[3].radius = req * 0.1

        self.bg_coll.set_paths(self.shapes[:-2])
        self.fg_coll.set_paths(self.shapes[-2:])
        self.fg_coll.set_zorder(1)

        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*R, 1.1*R)


class SingleGearFixedFulcrumCDM:

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax

        self.shapes = [
            # Turntable
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            # Pinion
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            # Canvas
            pat.Circle((0., 0.), 0., color='white', alpha=1.),
            # Fulcrum
            pat.Circle((0., 0.), 0., color='pink', alpha=1.),
            # Slider
            pat.Circle((0., 0.), 0., color='lightgreen', alpha=1.),
            # Connecting rod
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=1.),
            # Penholder
            pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[:-2], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[-2:], match_original=True))

    def redraw(self):
        R_t, R_g, d_f, theta_g, d_p, d_s = self.mecha.props
        C_g = (R_t + R_g) * np.array([math.cos(theta_g),
                                      math.sin(theta_g)])
        C_f = np.array([d_f, 0.])

        # Turntable
        self.shapes[0].radius = R_t
        # Pinion
        self.shapes[1].center = C_g
        self.shapes[1].radius = R_g
        self.shapes[2].center = C_g
        self.shapes[2].radius = R_g * 0.1
        # Canvas
        self.shapes[3].radius = R_t * 0.95
        # Fulcrum
        self.shapes[4].center = C_f
        self.shapes[4].radius = R_t * 0.1
        # Slider
        slider_pos = C_g + (d_s, 0.)
        self.shapes[5].center = slider_pos
        self.shapes[5].radius = R_t * 0.1
        # Connecting rod
        rod_vect = slider_pos - C_f
        rod_length = np.linalg.norm(rod_vect)
        rod_angle = math.atan2(rod_vect[1], rod_vect[0])
        rod_thickness = R_t * .03
        rectangle_offset = np.array([0., -rod_thickness / 2])
        self.shapes[6].xy = C_f + rectangle_offset
        self.shapes[6].set_width(rod_length*1.5)
        self.shapes[6].set_height(rod_thickness)
        rot = Affine2D().rotate_around(*C_f, theta=rod_angle)
#        self.shapes[6].get_transform().get_affine().rotate_around(
#            C_f[0], C_f[1], rod_angle)
        self.shapes[6].set_transform(rot)
        # Penholder
        penholder_pos = C_f + rod_vect * d_p / rod_length
        self.shapes[7].center = penholder_pos
        self.shapes[7].radius = R_t * 0.1

        self.bg_coll.set_paths(self.shapes[:-2])
        self.fg_coll.set_paths(self.shapes[-2:])
        self.fg_coll.set_zorder(3)

        self.ax.set_xlim(1.1*min(C_g[0] - R_g, -R_t), 1.1*max(d_f, R_t))
        self.ax.set_ylim(-1.1*R_t, 1.1*max(C_g[1] + R_g, R_t))
