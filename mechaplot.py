# -*- coding: utf-8 -*-
"""
Rendering and animation of mechanisms.

@author: Robin Roussel
"""
import math
import matplotlib.animation as manim
import matplotlib.patches as pat
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
import numpy as np
#import shapely.geometry as geom
#import descartes

import mecha


def mechaplot_factory(mechanism, ax):
    if type(mechanism) is mecha.BaseSpirograph:
        return BaseSpirograph(mechanism, ax)
    elif type(mechanism) is mecha.EllipticSpirograph:
        return EllipticSpirograph(mechanism, ax)
    elif type(mechanism) is mecha.SingleGearFixedFulcrumCDM:
        return SingleGearFixedFulcrumCDM(mechanism, ax)
    elif type(mechanism) is mecha.HootNanny:
        return HootNanny(mechanism, ax)


class AniMecha:
    """Base class for mechanism animation."""
    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.play = False
        self.anim = manim.FuncAnimation(
            ax.figure, self.animate, frames=self.get_anim_time,
            interval=30, blit=True)
        ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_anim_time(self):
        t_max = self.mecha._simulator.get_cycle_length()
        dt = 1. / (2*math.pi)
        t = 0.
        while t < t_max:
            if self.play:
                t += dt
            yield t

#    def init_anim(self):
#        raise NotImplementedError

    def animate(self, t):
        """Compute new shape positions and orientations at time t.
        Returns an iterable of Artists to update.
        """
        raise NotImplementedError

    def on_key_press(self, event):
        """Manage key press events."""
        if event.key == ' ':
            self.play = not self.play


class BaseSpirograph(AniMecha):

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

        super().__init__(mechanism, ax)

    def redraw(self):
        R, r, d = self.mecha.props

        # Create new static ring
        self.shapes[0].xy = np.array([-1.1 * R, -1.1 * R])
        self.shapes[0].set_width(R * 2.2)
        self.shapes[0].set_height(R * 2.2)
        self.shapes[1].radius = R
        # Create new rolling gear.
        self.shapes[2].center = self.mecha.assembly['rolling_gear']['pos']
        self.shapes[2].radius = r
        self.shapes[3].center = self.mecha.assembly['penhole']['pos']
        self.shapes[3].radius = r * 0.1
        # Update patches.
        self.bg_coll.set_paths(self.shapes[:-2])
        self.fg_coll.set_paths(self.shapes[-2:])
        self.fg_coll.set_zorder(1)
        # Compute new limits.
        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*R, 1.1*R)
        # Use a dirty hack to reset the blit cache of the animation.
        self.anim._handle_resize()
        # Reset the timer as well.
        self.anim.frame_seq = self.anim.new_frame_seq()

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        self.mecha.set_state(t)
        self.shapes[2].center = self.mecha.assembly['rolling_gear']['pos']
        self.shapes[3].center = self.mecha.assembly['penhole']['pos']
        self.fg_coll.set_paths(self.shapes[-2:])
        return self.fg_coll,


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

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)


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
        self.shapes[7].radius = R_t * 0.05

        self.bg_coll.set_paths(self.shapes[:-2])
        self.fg_coll.set_paths(self.shapes[-2:])
        self.fg_coll.set_zorder(3)

        self.ax.set_xlim(1.1*min(C_g[0] - R_g, -R_t), 1.1*max(d_f, R_t))
        self.ax.set_ylim(-1.1*R_t, 1.1*max(C_g[1] + R_g, R_t))

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)


class HootNanny:

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax

        self.shapes = [
            # Turntable
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            # Gears
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            pat.Circle((0., 0.), 0., color='grey', alpha=.7),
            # Canvas
            pat.Circle((0., 0.), 0., color='white', alpha=1.),
            # Pivots
            pat.Circle((0., 0.), 0., color='pink', alpha=1.),
            pat.Circle((0., 0.), 0., color='pink', alpha=1.),
            # Linkages
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=1.),
            pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                          color='grey', alpha=1.),
            # Penholder
            pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[:6], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[6:], match_original=True))

    def redraw(self):
        r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2 = self.mecha.props
        C_G1 = np.array([r_T + r_G1, 0.])
        C_G2 = (r_T + r_G2) * np.array([math.cos(theta_12),
                                        math.sin(theta_12)])
        # TODO: stop accessing _simulator here, for we don't know if the data
        # matches the current property values. Instead, move 'assembly' to
        # the Mechanism instance, and whenever 'update_prop' or 'reset' is
        # called, update the assembly data as well. We don't need to simulate
        # a full cycle, only to simulate t=0.
        C_P1 = self.mecha._simulator.assembly['pivot_1'][:, 0]
        C_P2 = self.mecha._simulator.assembly['pivot_2'][:, 0]
        C_PH = self.mecha._simulator.assembly['pen'][:, 0]

        # Turntable
        self.shapes[0].radius = r_T
        # Gears
        self.shapes[1].center = C_G1
        self.shapes[1].radius = r_G1
        self.shapes[2].center = C_G1
        self.shapes[2].radius = r_G1 * 0.1
        self.shapes[3].center = C_G2
        self.shapes[3].radius = r_G2
        self.shapes[4].center = C_G2
        self.shapes[4].radius = r_G2 * 0.1
        # Canvas
        self.shapes[5].radius = r_T * 0.95
        # Pivots
        self.shapes[6].center = C_P1
        self.shapes[6].radius = r_G1 * 0.1
        self.shapes[7].center = C_P2
        self.shapes[7].radius = r_G2 * 0.1
        # Linkages
        rod_thickness = r_T * .02
        rectangle_offset = np.array([0., -rod_thickness / 2])
        # 1
        P1PH = C_PH - C_P1
        self.shapes[8].xy = C_P1 + rectangle_offset
        self.shapes[8].set_width(l1)
        self.shapes[8].set_height(rod_thickness)
        rot = Affine2D().rotate_around(
            *C_P1, theta=math.atan2(P1PH[1], P1PH[0]))
        self.shapes[8].set_transform(rot)
        # 2
        P2PH = C_PH - C_P2
        self.shapes[9].xy = C_P2 + rectangle_offset
        self.shapes[9].set_width(l2)
        self.shapes[9].set_height(rod_thickness)
        rot = Affine2D().rotate_around(
            *C_P2, theta=math.atan2(P2PH[1], P2PH[0]))
        self.shapes[9].set_transform(rot)
        # Penholder
        self.shapes[10].center = C_PH
        self.shapes[10].radius = r_T * 0.05

        self.bg_coll.set_paths(self.shapes[:6])
        self.fg_coll.set_paths(self.shapes[6:])
        self.fg_coll.set_zorder(3)

        self.ax.set_xlim(1.1*min(C_G2[0] - r_G2, -r_T),
                         1.1*max(C_G1[0] + r_G1, r_T))
        self.ax.set_ylim(-1.1*r_T, 1.1*max(C_G2[1] + r_G2, r_T))

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)
