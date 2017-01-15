# -*- coding: utf-8 -*-
"""
Rendering and animation of mechanisms.

@author: Robin Roussel
"""
from odictliteral import odict
import math
import matplotlib.animation as manim
import matplotlib.patches as pat
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
import numpy as np
#import shapely.geometry as geom
#import descartes

import sys


def mechaplot_factory(mechanism, *args, **kwargs):
    """Create and return an instance of the corresponding graphical class."""
    # Hurray for introspection
    assert(mechanism.__module__.startswith('mecha'))
    cls = getattr(sys.modules[__name__], type(mechanism).__name__)
    return cls(mechanism, *args, **kwargs)
    # But seriously if this feels too hackish, this is +/- equivalent to:
#    import mecha
#    if type(mechanism) is mecha.BaseSpirograph:
#        return BaseSpirograph(mechanism, ax)
#    elif type(mechanism) is mecha.EllipticSpirograph:
#        return EllipticSpirograph(mechanism, ax)
#    elif type(mechanism) is mecha.SingleGearFixedFulcrumCDM:
#        return SingleGearFixedFulcrumCDM(mechanism, ax)
#    elif type(mechanism) is mecha.HootNanny:
#        return HootNanny(mechanism, ax)
#    elif type(mechanism) is mecha.Kicker:
#        return Kicker(mechanism, ax)


class AniMecha:
    """Base class for mechanism animation."""
    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.play = False
        self.anim = manim.FuncAnimation(
            ax.figure, self.animate, frames=self.get_anim_time,
            interval=40, blit=True)
        self.anim_plt = anim_plt
        if self.anim_plt is not None:
            self.anim_plt_init = self.anim_plt.get_data()
        ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)

    def get_anim_time(self):
        """Generator for the animation time."""
        t_max = self.mecha._simulator.get_cycle_length()
        dt = 1. / (4.*math.pi)
        t = 0.
        while t < t_max:
            if self.play:
                t += dt
            yield t

    def reset_anim(self):
        """Reset the animation. Will be visible after Axes redraw."""
        # Use a dirty hack to reset the blit cache of the animation.
        self.anim._handle_resize()
        # Reset the timer.
        self.anim.frame_seq = self.anim.new_frame_seq()
        if self.anim_plt is not None:
            self.anim_plt_init = self.anim_plt.get_data()

#    def init_anim(self):
#        raise NotImplementedError

    def animate(self, t):
        """Compute new shape positions and orientations at time t.
        Returns an iterable of Artists to update.
        """
        raise NotImplementedError

    def on_key_press(self, event):
        """Manage play/pause with the spacebar."""
        if event.key == ' ':
            self.play = not self.play


class BaseSpirograph(AniMecha):

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.ax = ax
        self.shapes = odict[
            'ring': [
                pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                              color='grey', alpha=.8),
                pat.Circle((0., 0.), 0., color='white', alpha=1.)
                ],
            'gear': [
                pat.Circle((0., 0.), 0., color='green', alpha=.6),
                pat.Circle((0., 0.), 0., edgecolor='green', facecolor='w',
                           alpha=1.)
                ]
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes['ring'], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes['gear'], match_original=True))

        super().__init__(mechanism, ax)

    def redraw(self):
        R, r, d = self.mecha.props

        # Create new static ring
        self.shapes['ring'][0].xy = np.array([-1.1 * R, -1.1 * R])
        self.shapes['ring'][0].set_width(R * 2.2)
        self.shapes['ring'][0].set_height(R * 2.2)
        self.shapes['ring'][1].radius = R
        # Create new rolling gear.
        self.shapes['gear'][0].center = self.mecha.assembly['rolling_gear']['pos']
        self.shapes['gear'][0].radius = r
        self.shapes['gear'][1].center = self.mecha.assembly['penhole']['pos']
        self.shapes['gear'][1].radius = r * 0.1
        # Update patches.
        self.bg_coll.set_paths(self.shapes['ring'])
        self.fg_coll.set_paths(self.shapes['gear'])
        self.fg_coll.set_zorder(1)
        # Compute new limits.
        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*R, 1.1*R)
        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        self.shapes['gear'][0].center = asb['rolling_gear']['pos']
        self.shapes['gear'][1].center = asb['penhole']['pos']


    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            self._redraw_moving_parts()
            self.fg_coll.set_paths(self.shapes['gear'])
        return self.fg_coll,


class EllipticSpirograph(AniMecha):

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.ax = ax
        self.shapes = odict[
            'ring': [
                pat.Rectangle((0., 0.), width=0., height=0., angle=0.,
                              color='grey', alpha=.8),
                pat.Circle((0., 0.), 0., color='white', alpha=1.),
                ],
            'gear': [
                pat.Ellipse((0., 0.), 0., 0., color='green', alpha=.6),
                pat.Circle((0., 0.), 0., edgecolor='green', facecolor='w',
                           alpha=1.)
                ]
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes['ring'], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes['gear'], match_original=True))

        super().__init__(mechanism, ax)

    def redraw(self):
        R, req, e2, d = self.mecha.props
        a = self.mecha._simulator.roul.m_obj.a
        # Static ring
        self.shapes['ring'][0].xy = np.array([-1.1 * R, -1.1 * R])
        self.shapes['ring'][0].set_width(R * 2.2)
        self.shapes['ring'][0].set_height(R * 2.2)
        self.shapes['ring'][1].radius = R
        # Rolling gear.
        self.shapes['gear'][0].width = 2 * a
        self.shapes['gear'][0].height = 2 * a * math.sqrt(1 - e2)
        self.shapes['gear'][1].radius = req * 0.1
        # Moving parts
        self._redraw_moving_parts()
        # Update patches.
        self.bg_coll.set_paths(self.shapes['ring'])
        self.fg_coll.set_paths(self.shapes['gear'])
        self.fg_coll.set_zorder(1)
        # Compute new limits.
        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*R, 1.1*R)
        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        self.shapes['gear'][0].center = asb['rolling_gear']['pos']
        self.shapes['gear'][0].angle = asb['rolling_gear']['or'] * 180. / math.pi
        self.shapes['gear'][1].center = asb['penhole']['pos']

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            self._redraw_moving_parts()
            self.fg_coll.set_paths(self.shapes['gear'])
        return self.fg_coll,


def _align_linkage_to_joints(p1, p2, linkage, offset):
    vec = p2 - p1
    linkage.xy = p1 + offset
    rot = Affine2D().rotate_around(*p1, theta=math.atan2(vec[1], vec[0]))
    linkage.set_transform(rot)


class SingleGearFixedFulcrumCDM(AniMecha):
    rod_thickness = .2

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.ax = ax
        self.shapes = odict[
            'turntable': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'gear': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7),
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'canvas': [
                pat.Circle((0., 0.), 0., color='white', alpha=1.)
                ],
            'fulcrum': [
                pat.Circle((0., 0.), 0., color='pink', alpha=1.)
                ],
            'slider': [
                pat.Circle((0., 0.), 0., color='lightgreen', alpha=1.)
                ],
            'link': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='grey', alpha=1.)
                ],
            'pen-holder': [
                pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
                ]
            ]
        self.bg_coll = self.ax.add_collection(PatchCollection(
            [patch for label in ['turntable', 'gear', 'canvas', 'fulcrum']
             for patch in self.shapes[label]],
            match_original=True))
        self.fg_coll = self.ax.add_collection(PatchCollection(
            [patch for label in ['slider', 'link', 'pen-holder']
             for patch in self.shapes[label]],
            match_original=True))

        super().__init__(mechanism, ax, anim_plt)

    def redraw(self):
        R_t, R_g, d_f, theta_g, d_p, d_s = self.mecha.props
        C_g = (R_t + R_g) * np.array([math.cos(theta_g),
                                      math.sin(theta_g)])
        C_f = np.array([d_f, 0.])

        # Static properties
        self.shapes['turntable'][0].radius = R_t
        self.shapes['gear'][0].center = C_g
        self.shapes['gear'][0].radius = R_g
        self.shapes['gear'][1].center = C_g
        self.shapes['gear'][1].radius = R_g * 0.1
        self.shapes['canvas'][0].radius = R_t * 0.95
        self.shapes['fulcrum'][0].center = C_f
        self.shapes['fulcrum'][0].radius = R_t * 0.1
        self.shapes['slider'][0].radius = R_t * 0.1
        self.shapes['link'][0].set_width(R_t + 2*R_g + d_f)
        self.shapes['pen-holder'][0].radius = R_t * 0.05
        # Moving parts
        self._redraw_moving_parts()
        # Update patches.
        self.bg_coll.set_paths(
            [patch for label in ['turntable', 'gear', 'canvas', 'fulcrum']
             for patch in self.shapes[label]])
        self.fg_coll.set_paths(
            [patch for label in ['slider', 'link', 'pen-holder']
             for patch in self.shapes[label]])
        self.fg_coll.set_zorder(3)
        # Compute new limits.
        self.ax.set_xlim(1.1*min(C_g[0] - R_g, -R_t), 1.1*max(d_f, R_t))
        self.ax.set_ylim(-1.1*R_t, 1.1*max(C_g[1] + R_g, R_t))
        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        OF = np.array([[self.mecha.props[2]],[0.]])
        OS = asb['slider']['pos']
        # Pivots
        self.shapes['slider'][0].center = OS
        self.shapes['pen-holder'][0].center = asb['pen-holder']['pos']
        # Link
        rectangle_offset = np.array([[0.], [-self.rod_thickness/2]])
        _align_linkage_to_joints(OF, OS, self.shapes['link'][0],
                                 rectangle_offset)

    def _rotate_plot(self):
        theta = self.mecha.assembly['turntable']['or']
        cos = np.cos(theta)
        sin = np.sin(theta)
        rot = np.array([[cos, -sin], [sin, cos]])
        points = rot.dot(self.anim_plt_init)

        self.anim_plt.set_data(*points)

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            self._redraw_moving_parts()
            self._rotate_plot()
            self.fg_coll.set_paths(
                [patch for label in ['slider', 'link', 'pen-holder']
                 for patch in self.shapes[label]])
        return self.fg_coll, self.anim_plt

class HootNanny(AniMecha):
    rod_thickness = .2

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.ax = ax
        self.shapes = odict[
            'turntable': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'gear_1': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7),
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'gear_2': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7),
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'canvas': [
                pat.Circle((0., 0.), 0., color='white', alpha=1.)
                ],
            'pivot_1': [
                pat.Circle((0., 0.), 0., color='pink', alpha=1.)
                ],
            'pivot_2': [
                pat.Circle((0., 0.), 0., color='pink', alpha=1.)
                ],
            'link_1': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='grey', alpha=1.)
                ],
            'link_2': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='grey', alpha=1.)
                ],
            'pen-holder': [
                pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
                ]
            ]
        self.bg_coll = self.ax.add_collection(PatchCollection(
            [patch for label in ['turntable', 'gear_1', 'gear_2', 'canvas']
             for patch in self.shapes[label]],
            match_original=True))
        self.fg_coll = self.ax.add_collection(PatchCollection(
            [patch for label in ['pivot_1', 'pivot_2', 'link_1', 'link_2',
                                 'pen-holder']
             for patch in self.shapes[label]],
            match_original=True))

        super().__init__(mechanism, ax, anim_plt)

    def redraw(self):
        r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2 = self.mecha.props
        C_G1 = np.array([r_T + r_G1, 0.])
        C_G2 = (r_T + r_G2) * np.array([math.cos(theta_12),
                                        math.sin(theta_12)])
        # Static properties
        self.shapes['turntable'][0].radius = r_T
        self.shapes['gear_1'][0].center = C_G1
        self.shapes['gear_1'][0].radius = r_G1
        self.shapes['gear_1'][1].center = C_G1
        self.shapes['gear_1'][1].radius = r_G1 * .1
        self.shapes['gear_2'][0].center = C_G2
        self.shapes['gear_2'][0].radius = r_G2
        self.shapes['gear_2'][1].center = C_G2
        self.shapes['gear_2'][1].radius = r_G2 * .1
        self.shapes['canvas'][0].radius = r_T * .95
        self.shapes['pivot_1'][0].radius = r_G1 * .1
        self.shapes['pivot_2'][0].radius = r_G1 * .1
        self.shapes['link_1'][0].set_width(l1)
        self.shapes['link_2'][0].set_width(l2)
        self.shapes['pen-holder'][0].radius = r_T * .05
        # Moving parts
        self._redraw_moving_parts()
        # Update patches
        self.bg_coll.set_paths(
            [patch for label in ['turntable', 'gear_1', 'gear_2', 'canvas']
             for patch in self.shapes[label]])
        self.fg_coll.set_paths(
            [patch for label in ['pivot_1', 'pivot_2', 'link_1', 'link_2',
                                 'pen-holder']
             for patch in self.shapes[label]])
        self.fg_coll.set_zorder(3)
        # Compute new limits.
        self.ax.set_xlim(1.1*min(C_G2[0] - r_G2, -r_T),
                         1.1*max(C_G1[0] + r_G1, r_T))
        self.ax.set_ylim(-1.1*r_T, 1.1*max(C_G2[1] + r_G2, r_T))
        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        OP1 = asb['pivot_1']['pos']
        OP2 = asb['pivot_2']['pos']
        OH = asb['pen-holder']['pos']

        self.shapes['pivot_1'][0].center = OP1
        self.shapes['pivot_2'][0].center = OP2
        self.shapes['pen-holder'][0].center = OH
        # Linkage
        rectangle_offset = np.array([[0.], [-self.rod_thickness/2.]])
        _align_linkage_to_joints(OP1, OH, self.shapes['link_1'][0],
                                 rectangle_offset)
        _align_linkage_to_joints(OP2, OH, self.shapes['link_2'][0],
                                 rectangle_offset)

    def _rotate_plot(self):
        theta = self.mecha.assembly['turntable']['or']
        cos = np.cos(theta)
        sin = np.sin(theta)
        rot = np.array([[cos, -sin], [sin, cos]])
        points = rot.dot(self.anim_plt_init)

        self.anim_plt.set_data(*points)

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            self._redraw_moving_parts()
            self._rotate_plot()
            self.fg_coll.set_paths(
                [patch for label in ['pivot_1', 'pivot_2', 'link_1', 'link_2',
                                     'pen-holder']
                 for patch in self.shapes[label]])
        return self.fg_coll, self.anim_plt


class Kicker(AniMecha):
    rod_thickness = .2

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.ax = ax

        self.shapes = odict[
            'gear_1': [
                pat.Circle((0., 0.), 1., color='grey', alpha=.7),
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'gear_2': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7),
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'arm_1': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='grey', alpha=1.)
                ],
            'arm_2': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='grey', alpha=1.)
                ],
            'thigh': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='m', alpha=1.)
                ],
            'calf': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='m', alpha=1.)
                ],
            'foot': [
                pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                              angle=0., color='m', alpha=1.)
                ],
            'pivot_1': [
                pat.Circle((0., 0.), 0., color='pink', alpha=1.)
                ],
            'pivot_2': [
                pat.Circle((0., 0.), 0., color='pink', alpha=1.)
                ],
            'pivot_12': [
                pat.Circle((0., 0.), 0., color='pink', alpha=1.)
                ],
            'pivot_hip': [
                pat.Circle((0., 0.), 0., color='lightgreen', alpha=1.)
                ],
            'pivot_knee': [
                pat.Circle((0., 0.), 0., color='lightgreen', alpha=1.)
                ],
            'pivot_ankle': [
                pat.Circle((0., 0.), 0., color='lightgreen', alpha=1.)
                ],
            'connector': [
                pat.Circle((0., 0.), 0., color='blue', alpha=1.)
                ],
            'end_effector': [
                pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
                ]
            ]

        self.bg_coll = self.ax.add_collection(
            PatchCollection(
                [patch for shape in self.shapes.values() for patch in shape],
                match_original=True))
#        self.fg_coll = self.ax.add_collection(
#            PatchCollection(self.shapes[6:], match_original=True))

        super().__init__(mechanism, ax)

    def redraw(self):
        r1, r2, _, l1, l2, d = self.mecha.props
        asb = self.mecha.assembly
        OG1 = asb['gear_1']['pos']
        OG2 = asb['gear_2']['pos']
        OH = asb['hip']['pos']
        # Static properties
        pivot_size = (r1+r2) * .05
        self.shapes['gear_1'][0].center = OG1
        self.shapes['gear_1'][0].radius = r1 * 1.2
        self.shapes['gear_1'][1].center = OG1
        self.shapes['gear_1'][1].radius = pivot_size
        self.shapes['gear_2'][0].center = OG2
        self.shapes['gear_2'][0].radius = r2 * 1.2
        self.shapes['gear_2'][1].center = OG2
        self.shapes['gear_2'][1].radius = pivot_size
        self.shapes['pivot_1'][0].radius = pivot_size
        self.shapes['pivot_2'][0].radius = pivot_size
        self.shapes['pivot_12'][0].radius = pivot_size
        self.shapes['pivot_hip'][0].center = OH
        self.shapes['pivot_hip'][0].radius = pivot_size
        self.shapes['pivot_knee'][0].radius = pivot_size
        self.shapes['pivot_ankle'][0].radius = pivot_size
        self.shapes['arm_1'][0].set_width(l1+d)
        self.shapes['arm_2'][0].set_width(l2)
        self.shapes['thigh'][0].set_width(l1+d)
        self.shapes['calf'][0].set_width(l1+d)
        self.shapes['foot'][0].set_width(1.5*(l1+d)/5)
        self.shapes['connector'][0].radius = pivot_size
        self.shapes['end_effector'][0].radius = pivot_size
        # Moving parts
        self._redraw_moving_parts()

        self.bg_coll.set_paths(
            [patch for shape in self.shapes.values() for patch in shape])
#        self.fg_coll.set_paths(self.shapes[6:])
#        self.fg_coll.set_zorder(3)

        OH = asb['hip']['pos']
        OE = asb['end_effector']['pos']
        self.ax.set_xlim(1.1*min(OG1[0]-r1-l1, OG2[0]-r2-l2, OH[0], OE[0]),
                         1.1*max(OG1[0]+r1+l1, OG2[0]+r2+l2, OH[0], OE[0]))
        self.ax.set_ylim(1.1*min(OG1[1]-r1-l1, OG2[1]-r2-l2, OH[1], OE[1]),
                         1.1*max(OG1[1]+r1+l1, OG2[1]+r2+l2, OH[1], OE[1]))

        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        OP1 = asb['pivot_1']['pos']
        OP2 = asb['pivot_2']['pos']
        OP12 = asb['pivot_12']['pos']
        OH = asb['hip']['pos']
        OK = asb['knee']['pos']
        OA = asb['ankle']['pos']
        OE = asb['end_effector']['pos']
        # Pivots
        self.shapes['pivot_1'][0].center = OP1
        self.shapes['pivot_2'][0].center = OP2
        self.shapes['pivot_12'][0].center = OP12
        self.shapes['pivot_knee'][0].center = OK
        self.shapes['pivot_ankle'][0].center = OA
        self.shapes['connector'][0].center = asb['connector']['pos']
        self.shapes['end_effector'][0].center = OE
        # Linkage
        rectangle_offset = np.array([[0.], [-self.rod_thickness/2.]])
        _align_linkage_to_joints(OP1, OP12, self.shapes['arm_1'][0],
                                 rectangle_offset)
        _align_linkage_to_joints(OP2, OP12, self.shapes['arm_2'][0],
                                 rectangle_offset)
        _align_linkage_to_joints(OH, OK, self.shapes['thigh'][0],
                                 rectangle_offset)
        _align_linkage_to_joints(OK, OA, self.shapes['calf'][0],
                                 rectangle_offset)
        _align_linkage_to_joints(OA, OE, self.shapes['foot'][0],
                                 rectangle_offset)

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
#        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(-t)
            self._redraw_moving_parts()
            self.bg_coll.set_paths(
                [patch for shape in self.shapes.values() for patch in shape])
        return self.bg_coll,


class Thing(AniMecha):
    rod_thickness = .02

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        nb_props = len(mechanism.props)
        self.ax = ax
        self.shapes = odict[
            'turntable': [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ],
            'canvas': [
                pat.Circle((0., 0.), 0., color='white', alpha=1.)
                ],
            'pen-holder': [
                pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
                ]
            ]
        self.shapes.update(
            {'gear_{}'.format(i): [
                pat.Circle((0., 0.), 0., color='grey', alpha=.7),
                pat.Circle((0., 0.), 0., color='grey', alpha=.7)
                ]
             for i in range(nb_props)
            })
        self.shapes.update(
            {'pivot_{}'.format(i): [
                pat.Circle((0., 0.), .03, color='pink', alpha=1.)
                ]
             for i in range(nb_props)
            })
        self.shapes.update(
            {'joint_{}'.format(i): [
                pat.Circle((0., 0.), .03, color='lightgreen', alpha=1.)
                ]
             for i in range(nb_props-1)
            })
        self.shapes.update(
            {'link_{}'.format(i): [
                pat.Rectangle((0., 0.), width=3., height=self.rod_thickness,
                              angle=0., color='grey', alpha=1.)
                ]
             for i in range(nb_props)
            })
        self.bg_labels = ['turntable', 'canvas'] + [
            'gear_{}'.format(i) for i in range(nb_props)]
        self.bg_coll = self.ax.add_collection(PatchCollection(
            [patch for label in self.bg_labels
             for patch in self.shapes[label]],
            match_original=True))
        self.fg_labels = ['pen-holder'] + [
            'link_{}'.format(i) for i in range(nb_props)] + [
            'joint_{}'.format(i) for i in range(nb_props-1)] + [
            'pivot_{}'.format(i) for i in range(nb_props)]
        self.fg_coll = self.ax.add_collection(PatchCollection(
            [patch for label in self.fg_labels
             for patch in self.shapes[label]],
            match_original=True))

        super().__init__(mechanism, ax, anim_plt)

    def redraw(self):
        r_T = 1.
        asb = self.mecha.assembly
        nb_props = len(self.mecha.props)
        r_G = self.mecha.constraint_solver.get_radius
        # Static properties
        self.shapes['turntable'][0].radius = r_T
        self.shapes['canvas'][0].radius = r_T * .95
        self.shapes['pen-holder'][0].radius = r_T * .05
        for i in range(nb_props):
            gear = 'gear_{}'.format(i)
            self.shapes[gear][0].center = asb[gear]['pos']
            self.shapes[gear][0].radius = r_G(i)
            self.shapes[gear][1].center = asb[gear]['pos']
            self.shapes[gear][1].radius = r_G(i) * .1
        self.shapes['link_{}'.format(nb_props-1)][0].set_width(np.linalg.norm(
            asb['pen-holder']['pos']
            - asb['joint_{}'.format(nb_props-2)]['pos']))
        # Moving parts
        self._redraw_moving_parts()
        # Update patches
        self.bg_coll.set_paths(
            [patch for label in self.bg_labels
             for patch in self.shapes[label]])
        self.fg_coll.set_paths(
            [patch for label in self.fg_labels
             for patch in self.shapes[label]])
        self.fg_coll.set_zorder(3)
        # Compute new limits.
        self.ax.set_xlim(1.1*(-r_T - 2*r_G(nb_props-1)), 1.1*(r_T + 2*r_G(0)))
        self.ax.set_ylim(-1.1*r_T, 1.5*r_T)
        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        nb_props = len(self.mecha.props)
        rectangle_offset = np.array([[0.], [-self.rod_thickness/2.]])

        for i in range(nb_props):
            pivot = 'pivot_{}'.format(i)
            self.shapes[pivot][0].center = asb[pivot]['pos']
            if i < nb_props - 1:
                joint = 'joint_{}'.format(i)
                self.shapes[joint][0].center = asb[joint]['pos']
                link = 'link_{}'.format(i)
                if i > 0:
                    prev_joint = 'joint_{}'.format(i-1)
                    next_pivot = 'pivot_{}'.format(i+1)
                    _align_linkage_to_joints(asb[prev_joint]['pos'],
                                             asb[next_pivot]['pos'],
                                             self.shapes[link][0],
                                             rectangle_offset)
        _align_linkage_to_joints(
            asb['pivot_0']['pos'],
            asb['pivot_1']['pos'],
            self.shapes['link_0'][0],
            rectangle_offset
            )
        _align_linkage_to_joints(
            asb['joint_{}'.format(nb_props-2)]['pos'],
            asb['pen-holder']['pos'],
            self.shapes['link_{}'.format(nb_props-1)][0],
            rectangle_offset
            )
        self.shapes['pen-holder'][0].center = asb['pen-holder']['pos']

    def _rotate_plot(self):
        theta = self.mecha.assembly['turntable']['or']
        cos = np.cos(theta)
        sin = np.sin(theta)
        rot = np.array([[cos, -sin], [sin, cos]])
        points = rot.dot(self.anim_plt_init)

        self.anim_plt.set_data(*points)

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            self._redraw_moving_parts()
            self._rotate_plot()
            self.fg_coll.set_paths(
                [patch for label in self.fg_labels
                 for patch in self.shapes[label]])
        return self.fg_coll, self.anim_plt

