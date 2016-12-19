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

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax
        # TODO use odict
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
        # Reset animation.
        self.reset_anim()

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            # TODO put this in _redraw_moving_parts
            self.shapes[2].center = self.mecha.assembly['rolling_gear']['pos']
            self.shapes[3].center = self.mecha.assembly['penhole']['pos']
            self.fg_coll.set_paths(self.shapes[-2:])
        return self.fg_coll,


class EllipticSpirograph(AniMecha):

    def __init__(self, mechanism, ax):
        self.mecha = mechanism
        self.ax = ax
        # TODO use odict
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

        super().__init__(mechanism, ax)

    def redraw(self):
        R, req, e2, d = self.mecha.props
        a = self.mecha._simulator.roul.m_obj.a
        # Static ring
        self.shapes[0].xy = np.array([-1.1 * R, -1.1 * R])
        self.shapes[0].set_width(R * 2.2)
        self.shapes[0].set_height(R * 2.2)
        self.shapes[1].radius = R
        # Rolling gear.
        self.shapes[2].width = 2 * a
        self.shapes[2].height = 2 * a * math.sqrt(1 - e2)
        self.shapes[3].radius = req * 0.1
        # Moving parts
        self._redraw_moving_parts()
        # Update patches.
        self.bg_coll.set_paths(self.shapes[:-2])
        self.fg_coll.set_paths(self.shapes[-2:])
        self.fg_coll.set_zorder(1)
        # Compute new limits.
        self.ax.set_xlim(-1.1*R, 1.1*R)
        self.ax.set_ylim(-1.1*R, 1.1*R)
        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        asb = self.mecha.assembly
        self.shapes[2].center = asb['rolling_gear']['pos']
        self.shapes[2].angle = asb['rolling_gear']['or'] * 180. / math.pi
        self.shapes[3].center = asb['penhole']['pos']

    def set_visible(self, b):
        self.bg_coll.set_visible(b)
        self.fg_coll.set_visible(b)

    def animate(self, t):
        if self.play:
            self.mecha.set_state(t)
            self._redraw_moving_parts()
            self.fg_coll.set_paths(self.shapes[-2:])
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
        # TODO use odict
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
            # Link
            pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                          angle=0., color='grey', alpha=1.),
            # Penholder
            pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[:-3], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[-3:], match_original=True))

        super().__init__(mechanism, ax, anim_plt)

    def redraw(self):
        R_t, R_g, d_f, theta_g, d_p, d_s = self.mecha.props
        C_g = (R_t + R_g) * np.array([math.cos(theta_g),
                                      math.sin(theta_g)])
        C_f = np.array([d_f, 0.])

        # Static properties

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
        self.shapes[5].radius = R_t * 0.1
        # Arm
        self.shapes[6].set_width(R_t + 2*R_g + d_f)
        # Penholder
        self.shapes[7].radius = R_t * 0.05

        # Moving parts
        self._redraw_moving_parts()

        # Update patches.
        self.bg_coll.set_paths(self.shapes[:-3])
        self.fg_coll.set_paths(self.shapes[-3:])
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
        self.shapes[-3].center = OS
        self.shapes[-1].center = asb['pen-holder']['pos']
        # Link
        rectangle_offset = np.array([[0.], [-self.rod_thickness/2]])
        _align_linkage_to_joints(OF, OS, self.shapes[-2], rectangle_offset)

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
            self.fg_coll.set_paths(self.shapes[-3:])
        return self.fg_coll, self.anim_plt

class HootNanny(AniMecha):
    rod_thickness = .2

    def __init__(self, mechanism, ax, anim_plt=None):
        self.mecha = mechanism
        self.ax = ax
        # TODO use odict
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
            pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                          angle=0., color='grey', alpha=1.),
            pat.Rectangle((0., 0.), width=0., height=self.rod_thickness,
                          angle=0., color='grey', alpha=1.),
            # Penholder
            pat.Circle((0., 0.), 0., color='lightblue', alpha=1.)
            ]
        self.bg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[:6], match_original=True))
        self.fg_coll = self.ax.add_collection(
            PatchCollection(self.shapes[6:], match_original=True))

        super().__init__(mechanism, ax, anim_plt)

    def redraw(self):
        r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2 = self.mecha.props
        C_G1 = np.array([r_T + r_G1, 0.])
        C_G2 = (r_T + r_G2) * np.array([math.cos(theta_12),
                                        math.sin(theta_12)])

        # Static properties

        # Turntable
        self.shapes[0].radius = r_T
        # Gears
        self.shapes[1].center = C_G1
        self.shapes[1].radius = r_G1
        self.shapes[2].center = C_G1
        self.shapes[2].radius = r_G1 * .1
        self.shapes[3].center = C_G2
        self.shapes[3].radius = r_G2
        self.shapes[4].center = C_G2
        self.shapes[4].radius = r_G2 * .1
        # Canvas
        self.shapes[5].radius = r_T * .95
        # Pivots
        self.shapes[6].radius = r_G1 * .1
        self.shapes[7].radius = r_G1 * .1
        # Linkage
        self.shapes[8].set_width(l1)
        self.shapes[9].set_width(l2)
        # Pen-holder
        self.shapes[10].radius = r_T * .05

        # Moving parts
        self._redraw_moving_parts()

        # Update patches
        self.bg_coll.set_paths(self.shapes[:6])
        self.fg_coll.set_paths(self.shapes[6:])
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
        # Pivots
        self.shapes[6].center = OP1
        self.shapes[7].center = OP2
        # Pen-holder
        self.shapes[10].center = OH
        # Linkage
        rectangle_offset = np.array([[0.], [-self.rod_thickness/2.]])
        _align_linkage_to_joints(OP1, OH, self.shapes[8], rectangle_offset)
        _align_linkage_to_joints(OP2, OH, self.shapes[9], rectangle_offset)

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
            self.fg_coll.set_paths(self.shapes[6:])
        return self.fg_coll, self.anim_plt


class Kicker(AniMecha):
    rod_thickness = .2

    def __init__(self, mechanism, ax):
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
        r1, r2, x2, y2, l1, l2, d = self.mecha.props
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
        self.shapes['foot'][0].set_width((l1+d)/5)
        self.shapes['end_effector'][0].radius = pivot_size
        # Moving parts
        self._redraw_moving_parts()

        self.bg_coll.set_paths(
            [patch for shape in self.shapes.values() for patch in shape])
#        self.fg_coll.set_paths(self.shapes[6:])
#        self.fg_coll.set_zorder(3)

        self.ax.set_xlim(OG1[0] - r1 - l1, OG2[0] + r2 + l2 + d)
        self.ax.set_ylim(1.2*(OG1[1] - r1), 1.1*OH[1])

        # Reset animation.
        self.reset_anim()

    def _redraw_moving_parts(self):
        r1, r2, x2, y2, l1, l2, d = self.mecha.props
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
