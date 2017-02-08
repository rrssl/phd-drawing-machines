#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sketcher class.

@author: Robin Roussel
"""
from enum import Enum
import math
import numpy as np
from matplotlib.patches import Circle #, Rectangle


# TODO: replace 'Actions' with pickable patches.
Actions = Enum('Actions', 'none sketch set_min_bound set_max_bound')


def remove_axes(ax):
    """Remove the actual 'axes' from a matplotlib Axes object."""
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    for s in ax.spines.values():
        s.set_color('none')


class Sketcher:
    """Turn an Axes into a canvas.

    Requirements on 'data' fields:
        -- crv_bnds (pair of floats or pair of None)
        -- sym_order (int)
        -- strokes (sequence of Nx2 arrays)
        -- undone_strokes (same as 'strokes')
    """

    def __init__(self, ax, data):
        self.data = data
        # State and callbacks
        self.action = Actions.none
        self._init_ctrl(ax)
        # Graphical objects
        self.undone_plots = []
        self.bounds = [None, None]
        self.sym_lines = []
        self.canvas = ax
        self._init_view(ax)

    ### Controller

    def _init_ctrl(self, ax):
        connect = ax.figure.canvas.mpl_connect
        connect('button_press_event', self.on_press)
        connect('button_release_event', self.on_release)
        connect('motion_notify_event', self.on_move)
        connect('scroll_event', self.zoom)

    def set_sketch_bounds(self, event):
        print("Set the bounds of the drawing.")
        self.action = Actions.set_min_bound
        # Flush previous bounds.
        if self.data.crv_bnds[0] is not None:
            self.data.crv_bnds = [None, None]
            self.remove_bounds()

    def set_symmetry(self, value):
        print("Set the symmetry order of the drawing: {}".format(value))
        self.data.sym_order = value
        self.remove_sym_lines()
        self.draw_sym_lines(value)
        self.redraw_axes()

    def undo_stroke(self, event):
        if not len(self.data.strokes):
            print("There is no stroke to undo.")
        else:
            print("Undo the last sketch stroke.")
            self.data.undone_strokes.append(self.data.strokes.pop())
            self.undone_plots.append(self.canvas.lines.pop())
            self.redraw_axes()

    def redo_stroke(self, event):
        if not len(self.data.undone_strokes):
            print("There is no undone stroke.")
        else:
            print("Redo the last undone sketch stroke.")
            self.data.strokes.append(self.data.undone_strokes.pop())
            self.canvas.lines.append(self.undone_plots.pop())
            self.redraw_axes()

    def add_sketch_point(self, xy, start=False):
        nb_sym = self.data.sym_order
        if start:
            for _ in range(nb_sym):
                # Careful to put this append in the loop, rather than appending
                # N * [], since the latter appends N copies of the same list.
                self.data.strokes.append([])
                self.canvas.plot([], [], lw=2, c='b', alpha=.8)
            # Flush history
            self.data.undone_strokes.clear()
            self.undone_plots.clear()
        else:
            for i in range(nb_sym):
                angle = 2 * math.pi * i / nb_sym
                cos_, sin_ = math.cos(angle), math.sin(angle)
                xy_ = [xy[0]*cos_ + xy[1]*sin_, -xy[0]*sin_ + xy[1]*cos_]
                self.data.strokes[-nb_sym+i].append(xy_)
                self.canvas.lines[-nb_sym+i].set_data(
                    *np.asarray(self.data.strokes[-nb_sym+i]).T)
        self.redraw_axes()

    def update_sketch_bound(self, radius):
        if self.action is Actions.set_min_bound:
            if self.bounds[0] is None:
                self.draw_inner_bound(radius)
            else:
                self.bounds[0].radius = radius
        if self.action is Actions.set_max_bound:
            if self.bounds[1] is None:
                self.draw_outer_bound(radius)
            else:
                self.bounds[1].radius = radius
        self.redraw_axes()

    def on_move(self, event):
        if event.inaxes != self.canvas:
            return
        if event.button == 3:
            self.shift(event)
        elif self.action in (Actions.set_min_bound, Actions.set_max_bound):
            radius = math.sqrt(event.xdata**2 + event.ydata**2)
            self.update_sketch_bound(radius)
        elif self.action is Actions.sketch:
            self.add_sketch_point([event.xdata, event.ydata])

    def on_press(self, event):
        if event.inaxes != self.canvas:
            return
        event.canvas.grab_mouse(self.canvas)
        if event.button in (2, 3):
            self.init_xy = event.xdata, event.ydata
        elif self.action not in (Actions.set_min_bound, Actions.set_max_bound):
            # Default left button action is sketching.
            self.action = Actions.sketch
            self.add_sketch_point([event.xdata, event.ydata], start=True)

    def on_release(self, event):
        if event.inaxes != self.canvas:
            return
        event.canvas.release_mouse(self.canvas)
        if event.button in (2, 3):
            return
        if self.action is Actions.set_min_bound:
            self.data.crv_bnds[0] = self.bounds[0].radius
            print("Inner bound set: {}".format(
                  self.data.crv_bnds[0]))
            self.action = Actions.set_max_bound
        elif self.action is Actions.set_max_bound:
            self.data.crv_bnds[1] = self.bounds[1].radius
            print("Outer bound set: {}".format(
                  self.data.crv_bnds[1]))
            self.action = Actions.none
        elif self.action is Actions.sketch:
            self.action = Actions.none

    def shift(self, event):
        ax = self.canvas
        # Get the points and limits.
#        xlim = np.array(ax.get_xlim())
#        ylim = np.array(ax.get_ylim())
        # Set new limits.
        ax.set_xlim(ax.get_xlim() - event.xdata + self.init_xy[0])
        ax.set_ylim(ax.get_ylim() - event.ydata + self.init_xy[1])
        self.redraw_axes()

    def zoom(self, event, scale_factor=1.5):
        if event.inaxes != self.canvas:
            return
        ax = self.canvas
        # Get the range.
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        cur_xrange = ax.get_xlim() - xdata
        cur_yrange = ax.get_ylim() - ydata
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1. / scale_factor
        elif event.button == 'down':
            # deal with zoom out
            pass
        # set new limits
        ax.set_xlim(xdata + cur_xrange*scale_factor)
        ax.set_ylim(ydata + cur_yrange*scale_factor)
        self.redraw_axes()

    ### View

    def _init_view(self, ax):
        remove_axes(ax)
        ax.set_xlim([-1.1, 1.1])
        ax.set_ylim([-1.1, 1.1])
        ax.set_aspect('equal', 'datalim')
#        ax.add_patch(Rectangle((-1.,-1.), 2., 2., lw=0, fc='1.'))
#        ax.set_axis_bgcolor('.9')

    def draw_inner_bound(self, radius):
        self.bounds[0] = Circle((0, 0), radius, fc='.9', ec='.7', ls='dashed')
        self.canvas.add_patch(self.bounds[0])

    def draw_outer_bound(self, radius):
        self.canvas.set_axis_bgcolor('.9')
        self.bounds[1] = Circle(
            (0, 0), radius, fc='1.', ec='.7', ls='dashed', zorder=-1)
        self.canvas.add_patch(self.bounds[1])

    def draw_sym_lines(self, order):
        if order > 1:
            radius = 10 * math.sqrt(
                max(np.abs(self.canvas.get_xlim()))**2 +
                max(np.abs(self.canvas.get_ylim()))**2
                )
            for i in range(order):
                angle = 2 * math.pi * i / order
                self.sym_lines.append(
                    self.canvas.plot(
                        (0., radius*math.cos(angle)),
                        (0., radius*math.sin(angle)),
                        c='.7', ls='dashed')[0]
                    )
            # Move these plots at the beginning so that they don't disturb the
            # undo/redo.
            self.canvas.lines = (self.canvas.lines[-order:] +
                                 self.canvas.lines[:-order])

    def redraw_axes(self):
        self.canvas.redraw_in_frame()
#        self.fig.canvas.blit(ax)
        self.canvas.figure.canvas.update()

    def remove_bounds(self):
        self.canvas.set_axis_bgcolor('1.')
        for patch in self.bounds:
            patch.remove()
        self.bounds = [None, None]

    def remove_sym_lines(self):
        for line in self.sym_lines:
            line.remove()
        self.sym_lines = []