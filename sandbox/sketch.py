#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo for sketch-based curve retrieval.

@author: Robin Roussel
"""
from enum import Enum
import math

from matplotlib.patches import Circle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import context
from controlpane import make_slider
import mecha
TYPES = mecha.EllipticSpirograph, # mecha.SingleGearFixedFulcrumCDM
import curvedistances as cdist

DEBUG = True
if DEBUG:
    from curveplotlib import distshow


Actions = Enum('Actions', 'none sketch set_min_bound set_max_bound')

#import warnings
#warnings.filterwarnings("error")


# TODO FIXME distance field is sometimes broken

class DrawingFinder:

    def __init__(self):
        ## Sketcher
        self.crv_bnds = [None, None]
        self.sym_order = 1
        self.strokes = [] # List of N*2 lists of points
        self.undone_strokes = []
        # Mechanism retrieval
        self.pts_per_dim = 4
        self.samples = self.get_global_sampling()
        self.distance = cdist.DistanceField().get_dist
        self.search_res = [] # list of  {'type', 'props', 'curve'} dicts
        # Mechanism
        self.mecha = None
        self.nb_crv_pts = 2**6
        self.crv = None

    def get_global_sampling(self):
        """Sample feasible parameters across all mechanisms."""
        samples = {}
        for t in TYPES:
            size = [self.pts_per_dim]*t.ConstraintSolver.nb_cprops
            samples[t] = np.array(list(
                t.ConstraintSolver.sample_feasible_domain(grid_resol=size)))
        return samples

    def search_mecha(self, nb):
        """Retrieve the closest drawings and their associated mechanisms."""
        if not len(self.strokes):
            return
        sketch = np.array(self.strokes).swapaxes(1, 2)

        self.search_res.clear()
        ranges = [0]
        # Convert types and samples to lists to keep the order.
        types = list(self.samples.keys())
        samples = list(self.samples.values())
        # Pre-filter the samples.
        if self.sym_order > 1:
            samples = [s[s[:, 0] == self.sym_order] for s in samples]
        # Compute distances.
        distances = []
        for type_, type_samples in zip(types, samples):
            ranges.append(ranges[-1] + type_samples.shape[1])
            mecha = type_(*type_samples[0])
            for sample in type_samples:
                mecha.reset(*sample)
                crv = mecha.get_curve(self.nb_crv_pts)
                distances.append(max(self.distance(crv, sketch), self.distance(sketch, crv)))
        distances = np.array(distances)
        best = distances.argpartition(nb)[:nb]
        # Sort the best matching curves.
        best = best[distances[best].argsort()]
        print(distances[best])
        # Build the result.
        ranges = np.array(ranges)
        for id_ in best:
            # Find index in ranges with a small trick: argmax gives id of the
            # first max value, here True.
            # TODO FIXME
#            typeid = np.argmax(ranges > id_) - 1
#            print(id_, id_-ranges[typeid])
#            type_ = types[typeid]
#            mecha = type_(*samples[typeid][id_-ranges[typeid]])
            type_ = types[0]
            mecha = type_(*samples[0][id_])
            self.search_res.append({
                'type': type_,
                'props': mecha.props.copy(),
                'curve': mecha.get_curve(self.nb_crv_pts)
                })


class View:

    def __init__(self, fig_size=(12,8)):
        self.fig = plt.figure(figsize=fig_size, facecolor='.2')
        self.fig.canvas.set_window_title('Sketcher')
        plt.subplots_adjust(left=0., right=1., bottom=0., top=1.,
                            wspace=0., hspace=0.)
        self.grid_size = (3*fig_size[1], 3*fig_size[0])
        self.colors = {
            'light': '.9',
            'bg': '.2'
            }

        self.sk_canvas = self.draw_sketch_canvas()

        self.widgets = {}

        self.sk_layer = []
        self.draw_sketcher_panel()

        self.max_nb_props = 5
        self.nb_props = 0
        self.max_nb_cstr_per_poi = 3

        self.draw_bottom_pane()

        self.undone_plots = []
        self.borders = [None, None]
        self.sym_lines = []

    @staticmethod
    def remove_axes(ax):
        """Remove the actual 'axes' from a matplotlib Axes object."""
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for s in ax.spines.values():
            s.set_color('none')

    ### Low-level graphical elements (buttons, separators, etc.)

    def draw_sketch_canvas(self):
        canvas = plt.subplot2grid(
            self.grid_size, (0, 0), rowspan=self.grid_size[0],
            colspan=self.grid_size[0])
        self.remove_axes(canvas)
        canvas.set_xlim([-1, 1])
        canvas.set_ylim([-1, 1])
        canvas.set_aspect('equal', 'datalim')
        return canvas

    def draw_section_title(self, grid_pos, label):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='none')
        self.remove_axes(ax)
        ax.set_navigate(False)
        title = ax.text(0.5, 0.5, label,
                        verticalalignment='bottom',
                        horizontalalignment='center',
                        transform=ax.transAxes)
        title.set_fontsize(15)
        title.set_weight('light')
        title.set_color('.9')
        return ax

    def draw_slider(self, grid_pos, slider_args, width):
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='.9')
        self.remove_axes(ax)
        slider_args['color'] = 'lightgreen'
        slider = make_slider(ax, **slider_args)
        slider.label.set_weight('bold')
        slider.label.set_color('.9')
        slider.label.set_x(-0.25)
        slider.label.set_horizontalalignment('center')
        slider.valtext.set_weight('bold')
        slider.valtext.set_color('.9')
        return slider

    def draw_button(self, grid_pos, width, height, label):
        bt_ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=height, colspan=width)
        bt = Button(bt_ax, label, color='.9', hovercolor='lightgreen')
        bt.label.set_fontsize(12)
        bt.label.set_weight('bold')
        bt.label.set_color('.2')
        return bt

    def draw_separator(self, grid_pos):
        width = (self.grid_size[1] - self.grid_size[0])
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='none')
        self.remove_axes(ax)
        ax.set_navigate(False)
        ax.axhline(.1, 0.01, .99, color='.5', linestyle='dashed')
        return ax

    def draw_inner_bound(self, radius):
        self.borders[0] = Circle((0, 0), radius, fc='.9', ec='.7', ls='dashed')
        self.sk_canvas.add_patch(self.borders[0])

    def draw_outer_bound(self, radius):
        self.sk_canvas.set_axis_bgcolor('.9')
        self.borders[1] = Circle(
            (0, 0), radius, fc='1.', ec='.7', ls='dashed', zorder=-1)
        self.sk_canvas.add_patch(self.borders[1])

    def draw_sym_lines(self, order):
        if order > 1:
            radius = 2 * math.sqrt(
                max(np.abs(self.sk_canvas.get_xlim()))**2 +
                max(np.abs(self.sk_canvas.get_ylim()))**2
                )
            for i in range(order):
                angle = 2 * math.pi * i / order
                self.sym_lines.append(
                    self.sk_canvas.plot(
                        (0., radius*math.cos(angle)),
                        (0., radius*math.sin(angle)),
                        c='.7', ls='dashed')[0]
                    )
            # Move these plots at the beginning so that they don't disturb the
            # undo/redo.
            self.sk_canvas.lines = (self.sk_canvas.lines[-order:] +
                                 self.sk_canvas.lines[:-order])

    ### High-level graphical elements (tabs and panels)

    def draw_sketcher_panel(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2
        row_id = 1

        self.sk_layer.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Construction lines")
            )
        row_id += 1

        self.widgets['sk_bnd'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//2), tab_width, 1,
            "Set boundaries")
        self.sk_layer.append(self.widgets['sk_bnd'].ax)
        row_id += 2

        slider_args = {'valmin': 0, 'valmax': 25, 'valinit': 1,
                       'label': "Symmetry\norder"}
        self.widgets['sk_sym'] = self.draw_slider(
            (row_id, self.grid_size[0]+tab_width*3//4), slider_args,
            tab_width*5//4)
        self.sk_layer.append(self.widgets['sk_sym'].ax)
        row_id += 1

        self.sk_layer.append(
            self.draw_separator((row_id, self.grid_size[0]))
            )
        row_id += 2

        self.sk_layer.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Sketch")
            )
        row_id += 1

        self.widgets['sk_undo'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//4), tab_width*3//4, 2,
            "Undo\nlast stroke")
        self.sk_layer.append(self.widgets['sk_undo'].ax)
        self.widgets['sk_redo'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width*5//4), tab_width*3//4, 2,
            "Redo\nlast stroke")
        self.sk_layer.append(self.widgets['sk_redo'].ax)

    def draw_bottom_pane(self):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        row_id = self.grid_size[0] * 2 // 3
        self.draw_separator((row_id, self.grid_size[0]))
        row_id += 2
        self.draw_section_title((row_id, self.grid_size[0]+width//2),
                                "Drawing machine")
        row_id += 1
        self.widgets['search'] = self.draw_button(
            (row_id, self.grid_size[0]+width//2), width, 1, "Search database")
        row_id += 2
        self.widgets['show'] = self.draw_button(
            (row_id, self.grid_size[0]+width//4), width*3//4, 2,
            "Show\nmechanism")
        self.widgets['export'] = self.draw_button(
            (row_id, self.grid_size[0]+width*5//4), width*3//4, 2,
            "Export\nmechanism")

    ### Update view

    def redraw_axes(self, ax):
        ax.redraw_in_frame()
#        self.fig.canvas.blit(ax)
        self.fig.canvas.update()

    def remove_borders(self):
        self.sk_canvas.set_axis_bgcolor('1.')
        for patch in self.borders:
            patch.remove()
        self.borders = [None, None]

    def remove_sym_lines(self):
        for line in self.sym_lines:
            line.remove()
        self.sym_lines = []


class SketchApp:

    def __init__(self):
        self.model = DrawingFinder()
        self.view = View()
        self.connect_widgets()
        self.connect_canvas()

        # Base state
        self.action = Actions.none

    def run(self):
        plt.ioff()
        plt.show()

    def connect_widgets(self):
        self.view.widgets['sk_bnd'].on_clicked(self.set_sketch_bounds)
        self.view.widgets['sk_undo'].on_clicked(self.undo_stroke)
        self.view.widgets['sk_redo'].on_clicked(self.redo_stroke)
        self.view.widgets['search'].on_clicked(self.search_mecha)
        self.view.widgets['sk_sym'].on_changed(self.set_symmetry)

    def connect_canvas(self):
        self.view.fig.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.view.fig.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.view.fig.canvas.mpl_connect(
            'motion_notify_event', self.on_move)

    def check_tb_inactive(self):
        """Check if the matplotlib toolbar plugin is inactive."""
        return self.view.fig.canvas.manager.toolbar._active is None

    def set_sketch_bounds(self, event):
        print("Set the bounds of the drawing.")
        self.action = Actions.set_min_bound
        # Flush previous bounds.
        if self.model.crv_bnds[0] is not None:
            self.model.crv_bnds = [None, None]
            self.view.remove_borders()

    def set_symmetry(self, value):
        print("Set the symmetry order of the drawing: {}".format(value))
        self.model.sym_order = value
        self.view.remove_sym_lines()
        self.view.draw_sym_lines(value)
        self.view.redraw_axes(self.view.sk_canvas)

    def undo_stroke(self, event):
        if not len(self.model.strokes):
            print("There is no stroke to undo.")
        else:
            print("Undo the last sketch stroke.")
            self.model.undone_strokes.append(self.model.strokes.pop())
            self.view.undone_plots.append(self.view.sk_canvas.lines.pop())
            self.view.redraw_axes(self.view.sk_canvas)

    def redo_stroke(self, event):
        if not len(self.model.undone_strokes):
            print("There is no undone stroke.")
        else:
            print("Redo the last undone sketch stroke.")
            self.model.strokes.append(self.model.undone_strokes.pop())
            self.view.sk_canvas.lines.append(self.view.undone_plots.pop())
            self.view.redraw_axes(self.view.sk_canvas)

    def search_mecha(self, event):
        if not len(self.model.strokes):
            print("There is no query sketch.")
        else:
            print("Search for the best matching mechanism.")
            self.model.search_mecha(6)
            if DEBUG:
                sketch = np.array(self.model.strokes).swapaxes(1, 2)
                fig = plt.figure(figsize=(6,12))
                ax1 = fig.add_subplot(211)
                ax1.set_aspect('equal')
                for stroke in sketch:
                    ax1.plot(*stroke, c='b', lw=2)
                ax2 = fig.add_subplot(212)
                distshow(ax2, sketch)
                fig.tight_layout()
                fig.show()

                fig = plt.figure(figsize=(12,6))
                for i, sr in enumerate(self.model.search_res):
                    ax = fig.add_subplot(2, 3, i+1)
                    distshow(ax, sketch, sr['curve'])
                fig.tight_layout()
                fig.show()

    def add_sketch_point(self, xy, start=False):
        nb_sym = self.model.sym_order
        if start:
            for _ in range(nb_sym):
                # Careful to put this append in the loop, rather than appending
                # N * [], since the latter appends N copies of the same list.
                self.model.strokes.append([])
                self.view.sk_canvas.plot([], [], lw=2, c='b', alpha=.8)
            # Flush history
            self.model.undone_strokes.clear()
            self.view.undone_plots.clear()
        else:
            for i in range(nb_sym):
                angle = 2 * math.pi * i / nb_sym
                cos_, sin_ = math.cos(angle), math.sin(angle)
                xy_ = [xy[0]*cos_ + xy[1]*sin_, -xy[0]*sin_ + xy[1]*cos_]
                self.model.strokes[-nb_sym+i].append(xy_)
                self.view.sk_canvas.lines[-nb_sym+i].set_data(
                    *np.asarray(self.model.strokes[-nb_sym+i]).T)
        self.view.redraw_axes(self.view.sk_canvas)

    def update_sketch_bound(self, radius):
        if self.action is Actions.set_min_bound:
            if self.view.borders[0] is None:
                self.view.draw_inner_bound(radius)
            else:
                self.view.borders[0].radius = radius
        if self.action is Actions.set_max_bound:
            if self.view.borders[1] is None:
                self.view.draw_outer_bound(radius)
            else:
                self.view.borders[1].radius = radius
        self.view.redraw_axes(self.view.sk_canvas)

    def on_move(self, event):
        if self.check_tb_inactive():
            if event.inaxes == self.view.sk_canvas:
                if (self.action is Actions.set_min_bound
                    or self.action is Actions.set_max_bound):
                    radius = math.sqrt(event.xdata**2 + event.ydata**2)
                    self.update_sketch_bound(radius)
                elif self.action is Actions.sketch:
                    self.add_sketch_point([event.xdata, event.ydata])

    def on_press(self, event):
        if self.check_tb_inactive():
            if event.inaxes == self.view.sk_canvas:
                event.canvas.grab_mouse(self.view.sk_canvas)

                if (self.action is Actions.set_min_bound
                    or self.action is Actions.set_max_bound):
                    pass
                else:
                    # Default action is sketching.
                    self.action = Actions.sketch
                    self.add_sketch_point([event.xdata, event.ydata],
                                          start=True)

    def on_release(self, event):
        if self.check_tb_inactive():
            if event.inaxes == self.view.sk_canvas:
                print("Canvas was clicked.")
                event.canvas.release_mouse(self.view.sk_canvas)

                if self.action is Actions.set_min_bound:
                    self.model.crv_bnds[0] = self.view.borders[0].radius
                    print("Inner bound set: {}".format(
                          self.model.crv_bnds[0]))
                    self.action = Actions.set_max_bound
                elif self.action is Actions.set_max_bound:
                    self.model.crv_bnds[1] = self.view.borders[1].radius
                    print("Outer bound set: {}".format(
                          self.model.crv_bnds[1]))
                    self.action = Actions.none
                elif self.action is Actions.sketch:
                    self.action = Actions.none


def main():
    app = SketchApp()
    app.run()

if __name__ == "__main__":
    main()
