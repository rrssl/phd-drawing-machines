#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demo for sketch-based curve retrieval.

@author: Robin Roussel
"""
#import sys
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
#from PyQt5.QtCore import Qt
#from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QApplication, QSizePolicy,
#                             QSlider)


import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

import _context
from controlpane import make_slider
import mecha
TYPES = mecha.EllipticSpirograph, # mecha.SingleGearFixedFulcrumCDM
import curvedistances as cdist
from sketcher import Sketcher, remove_axes

DEBUG = True
if DEBUG:
    from curveplotlib import distshow




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
                distances.append(max(self.distance(crv, sketch),
                                     self.distance(sketch, crv)))
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

        self.sk_canvas = plt.subplot2grid(
            self.grid_size, (0, 0), rowspan=self.grid_size[0],
            colspan=self.grid_size[0])

        self.widgets = {}

        self.axlist = []
        self.draw_sketcher_panel()
        self.draw_bottom_pane()

    ### Low-level graphical elements (buttons, separators, etc.)

    def draw_section_title(self, grid_pos, label):
        width = (self.grid_size[1] - self.grid_size[0]) // 2
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='none')
        remove_axes(ax)
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
        remove_axes(ax)
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
        remove_axes(bt_ax)
        bt.label.set_fontsize(12)
        bt.label.set_weight('bold')
        bt.label.set_color('.2')
        return bt

    def draw_separator(self, grid_pos):
        width = (self.grid_size[1] - self.grid_size[0])
        ax = plt.subplot2grid(
            self.grid_size, grid_pos, rowspan=1, colspan=width, axisbg='none')
        remove_axes(ax)
        ax.set_navigate(False)
        ax.axhline(.1, 0.01, .99, color='.5', linestyle='dashed')
        return ax

    ### High-level graphical elements (tabs and panels)

    def draw_sketcher_panel(self):
        tab_width = (self.grid_size[1] - self.grid_size[0]) // 2
        row_id = 1

        self.axlist.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Construction lines")
            )
        row_id += 1

        self.widgets['sk_bnd'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//2), tab_width, 1,
            "Set boundaries")
        self.axlist.append(self.widgets['sk_bnd'].ax)
        row_id += 2

        slider_args = {'valmin': 0, 'valmax': 25, 'valinit': 1,
                       'label': "Symmetry\norder"}
        self.widgets['sk_sym'] = self.draw_slider(
            (row_id, self.grid_size[0]+tab_width*3//4), slider_args,
            tab_width*5//4)
        self.axlist.append(self.widgets['sk_sym'].ax)
        row_id += 1

        self.axlist.append(
            self.draw_separator((row_id, self.grid_size[0]))
            )
        row_id += 2

        self.axlist.append(
            self.draw_section_title((row_id, self.grid_size[0]+tab_width//2),
                                    "Sketch")
            )
        row_id += 1

        self.widgets['sk_undo'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width//4), tab_width*3//4, 2,
            "Undo\nlast stroke")
        self.axlist.append(self.widgets['sk_undo'].ax)
        self.widgets['sk_redo'] = self.draw_button(
            (row_id, self.grid_size[0]+tab_width*5//4), tab_width*3//4, 2,
            "Redo\nlast stroke")
        self.axlist.append(self.widgets['sk_redo'].ax)

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
#        row_id += 2
#        self.widgets['show'] = self.draw_button(
#            (row_id, self.grid_size[0]+width//4), width*3//4, 2,
#            "Show\nmechanism")
#        self.widgets['export'] = self.draw_button(
#            (row_id, self.grid_size[0]+width*5//4), width*3//4, 2,
#            "Export\nmechanism")

    ### Update view

    def redraw_axes(self, ax):
        ax.redraw_in_frame()
#        self.fig.canvas.blit(ax)
        self.fig.canvas.update()



class App:

    def __init__(self):
        self.view = View()
        self.finder = DrawingFinder()
        self.sketcher = Sketcher(self.view.sk_canvas, self.finder)
        self._init_ctrl()

    def run(self):
        plt.ioff()
        plt.show()

    def _init_ctrl(self):
        wdg = self.view.widgets
        wdg['sk_bnd'].on_clicked(self.sketcher.set_sketch_bounds)
        wdg['sk_undo'].on_clicked(self.sketcher.undo_stroke)
        wdg['sk_redo'].on_clicked(self.sketcher.redo_stroke)
        wdg['sk_sym'].on_changed(self.sketcher.set_symmetry)
        wdg['search'].on_clicked(self.search_mecha)

    def search_mecha(self, event):
        if not len(self.finder.strokes):
            print("There is no query sketch.")
        else:
            print("Search for the best matching mechanism.")
            self.finder.search_mecha(6)
            if DEBUG:
                sketch = np.array(self.finder.strokes).swapaxes(1, 2)
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
                for i, sr in enumerate(self.finder.search_res):
                    ax = fig.add_subplot(2, 3, i+1)
                    distshow(ax, sketch, sr['curve'])
                fig.tight_layout()
                fig.show()


def main():
    app = App()
    app.run()

if __name__ == "__main__":
    main()
