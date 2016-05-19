#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application for the retrieval and editing of Spirograph curves.

@author: Robin Roussel
"""
from fractions import Fraction

import matplotlib.patches as mpat
import matplotlib.pyplot as plt
import numpy as np

import artist as art
import curvedistances as cdist
import curvegen as cg
import curvematching as cmat


class Spirou():
    """Main application."""

    def __init__(self, matcher=None, optimizer=None, init_curve=None):
        self.matcher = matcher
        self.optimizer = optimizer

        self.nb_retrieved = 6
        self.samples = cg.get_param_combinations()

        self.fig = plt.figure(figsize=(16, 9))

        self.init_draw()

    def init_draw(self):
        """Draw the initial frames."""
        plot_grid_size = (3, self.nb_retrieved)

        # Create the painter.
        self.paint_frame = plt.subplot2grid(
            plot_grid_size, (0, 0), rowspan=2, colspan=2, title="Draw here!")
        self.paint_frame.painter = art.Painter(self.paint_frame,
                                               self.update_data)

        # Create the sculptor.
        self.edit_frame = plt.subplot2grid(
            plot_grid_size, (0, 2), rowspan=2, colspan=2,
            title="Edit here!")
        self.edit_frame.sculptor = art.Sculptor(self.edit_frame,
                                                self.update_data)

        # Create the pickers.
        self.retrieved_frames = []
        for i in range(self.nb_retrieved):
            frame = plt.subplot2grid(plot_grid_size, (2, i), title=i+1)
            self.retrieved_frames.append(frame)
            frame.picker = art.ButtonDisplay(frame, self.update_data)

        # Create the final display.
        self.show_frame = plt.subplot2grid(
            plot_grid_size, (0, 4), rowspan=2, colspan=2, title="Result")
        self.show_frame.display = SpiroDisplay(self.show_frame,
                                               self.update_data)

        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95,
                            hspace=0.1)

    def update_data(self, from_ax):
        """Update data in frames in response to event."""
        if from_ax == self.paint_frame:
            self.transmit_painter_to_pickers()
        if from_ax in self.retrieved_frames:
            self.transmit_picker_to_sculptor(from_ax.picker)
            self.transmit_picker_to_display(from_ax.picker)
            self.transmit_picker_to_pickers(from_ax.picker)
        if from_ax == self.edit_frame:
            self.optimize_sculptor_curve()
            self.transmit_sculptor_to_pickers()
            self.transmit_sculptor_to_display()

    def transmit_painter_to_pickers(self):
        """Update pickers from hand drawing."""
        if self.paint_frame.lines:
            # Get hand-drawn curve.
            curve = np.array(self.paint_frame.lines[-1].get_data(orig=False))
            if curve.size <= 2:
                return
            # Retrieve closest matching curves.
            retrieved_args = (
                self.matcher(curve, self.samples)[:self.nb_retrieved, :])
            # Update pickers' plots.
            self.reset_pickers(retrieved_args)

    def transmit_picker_to_sculptor(self, picker):
        """Update sculptor data from selected picker."""
        scl = self.edit_frame.sculptor
        if picker.params is not None:
            # Perform deep copy of the args since they are actively modified in
            # the sculptor.
            args = picker.params.copy()
            data = picker.curve.get_data()
            sym_order = Fraction.from_float(
                args[0] / args[1]).limit_denominator(1000).numerator

            scl.reset(args, data, sym_order)
            scl.redraw()

    def transmit_picker_to_display(self, picker):
        """Update display data from selected picker."""
        if picker.params is not None:
            dsp = self.show_frame.display
            dsp.reset(picker.params, picker.curve.get_data())
            dsp.redraw()

# TODO: put in a ButtonDisplayRow class?
    def transmit_picker_to_pickers(self, picker):
        """Update picker selection state."""
        for retf in self.retrieved_frames:
            if retf.picker.hold and retf.picker != picker:
                retf.picker.hold = False
                retf.set_axis_bgcolor(art.ButtonDisplay.bg_colors[0])
                retf.picker.redraw()

    def optimize_sculptor_curve(self):
        """Optimize the sculptor after curve editing."""
        scl = self.edit_frame.sculptor
        # Optimization
        if scl.to_opt and self.optimizer is not None:
            opt_d = self.optimizer.optimize(target_curve=scl.data,
                                            init_guess=scl.params).x
            args = scl.params.copy()
            args[2:] = opt_d
            curve_pts = cg.get_curve(args)

            scl.reset(args, curve_pts, scl.sym_order)
            scl.redraw()

    def transmit_sculptor_to_pickers(self):
        """Update pickers from curve editing."""
        scl = self.edit_frame.sculptor
        args = scl.params
        if args is None:
            return
        # Get closest curves.
        dists = ((self.samples - args) ** 2).sum(axis=1)
        ids = np.argsort(dists)
        closest_args = self.samples[ids[:self.nb_retrieved]]
        # Update pickers' plots.
        self.reset_pickers(closest_args)

    def transmit_sculptor_to_display(self):
        """Update display data from curve editing."""
        scl = self.edit_frame.sculptor
        if scl.params is not None:
            dsp = self.show_frame.display
            dsp.reset(scl.params, scl.curve.get_data())
            dsp.redraw()

# TODO: put in a ButtonDisplayRow class?
    def reset_pickers(self, args):
        """Update display of the pickers."""
        # Update pickers' plots.
        for i, frame in enumerate(self.retrieved_frames):
            curve_pts = cg.get_curve(args[i, :])
            frame.picker.reset(args[i, :], curve_pts)
            frame.picker.redraw()


class SpiroDisplay(art.AnimDisplay):
    """Specialization of AnimDisplay to animate Spirograph machines."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shapes = []
        self.text = None

    def reset(self, params, data):
        """Reset the data."""
        super().reset(params, data)

        R, r, d = self.params
        if self.shapes:
            self.shapes[0].radius = R
            self.shapes[1].center = (R - r, 0.)
            self.shapes[1].radius = r
            self.shapes[2].center = (R - r + d, 0.)
            self.shapes[2].radius = r / 20

            self.text.set_text(self.params)
        else:
            self.init_draw()

        # Override the axes rescaling, taking the shapes into account.
        dim = max(R - r + d, R) * 1.1
        self.ax.set_xlim(-dim, dim)
        self.ax.set_ylim(-dim, dim)


    def init_draw(self):
        """Draw the spirograph corresponding to its arguments."""
        R, r, d = self.params
        ax = self.ax

        out_gear = mpat.Circle((0., 0.), R, color='r', fill=False)
        int_gear = mpat.Circle((R - r, 0.), r, color='g', fill=False)
        hole = mpat.Circle((R - r + d, 0.), r / 20, color='g', fill=False)

        self.shapes.append(ax.add_artist(out_gear))
        self.shapes.append(ax.add_artist(int_gear))
        self.shapes.append(ax.add_artist(hole))

        self.text = ax.text(0.95, 0.01, self.params,
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            transform=ax.transAxes)

    def animate(self):
        """Create the current frame."""
        idx = self.time % self.data.shape[1]
        theta = idx * 2 * np.pi * self.params[1] / self.data.shape[1]
        r = self.params[0] - self.params[1]
        self.shapes[1].center = [r * np.cos(theta), r * np.sin(theta)]

        self.shapes[2].center = self.data[:, idx]
        super().animate()


def main():
    """Entry point."""
    plt.ioff()

    distance = cdist.DistanceField()
    matcher = cmat.CurveMatcher(distance.get_dist)
    optimizer = cmat.CurveOptimizer(distance.get_dist)
    Spirou(matcher, optimizer)
    plt.show()


if __name__ == "__main__":
    main()
