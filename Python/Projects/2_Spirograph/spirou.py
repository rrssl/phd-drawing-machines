#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application for the retrieval and editing of Spirograph curves.

@author: Robin Roussel
"""
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
        self.paint_frame.painter = art.Painter(self.paint_frame, self)

        # Create the sculptor.
        self.edit_frame = plt.subplot2grid(
            plot_grid_size, (0, 2), rowspan=2, colspan=2,
            title="Edit here!")
        self.edit_frame.sculptor = art.Sculptor(self.optimizer, self.edit_frame,
                                                self)

        # Create the pickers.
        self.retrieved_frames = []
        for i in range(self.nb_retrieved):
            frame = plt.subplot2grid(plot_grid_size, (2, i), title=i+1)
            self.retrieved_frames.append(frame)
            frame.picker = art.ButtonDisplay(frame, self)

        # Create the final display.
        self.show_frame = plt.subplot2grid(
            plot_grid_size, (0, 4), rowspan=2, colspan=2, title="Result")
        self.show_frame.display = art.AnimDisplay(self.show_frame, self)

        plt.subplots_adjust(left=0.03, right=0.97, bottom=0.05, top=0.95, 
                            hspace=0.1)

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
            self.update_pickers_plots(retrieved_args, abs(curve).max())

    def transmit_picker_to_sculptor(self, picker):
        """Update sculptor data from selected picker."""
        scl = self.edit_frame.sculptor
        if picker.args is not None:
            scl.data = np.array(picker.curve.get_data())
            scl.curve.set_data(scl.data)
            # Perform deep copy of the args since they are actively modified in
            # the sculptor.
            scl.args = picker.args.copy()
            # Adapt the plot limits (keeping the frame square).
            dim = scl.data.max() * 1.5
            self.edit_frame.set_xlim(-dim, dim)
            self.edit_frame.set_ylim(-dim, dim)

            scl.redraw()

    def transmit_picker_to_display(self, picker):
        """Update display data from selected picker."""        
        if picker.args is not None:
            self.update_display(picker.args, picker.curve.get_data())

    def transmit_picker_to_pickers(self, picker):
        """Update picker selection state."""
        for retf in self.retrieved_frames:
            if retf.picker.hold and retf.picker != picker:
                retf.picker.hold = False
                retf.set_axis_bgcolor(art.ButtonDisplay.bg_colors[0])
                retf.picker.redraw()
                
    def transmit_sculptor_to_pickers(self):
        """Update pickers from curve editing."""
        scl = self.edit_frame.sculptor
        args = scl.args
        if args is None:
            return
        # Get closest curves.
        dists = ((self.samples - args) ** 2).sum(axis=1)
        ids = np.argsort(dists)
        closest_args = self.samples[ids[:self.nb_retrieved]]
        # Update pickers' plots.
        self.update_pickers_plots(closest_args, abs(scl.data).max())

    def transmit_sculptor_to_display(self):
        """Update display data from curve editing."""
        scl = self.edit_frame.sculptor
        if scl.args is not None:
            self.update_display(scl.args, scl.curve.get_data())

    def update_data(self, event):
        """Update data in frames in response to event."""
        ax = event.inaxes
        if ax == self.paint_frame:
            self.transmit_painter_to_pickers()
        if ax in self.retrieved_frames:
            self.transmit_picker_to_sculptor(ax.picker)
            self.transmit_picker_to_display(ax.picker)
            self.transmit_picker_to_pickers(ax.picker)
        if ax == self.edit_frame:
            self.transmit_sculptor_to_pickers()
            self.transmit_sculptor_to_display()

    def update_pickers_plots(self, args, scale):
        """Update display of the pickers."""
        # Update pickers' plots.
        for i, frame in enumerate(self.retrieved_frames):
            curve_pts = cg.get_curve(args[i, :])            
            frame.picker.curve.set_data(curve_pts[0], curve_pts[1])
            # Adapt the plot limits (keeping the frame square).
            dim = curve_pts.max() * 1.1
            frame.set_xlim(-dim, dim)
            frame.set_ylim(-dim, dim)

            frame.picker.args = args[i, :]
            # Unselect an eventual previously selected picker.
            if frame.picker.hold:
                frame.picker.hold = False
                frame.set_axis_bgcolor(art.ButtonDisplay.bg_colors[0])

            frame.picker.redraw()
            
    def update_display(self, args, data):
        """Update data and gears in the display pane."""
        dsp = self.show_frame.display
        dsp.args = args
        dsp.erase_spiro()
        dsp.draw_spiro()

        dsp.data = np.asarray(data)[:, :-1]
        
        dsp.redraw()


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
