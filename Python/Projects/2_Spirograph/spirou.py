#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application for the retrieval and editing of Spirograph curves.

@author: Robin Roussel
"""

import matplotlib.pyplot as plt
import numpy as np

import curvedistances as cdist
import curvegen as cg
import curvematching as cmat
#import curveplotlib as cplt

class Spirou():
    """Main application."""

    def __init__(self, matcher=None, optimizer=None, init_curve=None):
        self.matcher = matcher
        self.optimizer = optimizer

        self.nb_retrieved = 6
        self.samples = cg.get_param_combinations()

        self.fig = plt.figure(figsize=(16, 9))

        self.init_draw()
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

    def init_draw(self):
        """Draw the initial frames."""
        plot_grid_size = (3, self.nb_retrieved)

        # Create the painter.
        self.paint_frame = plt.subplot2grid(
            plot_grid_size, (0, 0), rowspan=2, colspan=2, title="Draw here!")
        self.paint_frame.painter = Painter(self.paint_frame)

        # Create the sculptor.
        self.edit_frame = plt.subplot2grid(
            plot_grid_size, (0, 2), rowspan=2, colspan=2,
            title="Edit here!")
        self.edit_frame.sculptor = Sculptor(self.edit_frame, self.optimizer)

        # Create the pickers.
        self.retrieved_frames = []
        for i in range(self.nb_retrieved):
            frame = plt.subplot2grid(plot_grid_size, (2, i), title=i+1)
            self.retrieved_frames.append(frame)
            frame.picker = Display(frame)

        # Create the final display.
        self.show_frame = plt.subplot2grid(
            plot_grid_size, (0, 4), rowspan=2, colspan=2, title="Result")
        self.show_frame.display = Display(self.show_frame)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, hspace=0.5)

    def on_release(self, event):
        """Manage global release events (data transmission between frames)."""
        if self.fig.canvas.manager.toolbar._active is None:
            scl = self.edit_frame.sculptor
            pnt = self.paint_frame.painter
            dsp = self.show_frame.display

            if event.inaxes == self.paint_frame and pnt.ax.lines:
                curve = np.array(pnt.ax.lines[-1].get_data(orig=False))
                if curve.size <= 2:
                    return
                retrieved_args = (
                    self.matcher(curve, self.samples)[:self.nb_retrieved, :])
                for i, frame in enumerate(self.retrieved_frames):
                    curve_pts = cg.get_curve(retrieved_args[i, :])
                    curve_pts *= abs(curve).max() / abs(curve_pts).max()
                    frame.picker.curve.set_data(curve_pts[0], curve_pts[1])
                    frame.picker.args = retrieved_args[i, :]
                    frame.picker.redraw()

            if event.inaxes in self.retrieved_frames:
                scl.data = np.array(event.inaxes.picker.curve.get_data())
                scl.curve.set_data(scl.data)
                scl.args = event.inaxes.picker.args
                scl.redraw()

            if event.inaxes == self.edit_frame:
                if scl.press:
                    scl.on_release(event)
                dsp.args = event.inaxes.sculptor.args
                R, r, d = dsp.args

                out_gear = plt.Circle((0, 0), R, color='r', fill=False)
                int_gear = plt.Circle((R - r, 0), r, color='g', fill=False)
                hole = plt.Circle((R - r + d, 0), r / 20, color='g', fill=False)

                dsp.ax.cla()
                dsp.ax.add_artist(out_gear)
                dsp.ax.add_artist(int_gear)
                dsp.ax.add_artist(hole)

                dim = max(R - r + d, R) + 1
                dsp.ax.set_xlim(-dim, dim)
                dsp.ax.set_ylim(-dim, dim)

                dsp.ax.text(0.95, 0.01, dsp.args,
                            verticalalignment='bottom',
                            horizontalalignment='right',
                            transform=dsp.ax.transAxes)
                dsp.redraw()

    def show(self):
        """Display the application."""
        plt.show()


class Artist():
    """Base class for frame interfaces."""

    def __init__(self, axes, output=None):
        self.ax = axes
        self.output = output

        self.press = False

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)

        self.ax.set_aspect('equal')
        self.ax.autoscale()
        self.ax.margins(0.1)

        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)

    def on_press(self, event):
        """Manage mouse press events."""
        pass

    def on_move(self, event):
        """Manage mouse move events."""
        pass

    def on_release(self, event):
        """Manage mouse release events."""
        pass

    def check_tb_inactive(self):
        """Check if the matplotlib toolbar plugin is inactive."""
        return self.ax.figure.canvas.manager.toolbar._active is None

    def redraw(self):
        """Redraw only the frame."""
        self.ax.redraw_in_frame()
        self.ax.figure.canvas.blit(self.ax.bbox)


class Painter(Artist):
    """Artist for curve drawing."""

    def __init__(self, axes):
        super().__init__(axes)
        self.ax.plot([], linewidth=2)
        self.xdata = []
        self.ydata = []

    def on_press(self, event):
        """Manage mouse press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            self.press = True

    def on_move(self, event):
        """Manage mouse move events."""
        if self.press:
            if event.inaxes == self.ax:
                self.xdata.append(event.xdata)
                self.ydata.append(event.ydata)
                self.ax.lines[-1].set_data(self.xdata, self.ydata)
                self.redraw()
            else:
                return

    def on_release(self, event):
        """Manage mouse release events."""
        if self.press:
            self.press = False
            self.xdata.clear()
            self.ydata.clear()


class Sculptor(Artist):
    """Artist for continuous curve editing."""

    def __init__(self, axes, optimizer=None):
        super().__init__(axes)

        self.data = np.array([])
        self.curve = axes.plot([], 'x-')[0]

        self.to_opt = False
        self.radius_coeff = 0.05
        self.moving_points = None
        self.coeffs = None
        self.mouse_pos = np.zeros(2)

        self.args = None
        self.opt = optimizer

    def on_press(self, event):
        """Manage mouse press events."""
        if (self.check_tb_inactive() and event.inaxes == self.ax and
            self.data.size):
            self.press = True

            self.mouse_pos[:] = [event.xdata, event.ydata]
            ids, dist = self.get_closest_points(self.mouse_pos)

            radius = self.get_radius()
            inside = dist < radius
            self.coeffs = 1. - dist[inside] / radius
            self.moving_points = ids[inside]

    def on_move(self, event):
        """Manage mouse move events."""
        if self.press and event.inaxes == self.ax:
            if self.moving_points is not None:
                displacement = [event.xdata, event.ydata] - self.mouse_pos
                self.data[:, self.moving_points] += (
                    displacement.reshape(2, 1) * self.coeffs)
                self.mouse_pos[:] = [event.xdata, event.ydata]

                self.to_opt = True
                self.curve.set_data(self.data)
                self.redraw()

    def on_release(self, event):
        """Manage mouse release events."""
        if self.press:
            self.press = False
            self.moving_points = None

            if self.to_opt and self.opt is not None:
                opt_d = self.opt.optimize(target_curve=self.data,
                                          init_guess=self.args).x
                self.args[2] = opt_d

                curve_pts = cg.get_curve(self.args)
                curve_pts *= abs(self.data).max() / abs(curve_pts).max()

                self.data = curve_pts
                self.curve.set_data(self.data)
                self.redraw()

                self.to_opt = False

    def get_radius(self):
        """Get the radius at the current scale."""
        dims = np.array([self.ax.get_xlim(), self.ax.get_ylim()])
        return self.radius_coeff * abs(dims[:, 0] - dims[:, 1]).sum()

    def get_closest_points(self, target):
        """Return the indices of the data points sorted wrt dist to target."""
        d = abs(self.data - target.reshape(2, 1)).sum(axis=0)
        ids = d.argsort()
        return ids, d[ids]


class Display(Artist):
    """Simple Artist able to hold a plot ant some parameters."""

    def __init__(self, axes):
        super().__init__(axes)

        self.curve = axes.plot([])[0]
        self.args = None


def main():
    """Entry point."""
    plt.ioff()

    distance = cdist.DistanceField()
    matcher = cmat.CurveMatcher(distance.get_dist)
    optimizer = cmat.CurveOptimizer(distance.get_dist)
    sp = Spirou(matcher, optimizer)
    sp.show()


if __name__ == "__main__":
    main()
