# -*- coding: utf-8 -*-
"""
Library of GUIs for curve drawing, manipulation and display.

@author: Robin Roussel
"""
from fractions import Fraction

import matplotlib as mpl
#import matplotlib.animation as manim
import matplotlib.patches as mpat
import numpy as np

import curvegen as cg


class Artist():
    """Base class for frame interfaces."""

    def __init__(self, axes, context):
        self.ax = axes
        self.ctxt = context

        self.press = False

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_aspect('equal')

        self.ax.figure.canvas.mpl_connect('axes_enter_event', self.on_enter)
        self.ax.figure.canvas.mpl_connect('axes_leave_event', self.on_leave)

        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.ax.figure.canvas.mpl_connect('key_press_event', self.on_key_press)
        self.ax.figure.canvas.mpl_connect('key_release_event',
                                          self.on_key_release)

    def on_enter(self, event):
        """Manage entering mouse events."""
        pass

    def on_leave(self, event):
        """Manage leaving mouse events."""
        pass

    def on_press(self, event):
        """Manage mouse press events."""
        pass

    def on_move(self, event):
        """Manage mouse move events."""
        pass

    def on_release(self, event):
        """Manage mouse release events."""
        pass

    def on_key_press(self, event):
        """Manage key press events."""
        pass

    def on_key_release(self, event):
        """Manage key release events."""
        pass

    def check_tb_inactive(self):
        """Check if the matplotlib toolbar plugin is inactive."""
        return self.ax.figure.canvas.manager.toolbar._active is None

    def redraw(self):
        """Redraw only the frame."""
        self.ax.redraw_in_frame()
#        self.ax.figure.canvas.blit(self.ax.bbox)
        self.ax.figure.canvas.update()


class Painter(Artist):
    """Artist for curve drawing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

            self.ctxt.update_data(event)


class Sculptor(Artist):
    """Artist for continuous curve editing."""
    init_radius_coeff = 0.05
    period = 40 # in milliseconds

    def __init__(self, optimizer=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = np.array([])
        self.curve = self.ax.plot([], 'x-')[0]

        self.to_opt = False
        self.radius_coeff = Sculptor.init_radius_coeff
        self.moving_points = None
        self.coeffs = None
        self.mouse_pos = np.zeros(2)

        self.grab_circle = None
        self.time = 0.
        self.timer = self.ax.figure.canvas.new_timer(interval=Sculptor.period)
        self.timer.add_callback(self.grow_radius)

        self.r_press = False
        self.aux_grab_circles = []
        self.aux_moving_points = []
        self.aux_coeffs = []

        self.args = None
        self.opt = optimizer

    def on_enter(self, event):
        """Manage entering mouse events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if self.grab_circle is None:
                self.mouse_pos[:] = [event.xdata, event.ydata]
                self.grab_circle = self.draw_circle(self.mouse_pos,
                                                    self.get_radius())
                if self.r_press: self.draw_aux_grab_circles()
                self.redraw()

    def on_leave(self, event):
        """Manage leaving mouse events."""
        if event.inaxes == self.ax:
            if not self.press and self.grab_circle is not None:
                # Cleanup
                self.grab_circle.remove()
                self.grab_circle = None
                self.erase_aux_grab_circles()
                self.redraw()

                self.radius_coeff = Sculptor.init_radius_coeff
                self.timer.stop()
                self.time = 0.

    def on_press(self, event):
        """Manage mouse press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if self.data.size:
                self.press = True

                self.mouse_pos[:] = [event.xdata, event.ydata]
                self.grab_points()
            self.timer.start()

    def on_move(self, event):
        """Manage mouse move events."""
        if event.inaxes == self.ax:
            redraw = False
            if self.grab_circle is not None:
                if self.press:
                    self.timer.stop()
                self.grab_circle.center = [event.xdata, event.ydata]
                redraw = True

                if self.aux_grab_circles:
                    aux_centers = self.get_rot_sym(self.grab_circle.center)[1:]
                    for gcirc, acnt in zip(self.aux_grab_circles, aux_centers):
                        gcirc.center = acnt

            if self.press and self.moving_points is not None:
                diff = [event.xdata, event.ydata] - self.mouse_pos
                self.data[:, self.moving_points] += (
                    diff.reshape(2, 1) * self.coeffs)
                self.mouse_pos[:] = [event.xdata, event.ydata]

                if self.aux_grab_circles:
                    aux_diffs = self.get_rot_sym(diff)[1:]
                    for adiff, ampts, acoeff in zip(aux_diffs,
                                                    self.aux_moving_points,
                                                    self.aux_coeffs):
                        self.data[:, ampts] += adiff.reshape(2, 1) * acoeff

                self.to_opt = True
                self.curve.set_data(self.data)
                redraw = True
            if redraw: self.redraw()

    def on_release(self, event):
        """Manage mouse release events."""
        if self.grab_circle is not None:
            self.timer.stop()
            self.time = 0.

        if self.press:
            redraw = False
            # Cleanup
            self.press = False
            self.moving_points = None
            self.coeffs = None
            if not self.r_press:
                self.erase_aux_grab_circles()
                redraw = True
                self.aux_moving_points = []
                self.aux_coeffs = []
            # Optimization
            if self.to_opt and self.opt is not None:
                opt_d = self.opt.optimize(target_curve=self.data,
                                          init_guess=self.args).x
                self.args[2] = opt_d

                curve_pts = cg.get_curve(self.args)
                curve_pts *= abs(self.data).max() / abs(curve_pts).max()

                self.data = curve_pts
                self.curve.set_data(self.data)
                redraw = True

                self.to_opt = False
            # Update
            if redraw: self.redraw()
            self.ctxt.update_data(event)

    def on_key_press(self, event):
        """Manage key press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if event.key == 'r':
                self.r_press = True

                if not self.press:
                    self.draw_aux_grab_circles()
                    self.redraw()

    def on_key_release(self, event):
        """Manage key release events."""
        if event.key == 'r' and self.r_press:
            self.r_press = False

            if not self.press:
                self.erase_aux_grab_circles()
                self.aux_moving_points = []
                self.aux_coeffs = []
                self.redraw()

    def draw_circle(self, center, radius):
        """Draw a circle."""
        circle = mpat.Circle(center, radius, color='r',
                             linestyle='dashed', fill=False)
        self.ax.add_artist(circle)
        return circle

    def draw_aux_grab_circles(self):
        """Draw the auxiliary selection circles."""
        if self.grab_circle is not None:
            centers = self.get_rot_sym(self.grab_circle.center)
            if centers:
                radius = self.get_radius()
                for center in centers[1:]:
                    self.aux_grab_circles.append(
                        self.draw_circle(center, radius))

    def erase_aux_grab_circles(self):
        """Erase the auxiliary selection circles."""
        if self.aux_grab_circles:
            for gc in self.aux_grab_circles:
                gc.remove()
            self.aux_grab_circles = []

    def get_rot_sym(self, point):
        """Get points that are rotationally symmetric around the center."""
        order = self.get_symmetry_order()
        if not order:
            return
        theta = 2 * np.pi / order
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        points = [np.array(point)]
        for i in range(order - 1):
            points.append(rot.dot(points[i]))

        return points

    def get_radius(self):
        """Get the radius at the current scale."""
        dims = np.array([self.ax.get_xlim(), self.ax.get_ylim()])
        return self.radius_coeff * abs(dims[:, 0] - dims[:, 1]).sum()

    def get_symmetry_order(self):
        """Get the order of symmetry of the current figure."""
        if self.args is None:
            return 0
        else:
            return Fraction.from_float(
                self.args[0] / self.args[1]).limit_denominator(1000).numerator

    def grow_radius(self):
        """Increase the grab circle radius after a while."""
        if self.grab_circle is not None:
            self.time += 1.

            if self.time > 20.:
                coeff = 1. + self.time / 1e4
                self.radius_coeff *= coeff
                self.grab_circle.radius *= coeff
                for gcirc in self.aux_grab_circles:
                    gcirc.radius *= coeff
                self.redraw()

                if self.data.size:
                    self.grab_points()

    def grab_points(self):
        """Grab the contiguous points closest to the center within radius."""
        target = self.mouse_pos.reshape(self.data.shape[0], 1)
        radius = self.get_radius()

        dist = abs(self.data - target).sum(axis=0)
        inside = dist <= radius
        if inside.any():
            nb_pts = self.data.shape[1]
            if inside.all():
                self.moving_points = np.arange(nb_pts)
            else:
                # Only keep the string of points containing the closest one.
                start = end = dist.argmin()
                while inside[(start - 1) % nb_pts]: start -= 1
                while inside[(end + 1) % nb_pts]: end += 1
                self.moving_points = np.arange(start, end) % nb_pts

            self.coeffs = 1. - dist[self.moving_points] / radius

        if self.aux_grab_circles:
            targets = np.array([gcirc.center for gcirc in self.aux_grab_circles])
            # Use broadcasting to efficiently compute all distances.
            targets = targets.reshape((targets.shape[0], self.data.shape[0], 1))
            distances = abs(self.data - targets).sum(axis=1)
#            # Keep only distances to the closest target point.
#            distances = distances.min(axis=0)

            # Find points within radius.
            inside = distances <= radius

            if inside.any():
                for row, ins in zip(distances, inside):
                    # Only keep the string of points containing the closest one.
                    start = end = row.argmin()
                    while ins[(start - 1) % nb_pts]: start -= 1
                    while ins[(end + 1) % nb_pts]: end += 1

                    ids = np.arange(start, end) % nb_pts
                    self.aux_moving_points.append(ids)
                    self.aux_coeffs.append(1. - row[ids] / radius)


class Display(Artist):
    """Simple Artist able to hold a plot and some parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.curve = self.ax.plot([])[0]
        self.args = None

        self.fixed_curve = None
        self.moving_curve = None
        self.tracing_point = None

        self.text = None

    def draw_spiro(self):
        """Draw the spirograph corresponding to its arguments."""
        R, r, d = self.args
        ax = self.ax

        out_gear = mpat.Circle((0, 0), R, color='r', fill=False)
        int_gear = mpat.Circle((R - r, 0), r, color='g', fill=False)
        hole = mpat.Circle((R - r + d, 0), r / 20, color='g', fill=False)

        self.fixed_curve = ax.add_artist(out_gear)
        self.moving_curve = ax.add_artist(int_gear)
        self.tracing_point = ax.add_artist(hole)

        dim = max(R - r + d, R) + 1
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)

        self.text = ax.text(0.95, 0.01, self.args, verticalalignment='bottom',
                            horizontalalignment='right',
                            transform=ax.transAxes)

    def erase_spiro(self):
        """Erase the spirograph."""
        if self.fixed_curve is not None:
            self.fixed_curve.remove()
            self.moving_curve.remove()
            self.tracing_point.remove()
            self.text.remove()


class AnimDisplay(Display):
    """Display able to play spirograph animations."""
    period = 40

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = None

        self.playing = False

        mpl.rcParams['keymap.pan'] = '' # Free the 'p' shortcut.

        self.time = 0.
        self.timer = self.ax.figure.canvas.new_timer(
            interval=AnimDisplay.period)
        self.timer.add_callback(self.animate)

    def on_key_press(self, event):
        """Manage key press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if event.key == 'p':
                if self.playing:
                    self.playing = False
                    self.timer.stop()

                else:
                    if self.moving_curve is not None:
                        self.playing = True
                        self.timer.start()

#    def play(self):
#        self.anim = manim.FuncAnimation(self.ax.figure, self.animate,
#                                        frames=200, init_func=self.init,
#                                        interval=20, blit=True)

    def animate(self):
        """Animate the spirograph."""
        idx = self.time % self.data.shape[1]
        self.curve.set_data(self.data[:, :idx + 1])

        theta = self.time * 2 * np.pi * self.args[1] / self.data.shape[1]
        r = self.args[0] - self.args[1]
        self.moving_curve.center = [r * np.cos(theta), r * np.sin(theta)]

        self.tracing_point.center = self.data[:, idx]

        self.redraw()
        self.time += 1.

    def erase_spiro(self):
        """Erase the spirograph."""
        super().erase_spiro()
        self.time = 0.


class ButtonDisplay(Display):
    """Selectable Display."""
    bg_colors = ('white', 'lightgreen')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hold = False

    def on_enter(self, event):
        """Manage entering mouse events."""
        if event.inaxes == self.ax:
            if not self.hold:
                self.ax.set_axis_bgcolor(ButtonDisplay.bg_colors[1])
                self.redraw()

    def on_leave(self, event):
        """Manage leaving mouse events."""
        if event.inaxes == self.ax:
            if not self.hold:
                self.ax.set_axis_bgcolor(ButtonDisplay.bg_colors[0])
                self.redraw()

    def on_press(self, event):
        """Manage mouse press events."""
        if event.inaxes == self.ax:
            if self.args is not None:
                self.hold = True

    def on_release(self, event):
        """Manage mouse release events."""
        if event.inaxes == self.ax:
            self.ctxt.update_data(event)
