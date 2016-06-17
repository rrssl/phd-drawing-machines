# -*- coding: utf-8 -*-
"""
Library of GUIs for curve drawing, manipulation and display.

@author: Robin Roussel
"""
import matplotlib as mpl
#import matplotlib.animation as manim
import matplotlib.patches as mpat
import numpy as np


class Artist():
    """Base class for frame interfaces."""

    def __init__(self, axes, update):
        self.ax = axes
        self.update = update

        self.press = False

        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
        axes.set_aspect('equal')

        connect = axes.figure.canvas.mpl_connect
        connect('axes_enter_event', self.on_enter)
        connect('axes_leave_event', self.on_leave)
        connect('button_press_event', self.on_press)
        connect('button_release_event', self.on_release)
        connect('motion_notify_event', self.on_move)
        connect('key_press_event', self.on_key_press)
        connect('key_release_event', self.on_key_release)

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

            self.update(self.ax)


# TODO: resorb the code duplication due to the useless distinction between
# grab_circle and aux_grab_circles.
class Selector(Artist):
    """Artist for advanced vertex selection."""
    init_radius_coeff = 0.05
    period = 40 # in milliseconds

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Logic
        self.r_press = False
        # Simple selection
        self.radius_coeff = Selector.init_radius_coeff
        self.grab_circle = None
        # Multiple selection
        self.sym_order = 0
        self.aux_grab_circles = []
        # Long selection
        self.time = 0.
        self.timer = self.ax.figure.canvas.new_timer(interval=Selector.period)
        self.timer.add_callback(self.grow_radius)

    def reset(self, sym_order=0):
        """Reset the data."""
        self.sym_order = sym_order

    def on_enter(self, event):
        """Manage entering mouse events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if self.grab_circle is None:
                self.grab_circle = self.draw_circle((event.xdata, event.ydata),
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

                self.radius_coeff = Selector.init_radius_coeff
                self.timer.stop()
                self.time = 0.

    def on_press(self, event):
        """Manage mouse press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            self.press = True
            self.timer.start()

    def on_move(self, event):
        """Manage mouse move events."""
        if event.inaxes == self.ax:
            if self.grab_circle is not None:
                if self.press:
                    self.timer.stop()
                self.grab_circle.center = (event.xdata, event.ydata)

                if self.aux_grab_circles:
                    aux_centers = self.get_rot_sym(self.grab_circle.center)[1:]
                    for gcirc, acnt in zip(self.aux_grab_circles, aux_centers):
                        gcirc.center = acnt

                self.redraw()

    def on_release(self, event):
        """Manage mouse release events."""
        if self.grab_circle is not None:
            self.timer.stop()
            self.time = 0.

        if self.press:
            # Cleanup
            self.press = False
            if not self.r_press:
                self.erase_aux_grab_circles()
                self.redraw()

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
            for agc in self.aux_grab_circles:
                agc.remove()
            self.aux_grab_circles.clear()

    def get_rot_sym(self, point):
        """Get points that are rotationally symmetric around the center."""
        if not self.sym_order:
            return
        theta = 2 * np.pi / self.sym_order
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        points = [np.array(point)]
        for i in range(self.sym_order - 1):
            points.append(rot.dot(points[i]))

        return points

    def get_radius(self):
        """Get the radius at the current scale."""
        dims = np.array([self.ax.get_xlim(), self.ax.get_ylim()])
        return self.radius_coeff * abs(dims[:, 0] - dims[:, 1]).sum()

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


# TODO: resorb the code duplication due to the useless distinction between
# grab_circle and aux_grab_circles.
class Sculptor(Selector):
    """Artist for continuous curve editing."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Plot
        self.params = None
        self.data = np.array([])
        self.curve = self.ax.plot([], 'x-')[0]
        # Logic
        self.to_opt = False
        # User input
        self.mouse_pos = np.zeros(2)
        # Simple selection
        self.moving_points = None
        self.coeffs = None
        # Multiple selection
        self.aux_moving_points = []
        self.aux_coeffs = []

    def reset(self, params, data, *args, **kwargs):
        """Reset the data."""
        super().reset(*args, **kwargs)
        self.params = params
        self.data = np.asarray(data)
        self.curve.set_data(self.data)
        self.to_opt = False
        # Adapt the plot limits (keeping the frame square).
        dim = self.data.max() * 1.5
        self.ax.set_xlim(-dim, dim)
        self.ax.set_ylim(-dim, dim)

    def on_press(self, event):
        """Manage mouse press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if self.data.size:
                self.mouse_pos[:] = (event.xdata, event.ydata)
                self.grab_points()
            super().on_press(event)

    def on_move(self, event):
        """Manage mouse move events."""
        if event.inaxes == self.ax:
            if self.press and self.moving_points is not None:
                diff = (event.xdata, event.ydata) - self.mouse_pos
                self.data[:, self.moving_points] += (
                    diff.reshape(2, 1) * self.coeffs)
                self.mouse_pos[:] = (event.xdata, event.ydata)

                if self.aux_grab_circles:
                    aux_diffs = self.get_rot_sym(diff)[1:]
                    for adiff, ampts, acoeff in zip(aux_diffs,
                                                    self.aux_moving_points,
                                                    self.aux_coeffs):
                        self.data[:, ampts] += adiff.reshape(2, 1) * acoeff

                self.to_opt = True
                self.curve.set_data(self.data)
            super().on_move(event)

    def on_release(self, event):
        """Manage mouse release events."""
        if self.press:
            # Cleanup
            self.moving_points = None
            self.coeffs = None
            self.aux_moving_points.clear()
            self.aux_coeffs.clear()
            super().on_release(event)
            # Update
            self.update(self.ax)

    def on_key_release(self, event):
        """Manage key release events."""
        if event.key == 'r' and self.r_press:
            if not self.press:
                self.aux_moving_points.clear()
                self.aux_coeffs.clear()
            super().on_key_release(event)

    def grow_radius(self):
        """Increase the grab circle radius after a while."""
        if self.grab_circle is not None:
            super().grow_radius()
            if self.time > 20. and self.data.size:
                if self.aux_moving_points:
                    self.aux_moving_points.clear()
                    self.aux_coeffs.clear()
                self.grab_points()

    def grab_points(self):
        """Grab the contiguous points closest to the center within radius."""
        target = self.mouse_pos.reshape(self.data.shape[0], 1)
        radius = self.get_radius()
        nb_pts = self.data.shape[1]

        dist = abs(self.data - target).sum(axis=0)
        inside = dist <= radius
        if inside.any():
            if inside.all():
                self.moving_points = np.arange(nb_pts)
            else:
                # Only keep the string of points containing the closest one.
                start = end = dist.argmin()
                while inside[(start - 1) % nb_pts]:
                    start -= 1
                while inside[(end + 1) % nb_pts]:
                    end += 1
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
                    while ins[(start - 1) % nb_pts]:
                        start -= 1
                    while ins[(end + 1) % nb_pts]:
                        end += 1

                    ids = np.arange(start, end) % nb_pts
                    self.aux_moving_points.append(ids)
                    self.aux_coeffs.append(1. - row[ids] / radius)


class Display(Artist):
    """Simple Artist able to hold a curve and some parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.curve = self.ax.plot([])[0]
        self.params = None


class AnimDisplay(Display):
    """Display able to show curve drawing animations."""
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
        
        self.nb_frames = 0

    def reset(self, params, data):
        """Reset the data."""
        self.params = params
        self.data = np.asarray(data)[:, :-1]
        self.curve.set_data([], [])
        self.time = 0.
        self.nb_frames = self.data.shape[1]
        # Adapt the plot limits (keeping the frame square).
        dim = self.data.max() * 1.1
        self.ax.set_xlim(-dim, dim)
        self.ax.set_ylim(-dim, dim)

    def on_key_press(self, event):
        """Manage key press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if event.key == 'p':
                if self.playing:
                    self.playing = False
                    self.timer.stop()

                else:
                    if self.data is not None:
                        self.playing = True
                        self.timer.start()

    def animate(self):
        """Create the current frame."""
        idx = self.time % self.nb_frames
        self.curve.set_data(self.data[:, :idx + 1])

        self.redraw()
        self.time += 1.

#    def play(self):
#        self.anim = manim.FuncAnimation(self.ax.figure, self.animate,
#                                        frames=200, init_func=self.init,
#                                        interval=20, blit=True)


class ButtonDisplay(Display):
    """Selectable Display."""
    bg_colors = ('white', 'lightgreen')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.hold = False

    def reset(self, params, data):
        """Reset the data."""
        self.curve.set_data(data[0], data[1])
        # Adapt the plot limits (keeping the frame square).
        dim = abs(data).max() * 1.2
        self.ax.set_xlim(-dim, dim)
        self.ax.set_ylim(-dim, dim)

        self.params = params
        # Unselect if previously selected.
        if self.hold:
            self.hold = False
            self.ax.set_axis_bgcolor(ButtonDisplay.bg_colors[0])

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
            if self.params is not None:
                self.hold = True

    def on_release(self, event):
        """Manage mouse release events."""
        if event.inaxes == self.ax:
            self.update(self.ax)
