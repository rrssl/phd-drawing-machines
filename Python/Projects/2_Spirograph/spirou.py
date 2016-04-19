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
        self.paint_frame.painter = Painter(self.paint_frame, self)

        # Create the sculptor.
        self.edit_frame = plt.subplot2grid(
            plot_grid_size, (0, 2), rowspan=2, colspan=2,
            title="Edit here!")
        self.edit_frame.sculptor = Sculptor(self.optimizer, self.edit_frame, 
                                            self)

        # Create the pickers.
        self.retrieved_frames = []
        for i in range(self.nb_retrieved):
            frame = plt.subplot2grid(plot_grid_size, (2, i), title=i+1)
            self.retrieved_frames.append(frame)
            frame.picker = ButtonDisplay(frame, self)

        # Create the final display.
        self.show_frame = plt.subplot2grid(
            plot_grid_size, (0, 4), rowspan=2, colspan=2, title="Result")
        self.show_frame.display = Display(self.show_frame, self)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.90, hspace=0.5)
                  
    def transmit_painter_to_pickers(self):
        if self.paint_frame.lines:
            curve = np.array(self.paint_frame.lines[-1].get_data(orig=False))
            if curve.size <= 2:
                return
            retrieved_args = (
                self.matcher(curve, self.samples)[:self.nb_retrieved, :])
            for i, frame in enumerate(self.retrieved_frames):
                curve_pts = cg.get_curve(retrieved_args[i, :])
                curve_pts *= abs(curve).max() / abs(curve_pts).max()
                frame.picker.curve.set_data(curve_pts[0], curve_pts[1])
                
                frame.picker.args = retrieved_args[i, :]
                
                if frame.picker.hold:
                    frame.picker.hold = False
                    frame.set_axis_bgcolor(ButtonDisplay.bg_colors[0])
                
                frame.picker.redraw()
                
    def transmit_picker_to_sculptor(self, picker):
        scl = self.edit_frame.sculptor
        scl.data = np.array(picker.curve.get_data())
        scl.curve.set_data(scl.data)
        scl.args = picker.args.copy()
        scl.redraw()
        
    def transmit_picker_to_display(self, picker):
        dsp = self.show_frame.display
        if picker.args is not None:
            dsp.args = picker.args
            dsp.ax.cla()
            dsp.draw_spiro()
            dsp.redraw()
    
    def transmit_picker_to_pickers(self, picker):
        for retf in self.retrieved_frames:
            if retf.picker.hold and retf.picker != picker:
                retf.picker.hold = False
                retf.set_axis_bgcolor(ButtonDisplay.bg_colors[0])
                retf.picker.redraw()
                
    def transmit_sculptor_to_display(self):
        scl = self.edit_frame.sculptor
        dsp = self.show_frame.display
        if scl.args is not None:
            dsp.args = scl.args
            dsp.ax.cla()
            dsp.draw_spiro()
            dsp.redraw()        

    def update_data(self, event):
        """Update data in frames after event (called by Artists)."""
        ax = event.inaxes
        if ax == self.paint_frame:
            self.transmit_painter_to_pickers()
        if ax in self.retrieved_frames:
            self.transmit_picker_to_sculptor(ax.picker)
            self.transmit_picker_to_display(ax.picker)
            self.transmit_picker_to_pickers(ax.picker)
        if ax == self.edit_frame:
            self.transmit_sculptor_to_display()

    def show(self):
        """Display the application."""
        plt.show()


class Artist():
    """Base class for frame interfaces."""

    def __init__(self, axes, context):
        self.ax = axes
        self.ctxt = context

        self.press = False

        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
        self.ax.set_aspect('equal')
        self.ax.autoscale()
        self.ax.margins(0.1)

        self.ax.figure.canvas.mpl_connect('axes_enter_event', self.on_enter)
        self.ax.figure.canvas.mpl_connect('axes_leave_event', self.on_leave)

        self.ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.ax.figure.canvas.mpl_connect('button_release_event',
                                          self.on_release)
        self.ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        
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

    def check_tb_inactive(self):
        """Check if the matplotlib toolbar plugin is inactive."""
        return self.ax.figure.canvas.manager.toolbar._active is None

    def redraw(self):
        """Redraw only the frame."""
        self.ax.redraw_in_frame()
        self.ax.figure.canvas.blit(self.ax.bbox)


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
        self.timer = self.ax.figure.canvas.new_timer(interval=40)
        self.timer.add_callback(self.grow_radius)

        self.args = None
        self.opt = optimizer
        
    def on_enter(self, event):
        """Manage entering mouse events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            if self.grab_circle is None:
                self.mouse_pos[:] = [event.xdata, event.ydata]
                self.grab_circle = plt.Circle(self.mouse_pos, 
                                              self.get_radius(), 
                                              color='r', linestyle='dashed', 
                                              fill=False)
                self.ax.add_artist(self.grab_circle)
                self.redraw()
                                      
    def on_leave(self, event):
        """Manage leaving mouse events."""
        if event.inaxes == self.ax:
            if self.grab_circle is not None:
                self.grab_circle.remove()
                self.grab_circle = None
                self.redraw()

                self.radius_coeff = Sculptor.init_radius_coeff
                self.timer.stop()
                self.time = 0.

    def on_press(self, event):
        """Manage mouse press events."""
        if self.check_tb_inactive() and event.inaxes == self.ax:
            self.timer.start()
            
            if self.data.size:
                self.press = True
    
                self.mouse_pos[:] = [event.xdata, event.ydata]
    
                radius = self.get_radius()
                self.moving_points, dist = self.grab_points(self.mouse_pos,
                                                            radius)
                if dist is not None:
                    self.coeffs = 1. - dist / radius

    def on_move(self, event):
        """Manage mouse move events."""
        if event.inaxes == self.ax:
            if self.grab_circle is not None:
                self.timer.stop()

                self.grab_circle.center = [event.xdata, event.ydata]
                self.redraw()
                
            if self.press and self.moving_points is not None:
                displacement = [event.xdata, event.ydata] - self.mouse_pos
                self.data[:, self.moving_points] += (
                    displacement.reshape(2, 1) * self.coeffs)
                self.mouse_pos[:] = [event.xdata, event.ydata]

                self.to_opt = True
                self.curve.set_data(self.data)
                self.redraw()

    def on_release(self, event):
        """Manage mouse release events."""
        if self.grab_circle is not None:
            self.timer.stop()
            self.time = 0.

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
                
            self.ctxt.update_data(event)

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
                self.redraw()

    def grab_points(self, target, radius):
        """Get the contiguous points closest to the target within radius."""
        dist = abs(self.data - target.reshape(2, 1)).sum(axis=0)
        inside = dist < radius
        if not inside.any():
            return None, None
        # Only keep the string of points containing the closest one.
        start = end = dist.argmin()
        while start > 0 and inside[start - 1]: start -= 1
        while end < inside.size - 1 and inside[end + 1]: end += 1
        ids = np.arange(start, end + 1)
        
        return ids, dist[ids]


class Display(Artist):
    """Simple Artist able to hold a plot and some parameters."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.curve = self.ax.plot([])[0]
        self.args = None

    def draw_spiro(self):
        """Draw the spirograph corresponding to its arguments."""
        R, r, d = self.args
        ax = self.ax

        out_gear = plt.Circle((0, 0), R, color='r', fill=False)
        int_gear = plt.Circle((R - r, 0), r, color='g', fill=False)
        hole = plt.Circle((R - r + d, 0), r / 20, color='g', fill=False)

        ax.add_artist(out_gear)
        ax.add_artist(int_gear)
        ax.add_artist(hole)

        dim = max(R - r + d, R) + 1
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)

        ax.text(0.95, 0.01, self.args, verticalalignment='bottom',
                horizontalalignment='right', transform=ax.transAxes)


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
