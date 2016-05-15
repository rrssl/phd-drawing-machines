#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spirograph plotting classes.

@author: Robin Roussel
"""

from fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from matplotlib.widgets import Slider
import numpy as np
import scipy.special as spec

import curvegen as cg
from curves import RouletteEllipseInCircle
from discreteslider import DiscreteSlider


class SpiroPlot:
    """Forward analytic simulation of a simple Spirograph."""

    def __init__(self, show_spiro=True):
        self.R_min_val = 1
        self.R_max_val = 15 # included
        self.R_vals = np.arange(self.R_min_val, self.R_max_val + 1, 1)
        self.R = 8
#        self.R = 5.0051

        self.r_min_val = 1
        self.r_max_val = 10 # included
        self.r_vals = np.arange(self.r_min_val, self.r_max_val + 1, 1)
        self.r = 5

        self.d_min_val = 0.01
        self.d_max_val = 10. # included
        self.d = 0.5

        self.fig, self.ax = plt.subplots()
#        fig = plt.figure()
        plt.subplots_adjust(bottom=0.25)
#        ax = fig.add_subplot(111)

        self.sliderax_R = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='pink')
        self.sliderax_r = plt.axes([0.15, 0.1, 0.65, 0.03],
                                   axisbg='lightgreen')
        self.sliderax_d = plt.axes([0.15, 0.15, 0.65, 0.03],
                                   axisbg='lightgreen')

        self.s_R = DiscreteSlider(self.sliderax_R, 'R', self.R_min_val,
                                  self.R_max_val, allowed_vals=self.R_vals,
                                  valinit=self.R)
#        self.s_R = Slider(self.sliderax_R, 'R', self.R_min_val, self.R_max_val,
#                          valinit=self.R)
        self.s_r = DiscreteSlider(self.sliderax_r, 'r', self.r_min_val,
                                  self.r_max_val, allowed_vals=self.r_vals,
                                  valinit=self.r)
        self.s_d = Slider(self.sliderax_d, 'd', self.d_min_val, self.d_max_val,
                          valinit=self.d)

        self.s_R.on_changed(self.update)
        self.s_r.on_changed(self.update)
        self.s_d.on_changed(self.update)

        self.show_spiro = show_spiro

        self.draw()

    def draw(self):
        """Draw the figure."""
        R = self.R
        r = self.r
        d = self.d
#        N = Fraction(int(R), int(r)).denominator # /!\ R & r are numpy floats
        N = Fraction.from_float(R/r).limit_denominator(100).denominator
        ax = self.ax

        hypo = cg.get_curve((R, r, d), N)
        if self.show_spiro:
            out_gear = pat.Circle((0,0), R, color='r', fill=False)
            int_gear = pat.Circle((R - r,0), r, color='g', fill=False)
            hole = pat.Circle((R - r + d,0), r / 20, color='g', fill=False)

        ax.cla()
        ax.set_aspect('equal')
        ax.grid(True)

        if self.show_spiro:
            ax.add_artist(out_gear)
            ax.add_artist(int_gear)
            ax.add_artist(hole)
        ax.plot(hypo[0], hypo[1])

        dim = max(R - r + d, R) + 1
        ax.set_xlim(-dim, dim)
        ax.set_ylim(-dim, dim)

        self.fig.canvas.draw()
#        self.fig.canvas.draw_idle()

    def update(self, value):
        """Callback function for parameter update."""
        self.R = self.s_R.val
        self.r = self.s_r.val
        self.d = self.s_d.val
        self.draw()


class SpiroGridPlot:
    """Plotting of Spirograph figures on a parametric grid."""

    def __init__(self):
        self.plot_increasing_ratios = True
        self.plot_increasing_r = False
        self.draw_grid()

    def draw_grid(self):
        """Draw the grid of figures."""
        num_R_vals = 10
        num_d_vals = 5

        # Compute combinations.
        combi = cg.get_param_combinations((num_R_vals, num_d_vals))
        if self.plot_increasing_ratios:
            ratios = (combi[:,1] / combi[:,0]) + 1e-6 * combi[:,2]
            sorted_indices = np.argsort(ratios)
            combi = combi[sorted_indices, :]
            axis_text = r'$\frac{{{1:.0f}}}{{{0:.0f}}}$'
        if self.plot_increasing_r:
            sorted_indices = np.argsort(
                combi[:,1] + 1e-3 * combi[:,0] + 1e-6 * combi[:,2])
            combi = combi[sorted_indices, :]
            axis_text = r'$({0:.0f},\ {1:.0f})$'
        else:
            axis_text = r'$({0:.0f},\ {1:.0f})$'

        # Draw curves.
        fig, frame = plt.subplots()

        for i, c in enumerate(combi):
            position = np.array([i // num_d_vals, i % num_d_vals]) * [3, 5]
            if c[1] / c[0] != combi[i - 1, 1] / combi[i - 1, 0]:
                frame.text(position[0] - 0.3, -6, axis_text.format(c[0], c[1]),
                           fontsize=12)
            self.draw_grid_cell(c, position)

        # Adjust figure size. Don't forget 'forward=True' to change window size
        # as well.
        base_size = fig.get_size_inches()
        fig.set_size_inches((2 * base_size[0], 2 * base_size[1]),
                            forward=True)

        plt.subplots_adjust(left=0.02, right=0.98)
        frame.set_xlim(-3, 3 * (combi.shape[0] // num_d_vals))
        frame.set_aspect('equal')
        frame.axis('off')

    def draw_grid_cell(self, parameters, position):
        """Draw a single cell of the grid."""
        curve = cg.get_curve(parameters)
        curve = curve / abs(curve).max()
        curve = curve + position.reshape(2, 1)

        plt.plot(curve[0], curve[1], 'b-')


# TODO: rather than updating the boundaries by changing the slider's scale, 
# add another slider in the same slideraxes symbolizing the upper limit.
class RoulettePlot:
    """Forward geometric simulation of a roulette curve (ellipse in circle)."""
    initial_parameters = {
        'R': {'valmin': 1,    # Radius of the outer circle.
              'valmax': 15,
              'valinit': 8},
        'S': {'valmin': 0,    # Uniform scale factor of the ellipse.
              'valmax': None,
              'valinit': None},
        'e': {'valmin': 0.,   # Eccentricity of the ellipse (0 < e < 1).
              'valmax': 1.,
              'valinit': 0.5},
        'd': {'valmin': 0.,   # Radius of the tracing point in the ellipse.
              'valmax': None,
              'valinit': None}
         }

    def __init__(self, show_gears=True):
        params = RoulettePlot.initial_parameters

        R_vals = np.arange(params['R']['valmin'], params['R']['valmax'] + 1, 1)
        self.R = params['R']['valinit']
        self.e = params['e']['valinit']

        # Deduce the reference values of the semiaxes.
        e2 = self.e * self.e
        self.a0 = np.pi / (2 * spec.ellipe(e2))
        self.b0 = self.a0 * np.sqrt(1 - e2)

        # Deduce the missing values for S.
        if params['S']['valmax'] is None:
            params['S']['valmax'] = int(self.get_S_max_val())
        S_vals = np.arange(params['S']['valmin'], params['S']['valmax'] + 1, 1)
        if params['S']['valinit'] is None:
            params['S']['valinit'] = int(
                (params['S']['valmin'] + params['S']['valmax']) / 2)
        self.S = params['S']['valinit']

        # Deduce the current values of the semiaxes.
        self.a = self.S * self.a0
        self.b = self.S * self.b0

        # Deduce the missing values for d.
        if params['d']['valmax'] is None:
            params['d']['valmax'] = self.a
        if params['d']['valinit'] is None:
            params['d']['valinit'] = (
                params['d']['valmin'] + params['d']['valmax']) / 2
        self.d = params['d']['valinit']

        # Create the sliders.
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.1 + 0.05 * len(params))
        sliderax_R = plt.axes([0.15, 0.2, 0.65, 0.03], axisbg='pink')
        sliderax_S = plt.axes([0.15, 0.15, 0.65, 0.03], axisbg='lightgreen')
        sliderax_e = plt.axes([0.15, 0.1, 0.65, 0.03], axisbg='lightgreen')
        sliderax_d = plt.axes([0.15, 0.05, 0.65, 0.03], axisbg='lightgreen')

        self.s_R = DiscreteSlider(sliderax_R, 'R', allowed_vals=R_vals,
                                  **params['R'])
        self.s_S = DiscreteSlider(sliderax_S, 'S', allowed_vals=S_vals,
                                  **params['S'])
        self.s_e = Slider(sliderax_e, 'e', closedmax=False, **params['e'])
        self.s_d = Slider(sliderax_d, 'd', **params['d'])

        self.s_R.on_changed(self.update_R)
        self.s_S.on_changed(self.update_S)
        self.s_e.on_changed(self.update_e)
        self.s_d.on_changed(self.update_d)

        self.show_gears = show_gears

        self.draw()

    def draw(self, update_tracer_only=False):
        """Draw the figure."""
        R = self.R
        S = self.S
        d = self.d
        a = self.a
        b = self.b
        ax = self.ax

        ax.cla()
        ax.set_aspect('equal')
        ax.grid(True)

        if S:
            if update_tracer_only:
                curve = self.roulette.update_tracer(self.d)
            else:
                N = Fraction.from_float(R / S).limit_denominator(100).denominator
                t = np.linspace(0., N * 2 * np.pi, N * 100)

                self.roulette = RouletteEllipseInCircle(R, a, b, d)
                curve = self.roulette.get_point(t)

            if self.show_gears:
                out_gear = pat.Circle((0., 0.), R, color='r', fill=False)
                int_gear = pat.Ellipse((R - a, 0.), 2 * a, 2 * b, color='g',
                                       fill=False)
                hole = pat.Circle((R - a + d, 0.), R / 40, color='g',
                                  fill=False)

            if self.show_gears:
                ax.add_artist(out_gear)
                ax.add_artist(int_gear)
                ax.add_artist(hole)
            ax.plot(curve[0], curve[1])

            dim = max(R - a + d, R) + 1
            ax.set_xlim(-dim, dim)
            ax.set_ylim(-dim, dim)
        else:
            if self.show_gears:
                out_gear = pat.Circle((0., 0.), R, color='r', fill=False)
                ax.add_artist(out_gear)

                dim = R + 1
                ax.set_xlim(-dim, dim)
                ax.set_ylim(-dim, dim)

        self.fig.canvas.draw()

    def update(self, parameter):
        """Callback function for parameter update.

        We use a single function to avoid multiple calls to self.draw().
        """
        if parameter == 'd':
            self.d = self.s_d.val
            self.draw(update_tracer_only=True)
        else:
            if parameter == 'S':
                self.S = self.s_S.val
            elif parameter == 'e':
                self.e = self.s_e.val
                e2 = self.e * self.e
                self.a0 = np.pi / (2 * spec.ellipe(e2))
                self.b0 = self.a0 * np.sqrt(1 - e2)
            elif parameter == 'R':
                self.R = self.s_R.val
            else:
                return

            # Temporarily freeze the widget callbacks.
            self.s_S.eventson = False
            self.s_S.drawon = False
            self.s_d.eventson = False
            self.s_d.drawon = False

            # Update the upper bound of S if necessary.
            S_max_val = self.get_S_max_val()
            S_max_val = int(S_max_val) - (1 if S_max_val.is_integer() else 0)
            if self.s_S.valmax != S_max_val:
                self.s_S.valmax = S_max_val
                if S_max_val:
                    self.s_S.ax.set_xlim(right=S_max_val)
                if self.S > S_max_val:
                    self.s_S.set_val(S_max_val)
                    self.S = S_max_val

            self.a = self.S * self.a0
            self.b = self.S * self.b0

            # Update the upper bound of d if necessary.
            self.s_d.valmax = self.a
            if self.a:
                self.s_d.ax.set_xlim(right=self.a)
            if self.d > self.a:
                self.s_d.set_val(self.a)
                self.d = self.a

            # Un-freeze the widgets callbacks.
            self.s_S.eventson = True
            self.s_S.drawon = True
            self.s_d.eventson = True
            self.s_d.drawon = True

            self.draw()

    def update_R(self, value, redraw=True):
        """Callback function for parameter 'R' update."""
        self.update('R')

    def update_e(self, value, redraw=True):
        """Callback function for parameter 'e' update."""
        self.update('e')

    def update_S(self, value):
        """Callback function for parameter 'S' update."""
        self.update('S')

    def update_d(self, value):
        """Callback function for parameter 'd' update."""
        self.update('d')

    def get_S_max_val(self):
        """Get the maximum S value so that the curvature constraint is met."""
        return self.R * np.sqrt(1 - self.e ** 2) / self.a0


def main():
    """Entry point."""
    plt.ioff()

#    SpiroPlot(show_spiro=True)

#    SpiroGridPlot()

    RoulettePlot(show_gears=True)

    plt.show()


if __name__ == "__main__":
    main()
