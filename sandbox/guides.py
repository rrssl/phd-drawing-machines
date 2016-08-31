# -*- coding: utf-8 -*-
"""
Playing with visual cues and constrained editing.

@author: robin
"""
from itertools import chain
import math
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np

import context

from mecha import BaseSpirograph
from controlpane import ControlPane


class SpiroGuides:
    """Guides."""

    def __init__(self):
        self.mecha = BaseSpirograph(11, 7, 1.5)

        self.pid = 2
        bounds = self.mecha.get_prop_bounds(self.pid)
        self.prop_data = (self.pid, {'valmin': bounds[0],
                                     'valmax': bounds[1],
                                     'valinit': self.mecha.props[2],
                                     'label': "Continuous\nparam."
                                     }),
        self.init_draw()

    def init_draw(self):
        """Initialize the canvas."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.margins(0.1)

        self.control_pane = ControlPane(self.fig, self.prop_data, self.update)

        self.draw_sectors()

        self.draw_ghost_curves(0)

        self.draw_arrows(0)

        curve = self.mecha.get_curve()
        self.plot = self.ax.plot(curve[0], curve[1], c='b')[0]

    def draw_sectors(self):
        """Draw rotationally symmetric sectors."""
        R, r, d = self.mecha.props
        dmax = self.mecha.get_prop_bounds(self.pid)[1]
        sym_order = R
        radius = (R - r + dmax) * 1.1
        t1 = 180 / sym_order
        t2 = t1 + 360 / sym_order
        for i in range(int(sym_order)):
            color = 'lightgrey' if i else 'white'
            zorder = 3 if i else 1
            self.ax.add_patch(
                pat.Wedge((0,0), radius, t1, t2, color=color, alpha=0.7,
                          zorder=zorder))
            t1 += 360 / sym_order
            t2 += 360 / sym_order

    def draw_ghost_curves(self, direction=1):
        """Draw the 'ghosts' of the neighboring curves."""
        nb_ghosts = 10
        alpha_bounds = (0.1, 1.)

        self.ghost_curves = []
        if direction == 0:
            return

        # Get bound and init value.
        current_val = self.mecha.props[self.pid]
        bound = self.mecha.get_prop_bounds(self.pid)[direction > 0]
        # Sample [0., 1.] with a Gaussian.
        rng = np.linspace(0, math.sqrt(2 * math.log(2)), nb_ghosts)
        rng = np.exp(- rng ** 2 / 2)
        rng = rng if direction > 0 else rng[::-1]
        # Compute transparencies.
        alphas = np.interp(rng, [0.5, 1.], alpha_bounds)
        # Map to property range.
        rng = np.interp(rng, [0.5, 1.], [bound, current_val])

#        # Get bounds and init value.
#        current_val = self.mecha.props[self.pid]
#        bounds = self.mecha.get_prop_bounds(self.pid)
#        # Sample [0., 1.] with a Gaussian.
#        rng = np.linspace(0, math.sqrt(2 * math.log(2)), nb_ghosts)
#        rng = np.exp(- rng ** 2 / 2)
#        # Symmetrize around 0.
#        rng_left = rng[::-1]
#        rng_right = rng
#        # Compute transparencies.
#        alphas = np.hstack([rng_left[:-1], rng[1:]])
#        alphas = np.interp(alphas, [0.5, 1.], alpha_bounds)
#        # Map to property range.
#        rng_left = np.interp(rng_left, [0.5, 1.], [bounds[0], current_val])
#        rng_right = np.interp(rng_right, [0.5, 1.], [bounds[1], current_val])
#        rng = np.hstack([rng_left[:-1], rng_right[1:]])

        # Plot.
        for val, alp in zip(rng, alphas):
            self.mecha.update_prop(self.pid, val)
            curve = self.mecha.get_curve()
            self.ghost_curves.append(
                self.ax.plot(curve[0], curve[1], color='orange', alpha=alp)[0])
        # Reset mechanism to current value.
        self.mecha.update_prop(self.pid, current_val)

    def draw_arrows(self, direction=1):
        """Draw the gradient arrows.

        direction = 1: prop is increasing
        direction = -1: prop is decreasing
        """
        prop_step = 0.1
        arrow_width = 0.1
        petal_id = 1

        self.arrows = []
        R, r, d = self.mecha.props
        bounds = self.mecha.get_prop_bounds(self.pid)
        if (direction == 0 or
            (direction > 0 and d + prop_step > bounds[1]) or
            (direction < 0 and d - prop_step < bounds[0])):
            return

        # Finding the index of the tip of a petal requires solving a simple
        # linear congruence problem. We can do it by brute force.
        x = 0
        while (R * x + petal_id) % r and x < r: x += 1
        x = (R * x + petal_id) / r
        pt_id = round(x * self.mecha._simulator.nb_samples * r / R)
        # Compute the arrows by evaluating the discrete gradient.
        curve_base = self.mecha.get_curve()
        if direction > 0:
            # Above the current d value.
            self.mecha.update_prop(self.pid, d + prop_step)
            curve_diff = self.mecha.get_curve()
        elif direction < 0:
            # Below the current d value.
            self.mecha.update_prop(self.pid, d - prop_step)
            curve_diff = self.mecha.get_curve()
        self.mecha.update_prop(self.pid, d)

        direction = (curve_diff - curve_base) * R
        # Draw the arrows.
        for i in chain(range(pt_id - 11, pt_id), range(pt_id, pt_id + 11)):
            self.arrows.append(self.ax.add_patch(pat.Arrow(
                curve_base[0, i], curve_base[1, i],
                direction[0, i], direction[1, i],
                arrow_width, zorder=4, lw=0, color='g')))

    def update(self, pid, val):
        """Update the curve."""
        props = self.mecha.props
        direction = val - props[pid]
        self.mecha.update_prop(pid, val)
        if direction != 0:
            # Update ghost curves.
            for ghost in self.ghost_curves:
                ghost.remove()
            self.draw_ghost_curves(direction)
        # Update arrows.
            for arrow in self.arrows:
                arrow.remove()
            self.draw_arrows(direction)
        # Update main curve.
        curve = self.mecha.get_curve()
        self.plot.remove()
        self.plot = self.ax.plot(curve[0], curve[1], c='b')[0]

def main():
    """Entry point."""
    plt.ioff()

    SpiroGuides()

    plt.show()

if __name__ == "__main__":
    main()
