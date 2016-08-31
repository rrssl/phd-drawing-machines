# -*- coding: utf-8 -*-
"""
Exploring the property space.

@author: Robin Roussel
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import context
from mecha import EllipticSpirograph

class PropExplorer:
    """Sample continuous properties."""

    def __init__(self):
        self.R = 5
        self.S = 3
        self.num_e2_vals = 10
        self.num_d_vals = 10
        self.mecha = EllipticSpirograph(self.R, self.S, .2, .2)
        self.nb = 2**4

        self.init_draw()

    def init_draw(self):
        """Initialize canvas."""
        self.fig, self.ax = plt.subplots(1, 2)

        self.draw_grid(self.ax[0])

        self.draw_prop_space(self.ax[1], None)

    def sample_properties(self, grid_size=(5,5)):
        """Sample the space of continuous properties."""
        n_e = grid_size[0]
        n_d = grid_size[1]
        bnd_e2 = self.mecha.get_prop_bounds(2)
        eps = 2 * self.mecha.constraint_solver.eps

        for e2 in np.linspace(bnd_e2[0], bnd_e2[1] - eps, n_e):
            bnd_d = self.mecha.constraint_solver.get_bounds(
                (self.R, self.S, e2, .2), 3)
            for d in np.linspace(bnd_d[0], bnd_d[1] - eps, n_d):
                yield e2, d

    def draw_grid(self, frame):
        """Draw the grid of figures."""
        frame.set_aspect('equal')
        frame.margins(0.1)
        frame.set_xlabel('e2')
        frame.set_ylabel('d')

        # Draw curves.
        samples = self.sample_properties((self.num_e2_vals, self.num_d_vals))
        for i, c in enumerate(samples):
            position = np.array([i // self.num_d_vals,
                                 i % self.num_d_vals]) * [3, 3]
            self.draw_grid_cell(frame, c, position)

    def draw_grid_cell(self, frame, properties, position):
        """Draw a single cell of the grid."""
        e2, d = properties
        self.mecha.update_prop(2, e2)
        self.mecha.update_prop(3, d)
        curve = self.mecha.get_curve(self.nb)

        curve = curve / abs(curve).max()
        curve = curve + position.reshape(2, 1)

        frame.plot(curve[0], curve[1], 'b-')

    def draw_prop_space(self, frame, samples):
        """Draw the property space and samples."""
        frame.margins(0.1)
        frame.set_xlabel('e2')
        frame.set_ylabel('d')

        bnd_e2 = self.mecha.get_prop_bounds(2)
        pts_top = []
        pts_btm = []
        for e2 in np.linspace(bnd_e2[0], bnd_e2[1], self.num_e2_vals):
            bnd_d = self.mecha.constraint_solver.get_bounds(
                (self.R, self.S, e2, .2), 3)
            pts_btm.append((e2, bnd_d[0]))
            pts_top.append((e2, bnd_d[1]))

        feasible_space = Polygon(pts_btm + pts_top[::-1], alpha=0.5)
        frame.add_patch(feasible_space)

        samples = self.sample_properties((self.num_e2_vals, self.num_d_vals))
        samples = np.array(list(samples)).T
        frame.scatter(samples[0], samples[1], c='r')

        frame.relim()
        frame.autoscale()


def main():
    """Entry point."""
    plt.ioff()

    PropExplorer()

    plt.show()

if __name__ == "__main__":
    main()
