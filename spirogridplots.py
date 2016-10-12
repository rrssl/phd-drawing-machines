#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spirograph plotting classes.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
import numpy as np

import curves as cu


class SpiroGridPlot:
    """Plotting of Spirograph figures on a parametric grid."""

    def __init__(self):
        self.plot_increasing_ratios = True
        self.plot_increasing_r = False
        self.draw_grid()

    def draw_grid(self):
        """Draw the grid of figures."""
        num_R_vals = 7
        num_d_vals = 7

        # Compute combinations.
        combi = np.array(list(
            cu.Hypotrochoid.sample_parameters((num_R_vals, num_d_vals))))
        if self.plot_increasing_ratios:
            ratios = (combi[:,1] / combi[:,0]) + 1e-6 * combi[:,2]
            sorted_indices = np.argsort(ratios)
            combi = combi[sorted_indices, :]
            axis_text = r'$\frac{{{1:.0f}}}{{{0:.0f}}}$'
        elif self.plot_increasing_r:
            sorted_indices = np.argsort(
                combi[:,1] + 1e-3 * combi[:,0] + 1e-6 * combi[:,2])
            combi = combi[sorted_indices, :]
            axis_text = '({0:.0f},{1:.0f})'
        else:
            axis_text = '({0:.0f},{1:.0f})'

        # Draw curves.
#        fig = plt.figure(figsize=(8, 6))
        fig = plt.figure(figsize=(16, 12))
        frame = fig.add_subplot(111)
        frame.set_xlim(-3, 3 * (combi.shape[0] // num_d_vals))
        frame.set_aspect('equal')
        frame.set_axis_bgcolor('none')
        frame.set_xlabel('r/R', labelpad=20, fontsize='xx-large')
        plt.tick_params(
            axis='both',            # changes apply to both awes
            which='both',           # both major and minor ticks are affected
            right='off',            # ticks along the bottom edge are off
            top='off',              # ticks along the top edge are off
            labelbottom='on')       # labels along the bottom edge are on
        frame.set_ylabel('d/r', labelpad=20, fontsize='xx-large')
        frame.spines['top'].set_edgecolor('none')
        frame.spines['right'].set_edgecolor('none')

        positions  = np.tile(np.arange(len(combi)), (2,1))
        positions[0] //= num_d_vals
        positions[1] %= num_d_vals
        positions *= np.array([3, 4]).reshape(2, 1)
        xlabels = []
        for i, (p, c) in enumerate(zip(positions.T, combi)):
            if c[1] / c[0] != combi[i - 1, 1] / combi[i - 1, 0]:
                xlabels.append(axis_text.format(*c))
            if c[2]:
                self.draw_grid_cell(c, p)
        frame.set_xticks(positions[0].reshape(-1, num_d_vals)[:, 0])
        frame.set_xticklabels(xlabels, fontsize='xx-large')
        ylabels = [r'${:.2f}$'.format(val) for val in
                   np.linspace(0., 1., num_d_vals, endpoint=False)[1:]]
        frame.set_yticks(positions[1, 1:num_d_vals])
        frame.set_yticklabels(ylabels, fontsize='x-large')

        frame.arrow(51, -.05, .1, 0., width=.01, color="k", clip_on=False,
                    head_width=.5, head_length=.5, lw=.5)
        frame.arrow(-2.97, 25., 0., 1., width=.01, color="k", clip_on=False,
                    head_width=.5, head_length=.5, lw =.5)

    def draw_grid_cell(self, parameters, position):
        """Draw a single cell of the grid."""
        N = parameters[1] if parameters[2] else 1
        nb_samples = N * 2 ** 5 + 1
        interval_length = N * 2 * np.pi
        param_range = np.linspace(0., interval_length, nb_samples)
        curve = cu.Hypotrochoid(*parameters).get_point(param_range)

        curve = curve / abs(curve).max()
        curve = curve + position.reshape(2, 1)

        plt.plot(curve[0], curve[1], 'b-', lw=2)


def main():
    """Entry point."""
    plt.ioff()

    SpiroGridPlot()

    plt.show()


if __name__ == "__main__":
    main()
