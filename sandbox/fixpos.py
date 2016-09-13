# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The position of the corresponding PoIs is constant."
Meaning that the invariant 'feature' of the PoIs here is the (x,y) coordinates:
i.e. for this simple example, curve space and feature space coincide.

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs have the same parameter value."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff t1 = t2.
This simplifies the correspondence tracking for this simple demonstration;
however there is no loss of generality.

Lastly this criterion allows us to use index value as a proxy for parameter
value.

@author: Robin Roussel
"""
import math
import numpy as np
import numpy.polynomial.polynomial as npol
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from scipy.special import ellipe
import scipy.interpolate as interp

import context
from controlpane import ControlPane
from mecha import EllipticSpirograph


def get_corresp_param(ref_crv, ref_par, curves):
    """Return the parameter value of the correspondance in each curve.

    Here this is trivial but with other correspondence criteria the
    algorithm would be more involved.
    """
    return [ref_par] * len(curves)


def get_features(curves, params):
    """Return the list of features f[i] = curves[i](params[i]).

    Trivial because of our choice of features.
    """
    # Can't use numpy here because curves may have different sizes.
    return [crv[:, par] for crv, par in zip(curves, params)]


def get_dist(ref, cand):
    """Get the L2 distance from each cand point to ref point."""
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    return np.linalg.norm(cand - ref.reshape((-1, 1)), axis=0)


def interp2d(x, y, z, gridsize=(200,200)):
    """Interpolate data points on a grid."""
    # Define grid.
    xi = np.linspace(x.min(), x.max(), gridsize[0])
    yi = np.linspace(y.min(), y.max(), gridsize[1])
    # Interpolate values.
    xi, yi = np.meshgrid(xi, yi)
    zi = interp.griddata((x, y), z, (xi, yi), method='cubic')
    return xi, yi, zi


def get_highest_quantile(vals, q=50):
    """Returns the indexes of the q-quantile of highest values."""
    imax = int(len(vals) / q)
    return np.argpartition(-vals, imax)[:imax]


# TODO: make it a parametric curve to be more generic.
def fit_curve(s, d=2, w=None):
    """Apply polynomial least-squares fitting to the input samples.

    Weights, if given, should be positive.
    """
    c = npol.polyfit(s[0], s[1], d, w=w)
    # polyfit returns coeffs in increasing powers, while poly1d expects them
    # in decreasing powers.
    c = c[::-1]
    return np.poly1d(c)


class FixPosDemo:
    """Find the position-invariant subspace."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (5, 3)
        self.cont_prop = (.2, 1.)
        self.num_e2_vals = 20
        self.num_d_vals = 20
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
#        self.nb = 2**5
        # Reference curve and parameter.
        self.ref_crv = self.mecha.get_curve()
        self.ref_par = 0
        # Regression data.
        self.samples = np.array(list(
            self.sample_properties((self.num_e2_vals, self.num_d_vals))
            )).T
        self.scores = self.get_invar_scores()
        # Filter out low scores and fit curve.
        ids = get_highest_quantile(self.scores) # len(ids) sould be > degree+1
        self.inv_crv = fit_curve(self.samples[:, ids], w=self.scores[ids])

        self.init_draw()

    def init_draw(self):
        """Initialize canvas."""
        self.fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(10, 2)
        self.ax = [
            self.fig.add_subplot(gs[:-2, 0]),
            self.fig.add_subplot(gs[:, 1]),
            ]
        self.fig.subplots_adjust(left=0.1, right=0.96, wspace=.3)
        self.cmap = plt.cm.YlGnBu_r

        self.new_crv_plt = None
        self.draw_curve_space(self.ax[0])

        self.new_crv_pos = None
        self.draw_prop_space(self.ax[1])

        self.slider = self.create_slider(gs[-1, 0])

    def create_slider(self, subplot_spec):
        """Create the slider to explore the invariant space."""
        bounds = self.mecha.get_prop_bounds(2)
        data = (None, {'valmin': bounds[0],
                       'valmax': bounds[1],
                       'valinit': self.cont_prop[0],
                       'label': "Subspace\nparametrization"
                      }),
        return ControlPane(self.fig, data, self.update, subplot_spec)

    def update(self, label, value):
        """Update the data."""
        self.mecha.update_prop(2, value)
        self.mecha.update_prop(3, self.inv_crv(value))
        self.redraw()

    def redraw(self):
        """Redraw dynamic elements."""
        new_crv = self.mecha.get_curve()
        self.new_crv_plt.set_data(new_crv[0], new_crv[1])
        self.new_crv_pos.set_offsets([self.mecha.props[2], self.mecha.props[3]])

    def draw_curve_space(self, frame):
        """Draw the curve."""
        frame.set_aspect('equal')
        frame.margins(0.1)
        frame.set_xlabel('$x$')
        frame.set_ylabel('$y$')
        frame.set_title("Curve space (visible in the UI).\n"
                        "The green point of interest is fixed by the user.")
        # Draw reference curve and reference point.
        frame.plot(self.ref_crv[0], self.ref_crv[1], 'b-',
                   label="Reference curve")
        ref_point = self.ref_crv[:, self.ref_par]
        frame.scatter(*ref_point, s=100, c='g', marker='o', edgecolor='w',
                      zorder=3, label="Point of interest")
        # Draw new curve (empty for now).
        self.new_crv_plt = frame.plot([], [], 'r-', label="New curve")[0]
        # Draw the legend.
        frame.legend(loc='upper left', scatterpoints=1, fontsize='medium')

    def draw_prop_space(self, frame):
        """Draw the property space, the samples, and the reference point."""
        frame.margins(0.1)
        frame.set_xlabel('$e^2$')
        frame.set_ylabel('$d$')
        frame.set_title("Continuous property space (hidden).\n"
                        "The invariant subspace is computed.")
        samples = self.samples
        # Draw the boundary.
        frame.add_patch(self.get_bound_poly())
        # Draw the interpolated scores.
        xi, yi, zi = interp2d(samples[0], samples[1], self.scores)
        zi = np.ma.masked_invalid(zi)
        pcol = frame.pcolormesh(xi, yi, zi, cmap=self.cmap)
        # Draw the samples.
        frame.scatter(samples[0], samples[1], marker='+', c='k',
                      label="Samples")
        # Draw the optimal solution.
        e2_an, d_an = self.get_optimal_path()
        frame.plot(e2_an, d_an, 'g--', linewidth=3, label="Optimal sol.")
        # Draw the approximate solution.
        e2_ap, d_ap = self.get_approx_path()
        frame.plot(e2_ap, d_ap, 'b-', linewidth=1, label="Approximate sol.")
        # Draw the position of the reference curve.
        frame.scatter(*self.cont_prop, s=200, c='lightyellow', marker='*',
                      edgecolor='k', zorder=3, label="Reference curve")
        # Draw the position of the new curve.
        self.new_crv_pos = frame.scatter([], [], s=200, c='k', marker='*',
                                         edgecolor='none', zorder=3,
                                         label="New curve")
        # Draw the legend.
        # Dummy plot to avoid the ugly patch symbol of the boundary.
        frame.plot([], [], c='r', linewidth=2, label="Boundary")
        frame.legend(loc='upper left', scatterpoints=1, fontsize='medium')
        # Draw the colorbar.
        cbar = self.fig.colorbar(pcol, ticks=[0.01, 1.])
        cbar.ax.set_ylabel("Invariance score", rotation=270)

    def get_bound_poly(self):
        """Return the boundary of the feasible space as a Patch."""
        bnd_e2 = self.mecha.get_prop_bounds(2)
        pts_top, pts_btm = [], []
        for e2 in np.linspace(bnd_e2[0], bnd_e2[1], self.num_e2_vals):
            bnd_d = self.mecha.constraint_solver.get_bounds(
                self.disc_prop+(e2, .2), 3)
            pts_btm.append((e2, bnd_d[0]))
            pts_top.append((e2, bnd_d[1]))
        return Polygon(pts_btm+pts_top[::-1], alpha=0.9, facecolor='none',
                       edgecolor='r', linewidth=2)

    def get_optimal_path(self):
        """Return the invariant space computed optimalally."""
        bnd_e2 = self.mecha.get_prop_bounds(2)
        e2 = np.linspace(bnd_e2[0], bnd_e2[1], self.num_e2_vals)
        # Sol: r - a(e2) + d = x_ref with 2aE(e2) = pi*req
        x_ref = self.ref_crv[0, self.ref_par]
        r, req = self.disc_prop
        a = math.pi * req / (2 * ellipe(e2))
        d = x_ref - r + a

        return e2, d

    def get_approx_path(self):
        """Return the estimated invariant space."""
        bnd_e2 = self.mecha.get_prop_bounds(2)
        e2 = np.linspace(bnd_e2[0], bnd_e2[1], self.num_e2_vals)
        return e2, self.inv_crv(e2)

#==============================================================================
# The following methods should be extracted from this class.
#==============================================================================
    def sample_properties(self, grid_size=(5,5)):
        """Sample the space of continuous properties."""
        n_e, n_d = grid_size
        bnd_e2 = self.mecha.get_prop_bounds(2)
        eps = 2 * self.mecha.constraint_solver.eps

        for e2 in np.linspace(bnd_e2[0], bnd_e2[1] - eps, n_e):
            bnd_d = self.mecha.constraint_solver.get_bounds(
                self.disc_prop+(e2, .2), 3)
            for d in np.linspace(bnd_d[0], bnd_d[1] - eps, n_d):
                yield e2, d

    def get_invar_scores(self):
        """Return scores of prop. space samples wrt the invariance criterion."""
        curves = [self.ref_crv]
        init = self.mecha.props.copy()
        for e2, d in self.samples.T:
            self.mecha.update_prop(2, e2)
            self.mecha.update_prop(3, d)
            # TODO: only recompute around the reference parameter instead of
            # the whole curve.
            curves.append(self.mecha.get_curve())
        # Reset the mechanism to its initial properties.
        self.mecha.reset(*init)
        # Find the correspondences and get the scores.
        params = [self.ref_par] + get_corresp_param(curves[0], self.ref_par,
                                                    curves[1:])
        feats = np.asarray(get_features(curves, params))
        dists = get_dist(feats[0], feats[1:].T)
        return np.exp(-dists)


def main():
    """Entry point."""
    plt.ioff()

    FixPosDemo()

    plt.show()

if __name__ == "__main__":
    main()
