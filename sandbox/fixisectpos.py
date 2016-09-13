# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The position of the corresponding PoIs is constant."
Except this time, the PoI is an intersection point.

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs are the closest PoIs of the same type."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

Lastly we use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
#import math
from itertools import compress
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

#import matplotlib as mpl
#mpl.rcParams['toolbar'] = 'None'

import context
from controlpane import ControlPane
from mecha import EllipticSpirograph
from fixpos import get_dist, interp2d, fit_curve, get_highest_quantile
#from poitracking import get_self_isect
#from isect import get_seg_intersection

import poly_point_isect as ppi
ppi.USE_DEBUG = False
ppi.USE_IGNORE_SEGMENT_ENDINGS = False
ppi.USE_RETURN_SEGMENT_ID = True


def get_self_isect(poly):
    """Get self-intersections.

    We use a version simplified from poitracking.py (we don't need to be as
    robust).
    """
    poly1 = np.asarray(poly)

    # Find intersections.
    inter1, seg_ids = ppi.isect_polygon(poly1.T)
    if not seg_ids:
        return None, None
    inter1 = np.array(inter1).T
    N = poly1.shape[1] - 2
    # Valid points give exactly two non-consecutive segments.
    valid_points = np.array(
        [len(pair) == 2 and abs(pair[0] % N - pair[1] % N) > 1
         for pair in seg_ids])
    seg_ids = np.array(list(compress(seg_ids, valid_points))).T

    return inter1[:, valid_points], seg_ids


def get_corresp(ref_crv, ref_par, curves):
    """Return the corresponding PoI + param value(s) in each curve.

    ref_par is a list of indexes.

    Returns
     -- an N_curves-list of PoIs,
     -- an N_curves-list of N_ref_par-lists.
    """
    cor_poi = []
    cor_par = []
    loc_size = 15
    ref_par = np.sort(ref_par)
    for crv in curves:
        # Extract candidate PoIs.
        # We only consider candidates in the parametric neighborhood of the
        # reference PoI-- saves a lot of time.
        n = crv.shape[1]
        # We get a concatenation of segments of size 2*loc_size.
        crv_loc = np.hstack([crv[:, par-loc_size:(par+loc_size)%n]
                             for par in ref_par])

        isect, ids = get_self_isect(crv_loc)

        if ids is None or not ids.size:
            cor_poi.append(None)
            cor_par.append([None] * len(ref_par))
            continue
        # Remove the erroneous intersections.
        # (Because we concatenate the arcs, but the input of get_self_isect
        # is still considered to be a polygon, a 'ghost' edge is added between
        # the end of an arc and the beginning of the next one, adding
        # erroneous intersections. Fortunately we know the id of the ghost
        # segment.
        valid = (ids % (2*loc_size - 1) != 0).all(axis=0)
        if not valid.any():
            cor_poi.append(None)
            cor_par.append([None] * len(ref_par))
            continue
        ids = ids[:, valid]
        isect = isect[:, valid]
        # Convert the ids back to their absolute value in the curve.
        pos = ids // (2*loc_size)
        ids += ref_par[pos] - (1 + pos*2)*loc_size
        # Find the corresponding PoI  and param values.
        ids = np.sort(ids, axis=0)
        id_ = np.argmin(get_dist(ref_par, ids))
        cor_par.append(ids[:, id_])
        cor_poi.append(isect[:, id_])

    return cor_poi, cor_par


def get_features(curves, params, pois):
    """Return the list of features f[i] = f(curves[i](params[i]), pois[i])."""
    feats = [np.array([1e6, 1e6]) if poi is None else poi for poi in pois]
    return feats


class FixIsectPosDemo:
    """Find the subspace where the PoIs coincide."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (5, 3)
        self.cont_prop = (.31, .48) # Quasi zero angle between segments
        self.num_e2_vals = 15
        self.num_d_vals = 25
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
#        self.nb = 2**5
        # Reference curve and parameter(s).
        self.ref_crv = self.mecha.get_curve()
#        self.ref_par = (11, 117)
        self.ref_par = (53, 267)
        self.ref_poi, self.ref_par = get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
        print(self.ref_par)
        # New curve and parameter(s).
        self.new_crv = None
        self.new_poi = None
        # Regression data.
        self.samples = np.array(list(
            self.sample_properties((self.num_e2_vals, self.num_d_vals))
            )).T
        self.scores = self.get_invar_scores()
        # Filter out low scores and fit curve.
        ids = get_highest_quantile(self.scores, q=40) # len(ids) sould be > degree+1
        print(len(ids))
        self.inv_crv = fit_curve(
            np.hstack([self.samples[:, ids], self.ref_par.reshape(-1, 1)]),
            w=np.hstack([self.scores[ids], 1.]),
            d=4)
        # Redefine bounds.
        self.inv_crv_bnds = (self.samples[0, ids].min(),
                             self.samples[0, ids].max())
        print(self.inv_crv_bnds)
        # Optimal solution.
        self.opt_path = np.asarray(self.get_optimal_path())
        self.inv_crv_opt = interp.interp1d(*self.opt_path)

        self.init_draw()
#        self.make_gif()

    def make_gif(self):
        """Export demo as a GIF."""
        e2 = np.linspace(*self.inv_crv_bnds, num=15)
        for i, val in enumerate(e2):
            self.control.sliders[0].set_val(val)
            plt.savefig("fixisectpos_explore_{0:02d}".format(i))

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
        self.new_poi = None
        self.draw_curve_space(self.ax[0])

        self.new_crv_pos = None
        self.draw_prop_space(self.ax[1])

        self.control = self.create_slider(gs[-1, 0])

    def create_slider(self, subplot_spec):
        """Create the slider to explore the invariant space."""
#        bounds = self.mecha.get_prop_bounds(2)
        bounds = self.inv_crv_bnds
        data = (
            ('app', {'valmin': bounds[0],
                     'valmax': bounds[1],
                     'valinit': self.cont_prop[0],
                     'label': "Subspace\nparametrization"
                    }),
            ('opt', {'valmin': bounds[0],
                     'valmax': bounds[1],
                     'valinit': self.cont_prop[0],
                     'label': "'Optimal'"
                    })
            )
        return ControlPane(self.fig, data, self.update, subplot_spec)

    def update(self, id_, value):
        """Update the data."""
        if id_ == 'app':
            inv_crv = self.inv_crv
        elif id_ == 'opt':
            inv_crv = self.inv_crv_opt

        self.mecha.update_prop(2, value)
        self.mecha.update_prop(3, inv_crv(value))
        self.new_crv = self.mecha.get_curve()
        self.new_poi = get_corresp(
            self.ref_crv, self.ref_par, [self.new_crv])[0][0]
        self.redraw()

    def redraw(self):
        """Redraw dynamic elements."""
        self.new_crv_plt.set_data(self.new_crv[0], self.new_crv[1])
        self.new_crv_pos.set_offsets([self.mecha.props[2], self.mecha.props[3]])
        self.new_poi_plt.set_offsets([self.new_poi])

    def draw_curve_space(self, frame):
        """Draw the curve."""
        frame.set_aspect('equal')
        frame.margins(0.3)
        frame.set_xlabel('$x$')
        frame.set_ylabel('$y$')
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's position is fixed by the "
                        "user.")
        # Draw the reference curve...
        frame.plot(self.ref_crv[0], self.ref_crv[1], 'b-', alpha=1.,
                   label="Reference curve")
        ref_point = self.ref_crv[:, self.ref_par]
        # ... and its point of interest.
        frame.scatter(*ref_point, s=100, c='b', marker='o', edgecolor='w',
                      zorder=3, label="Ref. pts of interest")
        # Draw the new curve (empty for now)...
        self.new_crv_plt = frame.plot([], [], 'r-', alpha=1.,
                                      label="New curve")[0]
        # ... and its point of interest.
        self.new_poi_plt = frame.scatter([], [], s=100, c='r', marker='o',
                                         edgecolor='w', zorder=3,
                                         label="New pts of interest")
        # Draw the legend.
        frame.legend(loc='upper left', scatterpoints=1, ncol=2,
                     fontsize='medium')

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
        frame.plot(self.opt_path[0], self.opt_path[1], 'g--', linewidth=3,
                   label="Optimal sol.")
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
        cbar = self.fig.colorbar(pcol, ticks=[1e-2, 1.])
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
        """Return the invariant space computed optimally."""
#        bnd_e2 = self.mecha.get_prop_bounds(2)
        bnd_e2 = self.inv_crv_bnds
        e2 = np.linspace(*bnd_e2, num=self.num_e2_vals)
        init = self.mecha.props.copy()
        ref_ft = get_features([self.ref_crv], [self.ref_par], [self.ref_poi])[0]
        print("Ref. feature", ref_ft)
        d = []
        for val in e2:
            self.mecha.update_prop(2, val)
            def obj_func(x):
                self.mecha.update_prop(3, x[0])
                crv = self.mecha.get_curve()[:, :-1]
                poi, par = get_corresp(self.ref_crv, self.ref_par, [crv])
                ft = get_features([crv], par, poi)[0]
                return np.linalg.norm(ft - ref_ft)
            bnd_d = list(self.mecha.constraint_solver.get_bounds(
                self.disc_prop+(val, None), 3))
            # Adjust upper bound (solver tends to exceed it slightly).
            bnd_d[1] -= 1e4 * self.mecha.constraint_solver.eps
            d.append(
                opt.minimize(
                    obj_func, init[3], method='L-BFGS-B', bounds=[bnd_d]).x)
#                opt.fsolve(obj_func, init[3]))
        self.mecha.reset(*init)
        return e2, d

    def get_approx_path(self):
        """Return the estimated invariant space."""
        e2 = np.linspace(*self.inv_crv_bnds, num=self.num_e2_vals)
        d = self.inv_crv(e2)
        # TODO: make it clean, call get_prop_bounds
        valid = np.logical_and(self.samples[1].min() <= d,
                               d <= self.samples[1].max())
        return e2[valid], d[valid]

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
                self.disc_prop+(e2, None), 3)
            for d in np.linspace(bnd_d[0], bnd_d[1] - eps, n_d):
                yield e2, d

    def get_invar_scores(self):
        """Return scores of prop. space samples wrt the invariance criterion."""
        curves = []
        init = self.mecha.props.copy()
        for e2, d in self.samples.T:
            self.mecha.update_prop(2, e2)
            self.mecha.update_prop(3, d)
            # TODO: only recompute around the reference parameter instead of
            # the whole curve.
            curves.append(self.mecha.get_curve())
        curves.append(self.ref_crv)
        # Reset the mechanism to its initial properties.
        self.mecha.reset(*init)
        # Find the correspondences and get the scores.
        poi, par = get_corresp(curves[-1], self.ref_par, curves[:-1])
        poi.append(self.ref_poi)
        par.append(self.ref_par)
        feats = np.asarray(get_features(curves, par, poi))
        dists = get_dist(feats[-1], feats[:-1].T)
        return np.exp(-dists)


def main():
    """Entry point."""
    plt.ioff()

    FixIsectPosDemo()

#    plt.pause(1)
    plt.show()

if __name__ == "__main__":
    main()
