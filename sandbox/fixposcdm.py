# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The position of the PoI is constant."

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs are the closest PoIs of the same type."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

Lastly we use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
#import math
from itertools import product
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
#from matplotlib.patches import Polygon
import numpy as np
#import scipy.interpolate as interp
import scipy.optimize as opt
from wpca import WPCA

#import matplotlib as mpl
#mpl.rcParams['toolbar'] = 'None'

import context
from controlpane import ControlPane
from mecha import SingleGearFixedFulcrumCDM
from fixpos import get_dist, get_highest_quantile#, interp2d
#from fixisectpos import get_corresp

def get_corresp(ref_crv, ref_par, curves):
    """Return the corresponding PoI + param value(s) in each curve.

    Parameters
    ----------
    ref_par: sequence
        List of indexes.

    Returns
    -------
    cor_poi: sequence
        N_curves-list of PoIs.
    cor_par: sequence
        N_curves-list of N_ref_par-lists.
    """
    cor_poi = [crv[:, ref_par] for crv in curves]
    cor_par = [ref_par] * len(curves)

    return cor_poi, cor_par


def get_features(curves, params, pois):
    """Return the list of features f[i] = f(curves[i](params[i]), pois[i])."""
    return pois


def fit_linear_map(s, w=None, ndim=2):
    """Fit a linear model to the (optionally) weighted samples.

    Parameters
    ----------
    s: N_samples x N_dims numpy array
        Samples.
    w: N_samples numpy array, optional
        Weights.
    ndim: int, optional
        Number of dimensions _of the new basis_.

    Returns
    -------
    phi: callable
        Linear map from the new basis to the old one.
    phi_inv: callable
        Inverse of phi.
    pca: WPCA
        Instance of WPCA used for the fitting.
    """
    assert(s.shape[0] == w.size)
    pca = WPCA(n_components=ndim)
    if w is not None:
        w = np.tile(w.reshape(-1, 1), (1, s.shape[1]))
    pca.fit(s, weights=w)

    def phi(p):
        """Transform data from the new space to the old.

        Accepts a N_samples x N_new_dims input.
        """
        return pca.inverse_transform(np.asarray(p).reshape(-1, ndim))
    def phi_inv(p):
        """Transform data from the old space to the new.

        Accepts a N_samples x N_old_dims input.
        """
        return pca.transform(np.asarray(p).reshape(-1, s.shape[1]))

    return phi, phi_inv, pca


def anyclose(l, tol=1e-1):
    """Check if one element from the list is sufficiently close to another."""
    l = np.asarray(l)
    d = np.abs(l - l[:, np.newaxis])
    d = d[np.triu_indices_from(d, 1)]
    return np.any(d <= tol)


def make_frames_from_slider(filename, slider, nb_frames=15):
    """Save a batch of figure screenshots from successive slider values."""
    if slider.slidermin is not None:
        min_ = slider.slidermin.val
    else:
        min_ = slider.valmin
    if slider.slidermax is not None:
        min_ = slider.slidermax.val
    else:
        max_ = slider.valmax
    rng = np.linspace(min_, max_, nb_frames)
    for i, val in enumerate(rng):
        slider.set_val(val)
        plt.savefig("{}_{0:02d}".format(filename, i))


class FixPosCDM:
    """Find the subspace where the PoIs coincide."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (4, 3)
        self.cont_prop = (7., 2.4, 6., 1.5)
        self.pts_per_dim = 5
        self.nbhood_size = .1
        self.ndim_invar_space = 4
        self.mecha = SingleGearFixedFulcrumCDM(*self.disc_prop+self.cont_prop)
#        self.nb = 2**5
        # Reference curve and parameter(s).
        self.ref_crv = self.mecha.get_curve()
        print(self.ref_crv.shape)
        self.ref_par = 0
#        self.ref_par = (98, 183) # Intersection
        self.ref_poi, self.ref_par = get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
        print(self.ref_par)
        # New curve and parameter(s).
        self.new_crv = None
        self.new_poi = None
        # Solution space.
        self.phi = None
        self.phi_inv = None
        self.pca = None
        self.new_cont_prop = None
        self.invar_space_bnds = None
        self.compute_local_invar_space()
#        # Optimal solution.
#        self.opt_path = np.asarray(self.get_optimal_path())
#        self.inv_crv_opt = interp.interp1d(*self.opt_path)
#
        self.init_draw()

        # Controller
        self.slider_active = False
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)

#==============================================================================
# Model
#==============================================================================

    def compute_local_invar_space(self):
        """Compute the local solution space for the geometric invariant."""
        # Regression data.
        samples = np.array(list(
            self.sample_local_props(self.pts_per_dim, self.nbhood_size)
            ))
        scores = self.get_invar_scores(samples)
        # Filter out low scores and find linear map of invariant space.
        ids = get_highest_quantile(scores, q=20)
        if len(ids) < self.ndim_invar_space + 1:
            print("Warning: too few samples for the PCA ({})".format(len(ids)))
        self.phi, self.phi_inv, pca = fit_linear_map(samples[ids], scores[ids],
                                                     self.ndim_invar_space)
        # Ensure consistency between the old and new bases.
        if self.pca is None:
            self.pca = pca
        else:
            var_scores = pca.explained_variance_ratio_[:self.ndim_invar_space]
            if anyclose(var_scores):
                # TODO: in case of ambiguity, consider the last base vector
                # explored, x_i. Find x_j', a vector from the new base, such that
                # the cross product of x_i and x_j' is minimized. Permutate vectors
                # in the new base so that x_j' has now the index i.
                print("Warning: variances are close; PCA axes may swap.")
                print("Variance ratios: {}".format(var_scores))
            # Make sure that the directions are consistent.
            flip = np.sign(np.diag(np.dot(
                pca.components_, self.pca.components_.T
                ))).reshape(-1, 1)
            pca.components_ *= flip
            self.pca = pca
        # Compute the property vector in the new subspace.
        self.new_cont_prop = self.phi_inv(self.cont_prop).ravel()
        # Redefine bounds.
        self.invar_space_bnds = [(-2., 2.)]*len(self.new_cont_prop)

    def set_cont_prop_vect(self, cont_prop):
        """Set the continuous property vector, update data."""
        # We can't update everything in a single loop: we need to update all
        # the parameters before getting the bounds.
        for i, prop in enumerate(cont_prop):
            self.mecha.update_prop(2+i, prop)
            self.ref_control.set_val(i, prop)
        for i in range(len(cont_prop)):
            bnds = self.mecha.get_prop_bounds(i+2)
            self.ref_control.set_bounds(i, bnds)
        for i in range(len(self.new_cont_prop)):
            bnds = self.get_new_bounds(i)
            self.control.set_bounds(i, bnds)

        self.new_crv = self.mecha.get_curve()
        self.new_poi = get_corresp(
            self.ref_crv, self.ref_par, [self.new_crv])[0][0]

    def project_cont_prop_vect(self):
        """Project the continuous property vector on the solution space."""
        dp = self.disc_prop
        init = np.asarray(self.mecha.props[2:]).ravel()

        ref_feat = get_features(self.ref_crv, self.ref_par, self.ref_poi)

        cs = self.mecha.constraint_solver
        cstrs = cs.get_constraints()
        # Start at 4 to remove constraints on discrete props.
        def adapt(cstr):
            return lambda p: cstr(np.r_[dp, p])
        cstrs = [adapt(cstrs[i]) for i in range(4, len(cstrs))]

        def objective(p):
            self.mecha.reset(*np.r_[dp, p])
            crv = self.mecha.get_curve()
            poi, par = get_corresp(self.ref_crv, self.ref_par, [crv])
            feat = get_features(crv, par, poi)[0]
            d_feat = ref_feat - feat
            d_init = p - init
            return np.dot(d_feat, d_feat) + np.dot(d_init, d_init)
        return opt.fmin_cobyla(objective, init, cons=cstrs, disp=0)

    def get_new_bounds(self, pid):
        """Find the bounds in the solution subspace."""
        assert(0 <= pid < len(self.new_cont_prop))

        dp, ncp, phi = self.disc_prop, self.new_cont_prop, self.phi
        def adapt(cstr):
            return lambda x: cstr(
                np.r_[dp, phi(np.r_[ncp[:pid], x, ncp[pid+1:]]).ravel()])

        cs = self.mecha.constraint_solver
        cstrs = cs.get_constraints()
        # Start at 4 to remove constraints on discrete props.
        cstrs = [adapt(cstrs[i]) for i in range(4, len(cstrs))]

        min_ = opt.fmin_cobyla(
            lambda x: x, ncp[pid], cons=cstrs, disp=0) + 2*cs.eps
        max_ = opt.fmin_cobyla(
            lambda x: -x, ncp[pid], cons=cstrs, disp=0) - 2*cs.eps

        return min_, max_

    def sample_local_props(self, nb=5, extent=.1):
        """Locally sample the space of continuous properties.

        nb: number of points _per dimension_.
        extent: relative size of the neighb. wrt the space between the bounds.
        """
        # Get local bounds.
        bnd = np.array([self.mecha.get_prop_bounds(i) for i in range(2, 6)]).T
        rad = extent * (bnd[1] - bnd[0])
        props = self.mecha.props[2:]
        l_bnd = np.vstack([props - rad, props + rad])
        # Get valid samples.
        coords = [np.linspace(a, b, nb) for a, b in l_bnd.T]
        condition = lambda p: self.mecha.constraint_solver.check_constraints(
            self.mecha.props[:2]+list(p))
        samples = filter(condition, product(*coords))
        # TODO: Check/Extract connected region
        return samples

    def get_invar_scores(self, samples):
        """Return scores of prop. space samples wrt the invariance criterion.

        'samples' is a N_samples x N_props iterable.
        """
        curves = []
        init = self.mecha.props.copy()
        for props in samples:
            for i, prop in enumerate(props):
                self.mecha.update_prop(2+i, prop)
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

#==============================================================================
# View
#==============================================================================

    def init_draw(self):
        """Initialize canvas."""
        self.fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(12, 2)
        self.ax = [
            self.fig.add_subplot(gs[:-3, 0]),
#            self.fig.add_subplot(gs[:-3, 1]),
            ]
        self.fig.subplots_adjust(left=0.1, right=0.96, wspace=.3)
        self.cmap = plt.cm.YlGnBu_r

        self.new_crv_plt = None
        self.new_poi = None
        self.draw_curve_space(self.ax[0])

#        self.new_crv_pos = None
#        self.draw_prop_space(self.ax[1])

        self.control = self.create_controls(gs[-2:, 0])
        self.ref_control = self.create_ref_controls(gs[-2:, 1])

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
        self.new_crv_plt = frame.plot([], [], 'g-', alpha=1.,
                                      label="New curve")[0]
        # ... and its point of interest.
        self.new_poi_plt = frame.scatter([], [], s=100, c='g', marker='o',
                                         edgecolor='w', zorder=3,
                                         label="New pts of interest")
        # Draw the legend.
        frame.legend(loc='upper left', scatterpoints=1, ncol=2,
                     fontsize='medium')

    def create_controls(self, subplot_spec):
        """Create the slider to explore the invariant space."""
        bounds = [self.get_new_bounds(i)
                  for i in range(len(self.new_cont_prop))]
        data = [
            (i, {'valmin': self.invar_space_bnds[i][0],
                 'valmax': self.invar_space_bnds[i][1],
                 'valinit': self.new_cont_prop[i],
                 'label': "$x_{}$".format(i+1)
                 })
            for i in range(len(self.new_cont_prop))
            ]

        return ControlPane(self.fig, data, self.on_slider_update, subplot_spec,
                           bounds=bounds)

    def create_ref_controls(self, subplot_spec):
        """Create the slider to show the corresponding original properties."""
        data = []
        bounds = []
        labels = ["$d_f$", r"$ \theta_g$", "$d_p$", "$d_s$"]
        for i in range(len(self.cont_prop)):
            bounds.append(self.mecha.get_prop_bounds(i+2))
            data.append(
                (i, {'valmin': .5*bounds[-1][0],
                     'valmax': 2.*bounds[-1][1],
                     'valinit': self.cont_prop[i],
                     'label': labels[i],
                     'color': 'black'
                     })
                )
        cp = ControlPane(self.fig, data, None, subplot_spec, bounds=bounds)
        for _, s in cp.sliders.items():
            s.active = False
            s.eventson = False
            s.drawon = False

        return cp

    def redraw(self):
        """Redraw dynamic elements."""
        self.new_crv_plt.set_data(self.new_crv[0], self.new_crv[1])
#        self.new_crv_pos.set_offsets([self.mecha.props[2], self.mecha.props[3]])
        self.new_poi_plt.set_offsets([self.new_poi])

        self.fig.canvas.draw()

#    def draw_prop_space(self, frame):
#        """Draw the property space, the samples, and the reference point."""
#        frame.margins(0.1)
#        frame.set_xlabel('$e^2$')
#        frame.set_ylabel('$d$')
#        frame.set_title("Continuous property space (hidden).\n"
#                        "The invariant subspace is computed.")
#        samples = self.samples
#        # Draw the boundary.
#        frame.add_patch(self.get_bound_poly())
#        # Draw the interpolated scores.
#        xi, yi, zi = interp2d(samples[0], samples[1], self.scores)
#        zi = np.ma.masked_invalid(zi)
#        pcol = frame.pcolormesh(xi, yi, zi, cmap=self.cmap)
#        # Draw the samples.
#        frame.scatter(samples[0], samples[1], marker='+', c='k',
#                      label="Samples")
#        # Draw the optimal solution.
#        frame.plot(self.opt_path[0], self.opt_path[1], 'g--', linewidth=3,
#                   label="Optimal sol.")
#        # Draw the approximate solution.
#        e2_ap, d_ap = self.get_approx_path()
#        frame.plot(e2_ap, d_ap, 'b-', linewidth=1, label="Approximate sol.")
#        # Draw the position of the reference curve.
#        frame.scatter(*self.cont_prop, s=200, c='lightyellow', marker='*',
#                      edgecolor='k', zorder=3, label="Reference curve")
#        # Draw the position of the new curve.
#        self.new_crv_pos = frame.scatter([], [], s=200, c='k', marker='*',
#                                         edgecolor='none', zorder=3,
#                                         label="New curve")
#        # Draw the legend.
#        # Dummy plot to avoid the ugly patch symbol of the boundary.
#        frame.plot([], [], c='r', linewidth=2, label="Boundary")
#        frame.legend(loc='upper left', scatterpoints=1, fontsize='medium')
#        # Draw the colorbar.
#        cbar = self.fig.colorbar(pcol, ticks=[1e-2, 1.])
#        cbar.ax.set_ylabel("Invariance score", rotation=270)
#
#    def get_bound_poly(self):
#        """Return the boundary of the feasible space as a Patch."""
#        bnd_e2 = self.mecha.get_prop_bounds(2)
#        pts_top, pts_btm = [], []
#        for e2 in np.linspace(bnd_e2[0], bnd_e2[1], self.grid_size):
#            bnd_d = self.mecha.constraint_solver.get_bounds(
#                self.disc_prop+(e2, .2), 3)
#            pts_btm.append((e2, bnd_d[0]))
#            pts_top.append((e2, bnd_d[1]))
#        return Polygon(pts_btm+pts_top[::-1], alpha=0.9, facecolor='none',
#                       edgecolor='r', linewidth=2)
#
#    def get_optimal_path(self):
#        """Return the invariant space computed optimally."""
##        bnd_e2 = self.mecha.get_prop_bounds(2)
#        bnd_e2 = self.inv_crv_bnds
#        e2 = np.linspace(*bnd_e2, num=self.grid_size)
#        init = self.mecha.props.copy()
#        ref_ft = get_features([self.ref_crv], [self.ref_par], [self.ref_poi])[0]
#        print("Ref. feature", ref_ft)
#        d = []
#        for val in e2:
#            self.mecha.update_prop(2, val)
#            def obj_func(x):
#                self.mecha.update_prop(3, x[0])
#                crv = self.mecha.get_curve()[:, :-1]
#                poi, par = get_corresp(self.ref_crv, self.ref_par, [crv])
#                ft = get_features([crv], par, poi)[0]
#                return np.linalg.norm(ft - ref_ft)
#            bnd_d = list(self.mecha.constraint_solver.get_bounds(
#                self.disc_prop+(val, None), 3))
#            # Adjust upper bound (solver tends to exceed it slightly).
#            bnd_d[1] -= 1e4 * self.mecha.constraint_solver.eps
#            d.append(
#                opt.minimize(
#                    obj_func, init[3], method='L-BFGS-B', bounds=[bnd_d]).x)
##                opt.fsolve(obj_func, init[3]))
#        self.mecha.reset(*init)
#        return e2, d
#
#    def get_approx_path(self):
#        """Return the estimated invariant space."""
#        e2 = np.linspace(*self.inv_crv_bnds, num=self.grid_size)
#        d = self.inv_crv(e2)
#        # TODO: make it clean, call get_prop_bounds
#        valid = np.logical_and(self.samples[1].min() <= d,
#                               d <= self.samples[1].max())
#        return e2[valid], d[valid]

#==============================================================================
# Controller
#==============================================================================

    def on_slider_update(self, id_, value):
        """Callback function for slider update."""
        self.slider_active = True

        self.new_cont_prop[id_] = value
        cont_prop = self.phi(self.new_cont_prop).ravel()
        self.set_cont_prop_vect(cont_prop)

        self.redraw()

    def on_release(self, event):
        """Callback function for mouse button release."""
        if self.slider_active:
            self.slider_active = False

            cont_prop = self.project_cont_prop_vect()
            self.compute_local_invar_space()
            self.set_cont_prop_vect(cont_prop)
            self.new_cont_prop = self.phi_inv(cont_prop).ravel()
            for i, val in enumerate(self.new_cont_prop):
                self.control.set_val(i, val, incognito=True)

            self.redraw()


def main():
    """Entry point."""
    plt.ioff()

    FixPosCDM()

#    plt.pause(1)
    plt.show()

if __name__ == "__main__":
    main()
