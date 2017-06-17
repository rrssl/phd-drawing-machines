#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Smart editor
Note: this module will be deprecated and replaced by 'invarspace.py'. It's only
used by the Spyrou app anymway.

@author: Robin Roussel
"""
from itertools import product
import numpy as np
import scipy.optimize as opt
from wpca import WPCA


def get_dist(ref, cand):
    """Get the L2 distance from each cand point to ref point."""
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    return np.linalg.norm(cand - ref.reshape((-1, 1)), axis=0)


def get_highest_quantile(vals, q=50):
    """Returns the indexes of the q-quantile of highest values."""
    imax = int(len(vals) / q)
    return np.argpartition(-vals, imax)[:imax]


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


class SmartEditor:
    """Smart Editor

    Attributes
    ----------
    mecha: Mechanism
        Instance of Mechanism.
    dprops
    cprops

    nb_crv_pts: int
        Number of points passed to mecha.get_curve()
    pts_per_dim: int
        Number of points sampled per dimension.
    keep_ratio: float
        Fraction of the highest ranking samples to keep for the regression.
    nbhood_size: float
        Relative size of the local neighborhood in property space.

    ref_crv
    ref_poi

    pca: WPCA object
        Instance of WPCA used to fit the linear map.
    phi: callable
        Linear map fitted to the samples [phi(new) = old].
    phi_inv: callable
        Inverse of the linear map fitted to the samples [phi_inv(old) = new].

    cbounds_invar: list of pairs
        Contains the current bounds for each property in the invariant space.
    ndim_invar_space: int
        Number of dimensions of the invariant subspace.
        Depends on the number of algebraic constraints.
    cprops_invar: 1D numpy array
        Current continuous property vector expressed in the new coordinates
        (i.e. the coordinates of the current linear approximation).

    """

    def sample_cprops(self, nb=5, extent=.1):
        """Sample the space of continuous properties.

        Parameters
        ----------
        nb: int
            Number of points _per dimension_.
        extent: float
            Relative size of the neighb. wrt the space between the bounds.
        """
        ids = range(len(self.dprops),
                    len(self.dprops)+len(self.cprops))
        # Get local bounds.
        bnd = np.array([self.mecha.get_prop_bounds(i) for i in ids]).T
        rad = extent * (bnd[1] - bnd[0])
        props = self.mecha.props[ids[0]:]
        l_bnd = np.vstack([props - rad, props + rad])
        # Get valid samples.
        coords = [np.linspace(a, b, nb) for a, b in l_bnd.T]
        condition = lambda p: self.mecha.constraint_solver.check_constraints(
            self.mecha.props[:ids[0]]+list(p))
        samples = filter(condition, product(*coords))
        # TODO: Check/Extract connected region
        return samples

    def get_corresp(self, ref_crv, ref_poi, curves):
        """Return the corresponding PoI + param value(s) in each curve.

        Parameters
        ----------
        ref_crv: 2 x N_pts numpy array
            Reference curve.
        ref_poi: numpy array
            PoI(s) in the reference curve.
        curves: sequence of 2 x N_pts_i numpy arrays
            List of curves in which to search for the corresponding PoIs.

        Returns
        -------
        cor_poi: sequence
            N_curves-list of PoIs.
        """
        raise NotImplementedError

    def get_features(self, curve, poi):
        """Return the features f(curve, poi)."""
        raise NotImplementedError

    def get_invar_scores(self, samples):
        """Return scores of prop. space samples wrt the invariance criterion.

        Parameters
        ----------
        samples: N_samples x N_props iterable
            List of samples to which attribute a score.
        """
        curves = []
        init = self.mecha.props.copy()
        for props in samples:
            for i, prop in enumerate(props):
                self.mecha.update_prop(len(self.dprops)+i, prop,
                                       check=False, update_state=False)
            # TODO: only recompute around the reference parameter instead of
            # the whole curve.
            curves.append(self.mecha.get_curve(self.nb_crv_pts))
        curves.append(self.ref_crv)
        # Reset the mechanism to its initial properties.
        self.mecha.reset(*init)
        # Find the correspondences and get the scores.
        pois = self.get_corresp(curves[-1], self.ref_poi, curves[:-1])
        pois.append(self.ref_poi)
        feats = np.array([self.get_features(crv, poi)
                          for crv, poi in zip(curves, pois)])
        dists = get_dist(feats[-1], feats[:-1].T)
        return np.exp(-dists)

    def compute_invar_space(self):
        """Compute the space corresponding to the invariance criterion."""
        # Regression data.
        samples = np.array(list(
            self.sample_cprops(self.pts_per_dim, self.nbhood_size)
            ))
        scores = self.get_invar_scores(samples)
        # Filter out low scores and find linear map of invariant space.
        ids = get_highest_quantile(scores, q=1/self.keep_ratio)
        if len(ids) < self.ndim_invar_space + 1:
            print("Warning: too few samples for the PCA ({})".format(len(ids)))
        self.phi, self.phi_inv, pca = fit_linear_map(samples[ids], scores[ids],
                                                     self.ndim_invar_space)
        # Ensure consistency between the old and new bases.
        if self.pca is None:
            self.pca = pca
        else:
#            var_scores = pca.explained_variance_ratio_[:self.ndim_invar_space]
#            if anyclose(var_scores):
#                print("Warning: variances are close; PCA axes may swap.")
#                print("Variance ratios: {}".format(var_scores))
            # Project previous components in the new subspace.
            proj = np.dot(np.dot(self.pca.components_, pca.components_.T),
                          pca.components_)
            proj /= np.linalg.norm(proj, axis=1).reshape(-1, 1)
            pca.components_ = proj
            # Make sure that the directions are consistent.
            flip = np.sign(np.diag(np.dot(
                pca.components_, self.pca.components_.T
                ))).reshape(-1, 1)
            pca.components_ *= flip
            self.pca = pca
        # Compute the property vector in the new subspace.
        self.cprops_invar = self.phi_inv(self.cprops).ravel()
        # Redefine bounds.
        self.cbounds_invar = [self.get_bounds_invar_space(i)
                              for i in range(self.ndim_invar_space)]

    def get_bounds_invar_space(self, pid):
        """Find the bounds of the solution subspace.

        This method is only needed if you add dynamic bounds to the solution
        subspace controller.
        """
        assert(0 <= pid < self.ndim_invar_space)

        dp, cpis, phi = self.dprops, self.cprops_invar, self.phi
        def adapt(cstr):
            return lambda x: cstr(
                np.r_[dp, phi(np.r_[cpis[:pid], x, cpis[pid+1:]]).ravel()])

        cs = self.mecha.constraint_solver
        cstrs = cs.get_constraints()
        # Start at 4 to remove constraints on discrete props.
        cstrs = [adapt(cstrs[i]) for i in range(4, len(cstrs))]

        min_ = opt.fmin_cobyla(
            lambda x: x, cpis[pid], cons=cstrs, disp=0) + 2*cs.eps
        max_ = opt.fmin_cobyla(
            lambda x: -x, cpis[pid], cons=cstrs, disp=0) - 2*cs.eps

        return min_, max_

    def project_cont_prop_vect(self):
        """Return the projection of the vector of continuous properties on the
        solution space."""
        dp = self.dprops
        init = np.asarray(self.mecha.props[len(dp):]).ravel()

        ref_feat = self.get_features(self.ref_crv, self.ref_poi)

        cstrs = self.mecha.constraint_solver.get_constraints()
        # Start at 2*n_disc_prop to remove constraints on discrete props.
        def adapt(cstr):
            return lambda p: cstr(np.r_[dp, p])
        cstrs = [adapt(cstrs[i]) for i in range(2*len(dp), len(cstrs))]

        def objective(p):
            self.mecha.reset(*np.r_[dp, p])
            crv = self.mecha.get_curve(self.nb_crv_pts)
            poi = self.get_corresp(self.ref_crv, self.ref_poi, [crv])
            feat = self.get_features(crv, poi[0])
            d_feat = ref_feat - feat
            d_init = p - init
            return np.dot(d_feat, d_feat) + np.dot(d_init, d_init)
        sol = opt.fmin_cobyla(objective, init, cons=cstrs, disp=0)
        self.mecha.reset(*np.r_[dp, init])
        return sol
