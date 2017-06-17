# -*- coding: utf-8 -*-
"""
Dynamically computing the invariant space of a given feature.

@author: Robin Roussel
"""
from itertools import product
import numpy as np
import scipy.optimize as opt
from wpca import WPCA


def anyclose(l, tol=1e-1):
    """Check if one element from the list is sufficiently close to another."""
    l = np.asarray(l)
    d = np.abs(l - l[:, np.newaxis])
    d = d[np.triu_indices_from(d, 1)]
    return np.any(d <= tol)


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
    s : N_samples x N_dims numpy array
        Samples.
    w : N_samples numpy array, optional
        Weights.
    ndim : int, optional
        Number of dimensions _of the new basis_.

    Returns
    -------
    phi : callable
        Linear map from the new basis to the old one.
    phi_inv : callable
        Inverse of phi.
    pca : WPCA
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


class InvariantSpaceFinder:
    """Find the invariant space of a constraint applied to mechanisms with 2
    continuous properties or more.

    Parameters
    ----------
    mecha_type : Mechanism
        Type of mechanism (not an instance).
    props : sequence
        Sequence of initial values for the mechanism's properties.
    init_poi_id : int
        Index of the point of interest when the machine is in its initial
        configuration.
    get_corresp : callable, optional
        Tracking function. See the 'get_corresp' method for the expected
        signature. This method should either be given as a parameter here, or
        inherit this class and override the method.
    get_features : callable, optional
        Function computing the features at the PoI of a given curve. See the
        'get_features' method for the expected signature. This method
        should either be given as a parameter here, or inherit this class and
        override the method.
    pts_per_dim : int, optional
        Number of points sampled per dimension. Default is 5.
    keep_ratio : float, optional
        Fraction of the highest ranking samples to be kept for the invariant
        space regression. If the PCA complains about not having enough samples,
        try increasing it. Default is 0.05.
    nbhood_size : float, optional
        Size of the local neighborhood used for sampling the valid property
        space, relative to its extent along each dimension. Default is 0.1.
    ndim_invar_space : int, optional
        Number of dimensions of the invariant subspace. Depends on the number
        of algebraic constraints. Fixed by hand in this version. Default is 2.
    nb_crv_pts : int, optional
        Density of points sampled along each curve. Passed to the 'get_curve'
        method of each mechanism. What 'density' means kind of depends on the
        parameterization of the mechanism. Default is 2**6.

    Attributes
    ----------
    phi: callable
        Linear map fitted to the samples [phi(new) = old].
    phi_inv : callable
        Inverse of the linear map fitted to the samples [phi_inv(old) = new].
    pca : WPCA object
        Instance of WPCA used to fit the linear map.
    cont_prop_invar_space : 1D numpy array
        Current continuous property vector expressed in the new coordinates
        (i.e. the coordinates of the current linear approximation).
    """
    def __init__(self, mecha_type, props, init_poi_id, get_corresp=None,
                 get_features=None, pts_per_dim=5, keep_ratio=.05,
                 nbhood_size=.1, ndim_invar_space=2, nb_crv_pts=2**6):
        # Initial parameters.
        nb_dprops = mecha_type.ConstraintSolver.nb_dprops
        self.disc_prop = props[:nb_dprops]
        self.cont_prop = props[nb_dprops:]
        self.pts_per_dim = pts_per_dim
        self.keep_ratio = keep_ratio
        self.nbhood_size = nbhood_size
        self.ndim_invar_space = ndim_invar_space
        self.mecha = mecha_type(*props)
        self.nb_crv_pts = nb_crv_pts
        self.labels = mecha_type.param_names[nb_dprops:]  # TODO: move to GUI
        if get_corresp is not None:
            self.get_corresp = get_corresp
        if get_features is not None:
            self.get_features = get_features
        # Reference curve and parameter(s).
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
        self.ref_par = init_poi_id
        self.ref_poi, self.ref_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
        # New curve and parameter(s).
        self.new_crv = None
        self.new_poi = None
        self.new_par = None
        # Solution space.
        self.phi = None
        self.phi_inv = None
        self.pca = None
        self.new_cont_prop = None  # TODO: either use it or remove it
        self.invar_space_bnds = None  # TODO: either use it or remove it
        self.compute_invar_space()

    def compute_invar_space(self):
        # Regression data.
        samples = np.array(list(
            self.sample_props(self.pts_per_dim, self.nbhood_size)
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
        self.cont_prop_invar_space = self.phi_inv(self.cont_prop).ravel()
        # Redefine bounds.
        self.bnds_invar_space = [self.get_bounds_invar_space(i)
                                 for i in range(self.ndim_invar_space)]

    def get_bounds_invar_space(self, pid):
        """Return the bounds of the solution space for parameter pid."""
        assert(0 <= pid < self.ndim_invar_space)
        # TODO: use the same optimizations as in ConstraintSolver.get_bounds
        dp, cpis, phi = self.disc_prop, self.cont_prop_invar_space, self.phi

        def adapt(cstr):
            return lambda x: cstr(
                np.r_[dp, phi(np.r_[cpis[:pid], x, cpis[pid+1:]]).ravel()])

        cs = self.mecha.constraint_solver
        cstrs = cs.get_constraints()
        # Start at 4 to remove constraints on discrete props.
        cstrs = [adapt(cstrs[i]) for i in range(len(dp), len(cstrs))]

        min_ = opt.fmin_cobyla(
            lambda x: x, cpis[pid], cons=cstrs, disp=0) + 2*cs.eps
        max_ = opt.fmin_cobyla(
            lambda x: -x, cpis[pid], cons=cstrs, disp=0) - 2*cs.eps

        return min_, max_

    def get_corresp(self, ref_crv, ref_par, curves):
        """Return the corresponding PoI + param value(s) in each curve.

        Parameters
        ----------
        ref_crv: 2 x N_pts numpy array
            Reference curve.
        ref_par: int or sequence of int
            Index(es) of the PoI(s) in the reference curve.
        curves: sequence of 2 x N_pts_i numpy arrays
            List of curves in which to search for the corresponding PoIs.

        Returns
        -------
        cor_poi: sequence
            N_curves-list of PoIs.
        cor_par: sequence
            N_curves-list of N_ref_par-lists.
        """
        raise NotImplementedError

    def get_features(self, curve, param, poi):
        """Return the features f(curve(param), poi)."""
        raise NotImplementedError

    def get_invar_scores(self, samples):
        """Return scores of prop. space samples wrt the invariance criterion.

        Parameters
        ----------
        samples: N_samples x N_props sequence
            List of samples to which attribute a score.
        """
        curves = []
        init = self.mecha.props.copy()
        for props in samples:
            for i, prop in enumerate(props):
                self.mecha.update_prop(len(self.disc_prop)+i, prop,
                                       check=False, update_state=False)
            # TODO: only recompute around the reference parameter instead of
            # the whole curve.
            curves.append(self.mecha.get_curve(self.nb_crv_pts))
        curves.append(self.ref_crv)
        # Reset the mechanism to its initial properties.
        self.mecha.reset(*init)
        # Find the correspondences and get the scores.
        pois, pars = self.get_corresp(curves[-1], self.ref_par, curves[:-1])
        pois.append(self.ref_poi)
        pars.append(self.ref_par)
        feats = np.array([self.get_features(crv, par, poi)
                          for crv, par, poi in zip(curves, pars, pois)])
        dists = get_dist(feats[-1], feats[:-1].T)
        return np.exp(-dists)

    def project_cont_prop_vect(self):
        """Return the projection of the vector of continuous properties on the
        solution space."""
        dp = list(self.disc_prop)
        init = np.asarray(self.mecha.props[len(dp):]).ravel()

        ref_feat = self.get_features(self.ref_crv, self.ref_par, self.ref_poi)

        cstrs = self.mecha.constraint_solver.get_constraints()
        # Start at 2*n_disc_prop to remove constraints on discrete props.
        # TODO: use the same optimizations as in ConstraintSolver.get_bounds

        def adapt(cstr):
            return lambda p: cstr(np.r_[dp, p])
        cstrs = [adapt(cstrs[i]) for i in range(2*len(dp), len(cstrs))]

        def objective(p):
            self.mecha.reset(*dp+list(p))
            crv = self.mecha.get_curve(self.nb_crv_pts)
            poi, par = self.get_corresp(self.ref_crv, self.ref_par, [crv])
            feat = self.get_features(crv, par[0], poi[0])
            d_feat = ref_feat - feat
            d_init = p - init
            return np.dot(d_feat, d_feat) + np.dot(d_init, d_init)

        valid = False
        optinit = init.copy()
        while not valid:
            sol = opt.fmin_cobyla(objective, optinit, cons=cstrs, disp=0)
            valid = self.mecha.reset(*dp+list(sol))
            if not valid:
                optinit = sol
        # Reset to initial state.
        self.mecha.reset(*dp+list(init))
        return sol

    def sample_props(self, nb=5, extent=.1):
        """Sample the space of continuous properties.

        Parameters
        ----------
        nb: int, optional
            Number of points _per dimension_. Default is 5.
        extent: float, optional
            Relative size of the neighb. wrt the space between the bounds.
            Default is 0.1.
        """
        ids = range(len(self.disc_prop),
                    len(self.disc_prop)+len(self.cont_prop))
        # Get local bounds.
        bnd = np.array([self.mecha.get_prop_bounds(i) for i in ids]).T
        rad = extent * (bnd[1] - bnd[0])
        props = self.mecha.props[ids[0]:]
        l_bnd = np.vstack([props - rad, props + rad])
        # Get valid samples.
        coords = [np.linspace(a, b, nb) for a, b in l_bnd.T]

        # TODO: now that constraints are vectorized, use numpy here
        def condition(p):
            return self.mecha.constraint_solver.check_constraints(
                self.mecha.props[:ids[0]]+list(p))
        samples = filter(condition, product(*coords))
        # TODO: Check/Extract connected region
        return samples

    def set_cont_prop(self, props):
        """Set the continuous property vector, update data."""
        self.cont_prop = props
        # We need to update all the parameters before getting the bounds.
        self.mecha.reset(*list(self.disc_prop)+list(props))
        # Update new curve and PoI.
        self.new_crv = self.mecha.get_curve(self.nb_crv_pts)
        new_poi, new_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.new_crv])
        self.new_poi = new_poi[0]
        self.new_par = new_par[0]
