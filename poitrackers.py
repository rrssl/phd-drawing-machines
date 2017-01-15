#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoI tracking functions.

@author: Robin Roussel
"""
from itertools import compress
import numpy as np
import scipy.signal as sig

import poly_point_isect as ppi
ppi.USE_DEBUG = False
ppi.USE_IGNORE_SEGMENT_ENDINGS = False
ppi.USE_RETURN_SEGMENT_ID = True

from curveproc import compute_curvature


def get_corresp_krvmax(ref_crv, ref_par, curves):
    """Return the corresponding curvature max + param value(s) in each curve.

    Supports several PoIs.

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
    cor_par = []
    cor_poi = []
    ref_par = np.atleast_1d(ref_par)
    for crv in curves:
        # Extract candidate PoIs' ids.
        krv = compute_curvature(crv)[:-1]
        peaks = sig.argrelmax(krv, mode='wrap')[0]
        # For each reference parameter find the corresponding PoI.
        try:
            crv_pars = [peaks[np.argmin(np.abs(peaks - par))]
                        for par in ref_par]
        except ValueError:
            cor_par.append(None)
            cor_par.append([None] * len(ref_par))
            continue
        if len(crv_pars) == 1:
            crv_pars = crv_pars[0]
        cor_par.append(crv_pars)
        cor_poi.append(crv[:, crv_pars])
    return cor_poi, cor_par


def _get_self_isect(poly):
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


def _get_dist(ref, cand):
    """Get the L2 distance from each cand point to ref point."""
    ref = np.asarray(ref)
    cand = np.asarray(cand)
    return np.linalg.norm(cand - ref.reshape((-1, 1)), axis=0)


def get_corresp_isect(ref_crv, ref_par, curves, loc_size=15):
    """Return the corresponding curvature max + param value(s) in each curve.

    Supports only one PoI.

    Parameters
    ----------
    ref_crv: 2 x N_pts numpy array
        Reference curve.
    ref_par: (int, int)
        Indexes of the PoI in the reference curve.
    curves: sequence of 2 x N_pts_i numpy arrays
        List of curves in which to search for the corresponding PoIs.

    Returns
    -------
    cor_poi: sequence
        N_curves-list of PoIs.
    cor_par: sequence
        N_curves-list of (int, int).
    """
    assert(ref_par[0] != ref_par[1])
    cor_poi = []
    cor_par = []
    ref_par = np.sort(ref_par)
    for crv in curves:
        # Extract candidate PoIs.
        # We only consider candidates in the parametric neighborhood of the
        # reference PoI-- saves a lot of time.
        # We extract a segment aroung each parameter.
        # First we test if segments overlap.
        n = crv.shape[1]
        if (ref_par[0] + loc_size)%n >= (ref_par[1] - loc_size)%n:
            # For now we just fall back to the full curve...
            local = False
            crv_ = crv
        else:
            local = True
            crv_ = np.hstack([
                np.take(crv, range(par-loc_size,par+loc_size), axis=1,
                        mode='wrap')
                for par in ref_par])

        isect, ids = _get_self_isect(crv_)

        if ids is None or not ids.size:
            cor_poi.append(None)
            cor_par.append([None] * len(ref_par))
            continue

        if local:
            # Remove the erroneous intersections.
            # (Because we concatenate the arcs, but the input of get_self_isect
            # is still considered to be a polygon, a 'ghost' edge is added
            # between the end of an arc and the beginning of the next one,
            # adding erroneous intersections. Fortunately we know the id of the
            # ghost segment.
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
        id_ = np.argmin(_get_dist(ref_par, ids))
        cor_par.append(ids[:, id_])
        cor_poi.append(isect[:, id_])

    return cor_poi, cor_par
