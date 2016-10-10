# -*- coding: utf-8 -*-
"""
Finding and tracking points of interest.

A point of interest is stored as a vector of features:
 -- its cartesian coordinates in curve space,
 -- its parameter value(s) (several if it is an intersection point),
 -- some PoI-specific features.

@author: Robin Roussel
"""
from itertools import compress
import numpy as np
import scipy.signal as sig

import poly_point_isect as ppi

from curveproc import compute_curvature

ppi.USE_DEBUG = False
ppi.USE_IGNORE_SEGMENT_ENDINGS = False
ppi.USE_RETURN_SEGMENT_ID = True


def find_krv_max(poly, closed=True):
    """Returns a list of curvature maxima from an input polygon.

    Parameters
    ----------
    poly: numpy array
        N_dim x N_points array.

    Returns
    -------
    pois: numpy array
        N_PoIs x N_features array of PoIs.
    """
    krv = compute_curvature(poly, closed)
    if closed: krv = krv[:-1]
    peaks = sig.argrelmax(krv, mode=('clip','wrap')[closed])[0]

    pois = np.vstack([poly[:, peaks], peaks, krv[peaks]]).T
    return pois


def find_isect(poly):
    """Get the self-intersections of an input polygon.

    We use a version simplified from poitracking.py (we don't need to be as
    robust).

    Parameters
    ----------
    poly: numpy array
        N_dim x N_points array.

    Returns
    -------
    pois: numpy array
        N_PoIs x N_features array of PoIs.
    """
    poly1 = np.asarray(poly)

    # Find intersections.
    inter1, seg_ids = ppi.isect_polygon(poly1.T)
    if not seg_ids:
        return None
    inter1 = np.array(inter1)
    N = poly1.shape[1] - 2
    # Valid points give exactly two non-consecutive segments.
    valid_points = np.array(
        [len(pair) == 2 and abs(pair[0] % N - pair[1] % N) > 1
         for pair in seg_ids])
    seg_ids = np.array(list(compress(seg_ids, valid_points)))
    if not seg_ids.size:
        return None
    else:
        seg_ids.sort(axis=1)

    pois = np.hstack([inter1[valid_points, :], seg_ids])
    return pois


class PoITracker:
    """Tracks corresponding PoIs in the property space of a mechanism."""

    def __init__(self, mecha, ref_par, prop_samples, poi_finder):
        self.mecha = mecha
        self.ref_par = ref_par
        n = len(self.mecha.props)
        self.prop_samples = np.asarray(prop_samples).reshape(-1, n)
        self.poi_finder = poi_finder

        self.loc_radius = np.pi / 2**3
        self.loc_size = 2**4 - 1

    def track(self):
        rho = self.loc_radius
        for props in self.prop_samples:
            self.mecha.reset(*props)
            crv_loc = np.hstack([
                self.mecha.get_range(par-rho, par+rho, self.loc_size)
                for par in self.ref_par
                ])
            pois = self.poi_finder(crv_loc, closed=False)



