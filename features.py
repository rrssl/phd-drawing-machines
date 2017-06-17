# -*- coding: utf-8 -*-
"""
Methods computing features on a point of interest.

@author: Robin Roussel
"""
import numpy as np

from curveproc import compute_curvature

# TODO: harmonize PoI structure to that defined in 'pois.py'.
# In the current version, 'param' and 'poi' have to be passed separately, and
# some useful features, like curvature, have to be recomputed whereas they were
# already computed previously. In the new version (the one in 'pois.py'),
# useful values are computed only once (during PoI detection) and concatenated
# in a single vector.


def get_position(curve, param, poi):
    return poi


def get_curvature(curve, param, poi):
    return compute_curvature(curve)[param]


def get_angle(curve, param, poi):
    return np.arctan2(poi[1], poi[0])


def get_curvature_and_angle(curve, param, poi):
    return np.r_[compute_curvature(curve)[param] * 3e-2,
                 np.arctan2(poi[1], poi[0])]
#    return np.r_[compute_curvature(curve)[param], np.arctan2(poi[1], poi[0])]


def get_isect_angle(curve, param, poi):
    if poi is None or not poi.size:
        feats = np.full(2, 1e6)
    else:
        curve = curve[:, :-1]  # Remove last point
        n = curve.shape[1]
        param = np.asarray(param)
        v = curve[:, (param+1) % n] - curve[:, param % n]
        v /= np.linalg.norm(v, axis=0)
        feats = v[:, 1] - v[:, 0]
    return feats


def get_dist(curve, param, poi):
    diff = poi[:, 1] - poi[:, 0]
    return diff[0]**2 + diff[1]**2


def get_dist_and_krv_diff(curve, param, poi):
    krv = compute_curvature(curve)[param]
    diffpos = (poi[:, 1] - poi[:, 0]) / poi[:, 0]
    diffkrv = (krv[1] - krv[0]) / krv[0]
    return diffpos[0]**2 + diffpos[1]**2 + diffkrv**2
