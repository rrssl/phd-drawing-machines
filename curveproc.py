# -*- coding: utf-8 -*-
"""
Library of functions for curve processing (smoothing, distorsions,
computations, etc.)

@author: Robin Roussel
"""
from matplotlib.path import Path
import numpy as np
import scipy.interpolate as itp


def compute_curvature(curve, is_closed=True):
    """Compute the curvature along the input curve."""
    if is_closed:
        # Extrapolate so that the curvature at the boundaries is correct.
        curve = np.hstack([curve[:, -3:-1], curve, curve[:, 1:3]])

    dx_dt = np.gradient(curve[0])
    dy_dt = np.gradient(curve[1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (np.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) /
                 (dx_dt * dx_dt + dy_dt * dy_dt)**1.5)

    if is_closed:
        curvature = curvature[2:-2]

    return curvature


def get_hand_drawn(curve, amplitude=0.01, wavelength=0.3, randomness=2):
    """Get a hand-drawn-like version of the input curve."""
    if amplitude == 0.:
        return curve
    width = curve[0].max() - curve[0].min()
    height = curve[1].max() - curve[1].min()
    diag = np.sqrt(width * width + height * height)
    amplitude *= diag
    wavelength *= curve.shape[1]

    path = Path(curve.T)
    path = path.cleaned(sketch=(amplitude, wavelength, randomness))
    curve2 = path.to_polygons()[0].T[:, :-1]
    curve2 = smooth_curve(curve2)

    return curve2


def smooth_curve(curve, sample_fraction=2, step=3):
    """Return a smoothed, optionally up-sampled version of the input curve."""
    # Sample the curve points.
    size = curve.shape[1]
    # Version 1: evenly spaced sample.
    curve = curve[:, ::sample_fraction]
    # Version 2: random sample (keeping the first and last points).
#    ids = np.arange(size)
#    ids = np.sort(
#        np.random.choice(ids, size / sample_fraction, replace=False))
#    ids[0] = 0
#    ids[-1] = size - 1
#    curve = curve[:, ids]

    # Use a parametric representation {(x(t), y(t)), t in [0, 1]} of the curve.
    t = np.zeros(curve.shape[1])
    # Version 1: Range
#    t = np.arange(x.shape[0], dtype=float)
    # Version 2: Distance
    t[1:] = np.linalg.norm(curve.T[:-1] - curve.T[1:], axis=1)
    t = np.cumsum(t)
    t /= t[-1]
    # Fit 1D splines to x and y, and sample them.
    nt = np.linspace(0, 1, size * step)
    x2 = itp.spline(t, curve[0], nt)
    y2 = itp.spline(t, curve[1], nt)

    return np.vstack([x2, y2])
