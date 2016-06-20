#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the normalization step.

@author: Robin Roussel
"""

import context

import matplotlib.pyplot as plt
import numpy as np

from curvedistances import CurveDistance
from curvegen import get_curve
import curveplotlib as cplt


def show_normalization(curve):
    """Show the curve normalization."""
    angle = np.pi / 10.
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    transformed = R.dot(curve) * 2. + np.array([[2], [3]])
    normalized = CurveDistance.normalize_pose(transformed)

    plt.figure()
    cplt.plot(transformed, label='input')
    cplt.plot(normalized, label='normalized')
    plt.gca().set_aspect('equal')
    plt.legend(loc='best')
    plt.grid()

def main():
    """Entry point."""
    plt.ioff()

    # Get the reference curve.
    params = (5., 3., 1.5)
#    params = (2., 1., 0.5) # Ellipse
    ref_curve = get_curve(params)

    show_normalization(ref_curve)

    plt.show()

if __name__ == "__main__":
    main()
