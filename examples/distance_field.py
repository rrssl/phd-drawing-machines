#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the distance field dissimilarity measure.

@author: Robin Roussel
"""

import matplotlib.pyplot as plt

import context
from curvegen import get_curve
import curveplotlib as cplt
from curveproc import get_hand_drawn


def main():
    """Entry point."""
    if not cplt.CV2_IMPORTED:
        raise ImportError("OpenCV module could not be imported.")

    plt.ioff()

    # Get the reference curve.
    params = (5., 3., 1.5)
    ref_curve = get_curve(params)
    ref_curve = get_hand_drawn(ref_curve, amplitude=0.02, wavelength=0.5,
                               randomness=10)

    # Get the candidate curve.
    params = (5., 3., 1.5)
#    params = (2., 1., 0.5) # Ellipse
    cand_curve = get_curve(params)

    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    cplt.distshow(ax, ref_curve, cand_curve)

    plt.show()

if __name__ == "__main__":
    main()
