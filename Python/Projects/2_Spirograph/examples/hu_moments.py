#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the Hu Moments dissimilarity measure.

@author: Robin Roussel
"""

import context

import matplotlib.pyplot as plt
import numpy as np

import curvedistances as cdist
from curvegen import get_curve
import curveplotlib as cplt


def show_hu_moments(cand_curve, target_curve):
    """Test the Hu moments descriptor."""
    # Compute the Hu moments of the full curve.
    hm = cdist.HuMoments()
    mom_full = hm.get_desc(target_curve)
    # Compute the distance.
    hu_dist = hm.get_dist(cand_curve, target_curve)
    print("Distance between Hu Moments: {}".format(hu_dist))

    # Compute the Hu moments of the target external contour.
    hm.contour_method = cdist.USE_EXT_CONTOUR
    mom_ext = hm.get_desc(target_curve)
    # Compute the distance.
    hu_dist = hm.get_dist(cand_curve, target_curve)
    print("Distance between external Hu Moments: {}".format(hu_dist))

    # Compute the Hu moments of the target internal contour.
    hm.contour_method = cdist.USE_INT_CONTOUR
    mom_int = hm.get_desc(target_curve)
    # Compute the distance.
    hu_dist = hm.get_dist(cand_curve, target_curve)
    print("Distance between internal Hu Moments: {}".format(hu_dist))

    # Compute the Hu moments of the intext contour.
    hm.contour_method = cdist.USE_INTEXT_CONTOUR
    mom_intext = hm.get_desc(target_curve)
    # Compute the distance.
    hu_dist = hm.get_dist(cand_curve, target_curve)
    print("Distance between internal Hu Moments: {}".format(hu_dist))

    # Display the moments.
    fig = plt.figure()
    plt.suptitle("Absolute value of log10 of the Hu moments.")
    ax = fig.add_subplot(221, title="Full curve.")
    try:
        plt.bar(np.arange(len(mom_full)) + .6, abs(mom_full), log=False)
    except ValueError:
        print('Error: ', mom_full)
    fig.add_subplot(222, sharex=ax, sharey=ax, title="Exterior contour.")
    plt.bar(np.arange(len(mom_ext)) + .6, abs(mom_ext), log=False)
    fig.add_subplot(223, sharex=ax, sharey=ax, title="Interior contour.")
    plt.bar(np.arange(len(mom_int)) + .6, abs(mom_int), log=False)
    fig.add_subplot(224, sharex=ax, sharey=ax, title="Difference of contours.")
    plt.bar(np.arange(len(mom_intext)) + .6, abs(mom_intext), log=False)


def main():
    """Entry point."""
    if not cplt.CV2_IMPORTED:
        raise ImportError("OpenCV module could not be imported.")

    plt.ioff()

    # Get the reference curve.
    params = (5., 3., 1.5)
    ref_curve = get_curve(params)

    # Get the candidate curve.
    params = (5., 3., 2.5)
#    params = (2., 1., 0.5) # Ellipse
    cand_curve = get_curve(params)

    show_hu_moments(cand_curve, ref_curve)

    plt.show()

if __name__ == "__main__":
    main()
