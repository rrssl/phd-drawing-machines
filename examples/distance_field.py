#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the distance field dissimilarity measure.

@author: Robin Roussel
"""

import context

import matplotlib.pyplot as plt

from curvedistances import DistanceField
from curvegen import get_curve
from curveimproc import fit_in_box
import curveplotlib as cplt


def show_distance_field(cand_curve, target_curve):
    """Test the distance field descriptor."""
    # Show the candidate curve embedded in the distance field of the target
    # curve.
    df = DistanceField()
    desc = df.get_desc(target_curve)
    adapted_cand_curve = fit_in_box(cand_curve, desc.shape)

    plt.figure()
    cplt.imshow(desc, adapted_cand_curve)
    plt.title('Candidate curve in the distance field of the target curve.')

    # Compute the DF-distance.
    df_dist = df.get_dist(adapted_cand_curve, target_curve)
    print("DF-distance: {}".format(df_dist))


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

    show_distance_field(cand_curve, ref_curve)

    plt.show()

if __name__ == "__main__":
    main()
