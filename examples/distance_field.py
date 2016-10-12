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
from curveproc import get_hand_drawn


def show_distance_field(cand_curve, target_curve):
    """Test the distance field descriptor."""
    # Show the candidate curve embedded in the distance field of the target
    # curve.
    df = DistanceField()
    desc = df.get_desc(target_curve)
    shp = desc.shape
    shp = (shp[0]-10, shp[1]-10)
    adapted_cand_curve = fit_in_box(cand_curve, shp)
    adapted_cand_curve += 5

    plt.figure(figsize=(12,12))
    cplt.imshow(desc, adapted_cand_curve)
    plt.gca().lines[0].set_linewidth(2)
    plt.gca().lines[0].set_color('1.')
#    plt.title('Candidate curve in the distance field of the target curve.')

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
    ref_curve = get_hand_drawn(ref_curve, amplitude=0.02, wavelength=0.5, randomness=10)

    # Get the candidate curve.
    params = (5., 3., 1.5)
#    params = (2., 1., 0.5) # Ellipse
    cand_curve = get_curve(params)

    show_distance_field(cand_curve, ref_curve)

    plt.show()

if __name__ == "__main__":
    main()
