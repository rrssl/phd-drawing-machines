#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the extraction of spirographic contours.

@author: Robin Roussel
"""

import context

import matplotlib.pyplot as plt

from curvegen import get_curve
import curveimproc as cimp
import curveplotlib as cplt

def show_contours(img):
    """Show the different available versions of contour extraction."""

    fig = plt.figure()

    fig.add_subplot(332, title="Source image.")
    cplt.imshow(img)

    ext_ctr = cimp.get_ext_contour(img, filled=False)
    int_ctr = cimp.get_int_contour(img, filled=False)

    fig.add_subplot(334, title="External contour.")
    cplt.imshow(ext_ctr)
    fig.add_subplot(335, title="Internal contour.")
    cplt.imshow(int_ctr)
    fig.add_subplot(336, title="Union of the two contours.")
    cplt.imshow(ext_ctr + int_ctr)

    ext_ctr = cimp.get_ext_contour(img, filled=True)
    int_ctr = cimp.get_int_contour(img, filled=True)

    fig.add_subplot(337, title="Filled external contour.")
    cplt.imshow(ext_ctr)
    fig.add_subplot(338, title="Filled internal contour.")
    cplt.imshow(int_ctr)
    fig.add_subplot(339, title="Difference of the two contours.")
    cplt.imshow(ext_ctr - int_ctr)

def main():
    """Entry point."""
    if not cplt.CV2_IMPORTED:
        raise ImportError("OpenCV module could not be imported.")

    plt.ioff()

    # Generate the curve image.
    params = (5., 3., 2.5)
#    params = (2., 1., 0.5) # Ellipse
    cand_curve = get_curve(params)
    shp = (512, 512)
    cand_img = cimp.getim(cand_curve, shp)

    show_contours(cand_img)

    plt.show()

if __name__ == "__main__":
    main()
