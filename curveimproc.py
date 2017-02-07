# -*- coding: utf-8 -*-
"""
Library of functions for curve rasterization, curve extraction from images,
and processing of curve images.

@author: Robin Roussel
"""

import numpy as np
import cv2

def fit_in_box(curve, box_limits):
    """Resize and translate the input curves to fit inside the box limits.

    box_limits = (xmax, ymax) => [0, xmax) x [0, ymax)
                OR
               = (xmin, ymin, xmax, ymax) => [xmin, xmax) x [ymin, ymax)
    """
    # Compute the box center and half-dimensions.
    if len(box_limits) == 2:
        A = np.array([0, 0])
        B = np.array(box_limits)
    elif len(box_limits) == 4:
        A = np.array(box_limits[:2])
        B = np.array(box_limits[2:])
    box_center = (A + B) * 0.5 - 0.5    # /!\ Note: -0.5 offset for images.
    box_half_dims = (B - A) * 0.5 - 0.5

    # Compute transform
    curve = np.asarray(curve)
    if len(curve.shape) == 2 and curve.shape[0] == 2:
        ctr = curve.mean(axis=1).reshape(2, 1)
        den = np.abs(curve - ctr).max(axis=1)
    elif len(curve.shape) == 3 and curve.shape[1] == 2:
        join = np.hstack(curve)
        ctr = join.mean(axis=1).reshape(2, 1)
        den = np.abs(join - ctr).max(axis=1)
    den[den == 0.] = 1.
    scale = (box_half_dims / den).reshape(2, 1)

    return scale * (curve - ctr) + box_center.reshape(2, 1)

def getim(curve, resol):
    """Rasterize the input curve to the given resolution."""
    # Adapt the curve to be rasterized.
    adapted_curve = fit_in_box(curve, resol)

    # Create the image.
    img = np.zeros(resol, np.uint8)
    curve = np.asarray(curve)
    if len(curve.shape) == 2 and curve.shape[0] == 2:
        pts = [np.int32(adapted_curve.T)]
    elif len(curve.shape) == 3 and curve.shape[1] == 2:
        pts = [np.int32(arc.T) for arc in adapted_curve]
    color = 255
    is_closed = False

    return cv2.polylines(img, pts, is_closed, color)

def get_ext_contour(img, filled=True):
    """Compute and return the external contour of a binary image."""
    # Add a 1-pixel black border (the outer border is ignored by findContours).
    img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

    # Find the external contour.
    retrieval_mode = cv2.RETR_EXTERNAL
    approx_method = cv2.CHAIN_APPROX_SIMPLE # Other methods:
                                            # CHAIN_APPROX_TC89_L1,
                                            # CHAIN_APPROX_TC89_KCOS
    _, contour, _ = cv2.findContours(img2, retrieval_mode, approx_method)
    contour = contour[0] - 1    # Remove the border.

    # Draw its image.
    out = np.zeros(img.shape, np.uint8)
    contour_id = -1 # All contours, i.e. the only one.
    color = 255
    if filled:
        thickness = cv2.FILLED
    else:
        thickness = 1
    cv2.drawContours(out, [contour], contour_id, color, thickness)

    return out

def get_int_contour(img, filled=True):
    """Compute and return the internal contour of a binary image."""
    # Add a 1-pixel black border (the outer border is ignored by findContours).
    img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)

    # Find the contours.
    retrieval_mode = cv2.RETR_CCOMP # The 1st contour will be the outer one.
    approx_method = cv2.CHAIN_APPROX_SIMPLE # Cf. get_ext_contour for other opts.
    _, contours, _ = cv2.findContours(img2, retrieval_mode, approx_method)
#    contour = contour[0] - 1

    # Find (or not) the inner contour containing the center.
    center = (img.shape[0] / 2, img.shape[1] / 2)
    for c in contours[1:]:
        if cv2.pointPolygonTest(c, center, False) > 0:
            contour = c - 1     # Remove the border.
            break
    else:
        contour = None
#        print("No interior contour was found for this shape.")
        # Typically the case for (7, 5, 2)

    # Draw its image.
    out = np.zeros(img.shape, np.uint8)
    if contour is not None:
        contour_id = 0 # All contours, i.e. the only one.
        color = 255
        if filled:
            thickness = cv2.FILLED
        else:
            thickness = 1
        cv2.drawContours(out, [contour], contour_id, color, thickness)

    return out
