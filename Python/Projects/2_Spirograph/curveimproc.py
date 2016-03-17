# -*- coding: utf-8 -*-
"""
Library of functions for curve rasterization, curve extraction from images,
and processing of curve images.

@author: Robin Roussel
"""

import numpy as np
import cv2

def fitInBox(curve, box_limits):
    """ box_limits = (xmin, ymin, xmax, ymax) = [xmin, xmax) x [ymin, ymax) """
    
    # Center the curve.
    curve_average = curve.mean(axis=1)
    centered_curve = curve - curve_average.reshape(2, 1)
    
    # Compute the box center and half-dimensions.
    if len(box_limits) == 2:
        A = np.array([0, 0])
        B = np.array(box_limits)
    elif len(box_limits) == 4:
        A = np.array(box_limits[:2])
        B = np.array(box_limits[2:])    
    box_center = (A + B) * 0.5 - 0.5    # /!\ Note: -0.5 offset for images.
    box_half_dims = (B - A) * 0.5 - 0.5
    
    # Rescale and translate the curve to fit in the box.
    scale = np.diag(box_half_dims / np.abs(centered_curve).max(axis=1))
    return scale.dot(centered_curve) + box_center.reshape(2, 1)
        
def getim(curve, resol):
    # Adapt the curve to be rasterized.
    adapted_curve = fitInBox(curve, resol)
    
    # Create the image.
    img = np.zeros(resol, np.uint8)
    pts = np.int32(adapted_curve.T)
    color = 255
    is_closed = True
    img = cv2.polylines(img, [pts], is_closed, color)
    
    return img
    
def getExtContour(img, filled=True):    
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
    
def getIntContour(img, filled=True):    
    # Add a 1-pixel black border (the outer border is ignored by findContours).
    img2 = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_CONSTANT)
    
    # Find the contours.
    retrieval_mode = cv2.RETR_CCOMP # The 1st contour will be the outer one.
    approx_method = cv2.CHAIN_APPROX_SIMPLE # Cf. getExtContour for other opts.
    _, contours, hierarchy = cv2.findContours(img2, retrieval_mode, approx_method)  
#    contour = contour[0] - 1
    
    # Find (or not) the inner contour containing the center.
    center = (img.shape[0] / 2, img.shape[1] / 2)
    for c in contours[1:]:
        if cv2.pointPolygonTest(c, center, False) > 0:
            contour = c - 1     # Remove the border.
            break
    else:
        contour = None
        print("No interior contour was found for this shape.")
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