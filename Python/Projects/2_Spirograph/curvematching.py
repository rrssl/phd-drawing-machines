# -*- coding: utf-8 -*-
"""
Tools for the matching, retrieval and classification of curves.

@author: Robin Roussel
"""
import numpy as np

import curvegen as cg


class CurveMatcher:
    """Adapter transforming a curve distance into a curve matcher."""
    
    def __init__(self, distance):
        self.distance = distance

    def __call__(self, target_curve, cand_params):
        """Find the candidate best matching the input curve."""
        # Compare each candidate curve.
        distances = np.array([self.distance(cg.get_curve(cand), target_curve)
                              for cand in cand_params])
        return cand_params[np.argsort(distances)[0], :]
        
    
def classify_curve(target_curve, cand_params, curve_matcher, threshold):
    """Find the candidates in the same class as the input curve."""
    # Compare each candidate curve.
    belongs = np.zeros(cand_params.shape[0], dtype=bool)
    for i, cand in enumerate(cand_params):
        # Generate the curve.
        cand_curve = cg.get_curve(cand)
        # Compute the distance.
        dist = curve_matcher(cand_curve, target_curve)
#        print(dist, c)
        belongs[i] = dist <= threshold

    return belongs