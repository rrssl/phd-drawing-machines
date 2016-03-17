# -*- coding: utf-8 -*-
"""
Library of analytical curves.

@author: Robin Roussel
"""

from fractions import Fraction
import numpy as np

class Curve:
    """Base class for curves."""

    def __init__(self, samples):
        self.samples = np.array(samples)

    def getX(self):
        """Get the X coordinates of the curve samples."""
        pass

    def getY(self):
        """Get the Y coordinates of the curve samples."""
        pass


class Hypotrochoid(Curve):
    """Hypotrochoid class (parameters R, r, d)."""

    def __init__(self, samples, ext_gear_radius, int_gear_radius, hole_dist):
        super().__init__(samples)
        self.R = ext_gear_radius
        self.r = int_gear_radius
        self.d = hole_dist

    @staticmethod
    def get_param_combinations(num_R_vals, num_d_vals):
        """Get an array of all possible parameter combinations."""
        combi = np.empty((0, 3))
        for R in range(1, num_R_vals + 1):
            for r in range(1, R):
                if Fraction(R, r).denominator == r: # Avoid repeating patterns
                    for d in np.linspace(0, r, num_d_vals + 1, endpoint=False):
                        if d != 0.: # Exlude d=0 and d=r
                            combi = np.vstack([combi, np.array([R, r, d])])
        return combi

    def getX(self):
        """Get the X coordinates of the curve samples."""
        R = self.R
        r = self.r
        d = self.d
        s = self.samples
        X = (R - r) * np.cos(s) + d * np.cos(s * (R - r) / r)

        return X

    def getY(self):
        """Get the Y coordinates of the curve samples."""
        R = self.R
        r = self.r
        d = self.d
        s = self.samples
        Y = (R - r) * np.sin(s) - d * np.sin(s * (R - r) / r)

        return Y
