# -*- coding: utf-8 -*-
"""
Features corresponding to each type of invariant. for corresponding PoIs.

@author: Robin Roussel
"""
from enum import Enum
import numpy as np

Invariants = Enum('Invariants',
                  'position curvature intersection_angle distance on_line')

def get_feature(invar):
    if invar is Invariants.position:
        return Position
    if invar is Invariants.curvature:
        return Curvature
    if invar is Invariants.intersection_angle:
        return IntersectionAngle
    if invar is Invariants.distance:
        return SignedDistance
    if invar is Invariants.on_line:
        return OnRadialLine


defaultval = 1e6


class Position:
    def __call__(self, crv, poi):
        return np.full(2, defaultval) if poi is None else poi[:2]


class Curvature:
    def __call__(self, crv, poi):
        return np.full(1, defaultval) if poi is None else poi[3]


class IntersectionAngle:
    def __call__(self, crv, poi):
        if poi is None:
            return np.full(2, defaultval)
        n = crv.shape[1]
        par = poi[2:].astype(int)
        v = crv[:, (par+1)%n] - crv[:, par%n]
        v /= np.linalg.norm(v, axis=0)
        return v[:, 1] - v[:, 0]


class SignedDistance:
    def __call__(self, crv, poi):
        return np.full(2, defaultval) if poi is None else poi[0] - poi[1]


class OnRadialLine:
    def __call__(self, crv, poi):
        return np.full(1, defaultval) if poi is None else np.arctan2(poi[1],
                                                                     poi[0])
