# -*- coding: utf-8 -*-
"""
Testing self-intersection detection algorithms.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
import numpy as np

import context
from mecha import BaseSpirograph

import poly_point_isect as ppi
#ppi.USE_PARANOID = True
#ppi.USE_VERBOSE = True

def itv_overlap(i1, i2):
    """Check whether 2 intervals (as pairs of values) overlap."""
    i1min, i1max, i2min, i2max = min(i1), max(i1), min(i2), max(i2)
    return (i2min <= i1[0] <= i2max or
            i2min <= i1[1] <= i2max or
            i1min <= i2[0] <= i1max)    # 4th case is redundant.


def get_orientation(A, B, C):
    """Get the orientation of triangle ABC.

    Returns:
     -- 1. if the orientation is counterclockwise,
     -- -1. if the orientation is clockwise,
     -- 0. if the points are collinear.
    """
    # The orientation is sign(det(AB, AC)).
    return np.sign(np.linalg.det(np.array([B-A, C-A])))


def seg_intersect(s1, s2):
    """Check whether 2 segments intersect.

    Using int-to-bool casting you can use it as a simple True/False function,
    or have a more detailed look at the intersection type:
     -- 1 if two segments intersect,
     -- 2 if they intersect AND are collinear,
     -- 0 otherwise.
    """
    orient = np.array([
        get_orientation(s1[0], s2[0], s2[1]),
        get_orientation(s1[1], s2[0], s2[1]),
        get_orientation(s2[0], s1[0], s1[1]),
        get_orientation(s2[1], s1[0], s1[1])
        ])
    # General case.
    if (orient[0] != orient[1]) and (orient[2] != orient[3]):
        return 1
    # Specific collinearity cases.
    elif np.count_nonzero(orient) < 3:
        # All 4 points are aligned if there's at least 2 zero determinants.
        if (itv_overlap((s1[0][0], s1[1][0]), (s2[0][0], s2[1][0])) and
            itv_overlap((s1[0][1], s1[1][1]), (s2[0][1], s2[1][1]))):
            # Segments intersect if their x and y projections overlap.
            return 2
    else:
        return 0


def get_seg_intersection(s1, s2):
    """Get the intersection of s1 and s2, assuming they are not collinear."""
    s1_vect = s1[1] - s1[0]
    s2_vect = s2[1] - s2[0]
    s0_vect = s1[0] - s2[0]
    t = np.linalg.det((s2_vect, s0_vect)) / np.linalg.det((s1_vect, s2_vect))
    return s1[0] + t * s1_vect


def get_seg_intersection_collinear_no_overlap(s1, s2):
    """Get the intersection of s1 and s2, collinear but not overlapping."""
    # We only want non-overlapping segments. This means that for two segments
    # projected on the x or y axis, the maximum value of one interval is equal
    # to the minimum value of another.
    s1xmin, s1xmax = min(s1[0][0], s1[1][0]), max(s1[0][0], s1[1][0])
    s2xmin, s2xmax = min(s2[0][0], s2[1][0]), max(s2[0][0], s2[1][0])
    s1ymin, s1ymax = min(s1[0][1], s1[1][1]), max(s1[0][1], s1[1][1])
    s2ymin, s2ymax = min(s2[0][1], s2[1][1]), max(s2[0][1], s2[1][1])
    if s1xmax == s2xmin:
        if s1ymax == s2ymin:
            return np.array([s1xmax, s1ymax])
        if s1ymin == s2ymax:
            return np.array([s1xmax, s1ymin])
    if s1xmin == s2xmax:
        if s1ymax == s2ymin:
            return np.array([s1xmin, s1ymax])
        if s1ymin == s2ymax:
            return np.array([s1xmin, s1ymin])
    return None


def find_self_inter_v1(poly):
    """Find self-intersections in the input closed polygon."""
    points = []

    for i in range(poly.shape[-1] - 1):     # nb. segments = nb. points - 1
        s1 = (poly[:, i], poly[:, i + 1])

        for j in range(i + 2, poly.shape[-1] - (1 if i else 2)):
            s2 = (poly[:, j], poly[:, j + 1])

            intersection_type = seg_intersect(s1, s2)
            if intersection_type == 1:
                points.append(get_seg_intersection(s1, s2))
            elif intersection_type == 2:
                # Segments are collinear.
                print('This message is just to check '
                      'if this corner case really ever happens.')
                I = get_seg_intersection_collinear_no_overlap(s1, s2)
                if I is not None:
                    points.append(I)
            else:
                pass
    return np.array(points).T


def find_self_inter_v2(poly):
    """Find self-intersections in the input closed polygon."""
    ppi.USE_DEBUG = False
    ppi.USE_IGNORE_SEGMENT_ENDINGS = False
    ppi.USE_VERBOSE = False
    poly = np.asarray(poly).T
    return np.array(ppi.isect_polygon(poly)).T


class SelfIntersectionFinder:
    """Finds self-intersections."""

    def __init__(self):
        self.mecha = BaseSpirograph(8.,5.,2.)
#        self.mecha = BaseSpirograph(4.,3.,2.) # Buggy case
        self.nb = 2**6
        self.init_draw()

    def init_draw(self):
        """Initialize the canvas."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.margins(0.1)

        curve = self.mecha.get_curve(self.nb)
        self.plot = self.ax.plot(curve[0], curve[1], c='b')[0]

        print("Number of points: ", curve.shape[1])
        import time

        start = time.clock()
        inter = find_self_inter_v1(curve)
        end = time.clock()
        print("Version 1 - Elapsed time: ", end - start)

        if len(inter):
            plt.scatter(inter[0], inter[1], c='g', edgecolor='none', s=100)

        start = time.clock()
        inter = find_self_inter_v2(curve)
        end = time.clock()
        print("Version 2 - Elapsed time: ", end - start)
        if len(inter):
            plt.scatter(inter[0], inter[1], c='r', marker='*',
                        edgecolor='none', s=100)


def main():
    """Entry point."""
    plt.ioff()

    SelfIntersectionFinder()

    plt.show()

if __name__ == "__main__":
    main()
