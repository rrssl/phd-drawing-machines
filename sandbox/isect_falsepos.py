# -*- coding: utf-8 -*-
"""
Demonstrating false positives obtained with the Bentley-Ottman implementation,
and how  we can try to filter them out.

@author: Robin Roussel
"""
import math
import matplotlib.pyplot as plt
import numpy as np

import context
import poly_point_isect as ppi
from controlpane import ControlPane
from mecha import BaseSpirograph


ppi.USE_DEBUG = False
ppi.USE_IGNORE_SEGMENT_ENDINGS = False
ppi.USE_VERBOSE = False


def get_self_isect(poly):
    """Get self-intersections."""
    poly = np.asarray(poly).T
    return np.array(ppi.isect_polygon(poly)).T

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def get_self_isect_polar(poly):
    """Get self-intersections in polar space."""
    poly = np.asarray(poly)
    poly = np.vstack(cart2pol(poly[0], poly[1])).T
    isects = np.array(ppi.isect_polygon(poly)).T
    return np.vstack(pol2cart(isects[0], isects[1]))

def get_self_isect_robust(poly):
    """Get self_intersections in a more robust way.

    Our first approach is to run the sweep line algorithm on the initial
    polygon; then run it on its image reflected along the first bisector; then
    reflect the intersections found for the second polygon; then take the
    intersection of both results.
    While this seems efficient, it still fails to remove false positives near
    points of high curvature, especially if
        -- the curve displays a symmetry wrt the first bisector
        -- some points of high curvature lie on the first bisector
    because of course in that case our reflection does not change anything.

    However, since points of high curvature are points of interest, this means
    that those erroneous intersections will get lumped together with them,
    so this is less of a problem.
    """
    poly1 = np.asarray(poly)
    # Reflect polygon along first bisector.
    poly2 = poly1[[1, 0], :]

    inter1 = np.array(ppi.isect_polygon(poly1.T)).T
    inter2 = np.array(ppi.isect_polygon(poly2.T)).T
    # De-reflect intersection points.
    inter2 = inter2[[1, 0], :]

    # Find all (x,y) pairs which are both in inter1 and inter2.
    valid_points = np.logical_and(
        np.in1d(inter1[0], inter2[0]),
        np.in1d(inter1[1], inter2[1]))
    if not valid_points.all():
        print('False positive')

    return inter1[:, valid_points]


def get_self_isect_robust_alternative(poly):
    """Get self_intersections in another robust way.

    Rather than reflecting, here we rotate the auxiliary polygon by a small
    quantity; however the implementation is slightly more complicated (this
    introduces numerical differences between intersections found in both
    cases, therefore we need to use '< eps' rather than '=='.)
    """
    poly1 = np.asarray(poly)
    # Slightly rotate polygon.
    angle = math.pi * 0.001
    cos = math.cos(angle)
    sin = math.sin(angle)
    rot = np.array([[cos, -sin],
                    [sin,  cos]])
    poly2 = rot.dot(poly1)

    inter1 = np.array(ppi.isect_polygon(poly1.T)).T
    inter2 = np.array(ppi.isect_polygon(poly2.T)).T
    # De-rotate intersection points.
    inter2 = rot.T.dot(inter2)

    # Find all (x,y) pairs which are both in inter1 and inter2.
    eps = 1e-9
    valid_points = np.logical_and(
        [(inter2[0] - val < eps).any() for val in inter1[0]],
        [(inter2[1] - val < eps).any() for val in inter1[1]])
    if not valid_points.all():
        print('False positive')

    return inter1[:, valid_points]


def get_self_isect_super_robust(poly):
    """Get self_intersections in a more more robust way.

    This time we extend the 'robust' way to look directly at the data structure
    returned by the algorithm. We rely on a trick specific to our cyclic
    curves: no two consecutive segments intersect.

    This function has worked in all cases that were tested so far.
    """
    poly1 = np.asarray(poly)
    # Reflect polygon along first bisector.
    poly2 = poly1[[1, 0], :]

    # Find intersections.
    ppi.USE_RETURN_SEGMENT_ID = True
    inter1, seg_ids = ppi.isect_polygon(poly1.T)
    ppi.USE_RETURN_SEGMENT_ID = False
    inter1 = np.array(inter1).T
    N = poly1.shape[1] - 2
    # Valid points give exactly two non-consecutive segments.
    valid_points = np.array(
        [len(pair) == 2 and abs(pair[0] % N - pair[1] % N) > 1
         for pair in seg_ids])

    inter2 = np.array(ppi.isect_polygon(poly2.T)).T
    # De-reflect intersection points.
    try:
        inter2 = inter2[[1, 0], :]
    except IndexError:
        return None, None

    # Find all (x,y) pairs which are both in inter1 and inter2.
    valid_points = np.logical_and(
        valid_points,
        np.logical_and(
            np.in1d(inter1[0], inter2[0]),
            np.in1d(inter1[1], inter2[1])))
    if not valid_points.all():
        print('False positive')

    return inter1[:, valid_points]


class SelfIntersectionFinder:
    """Finds self-intersections."""

    def __init__(self):
        self.mecha = BaseSpirograph(8.,3.,1.)
        self.pid = 2
        bounds = self.mecha.get_prop_bounds(self.pid)
        self.prop_data = (self.pid, {'valmin': bounds[0],
                                     'valmax': bounds[1],
                                     'valinit': self.mecha.props[2],
                                     'label': "Pole dist."
                                     }),

        self.curve = self.mecha.get_curve()
        self.isect = get_self_isect_super_robust(self.curve)

        self.init_draw()

    def init_draw(self):
        """Initialize canvas and draw POIs."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.margins(0.1)

        self.control_pane = ControlPane(self.fig, self.prop_data, self.update)

        self.plot = self.ax.plot(self.curve[0], self.curve[1], c='b')[0]

        if self.isect is not None:
            self.isect_plt = self.ax.scatter(
                self.isect[0], self.isect[1], c='r', marker='*',
                edgecolor='none', s=100)

    def update(self, pid, val):
        """Update the curve."""
        self.mecha.update_prop(pid, val)
        self.curve = self.mecha.get_curve()

        self.isect = get_self_isect_super_robust(self.curve)

        self.redraw()

    def redraw(self):
        """Redraw canvas."""
        self.plot.set_data(self.curve[0], self.curve[1])

        if self.isect is not None:
            self.isect_plt.remove()
            self.isect_plt = self.ax.scatter(
                self.isect[0], self.isect[1], c='r', marker='*',
                edgecolor='none', s=100)

        self.ax.relim()
        self.ax.autoscale()


def main():
    """Entry point."""
    plt.ioff()

    SelfIntersectionFinder()

    plt.show()

if __name__ == "__main__":
    main()
