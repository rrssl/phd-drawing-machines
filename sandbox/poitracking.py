# -*- coding: utf-8 -*-
"""
Tracking points of interest in the property space.

@author: Robin Roussel
"""
from itertools import compress
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np

import context
from controlpane import ControlPane
from mecha import BaseSpirograph
import poly_point_isect as ppi

ppi.USE_DEBUG = False
ppi.USE_IGNORE_SEGMENT_ENDINGS = False
ppi.USE_VERBOSE = False
ppi.USE_RETURN_SEGMENT_ID = True


def get_self_isect(poly):
    """Get self-intersections.

    See isect_falsepos.py for details.
    """
    poly1 = np.asarray(poly)
    # Reflect polygon along first bisector.
    poly2 = poly1[[1, 0], :]

    # Find intersections.
    inter1, seg_ids = ppi.isect_polygon(poly1.T)
    if not seg_ids:
        return None, None
    inter1 = np.array(inter1).T
    N = poly1.shape[1] - 2
    # Valid points give exactly two non-consecutive segments.
    valid_points = np.array(
        [len(pair) == 2 and abs(pair[0] % N - pair[1] % N) > 1
         for pair in seg_ids])

    ppi.USE_RETURN_SEGMENT_ID = False
    inter2 = np.array(ppi.isect_polygon(poly2.T)).T
    ppi.USE_RETURN_SEGMENT_ID = True
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
#    if not valid_points.all():
#        print('False positive')
    seg_ids = np.array(list(compress(seg_ids, valid_points))).T

    return inter1[:, valid_points], seg_ids


def get_closest_id(target, points, output_dist=False):
    """Get id of point closest to target."""
    dists = np.linalg.norm(points - target.reshape((-1,1)), axis=0)
    if not output_dist:
        return np.argmin(dists)
    else:
        argmin = np.argmin(dists)
        return argmin, dists[argmin]


class POITracker:
    """Tracks points of interest."""

    def __init__(self):
        self.mecha = BaseSpirograph(5.,3.,1.3)
        self.nb = 2**5

        self.pid = 2
        bounds = self.mecha.get_prop_bounds(self.pid)
        self.prop_data = (self.pid, {'valmin': bounds[0],
                                     'valmax': bounds[1],
                                     'valinit': self.mecha.props[2],
                                     'label': "Parameter"
                                     }),

        self.curve = self.mecha.get_curve(self.nb)
        self.isect, self.isect_seg_ids = get_self_isect(self.curve)
        if self.isect is not None and self.isect.shape[1]:
            self.track = self.isect[:, 0]
            self.track_id = 0
        else:
            self.track = None
            self.track_id = None
        # The following distance is used as a threshold: if no correspondence
        # is found within a circle of radius 'max_dist' around the old
        # point, we consider that we temporarily lost its track.
        # This is not a bug, it's just that some points do disappear near
        # extremal property values (for instance try (5,3,2) and increase the
        # last property up to 3).
        # Be careful not to set this too low or the track will be considered
        # 'lost' every time the intersecting segments change.
        self.max_dist = 0.1
        # Approximate the arclength s in [0,1] for each point
        self.arclength_approx = (np.arange(self.curve.shape[1] + 1) /
                                 self.curve.shape[1])

        self.init_draw()

    def init_draw(self):
        """Initialize canvas and draw POIs."""
        self.fig, self.ax = plt.subplots()
        self.ax.set_aspect('equal')
        self.ax.margins(0.1)
        self.cmap = plt.cm.hsv
        # Add arclength colorbar
        dummy = plt.contourf([[0,0]], self.arclength_approx, cmap=self.cmap)
        self.cbar = self.fig.colorbar(dummy, ticks=[0, 1])
        self.cbar.ax.set_yticklabels(['0.', '1.'])
        self.cbar.ax.set_ylabel('arclength', rotation=270)
        # Add control pane.
        self.control_pane = ControlPane(self.fig, self.prop_data, self.update)
        # Add plot.
        self.plot = self.ax.plot([], [], 'k', alpha=0.1)[0]
        # Add intersections.
        if self.isect is not None and self.isect.shape[1]:
            self.isect_plt = self.ax.scatter([], [])

        lc = LineCollection([], cmap=self.cmap)
        self.isect_seg_plt = self.ax.add_collection(lc)

        self.redraw()

    def update(self, pid, val):
        """Update the curve."""
        self.mecha.update_prop(pid, val)
        self.curve = self.mecha.get_curve(self.nb)
        self.track_point()

        self.redraw()

    def track_point(self):
        """Get the new id of the tracked point."""
        if self.isect is not None and self.isect.shape[1]:
            if self.track_id is not None:
                arclength = np.sort(
                    self.arclength_approx[self.isect_seg_ids[:, self.track_id]])
    #            track = np.append(self.isect[:, self.track_id], arclength)
                self.track = arclength
        else:
            self.track_id = None

        self.isect, self.isect_seg_ids = get_self_isect(self.curve)

        if self.isect is not None and self.isect.shape[1]:
            arclength = np.sort(
                self.arclength_approx[self.isect_seg_ids], axis=0)
#            print(min_arclength.shape, self.isect.shape)
#            points = np.vstack([self.isect, arclength])
            points = arclength

            tid, dist = get_closest_id(self.track, points, output_dist=True)
            if dist < self.max_dist:
                self.track_id = tid
            else:
                self.track_id = None
                print("Point was lost.")

    def redraw(self):
        """Redraw canvas."""
        self.plot.set_data(self.curve[0], self.curve[1])

        if self.isect is not None and self.isect.shape[1]:
            self.isect_plt.remove()
            colors = ['w'] * self.isect.shape[1]
            if self.track_id is not None:
                colors[self.track_id] = 'red'
            self.isect_plt = self.ax.scatter(
                self.isect[0], self.isect[1], c=colors, marker='o',
                edgecolor='grey', s=100, zorder=3)

        self.isect_seg_plt.remove()
#            beg = self.curve[:, self.isect_seg_ids.T].T.reshape(-1, 1, 2)
#            end = self.curve[:, self.isect_seg_ids.T + 1].T.reshape(-1, 1, 2)
#            segments = np.concatenate([beg, end], axis=1)
        points = self.curve.T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

#            colors = self.arclength_approx[self.isect_seg_ids.ravel()]
        colors = self.arclength_approx
        lc = LineCollection(segments, cmap=self.cmap)
        lc.set_array(colors)
        lc.set_linewidth(3)
        self.isect_seg_plt = self.ax.add_collection(lc)

        self.ax.relim()
        self.ax.autoscale()


def main():
    """Entry point."""
    plt.ioff()

    POITracker()

    plt.show()

if __name__ == "__main__":
    main()
