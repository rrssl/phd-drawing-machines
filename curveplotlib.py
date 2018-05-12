# -*- coding: utf-8 -*-
"""
Library of functions to plot curves and their derivatives.

@author: Robin Roussel
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from curvedistances import DistanceField
from curveimproc import fit_in_box


def distshow(ax, crv, aux_crv=None):
    """Show the distance field descriptor."""
    # Show the candidate curve embedded in the distance field of the target
    # curve.
    df = DistanceField()
    desc = df.get_desc(crv)
    if aux_crv is None:
        adapted_cand_curve = None
    else:
        shp = desc.shape
        shp = (shp[0]-10, shp[1]-10)
        adapted_cand_curve = fit_in_box(aux_crv, shp)
        adapted_cand_curve += 5

    imshow(ax, desc, adapted_cand_curve)

    if aux_crv is not None:
        ax.lines[0].set_linewidth(2)
        ax.lines[0].set_color('1.')
        d = max(df.get_dist(aux_crv, crv), df.get_dist(crv, aux_crv))
        ax.set_title(
            'Distance value: {:.2f}\n'.format(d), fontsize='xx-large')


class PixelFormatter:
    """Coordinate formatter to show pixel value with pyplot.imshow."""

    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        try:
            z = self.im.get_array()[int(y), int(x)]
        except IndexError:
            z = np.nan
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)


def imshow(frame, img, curve=None):
    """Show a raster image, optionnally along with a superimposed curve."""
    if curve is not None:
        frame.plot(*curve)

    shp = img.shape
    if len(shp) == 2:
        pltim = frame.imshow(img, interpolation=None, cmap=plt.cm.gray)
    elif len(shp) == 3 and shp[2] == 3:
        # /!\ OpenCV uses BGR order, while Pyplot uses RGB!
        img2 = img[:, :, ::-1]
        pltim = frame.imshow(img2, interpolation='none')

    # Show pixel value when hovering over it.
    frame.format_coord = PixelFormatter(pltim)
    # Remove axis text.
    frame.axes.get_xaxis().set_ticks([])
    frame.axes.get_yaxis().set_ticks([])


def cvtshow(curve, curvature):
    """Plot the curvature along the input curve."""
    points = curve.T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    lc = LineCollection(segments, cmap=plt.cm.winter)
    lc.set_array(curvature)
    lc.set_linewidth(2)

    plt.gca().set_aspect('equal')
    plt.gca().add_collection(lc)
    plt.autoscale()
    plt.margins(0.1)
