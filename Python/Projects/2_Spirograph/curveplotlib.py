# -*- coding: utf-8 -*-
"""
Library of urve plotting functions.

@author: Robin Roussel
"""

import numpy as np

try:
    import cv2
except ImportError:
    CV2_IMPORTED = False
else:
    CV2_IMPORTED = True
    
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

def plot(curve, *args, **kwargs):
    plt.plot(curve[0], curve[1], *args, **kwargs)

class PixelFormatter:
    def __init__(self, im):
        self.im = im
    def __call__(self, x, y):
        try:
            z = self.im.get_array()[int(y), int(x)]
        except IndexError:
            z = np.nan
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

def imshow(img, curve=None, viewer='plt'):
    if viewer == 'plt':
        # Display using Pyplot.    
        if curve is not None:
            plot(curve)
            
        shp = img.shape        
        if len(shp) == 2:
            pltim = plt.imshow(img, interpolation='none', cmap=plt.cm.gray)
        elif len(shp) == 3 and shp[2] == 3:
            # /!\ OpenCV uses BGR order, while Pyplot uses RGB!
            img2 = img[:,:,::-1]
            pltim = plt.imshow(img2, interpolation='none')
        
        # Show pixel value when hovering over it.
        plt.gca().format_coord = PixelFormatter(pltim)
        
    elif viewer == 'qt':
        if CV2_IMPORTED:
            # Display using OpenCV's Qt viewer.
            cv2.namedWindow('Viewer',  cv2.WINDOW_NORMAL)
#            img2 = cv2.normalize(img, 0, 255, norm_type=cv2.NORM_MINMAX, 
#                                 dtype=cv2.CV_8UC1)
            img2 = cv2.normalize(img, 0, 1, norm_type=cv2.NORM_MINMAX)
            cv2.imshow('Viewer', img2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Qt viewer cannot be used: OpenCV module not found.")
        
def cvtshow(curve, curvature):
    # Plot curvature.
    points = curve.T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    lc = LineCollection(segments, cmap=plt.cm.winter)
    lc.set_array(curvature)
    lc.set_linewidth(2)

    plt.gca().set_aspect('equal')
    plt.gca().add_collection(lc)
    plt.autoscale()
    plt.margins(0.1)  