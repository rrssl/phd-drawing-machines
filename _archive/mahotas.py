# -*- coding: utf-8 -*-
"""
Test mahotas.

@author: robin
"""

import numpy as np
import mahotas as mh
import matplotlib.pyplot as plt

plt.ioff()

img = mh.imread('userinput_5_3_1.5.png')
img =mh.colors.rgb2grey(img[:,:,:3], dtype=np.uint)

plt.subplot(131)
plt.imshow(img, cmap=plt.cm.gray)

threshold = mh.thresholding.rc(img)
print(threshold)
plt.subplot(132)
plt.imshow(img > threshold, cmap=plt.cm.gray, interpolation='none')

seeds, _ = mh.label(img > threshold)
labeled = mh.cwatershed(img.max() - img, seeds)
plt.subplot(133)
plt.imshow(seeds)

#plt.show()

zm = mh.features.zernike_moments(img, radius=128)
print(zm)
