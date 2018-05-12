#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make a complex machine gnagnagna

@author: Robin Roussel
"""
import numpy as np
import matplotlib.pyplot as plt

nb_gears = 4

# Geometric parameters
radii = (1. / np.arange(2, 2+nb_gears)[::-1]).reshape(-1, 1, 1)
amplitudes = radii / 2.
angles = np.pi * np.linspace(0., 1., nb_gears).reshape(-1, 1, 1)
centers = (1.+radii) * np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
cdists = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
lengths = (cdists + (radii[:-1] - radii[1:]).reshape(-1, 1)) / 2.
# Time-dependent variables
t_rng = np.linspace(0., 2*np.pi, 2**8)
angles = -t_rng/radii
pivots = (amplitudes * np.concatenate((np.cos(angles), np.sin(angles)), axis=1)
          + centers)
motion = pivots[0]
for i in range(nb_gears-1):
    vec = pivots[i+1] - motion
    vec *=  lengths[i] / (vec[0]**2 + vec[1]**2)
    motion += vec
# Shift so that first position coincides with origin
motion -= motion[:, [0]]
# Frame change (Turntable rotation) -- Angle is theta = -t
cos = np.cos(t_rng)
sin = np.sin(-t_rng)
rot = np.array([[cos, -sin],
                [sin, cos]])
motion = np.einsum('ijk,jk->ik', rot, motion)
# Show
plt.figure()
plt.plot(*motion)
plt.ioff()
plt.show()