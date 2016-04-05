#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the curvature features dissimilarity measure.

@author: Robin Roussel
"""

import context

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sig

from curvedistances import compute_curvature
from curvegen import get_curve
import curveplotlib as cplt


def show_curvature_features(curve, R, r, sampling_rate):
    """Test the curvature features descriptor."""
    # Compute Fourier transform.
    cvt = compute_curvature(curve)
    fourier = np.fft.rfft(cvt)
    power = abs(fourier) * abs(fourier)
    freq = np.fft.rfftfreq(curve.shape[-1])
    # Find power peaks and keep the ones >= 1% of the main peak.
    argrelmax = sig.argrelmax(power)[0]
    argrelmax = argrelmax[power[argrelmax] >= 0.01 * power[0]]
    maxima = np.vstack([freq[argrelmax], power[argrelmax]])
    # Test the main frequency (the other are just harmonics).
    f1 = maxima[0, 0]
    print("Main frequence: {}".format(f1))
    theta_period = 1 / (f1  * sampling_rate)
    print("Corresponding main theta period: {}".format(theta_period))
    print("phi_(max, max) = 2pi * r / R = {}".format(2 * np.pi * r / R))

    # Show the curvature and Fourier peaks.
    plt.gcf().add_subplot(311, title="Curvature plot.")
    plt.plot(cvt)
    plt.xlim(xmax=len(cvt)-1)
    plt.gcf().add_subplot(312, title="Curvature on the trajectory.")
    cplt.cvtshow(curve, cvt)
    plt.gcf().add_subplot(313, title="Fourier transform.")
    plt.plot(freq, power)
    plt.scatter(maxima[0], maxima[1], c='r')
    plt.xlim(freq[0], freq[-1])
    plt.ylim(ymin=0)

    base_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches((base_size[0], base_size[1] * 1.5),
                              forward=True)


def main():
    """Entry point."""
    plt.ioff()

    # Get the reference curve.
    params = (5., 3., 1.5)
    samples_per_turn = 50
    ref_curve = get_curve(params, nb_samples_per_turn=samples_per_turn)
    sampling_rate = samples_per_turn / (2 * np.pi)

    show_curvature_features(ref_curve, params[0], params[1], sampling_rate)

    plt.show()


if __name__ == "__main__":
    main()
