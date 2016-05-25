# -*- coding: utf-8 -*-
"""
Helper module of factory functions for the uniform management of curve creation.

@author: Robin Roussel
"""

import math
import numpy as np
import scipy.optimize as opt
import scipy.special as spec
import curves as cu

def get_curve(params, nb_cycles=None, nb_samples_per_cycle=2**6 + 1,
              mov_type=cu.Circle, nmov_type=cu.Circle, mov_inside_nmov=True):
    """Generate the curve of given parameters."""
    if nmov_type is cu.Circle and mov_inside_nmov:
        if mov_type is cu.Circle:
            return get_hypotrochoid(params, nb_cycles, nb_samples_per_cycle)
        elif mov_type is cu.Ellipse:
            return get_roulette_ellipse_in_circle(params, nb_cycles, 
                                                  nb_samples_per_cycle)

def get_hypotrochoid(params, nb_cycles=None, nb_samples_per_cycle=50):
    """Get the Hypotrochoid points."""
    if nb_cycles is None:
        nb_cycles = params[1]
    nb_samples = nb_cycles * nb_samples_per_cycle
    interval_length = nb_cycles * 2 * np.pi
    theta = np.linspace(0., interval_length, nb_samples)
    hypo = cu.Hypotrochoid(*params)
    return hypo.get_point(theta)

    
def get_roulette_ellipse_in_circle(params, nb_cycles=None, 
                                   nb_samples_per_cycle=2 ** 6 + 1):
    """Get the Roulette(Ellipse, Circle) points."""
    if nb_cycles is None:
        nb_cycles = params[0]
    nb_samples = nb_cycles * nb_samples_per_cycle
    interval_length = nb_cycles * 2 * np.pi

    cir = cu.Circle(params[0])
    # Reparametrize.
    e2 = params[2]
    semiaxes_base = np.array([np.pi / (2 * spec.ellipe(e2)), 0.])
    semiaxes_base[1] = semiaxes_base[0] * math.sqrt(1 - e2)
    semiaxes = params[1] * semiaxes_base
    ell = cu.Ellipse(*semiaxes)
    roul = cu.Roulette(ell, cir, params[3], 'moving')

    return roul.get_range(0., interval_length, nb_samples)


def get_param_combinations(nb_grid_nodes=(10,), mov_type=cu.Circle, 
                           nmov_type=cu.Circle, mov_inside_nmov=True):
    """Sample the parameter space and get all possible combinations."""
    if nmov_type is cu.Circle and mov_inside_nmov:
        if mov_type is cu.Circle:
            if len(nb_grid_nodes) == 1:
                nb_grid_nodes *= 3
#            return cu.Hypotrochoid.get_param_combinations(*nb_grid_nodes)
            return np.array(list(
                cu.Hypotrochoid.sample_parameters(nb_grid_nodes)))
        elif mov_type is cu.Ellipse:
            if len(nb_grid_nodes) == 1:
                nb_grid_nodes *= 4
            return np.array(list(
                sample_parameters_roulette_ellipse_in_circle(nb_grid_nodes)))


def sample_parameters_roulette_ellipse_in_circle(nb_grid_nodes=(10,10,10)):
    """Get a regular sampling of the parameter space with a generator."""
    n_R, n_e, n_d = nb_grid_nodes[0], nb_grid_nodes[-2], nb_grid_nodes[-1]

    for S, R in cu.skipends(cu.farey(n_R)):
        
        emax2 = _get_emax2(R, S)
        for e2 in np.linspace(0, emax2 * (n_e - 1) / n_e, n_e):
            
            dmax = S * np.pi / (2 * spec.ellipe(e2)) # dmax = semimajor
            for d in np.linspace(0, dmax * (n_d - 1) / n_d, n_d):
                yield R, S, e2, d


def _get_emax2(R, S):
    """Get the square of the eccentricity's upper bound."""
    # Approximate this bound using an inversion of the Naive Approximation of
    # the ellipse circumference.
    emax2_approx = ((R - 4 * S) + math.sqrt(R * (R + 8 * S))) / (2 * R)
    # Compute the exact bound.
    return opt.fsolve(
        lambda x: x + (S * np.pi / (2 * spec.ellipe(x) * R)) ** 2 - 1, 
        emax2_approx)


# Finding the eccentricity's upper bound:
# emax = lambda a, b: fsolve(lambda t: t * t + (b * np.pi / (2 * ellipe(t * t) * a)) ** 2 - 1, init)
# emax = lambda a, b: fsolve(lambda t: t + (b * np.pi / (2 * ellipe(t) * a)) ** 2 - 1, init)
# Fast approximation:
# emax = sqrt(2)*sqrt((R*(R - 4*sigma) + sqrt(R**3*(R + 8*sigma)))/R**2)/2
# emax**2 = (R*(R - 4*sigma) + sqrt(R**3*(R + 8*sigma))) / (2*R**2)
#
# Then dmax = S*np.pi/(2*ellipe(emax**2))
