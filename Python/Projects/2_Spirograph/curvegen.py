# -*- coding: utf-8 -*-
"""
Helper functions for the uniform management of curve generation accross modules.

@author: Robin Roussel
"""

import numpy as np
from curves import Hypotrochoid

def get_curve(params, nb_turns=None, nb_samples_per_turn=50):
    """Generate the curve (hypotrochoid) of given parameters."""
    if nb_turns is None:
        nb_turns = params[1]
    nb_samples = nb_turns * nb_samples_per_turn
    interval_length = nb_turns * 2 * np.pi
    theta = np.linspace(0., interval_length, nb_samples)
    hypo = Hypotrochoid(*params)
    return hypo.get_point(theta)
    
    
def get_param_combinations(grid_size=(10,10)):
    """Generate combinations of parameters of (hypotrochoid) curves."""
    return Hypotrochoid.get_param_combinations(*grid_size)
