# -*- coding: utf-8 -*-
"""
Tools for the matching, retrieval and classification of curves.

@author: Robin Roussel
"""
import numpy as np
import scipy.optimize as opt


class CurveMatcher:
    """Adapter transforming a curve distance into a curve matcher."""
    
    def __init__(self, distance, get_curve):
        self.distance = distance
        self.get_curve = get_curve

    def __call__(self, target_curve, cand_params):
        """Find the candidate best matching the input curve."""
        # Compare each candidate curve.
        distances = np.array([self.distance(self.get_curve(cand), target_curve)
                              for cand in cand_params])
        return cand_params[np.argsort(distances), :]
        
    
def classify_curve(target_curve, cand_params, curve_matcher, threshold):
    """Find the candidates in the same class as the input curve."""
    # Compare each candidate curve.
    belongs = np.zeros(cand_params.shape[0], dtype=bool)
    for i, cand in enumerate(cand_params):
        # Generate the curve.
        cand_curve = curve_matcher.curve_func(cand)
        # Compute the distance.
        dist = curve_matcher(cand_curve, target_curve)
#        print(dist, c)
        belongs[i] = dist <= threshold

    return belongs


class CurveOptimizer:
    """Precise optimization of a candidate curve on a target curve."""
    
    def __init__(self, distance, get_curve, target_curve=None, bounds=None, 
                 constraints=None, init_guess=None):
        self.distance = distance
        self.get_curve = get_curve
        self.target = target_curve
        self.bounds = bounds
        self.constr = constraints
        self.init = init_guess
        
    def _get_objective(self, x):
        """Get f(x), with f the objective function."""
        return self.distance(self.get_curve(x), self.target)
        
    def optimize(self, target_curve=None, bounds=None, constraints=None,
                 init_guess=None, display=False):
        """Get the optimal parameters for the target curve."""
        self.target = target_curve if target_curve is not None else self.target
        self.bounds = bounds if bounds is not None else self.bounds
        self.constr = constraints if constraints is not None else self.constr
        self.init = init_guess if init_guess is not None else self.init
        
        options = {'options': {'disp': display}}
        if len(self.init) <= 1:
            opt_func = opt.minimize_scalar
            if self.bounds is not None:
                options['bounds'] = self.bounds
                options['method'] = 'bounded'
            else:
                options['method'] = 'brent'
        else:
            opt_func = opt.minimize
            options['x0'] = self.init
            if self.bounds is not None:
                options['bounds'] = self.bounds
                if self.constr is not None:
                    options['constraints'] = self.constr
                    options['method'] = 'SLSQP'
                else:
                    options['method'] = 'L-BFGS-B'
            elif self.constr is not None:
                options['constraints'] = self.constr
                options['method'] = 'COBYLA'
            else:
                options['method'] = 'BFGS'
        print("Method: ", options['method'])
        return opt_func(self._get_objective, **options)
