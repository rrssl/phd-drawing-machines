# -*- coding: utf-8 -*-
"""
Helper module of factory functions for the uniform management of curve creation.

@author: Robin Roussel
"""

# TODO: should we rename the module "spirogen"?
# Or a name reflecting the fact that it is only meant to create closed curves?

import math
import numpy as np
import scipy.optimize as opt
import scipy.special as spec

import curves as cu
from utils import skipends, farey


def get_curve(params, nb_samples_per_cycle=2**6 + 1,
              mov_type=cu.Circle, nmov_type=cu.Circle, mov_inside_nmov=True):
    """Generate the closed curve of given parameters.

    This function is a common interface to generate different roulettes.
    """
    if nmov_type is cu.Circle and mov_inside_nmov:
        if mov_type is cu.Circle:
            return Hypotrochoid(*params).get_curve(nb_samples_per_cycle)
        elif mov_type is cu.Ellipse:
            return RouletteEllipseInCircle(*params).get_curve(nb_samples_per_cycle)


def get_param_combinations(nb_grid_nodes=(10,), mov_type=cu.Circle,
                           nmov_type=cu.Circle, mov_inside_nmov=True):
    """Sample the parameter space and get all possible combinations.

    This function is a common interface to generate all parameters combinations
    that can be given to get_curve()."""
    if nmov_type is cu.Circle and mov_inside_nmov:
        if mov_type is cu.Circle:
            return Hypotrochoid.sample_parameters(nb_grid_nodes)
        elif mov_type is cu.Ellipse:
            return RouletteEllipseInCircle.sample_parameters(nb_grid_nodes)


class Hypotrochoid(cu.Hypotrochoid):
    """Class to generate physically valid closed hypotrochoids.

    This class makes more assumptions than the generic curves.Hypotrochoid:
     -- restriction to the subspace of trochoids closing in a finite number
     of turns,
     -- restriction to physically realizable curves (size and placement
     limits).
     """

    def __init__(self, ext_gear_radius, int_gear_radius, pole_dist):
        super().__init__(ext_gear_radius, int_gear_radius, pole_dist)

    def get_curve(self, nb_samples_per_cycle=2**6):
        nb_cycles = self.r if self.d else 1 # d = 0: Degenerate case.
        nb_samples = nb_cycles * nb_samples_per_cycle + 1
        interval_length = nb_cycles * 2 * np.pi
        theta = np.linspace(0., interval_length, nb_samples)

        return self.get_point(theta)

    def update_curve(self, params, nb_samples_per_cycle=2**6):
        self.R, self.r, self.d = params
        return self.get_curve(nb_samples_per_cycle)

    @staticmethod
    def sample_parameters(nb_grid_nodes=(10,)):
        """Sample the parameter space."""
        if len(nb_grid_nodes) == 1:
            nb_grid_nodes *= 3
        return np.array(list(
            cu.Hypotrochoid.sample_parameters(nb_grid_nodes)))

    def get_rot_sym_order(self):
        """Get the order of the rotational symmetry of the curve."""
        return int(self.R)

    def get_bounds(self, n):
        """Get the bounds of the nth parameter."""
        if n == 0:
            return (0., None)
        if n == 1:
            return (0., self.R)
        if n == 2:
            return (0., self.r)
        else:
            return None

    def get_continuous_optimization_constraints(self):
        """Get a dict of optim. constraints on the continuous parameters."""
        return {'bounds': self.get_bounds(2)}


class RouletteEllipseInCircle:
    """Class to generate physically valid closed roulette curves (ell in circ).

    This class makes more assumptions than the generic
    Roulette(Ellipse, Circle):
     -- restriction to the subspace of curves closing in a finite number
     of turns,
     -- restriction to physically realizable curves (size, curvature and
     placement limits).

     Warning: the parametrization of this class is not the same as
     Roulette(Ellipse, Circle).
     """

    def __init__(self, ext_gear_radius, ell_scale_factor, ell_sq_eccentricity,
                 pole_dist):
        self.R = ext_gear_radius
        self.S = ell_scale_factor
        self.e2 = ell_sq_eccentricity
        self.d = pole_dist

        if not self.check_bounds():
            self.print_bounds_warning()
        semiaxes = self.get_ellipse_semiaxes() # Reparametrize.

        cir = cu.Circle(self.R)
        ell = cu.Ellipse(*semiaxes)
        self.curve = cu.Roulette(ell, cir, self.d, 'moving')

    def check_bounds(self):
        """Check that current parameters are within valid bounds."""
        if not self.S:
            # Do not evaluate bounds if S == 0.
            return True
        else:
            bounds = [self.get_bounds(i) for i in range(4)]
            return (bounds[0][0] <= self.R and
                    bounds[1][0] <= self.S <= bounds[1][1] and
                    bounds[2][0] <= self.e2 <= bounds[2][1] and
                    bounds[3][0] <= self.d <= bounds[3][1])

    def print_bounds_warning(self):
        """Warn that a parameter is out of bounds."""
        print("Warning: at least one parameter is out of bounds" +
              "(R = {:.2f}, S = {:.2f}, ".format(self.R, self.S) +
              "e2 = {:.2f}, d = {:.2f}).".format(self.e2, self.d))

    def get_ellipse_semiaxes(self):
        """Get the semiaxes (a,b) of the ellipse."""
        semiaxes_base = np.array([np.pi / (2 * spec.ellipe(self.e2)), 0.])
        semiaxes_base[1] = semiaxes_base[0] * math.sqrt(1 - self.e2)
        semiaxes = self.S * semiaxes_base

        return semiaxes

    def get_curve(self, nb_samples_per_cycle=2**6):
        """Get the Roulette(Ellipse, Circle) points."""
        self.degenerate = False
        nb_cycles = self.R
        if not self.d:
            # Degenerate cases.
#            print('degenerate')
            if not self.e2:
                self.degenerate = True
                nb_cycles /= self.S
            elif not self.S % 2:
                # Since (S,R) = 1, if S is even then R is necessarily odd.
                self.degenerate = True
                nb_cycles /= 2
        nb_samples = nb_cycles * nb_samples_per_cycle + 1
        interval_length = nb_cycles * 2 * np.pi

        curve = self.curve.get_range(0., interval_length, nb_samples)
        if not np.isfinite(curve).all():
            print("Warning: curve contains NaNs.")
#            print(self.R, self.S, self.e2, self.d)

        return curve

    def update_curve(self, params, nb_samples_per_cycle=2**6):
        """Update the current parameters and return the corresponding curve."""
        if (params[0] == self.R and
            params[1] == self.S and
            params[2] == self.e2 and
            params[3] != self.d and
            hasattr(self.curve, 'n_pts') and
            not self.degenerate # Force recomputing if previously degenerate.
            ):
            self.d = params[3]
            return self.curve.update_pole(self.d)

        self.R, self.S, self.e2, self.d = params
        if not self.check_bounds():
            self.print_bounds_warning()
        semiaxes = self.get_ellipse_semiaxes() # Reparametrize.
        self.curve.m_obj.reset(*semiaxes)
        self.curve.n_obj.reset(self.R)
        self.curve.reset(self.d)

        return self.get_curve(nb_samples_per_cycle)

    def get_angles(self):
        """Get the angular deviations used for the last curve computation."""
        return np.arctan2(self.curve.rot[1, 0], self.curve.rot[0, 0])

    @staticmethod
    def sample_parameters(nb_grid_nodes=(10,)):
        """Sample the parameter space."""
        if len(nb_grid_nodes) == 1:
            nb_grid_nodes *= 4
        return np.array(list(
            RouletteEllipseInCircle._gen_sample_parameters(nb_grid_nodes)))

    def get_rot_sym_order(self):
        """Get the order of the rotational symmetry of the curve."""
        return int(self.R)

    def get_bounds(self, n):
        """Get the bounds of the nth parameter."""
        if n == 0:
            return (0., None)
        if n == 1:
            return (0., self.R)
        if n == 2:
            return (0., self._get_e2max(self.R, self.S))
        if n == 3:
            return (0., self._get_dmax(self.S, self.e2))

    def get_continuous_optimization_constraints(self):
        """Get a dict. of optim. constraints on the continuous parameters."""
        bounds = (self.get_bounds(2), (0., None))
        constr = {
            'type': 'ineq',
            'fun': lambda x: np.array(self._get_dmax(self.S, x[0]) - x[1]),
            'jac': lambda x: np.array([self._get_ddmax_de2(self.S, x[0]), -1.])
            }
        return {'bounds': bounds, 'constraints': constr}

    @staticmethod
    def _gen_sample_parameters(nb_grid_nodes=(10,10,10)):
        """Get a regular sampling of the parameter space with a generator."""
        n_R, n_e, n_d = nb_grid_nodes[0], nb_grid_nodes[-2], nb_grid_nodes[-1]

        for S, R in skipends(farey(n_R)):

            emax2 = RouletteEllipseInCircle._get_e2max(R, S)
            for e2 in np.linspace(0, emax2 * (n_e - 1) / n_e, n_e):

                dmax = RouletteEllipseInCircle._get_dmax(S, e2)
                for d in np.linspace(0, dmax * (n_d - 1) / n_d, n_d):
                    yield R, S, e2, d

    @staticmethod
    def _get_e2max(R, S):
        """Get the square of the eccentricity's upper bound given R and S."""
        # Approximate this bound using an inversion of the Naive Approximation
        #  of the ellipse circumference.
        emax2_approx = ((R - 4 * S) + math.sqrt(R * (R + 8 * S))) / (2 * R)
        # Compute the exact bound.
        return opt.fsolve(
            lambda x: x + (S * np.pi / (2 * spec.ellipe(x) * R)) ** 2 - 1,
            emax2_approx)[0]

    @staticmethod
    def _get_dmax(S, e2):
        """Get the upper bound of the pole distance given S and e2."""
        return S * np.pi / (2 * spec.ellipe(e2)) # dmax = semimajor

    @staticmethod
    def _get_ddmax_de2(S, e2):
        """Get the partial der.of the up. bound of the pole distance wrt e2."""
        E, K = spec.ellipe(e2), spec.ellipk(e2)
        return - S * np.pi * (E - K) / (2 * math.sqrt(e2) * E * E)


# Finding the eccentricity's upper bound:
# emax = lambda a, b: fsolve(lambda t: t * t + (b * np.pi / (2 * ellipe(t * t) * a)) ** 2 - 1, init)
# emax = lambda a, b: fsolve(lambda t: t + (b * np.pi / (2 * ellipe(t) * a)) ** 2 - 1, init)
# Fast approximation:
# emax = sqrt(2)*sqrt((R*(R - 4*sigma) + sqrt(R**3*(R + 8*sigma)))/R**2)/2
# emax**2 = (R*(R - 4*sigma) + sqrt(R**3*(R + 8*sigma))) / (2*R**2)
#
# Then dmax = S*np.pi/(2*ellipe(emax**2))
