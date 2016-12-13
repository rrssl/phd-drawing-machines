# -*- coding: utf-8 -*-
"""
Library of parametric curves.

@author: Robin Roussel
"""

#from fractions import Fraction
import math
import numpy as np
import scipy.optimize as opt
import scipy.special as spec

from utils import skipends, farey


class Curve:
    """Base class for curves."""

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        raise NotImplementedError


class Hypotrochoid(Curve):
    """Hypotrochoid class."""

    def __init__(self, ext_gear_radius, int_gear_radius, pole_dist):
        self.reset(ext_gear_radius, int_gear_radius, pole_dist)

    def reset(self, ext_gear_radius, int_gear_radius, pole_dist):
        """Reset the parameters."""
        self.R = ext_gear_radius
        self.r = int_gear_radius
        self.d = pole_dist

    @staticmethod
    def sample_parameters(nb_grid_nodes):
        """Get a regular sampling of the parameter space with a generator."""
        n_R, n_d = nb_grid_nodes[0], nb_grid_nodes[-1]
        d_arr = [np.linspace(0, l * (n_d - 1) / n_d, n_d)
                 for l in range(1, n_R)]
        for r, R in skipends(farey(n_R)):
            for d in d_arr[r - 1]:
                yield R, r, d

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        R = self.R
        r = self.r
        d = self.d
        return np.vstack([(R - r) * np.cos(t) + d * np.cos(t * (R - r) / r),
                          (R - r) * np.sin(t) - d * np.sin(t * (R - r) / r)])


class Epitrochoid(Curve):
    """Epitrochoid class."""

    def __init__(self, int_gear_radius, ext_gear_radius, pole_dist):
        self.reset(int_gear_radius, ext_gear_radius, pole_dist)

    def reset(self, int_gear_radius, ext_gear_radius, pole_dist):
        """Reset the parameters."""
        self.R = int_gear_radius
        self.r = ext_gear_radius
        self.d = pole_dist

    @staticmethod
    def sample_parameters(nb_grid_nodes):
        """Get a regular sampling of the parameter space with a generator."""
        n_R, n_d = nb_grid_nodes[0], nb_grid_nodes[-1]
        d_arr = [np.linspace(0, l * (n_d - 1) / n_d, n_d)
                 for l in range(1, n_R)]
        for r, R in skipends(farey(n_R)):
            for d in d_arr[r - 1]:
                yield R, r, d

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        R = self.R
        r = self.r
        d = self.d
        return np.vstack([(R + r) * np.cos(t) - d * np.cos(t * (R + r) / r),
                          (R + r) * np.sin(t) - d * np.sin(t * (R + r) / r)])


class Circle(Curve):
    """Parametric curve of a circle."""

    def __init__(self, radius):
        self.reset(radius)

    def reset(self, radius):
        """Reset the class fields."""
        self.r = radius

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        return self.r * np.vstack([np.cos(t), np.sin(t)])

    def get_jac(self, t):
        """Get the [x'(t), y'(t)] jacobian(s)."""
        return self.r * np.vstack([-np.sin(t), np.cos(t)])

    def get_perimeter(self):
        """Get the full perimeter of the circle."""
        return 2 * math.pi * self.r

    def get_arclength(self, t):
        """Get the arc length(s) s(t)."""
        return self.r * t

    def get_arclength_der(self, t):
        """Get the derivative(s) of the arc length s'(t)."""
        return np.full(t.shape, float(self.r))

    def get_arclength_inv(self, s):
        """Get the parameter t given s(t)."""
        return s / self.r

    def get_min_curvature(self):
        return 1 / self.r if self.r else np.inf

    def get_max_curvature(self):
        return 1 / self.r if self.r else np.inf

    def get_period(self):
        return 2 * math.pi

    def has_even_arclength(self):
        return True


class Ellipse(Curve):
    """Parametric curve of an ellipse."""

    def __init__(self, semimajor, semiminor):
        self.reset(semimajor, semiminor)

    def reset(self, semimajor, semiminor):
        """Reset the class fields."""
        self.a = semimajor
        self.b = semiminor
        if self.a < self.b:
            print("Warning: the semimajor axis must be greater than the"
                  "semiminor axis "
                  "(a = {:.2f}, b = {:.2f}).".format(self.a, self.b))

        if semimajor:
            self.e2 = 1. - semiminor * semiminor / (semimajor * semimajor)
            # /!\ scipy's implementation uses the convention E(phi, m) with m
            # the elliptic parameter, which in our case needs to be e**2.
            self.ellipe_val = spec.ellipe(self.e2)
        else:
            self.e2 = 1.
            self.ellipe_val = 0.

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        return np.vstack([self.a * np.cos(t), self.b * np.sin(t)])

    def get_normal(self, t):
        """Get the [n_x(t), n_y(t)] normals(s)."""
        return np.vstack([self.b * np.cos(t), self.a * np.sin(t)])

    def get_range(self, start, end, nb_pts):
        """Get [x(t), y(t)] with the t values evenly sampled in [start, end]."""
        return self.get_point(np.linspace(start, end, nb_pts))

    def get_jac(self, t):
        """Get the [x'(t), y'(t)] jacobian(s)."""
        return np.vstack([self.a * -np.sin(t), self.b * np.cos(t)])

    def get_perimeter(self):
        """Get the full perimeter of the ellipse."""
        # Use the value of the complete elliptic integral of the 2nd kind.
        return 4 * self.a * self.ellipe_val

    def get_arclength(self, t):
        """Get the arc length(s) s(t)."""
        # Use the incomplete elliptic integral of the 2nd kind.
        # /!\ The "amplitude" phi of E(phi, k) as it is often found in the
        # litterature is NOT the same as our ellipse parameter here. To get the
        # arc length right we have to use s(t) = E(t + pi/2, e) - E(e).
        return self.a * (spec.ellipeinc(t + (math.pi / 2), self.e2) -
                         self.ellipe_val)

    def get_arclength_der(self, t):
        """Get the derivative(s) of the arc length s'(t)."""
        # Easy with the fundamental theorem of calculus.
        return self.a * np.sqrt(1 - self.e2 * np.cos(t) ** 2)

    def get_arclength_inv(self, s):
        """Get the parameter t given s(t)."""
        obj_func = lambda t: self.get_arclength(t) - s
        obj_jac = lambda t: np.diag(self.get_arclength_der(t))
        # Initialize by approximating with a circle of equal perimeter.
        init_guess = 2 * math.pi * s / self.get_perimeter()

        return opt.fsolve(obj_func, init_guess, fprime=obj_jac)

    def get_min_curvature(self):
        return self.b / (self.a * self.a) if self.a else np.inf

    def get_max_curvature(self):
        return self.a / (self.b * self.b) if self.b else np.inf

    def get_period(self):
        return 2 * math.pi

    def has_even_arclength(self):
        return True


class Ellipse2(Ellipse):
    """Parametric curve of an ellipse, with alternative input arguments.

    The 'equivalent radius' is the radius of a circle with the same perimeter.
    The eccentricity is squared because it always appears as such in formulas.
    """

    def __init__(self, equiv_radius, eccentricity_sq):
        self.reset(equiv_radius, eccentricity_sq)

    def reset(self, equiv_radius, eccentricity_sq):
        """Reset the class fields."""
        if not equiv_radius:
            self.req = 0.
            self.e2 = 0.
            self.ellipe_val = math.pi / 2.
            self.a = 0.
            self.b = 0.
        elif not eccentricity_sq:
            self.req = equiv_radius
            self.e2 = 0.
            self.ellipe_val = math.pi / 2.
            self.a = equiv_radius
            self.b = equiv_radius
        else:
            self.req = equiv_radius
            self.e2 = eccentricity_sq
            self.ellipe_val = spec.ellipe(self.e2)
            self.a = math.pi * self.req / (2. * self.ellipe_val)
            self.b = self.a * math.sqrt(1 - self.e2)

            if self.a < self.b:
                print("Warning: the semimajor axis must be greater than the"
                      "semiminor axis "
                      "(a = {:.2f}, b = {:.2f}).".format(self.a, self.b))


def _get_rotation(u, v):
    """Get the 2D rotation matrix s.t. v = Ru."""
    norms = np.sqrt((u[0] ** 2 + u[1] ** 2) * (v[0] ** 2 + v[1] ** 2))
    cos = np.einsum('ij,ij->j', u, v) / norms # dot product
    sin = (u[0] * v[1] - u[1] * v[0]) / norms # cross
    return np.array([[cos, -sin],
                     [sin, cos]])


class Roulette(Curve):
    """Parametric roulette curve.

    The 2 first inputs are expected to be Curve objects.

    Two parametrizations are available: the contact point trajectory is
    described by the parameter of either the moving or the static curve.
    In some cases one is more intuitive than the other (i.e. the parameter of
    one curve has a more intuitive geometric meaning), but can also be more
    computationally expensive.
        ex: Ellipse rolling in Circle. The Circle's parameter is more intuitive
            (equal to the polar angle), but requires the inversion of the
            arc length function of the Ellipse.
    """
    def __init__(self, moving_obj, nonmoving_obj, pole_dist,
                 parametrization='nonmoving'):
        self.m_obj = moving_obj
        self.n_obj = nonmoving_obj

        if not self.check_curvature_constraint():
            self.print_curvature_warning()

        self.T = np.array([[pole_dist], [0.]])
        self.par = parametrization
        self._old = ()

    def check_curvature_constraint(self):
        """Check that the minimum curvature of the moving curve is strictly
        greater than the maximum curvature of the nonmoving curve.
        """
        m_K = self.m_obj.get_min_curvature()
        n_K = self.n_obj.get_max_curvature()
        if np.isfinite(m_K) and np.isfinite(n_K):
            return m_K > n_K
        else:
            return True

    def print_curvature_warning(self):
            min_K_m = self.m_obj.get_min_curvature()
            max_K_m = self.n_obj.get_max_curvature()
            print("Warning: The minimum curvature of the moving curve should "
                  "be greater than the maximum curvature of the nonmoving "
                  "curve (min(K_m) = "
                  "{:.2f}, max(K_n) = {:.2f}).".format(min_K_m, max_K_m))

    def reset(self, pole_dist):
        self.T[0] = pole_dist

        if not self.check_curvature_constraint():
            self.print_curvature_warning()

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s).

        See the class docstring to know how 't' is going to be interpreted.
        No assumptions are made on the order of the points, the periodicity
        and/or symmetry of the curve.
        """
        # Get the parameter values for both curves.
        if self.par == 'moving':
            tm = t
            svals = self.m_obj.get_arclength(tm)
            tn = self.n_obj.get_arclength_inv(svals)
        else:
            tn = t
            svals = self.n_obj.get_arclength(tn)
            tm = self.m_obj.get_arclength_inv(svals)
        # Get the points' coordinates from the parameter values.
        self.n_pts = self.n_obj.get_point(tn)
        self.m_pts = self.m_obj.get_point(tm)
        # Compute pairs of jacobians.
        n_jacs = self.n_obj.get_jac(tn) / self.n_obj.get_arclength_der(tn)
        m_jacs = self.m_obj.get_jac(tm) / self.m_obj.get_arclength_der(tm)
        # Compute the rotations between each pair of jacobians.
        self.rot = _get_rotation(m_jacs, n_jacs)

        return self._get_curve_from_data()

    def get_range(self, start, end, nb_pts, reuse=True):
        """Get [x(t), y(t)] with the t values evenly sampled in [start, end].

        Optimized computations will be used if:
            - the parametrizing curve is T-periodic and the sampling is s.t.
        T % step == 0,
            - under the previous condition, the parametrizing curve's arc
        length is an even function and (T / 2) % step == 0,
            - TODO: the curve is symmetric wrt the x-axis.
        """
        if (not reuse
            or self._check_stale_data([start, end, nb_pts])):
            self._compute_range_data(start, end, nb_pts)
        return self._get_curve_from_data()

    def _check_stale_data(self, new=()):
        """Returns True if the intermediary data should be updated.

        Used for partial memoization.
        """
        new += (tuple(vars(self.m_obj).values())
                + tuple(vars(self.n_obj).values())
                + (self.par,))
        if self._old:
            stale = new != self._old
        else:
            stale = True
        self._old = new
        return stale

    def _compute_range_data(self, start, end, nb_pts):
        """Compute range data with param values evenly sampled in [start, end].

        Optimized computations will be used if:
            - the parametrizing curve is T-periodic and the sampling is s.t.
        T % step == 0,
            - under the previous condition, the parametrizing curve's arc
        length is an even function and (T / 2) % step == 0,
            - TODO: the curve is symmetric wrt the x-axis.
        """
        if self.par == 'moving':
            ref_obj = self.m_obj
            aux_obj = self.n_obj
        else:
            ref_obj = self.n_obj
            aux_obj = self.m_obj

        # Check if optimizations can be done.
        step = abs(end - start) / (nb_pts - 1)
        per = ref_obj.get_period()
        if per is None or per >= end or per % step != 0:
            return self.get_point(np.linspace(start, end, nb_pts))

        # Get the parameter values for both curves.
        nb_cycles, rem = divmod(abs(end - start), per)
        nb_cycles = int(nb_cycles)
        nb_rem_pts = int(rem / step + 1)

        tr = np.linspace(start, per, per / step + 1)
        if ref_obj.has_even_arclength() and (per / 2) % step == 0:
            # Compute s(t) on [0, T/2].
            svals = ref_obj.get_arclength(tr[:(len(tr) + 1) // 2])
            # Mirror the variation, integrate and shift.
            svals_diff= np.cumsum((svals[1:] - svals[:-1])[::-1]) + svals[-1]
            svals = np.concatenate([svals, svals_diff])
        else:
            # Compute s(t) on [0, T].
            svals = ref_obj.get_arclength(tr)
        svals = np.concatenate(
            [svals[:-1] + i * svals[-1] for i in range(int(nb_cycles))] +
            [svals[:nb_rem_pts] + nb_cycles * svals[-1]])
        ta = aux_obj.get_arclength_inv(svals)

        # Get the points' coordinates from the parameter values.
        ref_pts = ref_obj.get_point(tr)
        ref_pts = np.hstack([np.tile(ref_pts[:, :-1], (1, nb_cycles)),
                             ref_pts[:, :nb_rem_pts]])
        aux_pts = aux_obj.get_point(ta)
        # Compute pairs of jacobians.
        ref_jacs = ref_obj.get_jac(tr) / ref_obj.get_arclength_der(tr)
        ref_jacs = np.hstack([np.tile(ref_jacs[:, :-1], (1, nb_cycles)),
                              ref_jacs[:, :nb_rem_pts]])
        aux_jacs = aux_obj.get_jac(ta) / aux_obj.get_arclength_der(ta)

        # Compute the rotations between each pair of jacobians.
        if self.par == 'moving':
            self.m_pts = ref_pts
            self.n_pts = aux_pts
            self.rot = _get_rotation(ref_jacs, aux_jacs)
        else:
            self.m_pts = aux_pts
            self.n_pts = ref_pts
            self.rot = _get_rotation(aux_jacs, ref_jacs)

    def _get_curve_from_data(self):
        # Apply the general roulette formula:
        # P = F + R(T-M)
        return self.n_pts + np.einsum('ijk,jk->ik',
                                      self.rot, self.T - self.m_pts)

    def update_pole(self, pole_dist):
        """Update the roulette to a different pole position.

        This convenience function is here to avoid recomputing the same
        intermediary data when all that is needed is a change to the pole
        position.
        In order to work, get_point() or get_range() must have been called
        earlier.
        """
        self.T[0] = pole_dist
        # This will raise an AttributeError if the function is called before
        # the data has been cached.
        return self._get_curve_from_data()


class CircleInvolute(Curve):
    """Parametric involute of a circle."""

    def __init__(self, radius, phase=0):
        self.reset(radius, phase)

    def reset(self, radius, phase):
        """Reset the class fields."""
        self.r = radius
        self.phi = phase

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        cos = np.cos(t + self.phi)
        sin = np.sin(t + self.phi)
        return self.r * np.vstack([cos + t * sin, sin - t * cos])

