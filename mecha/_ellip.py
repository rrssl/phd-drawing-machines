# -*- coding: utf-8 -*-
"""
Elliptic Spirograph

@author: Robin Roussel
"""
import math
import numpy as np
import scipy.optimize as opt
import scipy.special as spec

from ._mecha import Mechanism, DrawingMechanism
from utils import skipends, farey
import curves as cu


class EllipticSpirograph(DrawingMechanism):
    """Spirograph with an elliptic gear."""

    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 2
        max_nb_turns = 20  # Arbitrary value

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                cstr[cls] = (
                    lambda p: cls.max_nb_turns - p[0],
                    lambda p: p[0] - p[1] - 1,
                    lambda p: p[2],
                    lambda p: cls._get_e2max(p[0], p[1]) - p[2] - cls.eps,
                    lambda p: p[3],
                    lambda p: cls._get_dmax(p[1], p[2]) - p[3] - cls.eps
                    )
                return cstr[cls]

        @classmethod
        def get_bounds(cls, prop, pid):
            """Get the bounds of the property prop[pid]."""
            assert(0 <= pid < len(prop))
            if pid == 0:
                return prop[1] + 1, cls.max_nb_turns
            elif pid == 1:
                return max(1, cls._get_reqmin(*prop[2:])), prop[0] - 1
            elif pid == 2:
                return 0., cls._get_e2max(prop[0], prop[1]) - 2*cls.eps
            else:
                return 0., cls._get_dmax(prop[1], prop[2]) - 2*cls.eps

        @classmethod
        def sample_feasible_domain(cls, grid_resol=(10, 10),
                                   fixed_values=None):
            """Sample the feasible domain."""
            if fixed_values is not None:
                R_f, req_f, *_ = fixed_values
            else:
                R_f = None
                req_f = None
            if R_f is not None:
                if req_f is not None:
                    yield from cls.sample_feasible_continuous_domain(
                        R_f, req_f, grid_resol)
                else:
                    for req in range(1, R_f):
                        yield from cls.sample_feasible_continuous_domain(
                            R_f, req, grid_resol)
            else:
                if req_f is not None:
                    for R in range(req_f+1, cls.max_nb_turns):
                        yield from cls.sample_feasible_continuous_domain(
                            R, req_f, grid_resol)
                else:
                    for req, R in skipends(farey(cls.max_nb_turns)):
                        yield from cls.sample_feasible_continuous_domain(
                            R, req, grid_resol)

        @classmethod
        def sample_feasible_continuous_domain(
                cls, R, req, grid_resol=(10, 10)):
            n_e2, n_d = grid_resol[-2], grid_resol[-1]
            emax2 = cls._get_e2max(R, req) - 2*cls.eps
            for e2 in np.linspace(0, emax2, n_e2, endpoint=False):

                dmax = cls._get_dmax(req, e2) - 2*cls.eps
                for d in np.linspace(0, dmax, n_d, endpoint=False):
                    yield R, req, e2, d

        @staticmethod
        def _get_e2max(R, req):
            """Get the upper bound of the squared eccentricity."""
            # Approximate this bound using an inversion of the Naive
            # Approximation of the ellipse circumference.
            emax2_approx = ((R - 4*req) + np.sqrt(R*(R + 8*req))) / (2*R)
            # Compute the exact bound.
            return opt.fsolve(
                lambda x: x + (req*math.pi / (2*spec.ellipe(x)*R)) ** 2 - 1,
                emax2_approx)[0]

        @staticmethod
        def _get_dmax(req, e2):
            """Get the upper bound of the pole distance."""
            return req * math.pi / (2 * spec.ellipe(e2))  # dmax = semimajor

        @staticmethod
        def _get_reqmin(e2, d):
            """Get the lower bound of the equivalent radius."""
            return int((2 * d * spec.ellipe(e2) / math.pi) + 1)

    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.roul = None
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the class fields."""
            if self.roul:
                self.roul.n_obj.reset(properties[0])
                self.roul.m_obj.reset(properties[1], properties[2])
                self.roul.reset(properties[-1])
            else:
                cir = cu.Circle(properties[0])
                ell = cu.Ellipse2(properties[1], properties[2])
                self.roul = cu.Roulette(ell, cir, properties[-1], 'moving')
            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            gcd_ = math.gcd(int(self.roul.n_obj.r), int(self.roul.m_obj.req))
            nb_turns = self.roul.n_obj.r / gcd_
            if not self.roul.T[0]:
                # Degenerate cases.
                if not self.roul.m_obj.e2:
                    nb_turns /= self.roul.m_obj.req
                elif not self.roul.m_obj.req % 2:
                    # Since (r,req) = 1, if req is even then r is odd.
                    nb_turns /= 2
            return 2*math.pi*nb_turns

        def simulate_cycle(self, reuse=True):
            """Simulate one cycle of the assembly's motion."""
            length = self.get_cycle_length()
            if self.per_turn:
                nb_samples = int(length / (2*math.pi)) * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            curve = self.roul.get_range(0., length, nb_samples, reuse)

            assert(not np.isnan(curve.min()))
            return curve

        def compute_state(self, asb, t):
            """Compute the state of the assembly a time t."""
            d = float(self.roul.T[0])
            asb['penhole']['pos'] = self.roul.get_point(t)
            asb['rolling_gear']['pos'] = self.roul.update_pole(0.)
            asb['rolling_gear']['or'] = np.arctan2(
                self.roul.rot[1, 0], self.roul.rot[0, 0])
            self.roul.T[0] = d

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert (0 <= pid < 4)
            if pid == 0:
                self.roul.n_obj.reset(value)
            elif pid == 1:
                self.roul.m_obj.reset(value, self.roul.m_obj.e2)
            elif pid == 2:
                self.roul.m_obj.reset(self.roul.m_obj.req, value)
            else:
                self.roul.reset(value)

    def get_curve(self, nb=2**6, per_turn=True):
        """Return an array of points sampled on the full curve.

        By default the number of points is not absolute, but rather "per turn"
        of the input driver. In this case, powers of 2 can make the
        computation faster.
        The curve parameter is evenly sampled, but the curve points are not
        necessarily evenly spaced.
        """
        self._simulator.nb_samples = nb
        self._simulator.per_turn = per_turn
        return self._simulator.simulate_cycle()

    def get_point(self, t):
        """Return a specific curve point."""
        raise NotImplementedError

    def get_arc(self, start, stop, nb):
        """Return 'nb' curve points evenly sampled on a parameter interval.

        The curve parameter is evenly sampled, but the curve points are not
        necessarily evenly spaced.
        """
        raise NotImplementedError

    @staticmethod
    def _create_assembly():
        return {
            'rolling_gear': {'pos': None, 'or': None},
            'penhole': {'pos': None}
            }
