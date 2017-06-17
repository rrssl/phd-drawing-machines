# -*- coding: utf-8 -*-
"""
Basic Spirograph

@author: Robin Roussel
"""
import math
import numpy as np

from ._mecha import Mechanism, DrawingMechanism
from utils import skipends, farey
from curves import Hypotrochoid


class BaseSpirograph(DrawingMechanism):
    """Internal Spirograph with circular gears."""

    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 1
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
                    lambda p: p[1] - p[2] - cls.eps,
                    lambda p: p[2]
                    )
                return cstr[cls]

        @classmethod
        def get_bounds(cls, prop, pid):
            """Return the bounds of the pid-th element of the input list."""
            assert(0 <= pid < len(prop))
            if pid == 0:
                return prop[1] + 1, cls.max_nb_turns
            elif pid == 1:
                return max(1, int(prop[2]+1)), prop[0] - 1
            else:
                return 0., prop[1] - 2*cls.eps

        @classmethod
        def sample_feasible_domain(cls, grid_resol=(10,)):
            """Sample the feasible domain."""
            n_R, n_d = cls.max_nb_turns, grid_resol[-1]
            d_arr = [np.linspace(0, l - 2*cls.eps, n_d)
                     for l in range(1, n_R)]
            for r, R in skipends(farey(n_R)):
                for d in d_arr[r - 1]:
                    yield R, r, d

    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.trocho = None
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the class fields."""
            try:
                self.trocho.reset(*properties)
            except AttributeError:
                self.trocho = Hypotrochoid(*properties)
            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            if self.trocho.d:
                gcd_ = math.gcd(int(self.trocho.R), int(self.trocho.r))
                nb_turns = self.trocho.r / gcd_
            else:
                # Degenerate case.
                nb_turns = 1
            return 2*math.pi*nb_turns

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            length = self.get_cycle_length()
            if self.per_turn:
                nb_samples = (length / (2*math.pi)) * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            t_range = np.linspace(0., length, nb_samples)

            return self.trocho.get_point(t_range)

        def compute_state(self, asb, t):
            """Compute the state of the assembly a time t."""
            R, r = self.trocho.R, self.trocho.r
            asb['rolling_gear']['pos'] = (R - r) * np.vstack(
                [np.cos(t), np.sin(t)])
            asb['penhole']['pos'] = self.trocho.get_point(t)

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert(0 <= pid < 3)
            if pid == 0:
                self.trocho.R = value
            elif pid == 1:
                self.trocho.r = value
            else:
                self.trocho.d = value

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
            'rolling_gear': {'pos': None},
            'penhole': {'pos': None}
            }
