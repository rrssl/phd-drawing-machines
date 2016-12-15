#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball-kicking leg

@author: Robin Roussel
"""
import math
import numpy as np

import mecha.context
from mecha._mecha import Mechanism, DrawingMechanism
#from utils import skipends, farey


class Kicker(DrawingMechanism):
    """Ball-kicking leg."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 0
        nb_cprops = 7
        max_nb_turns = 1 # Arbitrary value

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                cstr[cls] = (
                    lambda p: 1,
                    )
                return cstr[cls]

        @classmethod
        def get_bounds(cls, prop, pid):
            """Return the bounds of the pid-th element of the input list."""
            assert(0 <= pid < len(prop))
            return 0., 10.

        @classmethod
        def sample_feasible_domain(cls, grid_resol=(10,)):
            """Sample the feasible domain."""
            yield None


    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""
        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the class fields."""
            self.props = list(properties)
            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            # Gears turn at the same speed: there's only 1 cycle.
            nb_turns = 1
            return 2*math.pi*nb_turns

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            r1, r2, x2, y2, l1, l2, d = self.props
            # Time range
            length = self.get_cycle_length()
            if self.per_turn:
                nb_samples = (length / (2*math.pi)) * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            t_range = np.linspace(0., length, nb_samples)
            return self._compute_vectors(t_range)[-1]

        def compute_state(self, asb, t):
            """Compute the state of the assembly a time t."""
            r1, r2, x2, y2, l1, l2, d = self.props
            OG1, OG2, OP1, OP2, OE = self._compute_vectors(t)
            asb['gear_1']['pos'] = OG1
            asb['gear_2']['pos'] = OG2
            asb['pivot_1']['pos'] = OP1
            asb['pivot_2']['pos'] = OP2
            asb['pivot_12']['pos'] = OP1 + (OE-OP1) * l1 / (l1 + d)
            asb['end_effector']['pos'] = OE

        def _compute_vectors(self, t):
            t = np.atleast_1d(t)
            r1, r2, x2, y2, l1, l2, d = self.props
            # Fixed points
            OG1 = np.tile([[0.],[0.]], t.shape)
            OG2 = np.tile([[x2],[y2]], t.shape)
            # Equations
            OP1 = r1 * np.vstack([np.cos(t), np.sin(t)])
            OP2 = r2 * np.vstack([np.cos(-t), np.sin(-t)]) + OG2
            P1P2 = OP2 - OP1
            d12_sq = P1P2[0]**2 + P1P2[1]**2
            d12 = np.sqrt(d12_sq)
            cos_a = (d12_sq + l1**2 - l2**2) / (2.*l1*d12)
            # Under the non-singularity constraint sin > 0 for all t
            sin_a = np.sqrt(1. - cos_a**2)
            rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            OE = OP1 + (d+l1) * np.einsum('ijk,jk->ik', rot_a, P1P2 / d12)
            return OG1, OG2, OP1, OP2, OE

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert(0 <= pid < 7)
            self.props[pid] = value


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
            'gear_1': {'pos': None},
            'gear_2': {'pos': None},
            'pivot_1': {'pos': None},
            'pivot_2': {'pos': None},
            'pivot_12': {'pos': None},
            'end_effector': {'pos': None}
            }
