#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Personal creation

@author: Robin Roussel
"""
#from itertools import product
import math
import numpy as np
#import scipy.optimize as opt

from ._mecha import Mechanism, DrawingMechanism
#from utils import skipends, farey

NB_GEARS = 5
RADII = 1. / np.arange(2, 2+NB_GEARS)[::-1]
RADII[2] = 6/7
#RADII[0] = 2


class Thing(DrawingMechanism):
    """Just a thing."""
    param_names = ["a_{}".format(i+1) for i in range(NB_GEARS)]


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 0
        nb_cprops = NB_GEARS
        max_nb_turns = 1 # Arbitrary value
        _prop_constraint_map = {
            i: ((i-1)*2+2, (i-1)*2+3) for i in range(NB_GEARS)
            }

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                temp = [(
                    lambda p, i=i: p[i],                        # a_i >= 0
                    lambda p, i=i: cls.get_radius(i) - p[i],    # a_i <= r_i
                    )
                    for i in range(NB_GEARS)
                    ]
                cstr[cls] = [fun for pair in temp for fun in pair]
                return cstr[cls]

        @classmethod
        def get_bounds(cls, prop, pid):
            """Get the bounds of the property prop[pid]."""
            return 0., cls.get_radius(pid)

        @classmethod
        def sample_feasible_domain(cls, grid_resol=(4,)*NB_GEARS):
            """Sample the feasible domain."""
            raise NotImplementedError

        @staticmethod
        def get_radius(i):
            return RADII[i]


    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**8, per_turn=True):
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**8, per_turn=True):
            """Reset the properties.

            properties = a_i
            with:
                a_i: amplitude of gear i
            """
            self.props = list(properties)

            self.nb_samples = nb_samples
            self.per_turn = per_turn

            self.offset = np.zeros((2, 1))
            self.offset = self._compute_vectors(0.)[-1]

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert (0 <= pid < NB_GEARS)
            self.props[pid] = value

            self.offset = np.zeros((2, 1))
            self.offset = self._compute_vectors(0.)[-1]

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            nb_turns = 6
            return 2*math.pi*nb_turns

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            length = self.get_cycle_length()
            if self.per_turn:
                nb_samples = (length / (2*math.pi)) * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            t_range = np.linspace(0., length, nb_samples)
            return self._compute_vectors(t_range)[-1]

        def compute_state(self, asb, t):
            """Compute the state of the assembly a time t."""
            gears, pivots, joints, motion, _ = self._compute_vectors(t)
            asb['turntable']['or'] = t
            for i, pos in enumerate(gears):
                asb['gear_{}'.format(i)]['pos'] = pos
            for i, pos in enumerate(pivots):
                asb['pivot_{}'.format(i)]['pos'] = pos
            for i, pos in enumerate(joints):
                asb['joint_{}'.format(i)]['pos'] = pos
            asb['pen-holder']['pos'] = motion

        def _compute_vectors(self, t):
            t = np.atleast_1d(t)
            # Geometric parameters
            radii = RADII.reshape(-1, 1, 1)
            amplitudes = np.asarray(self.props).reshape(-1, 1, 1)
            angles = np.pi * np.linspace(0., 1., NB_GEARS).reshape(-1, 1, 1)
            gears = (1.+radii) * np.concatenate(
                (np.cos(angles), np.sin(angles)), axis=1)
            gdists = np.linalg.norm(gears[1:] - gears[:-1], axis=1)
            lengths = (gdists + (radii[:-1] - radii[1:]).reshape(-1, 1)) * .5
            # Time-dependent variables
            angles = -t / radii
            pivots = amplitudes * np.concatenate(
                (np.cos(angles), np.sin(angles)), axis=1) + gears
            joints = [pivots[0]]
            for i in range(NB_GEARS-1):
                vec = pivots[i+1] - joints[-1]
                vec *=  lengths[i] / (vec[0]**2 + vec[1]**2)
                joints.append(joints[-1] + vec)
            del joints[0]
            # Shift so that first position coincides with origin
            motion = joints[-1].copy()
            motion -= self.offset
            # Frame change (Turntable rotation) -- Angle is theta = -t
            cos = np.cos(t)
            sin = np.sin(-t)
            rot = np.array([[cos, -sin],
                            [sin, cos]])
            drawing = np.einsum('ijk,jk->ik', rot, motion)

            return gears, pivots, joints, motion, drawing


    def get_curve(self, nb=2**8, per_turn=True):
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
        asb = {
            'turntable': {'or': None},
            'pen-holder': {'pos': None}
            }
        asb.update(
            {'gear_{}'.format(i): {'pos': None} for i in range(NB_GEARS)})
        asb.update(
            {'pivot_{}'.format(i): {'pos': None} for i in range(NB_GEARS)})
        asb.update(
            {'joint_{}'.format(i): {'pos': None} for i in range(NB_GEARS-1)})
        return asb
