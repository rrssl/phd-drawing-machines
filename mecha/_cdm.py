#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cycloid Drawing Machine

@author: Robin Roussel
"""
#from itertools import product
import math
import numpy as np
import scipy.optimize as opt

from ._mecha import Mechanism, DrawingMechanism
from utils import skipends, farey


class SingleGearFixedFulcrumCDM(DrawingMechanism):
    """Cycloid Drawing Machine with the 'simple setup'."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 4
        max_nb_turns = 10 # Arbitrary value

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                cstr[cls] = (
                    lambda p: p[0] - 1,                     # 00: R_t > 0
                    lambda p: cls.max_nb_turns - p[0],      # 01: R_t <= B
                    lambda p: p[1] - 1,                     # 02 R_g > 0
                    lambda p: cls.max_nb_turns - p[1],      # 03: R_g <= B
                    lambda p: p[2] - p[0] - cls.eps,        # 04: d_f > R_t
                    lambda p: 2*cls.max_nb_turns - p[2],    # 05: d_f <= 2B
                    lambda p: p[3],                         # 06: theta_g >= 0
                    lambda p: math.pi - p[3],               # 07: theta_g <= pi
                    lambda p: p[4] - cls.eps,               # 08: d_p > 0
                    lambda p: cls._get_FG(*p[:4]) - p[1] - p[4] - cls.eps,
                                                            # 09: d_p < FG-R_g
                    lambda p: p[5],                         # 10: d_s >= 0
                    lambda p: p[1] - p[5] - cls.eps,        # 11: d_s < R_g
                    lambda p: p[0]**2 - cls._get_OP2_max(*p)
                                                            # 12: R_t >= OP_max
                    )
                return cstr[cls]

        @classmethod
        def get_bounds(cls, prop, pid):
            """Get the bounds of the property prop[pid]."""
            assert(0 <= pid < len(prop))
            prop = list(prop)
            def adapt(cstr):
                return lambda x: cstr(prop[:pid] + [x] + prop[pid+1:])

            cstrs = cls.get_constraints()
            if pid == 0:
                ids = [0, 1, 4, 9, -1]
            elif pid == 1:
                ids = [2, 3, 9, -2, -1]
            elif pid == 2:
                ids = [4, 5, 9, -1]
            elif pid == 3:
                ids = [6, 7, 9, -1]
            elif pid == 4:
                ids = [8, 9, -1]
            else:
                ids = [10, 11, -1]
            cstrs = [adapt(cstrs[i]) for i in ids]

            min_ = opt.fmin_cobyla(
                lambda x: x, prop[pid], cons=cstrs, disp=0, catol=cls.eps)
            max_ = opt.fmin_cobyla(
                lambda x: -x, prop[pid], cons=cstrs, disp=0, catol=cls.eps)

#            if max_ - min_ > 4*cls.eps:
#                min_ += 2*cls.eps
#                max_ -= 2*cls.eps

            if pid == 0 or pid == 1:
                min_ = math.ceil(min_)
                max_ = math.floor(max_)

            return min_, max_

        @classmethod
        def sample_feasible_domain(cls, grid_resol=(5, 5, 5, 5)):
            """Sample the feasible domain."""
            for R_g, R_t in skipends(farey(cls.max_nb_turns)):
                for p in cls.sample_feasible_continuous_domain(
                    R_t, R_g, grid_resol):
                    yield p
                for p in cls.sample_feasible_continuous_domain(
                    R_g, R_t, grid_resol):
                    yield p
            for p in cls.sample_feasible_continuous_domain(
                1, 1, grid_resol):
                yield p

        @classmethod
        def sample_feasible_continuous_domain(cls, R_t, R_g,
                                              grid_resol=(5, 5, 5, 5)):
            """Sample the feasible continuous domain."""
            n = grid_resol[-4:]
            eps = 2*cls.eps

            for d_f in np.linspace(R_t+eps, 2*R_t, n[0]):

                for theta_g in np.linspace(
                    cls._get_theta_min(R_t, R_g, d_f)+eps, math.pi, n[1]):

                    for d_p in np.linspace(eps,
                        cls._get_FG(R_t, R_g, d_f, theta_g)-R_g-eps, n[2]):

                        if cls.get_constraints()[-1](
                            (R_t, R_g, d_f, theta_g, d_p, eps)) < 0:
                            # This constraint is strictly decreasing wrt d_s:
                            # If there's no solution for d_s = 0, there's
                            # no solution at all.
                            continue
                        d_s_bnds = cls.get_bounds(
                            (R_t, R_g, d_f, theta_g, d_p, 0.), 5)
                        if d_s_bnds[0] == d_s_bnds[1]:
                            # Only happens if min = max = 0, which produces
                            # a circle.
                            continue
                        for d_s in np.linspace(
                            d_s_bnds[0]+eps, d_s_bnds[1]-eps, n[3]):
                            yield R_t, R_g, d_f, theta_g, d_p, d_s

        @staticmethod
        def _get_FG(R_t, R_g, d_f, theta_g):
            """Get the distance between the fulcrum and the gear center."""
            OG =  R_t + R_g
            return np.sqrt(d_f**2 + OG**2 - 2*d_f*OG*np.cos(theta_g))

        @staticmethod
        def _get_OP2_max(R_t, R_g, d_f, theta_g, d_p, d_s):
            """Get the maximum distance between the center and the penholder."""
            OG = R_t + R_g
            OGx = OG*np.cos(theta_g)
            FG = np.sqrt(d_f**2 + OG**2 - 2*d_f*OGx)
            alpha = np.arctan2(d_s, FG)
            theta_fg = math.pi - np.arctan2(OG*np.sin(theta_g), d_f - OGx)
            return d_f**2 + d_p**2 + 2*d_f*d_p*np.cos(theta_fg - alpha)

        @staticmethod
        def _get_theta_min(R_t, R_g, d_f):
            """Get the minimum polar angle of the gear."""
            # Angle s.t FG is tangent to the turntable. H is the point of
            # tangency.
            OG = R_t + R_g
            OG2 = OG**2
            d_f2 = d_f**2
            R_t2 = R_t**2
            FG = np.sqrt(d_f2 - R_t2) + np.sqrt(OG2 - R_t2) # = FH + HG
            # Law of cosines
            return np.arccos((OG2 + d_f2 - FG**2) / (2*d_f*OG))


    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the properties.

            properties = R_t, R_g, d_f, theta_g, d_p, d_s
            with:
                R_t: Turntable gear radius
                R_g: External gear radius
                d_f: Distance from turntable center to fulcrum center
                theta_g: Polar angle of the external gear
                d_p: Fulcrum-penholder distance
                d_s: Distance from external gear center to slider
            """
            self.props = list(properties)

            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert (0 <= pid < 6)
            self.props[pid] = value

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            R_t, R_g, _, _, _, d_s = self.props
            gcd_ = math.gcd(int(R_t), int(R_g))
            nb_turns = R_g / gcd_
            if not d_s:
                # Degenerate case.
                nb_turns /= R_t / gcd_
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
            """Compute the state of the assembly at time t."""
            OS, OP, _ = self._compute_vectors(t)
            asb['turntable']['or'] = t
            asb['slider']['pos'] = OS
            asb['pen-holder']['pos'] = OP

        def _compute_vectors(self, t):
            t = np.atleast_1d(t)
            R_t, R_g, d_f, theta_g, d_p, d_s = self.props
            C_f = np.array([[d_f], [0.]])
            # Slider curve
            C_g = (R_t + R_g) * np.array([[math.cos(theta_g)],
                                          [math.sin(theta_g)]])
            theta = - R_t * t / R_g
            OS = d_s * np.vstack([np.cos(theta), np.sin(theta)]) + C_g
            # Connecting rod vector
            FS = OS - C_f
            # Penholder curve
            OP = C_f + FS * d_p / np.sqrt(FS[0]**2 + FS[1]**2)
            # Frame change (Turntable rotation) -- Angle is theta = -t
            cos = np.cos(t)
            sin = np.sin(-t)
            rot = np.array([[cos, -sin], [sin, cos]])
            OP_rot = np.einsum('ijk,jk->ik', rot, OP)

            return OS, OP, OP_rot


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
            'turntable': {'or': None},
            'slider': {'pos': None},
            'pen-holder': {'pos': None}
            }
