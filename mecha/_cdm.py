# -*- coding: utf-8 -*-
"""
Cycloid Drawing Machine

@author: Robin Roussel
"""
# from itertools import product
import math
import numpy as np
import scipy.optimize as opt

from ._mecha import Mechanism, DrawingMechanism
from utils import skipends, farey


class SingleGearFixedFulcrumCDM(DrawingMechanism):
    """Cycloid Drawing Machine with the 'simple setup'."""
    param_names = ["r_T", "r_G", "$d_F$", r"$ \theta_G$", "$d_P$", "$d_S$"]

    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 4
        max_nb_turns = 10  # Arbitrary value

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                cstr[cls] = (
                    lambda p: p[0] - 1,                     # 00: r_T > 0
                    lambda p: cls.max_nb_turns - p[0],      # 01: r_T <= B
                    lambda p: p[1] - 1,                     # 02 r_G > 0
                    lambda p: cls.max_nb_turns - p[1],      # 03: r_G <= B
                    lambda p: p[2] - p[0] - cls.eps,        # 04: d_F > r_T
                    lambda p: 2*cls.max_nb_turns - p[2],    # 05: d_F <= 2B
                    lambda p: p[3],                         # 06: theta_G >= 0
                    lambda p: math.pi - p[3],               # 07: theta_G <= pi
                    lambda p: p[4] - cls.eps,               # 08: d_P > 0
                    lambda p: (                             # 09: d_P < FG-r_G
                        cls._get_FG(*p[:4]) - p[1] - p[4] - cls.eps),
                    lambda p: p[5],                         # 10: d_S >= 0
                    lambda p: p[1] - p[5] - cls.eps,        # 11: d_S < r_G
                    lambda p: (                             # 12: r_T >= OP_max
                        p[0]**2 - cls._get_OP2_max(*p))
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
        def sample_feasible_domain(cls, grid_resol=(5, 5, 5, 5),
                                   fixed_values=None):
            """Sample the feasible domain."""
            if fixed_values is not None:
                r_T_f, r_G_f, *_ = fixed_values
            else:
                r_T_f = None
                r_G_f = None
            if r_T_f is not None:
                if r_G_f is not None:
                    yield from cls.sample_feasible_continuous_domain(
                            r_T_f, r_G_f, grid_resol)
                else:
                    for r_G in range(1, cls.max_nb_turns + 1):
                        if r_T_f == 1 or r_G != r_T_f:
                            yield from cls.sample_feasible_continuous_domain(
                                    r_T_f, r_G, grid_resol)
            else:
                if r_G_f is not None:
                    for r_T in range(1, cls.max_nb_turns + 1):
                        if r_G_f == 1 or r_T != r_G_f:
                            yield from cls.sample_feasible_continuous_domain(
                                    r_T, r_G_f, grid_resol)
                else:
                    for r_G, r_T in skipends(farey(cls.max_nb_turns)):
                        yield from cls.sample_feasible_continuous_domain(
                                r_T, r_G, grid_resol)
                        yield from cls.sample_feasible_continuous_domain(
                                r_G, r_T, grid_resol)
                    yield from cls.sample_feasible_continuous_domain(
                            1, 1, grid_resol)

        @classmethod
        def sample_feasible_continuous_domain(cls, r_T, r_G,
                                              grid_resol=(5, 5, 5, 5)):
            """Sample the feasible continuous domain."""
            n = grid_resol[-4:]
            eps = 2*cls.eps

            for d_F in np.linspace(r_T+eps, 2*r_T, n[0], endpoint=False):

                for theta_G in np.linspace(
                        cls._get_theta_min(r_T, r_G, d_F)+eps, 3.1415, n[1]):

                    for d_P in np.linspace(
                            eps, cls._get_FG(r_T, r_G, d_F, theta_G)-r_G-eps,
                            n[2], endpoint=False):

                        if cls.get_constraints()[-1](
                                (r_T, r_G, d_F, theta_G, d_P, eps)) < 0:
                            # This constraint is strictly decreasing wrt d_S:
                            # If there's no solution for d_S = 0, there's
                            # no solution at all.
                            continue
                        d_S_bnds = cls.get_bounds(
                            (r_T, r_G, d_F, theta_G, d_P, 0.), 5)
                        if d_S_bnds[0] == d_S_bnds[1]:
                            # Only happens if min = max = 0, which produces
                            # a circle.
                            continue
                        for d_S in np.linspace(
                                d_S_bnds[0]+eps, d_S_bnds[1]-eps, n[3]):
                            yield r_T, r_G, d_F, theta_G, d_P, d_S

        @staticmethod
        def _get_FG(r_T, r_G, d_F, theta_G):
            """Get the distance between the fulcrum and the gear center."""
            OG = r_T + r_G
            return np.sqrt(d_F**2 + OG**2 - 2*d_F*OG*np.cos(theta_G))

        @staticmethod
        def _get_OP2_max(r_T, r_G, d_F, theta_G, d_P, d_S):
            """Get the max distance between the center and the penholder."""
            OG = r_T + r_G
            OGx = OG*np.cos(theta_G)
            FG = np.sqrt(d_F**2 + OG**2 - 2*d_F*OGx)
            alpha = np.arctan2(d_S, FG)
            theta_fg = math.pi - np.arctan2(OG*np.sin(theta_G), d_F - OGx)
            return d_F**2 + d_P**2 + 2*d_F*d_P*np.cos(theta_fg - alpha)

        @staticmethod
        def _get_theta_min(r_T, r_G, d_F):
            """Get the minimum polar angle of the gear."""
            # Angle s.t FG is tangent to the turntable. H is the point of
            # tangency.
            OG = r_T + r_G
            OG2 = OG**2
            d_F2 = d_F**2
            r_T2 = r_T**2
            FG = np.sqrt(d_F2 - r_T2) + np.sqrt(OG2 - r_T2)  # = FH + HG
            # Law of cosines
            return np.arccos((OG2 + d_F2 - FG**2) / (2*d_F*OG))

    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the properties.

            properties = r_T, r_G, d_F, theta_G, d_P, d_S
            with:
                r_T: Turntable gear radius
                r_G: External gear radius
                d_F: Distance from turntable center to fulcrum center
                theta_G: Polar angle of the external gear
                d_P: Fulcrum-penholder distance
                d_S: Distance from external gear center to slider
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
            r_T, r_G, _, _, _, d_S = self.props
            gcd_ = math.gcd(int(r_T), int(r_G))
            nb_turns = r_G / gcd_
            if not d_S:
                # Degenerate case.
                nb_turns /= r_T / gcd_
            return 2*math.pi*nb_turns

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            length = self.get_cycle_length()
            if self.per_turn:
                nb_samples = int(length / (2*math.pi)) * self.nb_samples + 1
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
            r_T, r_G, d_F, theta_G, d_P, d_S = self.props
            C_f = np.array([[d_F], [0.]])
            # Slider curve
            C_g = (r_T + r_G) * np.array([[math.cos(theta_G)],
                                          [math.sin(theta_G)]])
            theta = - r_T * t / r_G
            OS = d_S * np.vstack([np.cos(theta), np.sin(theta)]) + C_g
            # Connecting rod vector
            FS = OS - C_f
            # Penholder curve
            OP = C_f + FS * d_P / np.sqrt(FS[0]**2 + FS[1]**2)
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
