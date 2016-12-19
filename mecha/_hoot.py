#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hoot-Nanny

@author: Robin Roussel
"""
#from itertools import product
import math
import numpy as np
import scipy.optimize as opt

from ._mecha import Mechanism, DrawingMechanism #, skipends, farey


class HootNanny(DrawingMechanism):
    """Hoot-Nanny (Magic Designer), aka the HTMLSpirograph."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 3
        nb_cprops = 5
        max_nb_turns = 10 # Arbitrary value

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                cstr[cls] = (
                    lambda p: p[0] - 1,                     # 00: r_T > 0
                    lambda p: cls.max_nb_turns - p[0],      # 01: r_T <= B
                    lambda p: p[1] - 1,                     # 02 r_G1 > 0
                    lambda p: cls.max_nb_turns - p[1],      # 03: r_G1 <= B
                    lambda p: p[2] - 1,                     # 04 r_G2 > 0
                    lambda p: cls.max_nb_turns - p[2],      # 05: r_G2 <= B
                    lambda p: p[3],                         # 06: theta12 >= 0
                    lambda p: math.pi - p[3],               # 07: theta12 <= pi
                    lambda p: p[4],                         # 08: d1 >= 0
                    lambda p: p[1] - p[4],                  # 09: d1 <= r_G1
                    lambda p: p[5],                         # 10: d2 >= 0
                    lambda p: p[2] - p[5],                  # 11: d2 <= r_G2
                    lambda p: p[6] - p[1] - p[4],           # 12: l1 >= r_G1+d1
                    lambda p: 2*p[0] + p[1] - p[4] - p[6],
                                                    # 13: l1 <= 2r_T+r_G1-d1
                    lambda p: p[7] - p[2] - p[5],           # 14: l2 >= r_G2+d2
                    lambda p: 2*p[0] + p[2] - p[5] - p[7],
                                                    # 15: l2 <= 2r_T+r_G2-d2
                    lambda p: cls._get_G1G2(*p[:4]) - p[1] - p[2],
                                                    # 16: G1G2 >= r_G1+r_G2
                    # Sufficient non-singularity condition.
                    lambda p: p[6] + p[7] - cls._get_G1G2(*p[:4]) - p[4] - p[5],
                                                    # 17: l1+l2 >= G1G2+d1+d2
                    # Drawing-inside condition.
                    lambda p: p[0] - cls._get_OH_max(*p)    # 18: OH_max <= r_T
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
                ids = [0, 1, 13, 15, 16, 17, 18]
            elif pid == 1:
                ids = [2, 3, 9, 12, 13, 16, 17, 18]
            elif pid == 2:
                ids = [4, 5, 11, 14, 15, 16, 17, 18]
            elif pid == 3:
                ids = [6, 7, 16, 17, 18]
            elif pid == 4:
                ids = [8, 9, 12, 13, 17, 18]
            elif pid == 5:
                ids = [10, 11, 14, 15, 17, 18]
            elif pid == 6:
                ids = [12, 13, 17, 18]
            else:
                ids = [14, 15, 17, 18]
            cstrs = [adapt(cstrs[i]) for i in ids]

            min_ = opt.fmin_cobyla(
                lambda x: x, prop[pid], cons=cstrs, disp=0, catol=cls.eps)
            max_ = opt.fmin_cobyla(
                lambda x: -x, prop[pid], cons=cstrs, disp=0, catol=cls.eps)

            if pid in (0, 1, 2):
                min_ = math.ceil(min_)
                max_ = math.floor(max_)

            return min_, max_

#        @classmethod
#        def sample_feasible_domain(cls, grid_resol=(10, 10, 10, 10)):
#            """Sample the feasible domain.
#            Works if the domain is convex (the necessary condition is a bit
#            more complex).
#            """
#            n = grid_resol[-4:]
#            p = [0, 0] +  4*[0.]
#            for R_t, R_g in skipends(farey(cls.max_nb_turns)):
#
#                p[:2] = R_t, R_g
#                for d_f in np.linspace(p[0]+cls.eps, 2*cls.max_nb_turns, n[0]):
#
#                    p[2] = d_f
#                    for theta_g in np.linspace(0., np.pi, n[1]):
#
#                        p[3] = theta_g
#                        for d_p in np.linspace(
#                            cls.eps, cls._get_FG(*p[:4]), n[2]):
#
#                            p[4] = d_p
#                            for d_s in np.linspace(
#                                *cls.get_bounds(p, 5), num=n[3]):
#                                p[5] = d_s
#                                yield p.copy()
#
#            for R_g, R_t in skipends(farey(cls.max_nb_turns)):
#                pass # Copy sub-loops

        @staticmethod
        def _get_G1G2(r_T, r_G1, r_G2, theta_12):
            """Get the distance between the gear centers."""
            d_OG1 = r_T + r_G1
            d_OG2 = r_T + r_G2
            return math.sqrt(
                d_OG1**2 + d_OG2**2 - 2*d_OG1*d_OG2*math.cos(theta_12))

        @classmethod
        def _get_OH_max(cls, r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2):
            """Get the maximum distance between the center and the pen."""
            # Compute the 4 candidate points for the extremum.
            OG2 = (r_T + r_G2) * np.vstack(
                [math.cos(theta_12), math.sin(theta_12)])
            G2G1 = np.vstack([r_T + r_G1, 0.]) - OG2
            d_G2G1_sq = G2G1[0]**2 + G2G1[1]**2
            d_G2G1 = math.sqrt(d_G2G1_sq)
            d_G1H = np.r_[l1 - d1, l1 + d1, l1 - d1, l1 + d1]
            d_G2H = np.r_[l2 - d2, l2 - d2, l2 + d2, l2 + d2]
            sgn = -1.
            cos_a = (d_G2G1_sq + d_G2H**2 - d_G1H**2) / (2 * d_G2G1 * d_G2H)
            if (np.abs(cos_a) > 1.).any():
                # I.e. triangle is impossible; happens when constraint 17 is
                # violated.
                # Return a value that will violate constraint 18 as well.
                return 2*r_T
            sin_a = sgn * np.sqrt(1 - cos_a**2)
            rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            OH = OG2 + d_G2H * np.einsum('ijk,jk->ik', rot_a, G2G1 / d_G2G1)
            # Return the maximum distance.
            return np.sqrt(OH[0]**2 + OH[1]**2).max()


    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.reset(properties, nb_samples, per_turn)
            # Structure dict. (useful if we do animations later)
            self.assembly = {
                'pivot_1': None,
                'pivot_2': None,
                'pen': None
            }

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the properties.

            properties = r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2
            with:
                r_T: Turntable radius
                r_G1: Gear 1 radius
                r_G2: Gear 2 radius
                theta_12: Polar angle between gears
                d1: Pivot-gear center distance for gear 1
                d2: Pivot-gear center distance for gear 2
                l1: Length of arm 1
                l2: Length of arm 2
            """
            self.props = list(properties)

            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert (0 <= pid < 8)
            self.props[pid] = value

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            r_T, r_G1, r_G2, _, d1, d2, _, _ = self.props
            gcd_1 = math.gcd(int(r_T), int(r_G1))
            gcd_2 = math.gcd(int(r_T), int(r_G2))
            gcd_ = math.gcd(int(r_G1) // gcd_1, int(r_G2) // gcd_2)
            nb_turns = (r_G1 / gcd_1) * (r_G2 / gcd_2) / gcd_
            if not d1:
                # Degenerate case.
                nb_turns /= r_G1 / (gcd_1*gcd_)
            if not d2:
                # Degenerate case.
                nb_turns /= r_G2 / (gcd_2*gcd_)
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
            OP1, OP2, OH, _ = self._compute_vectors(t)
            asb['turntable']['or'] = t
            asb['pivot_1']['pos'] = OP1
            asb['pivot_2']['pos'] = OP2
            asb['pen-holder']['pos'] = OH

        def _compute_vectors(self, t):
            t = np.atleast_1d(t)
            r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2 = self.props
            # Pivots positions
            # 1
            theta1 = -r_T * t / r_G1
            OG1 = np.array([[r_T + r_G1], [0.]])
            OP1 = OG1 + d1 * np.vstack([np.cos(theta1), np.sin(theta1)])
            # 2
            theta2 = -r_T * t / r_G2
            OG2 = (r_T + r_G2) * np.array(
                [[math.cos(theta_12)], [math.sin(theta_12)]])
            OP2 = OG2 + d2 * np.vstack([np.cos(theta2), np.sin(theta2)])
            # Vector between pivots
            P2P1 = OP1 - OP2
            d21_sq = P2P1[0]**2 + P2P1[1]**2
            d21 = np.sqrt(d21_sq)
            # Angle between arm 2 and P2P1 (law of cosines)
            cos_a = (d21_sq + l2**2 - l1**2)  / (2 * d21 * l2)
            sin_a = -np.sqrt(1 - cos_a**2) # sin < 0. for all t by convention
            # Pen-holder in referential R_0
            rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            OH = OP2 + l2 * np.einsum('ijk,jk->ik', rot_a, P2P1 / d21)
            # Frame change (Turntable rotation) -- Angle is theta = -t
            cos_ = np.cos(t)
            sin_ = np.sin(-t)
            rot = np.array([[cos_, -sin_], [sin_, cos_]])
            OH_rot = np.einsum('ijk,jk->ik', rot, OH)

            return OP1, OP2, OH, OH_rot


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
            'pivot_1': {'pos': None},
            'pivot_2': {'pos': None},
            'pen-holder': {'pos': None}
            }
