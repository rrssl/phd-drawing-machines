#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hoot-Nanny

@author: Robin Roussel
"""
from fractions import gcd
#from itertools import product
import math
import numpy as np
import scipy.optimize as opt
from mecha._mecha import Mechanism, DrawingMechanism #, skipends, farey


class HootNanny(DrawingMechanism):
    """Hoot-Nanny (Magic Designer), aka the HTMLSpirograph."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 3
        nb_cprops = 5
        max_nb_turns = 25 # Arbitrary value

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
                    lambda p: p[0] - cls._get_TM_max(*p)    # 18: TM_max <= r_T
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
                lambda x: x, prop[pid], cons=cstrs, disp=0) + 2*cls.eps
            max_ = opt.fmin_cobyla(
                lambda x: -x, prop[pid], cons=cstrs, disp=0) - 2*cls.eps

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
            d_TG1 = r_T + r_G1
            d_TG2 = r_T + r_G2
            return math.sqrt(
                d_TG1**2 + d_TG2**2 - 2*d_TG1*d_TG2*math.cos(theta_12))

        @classmethod
        def _get_TM_max(cls, r_T, r_G1, r_G2, theta_12, d1, d2, l1, l2):
            """Get the maximum distance between the center and the pen."""
            # Compute the 4 candidate points for the extremum.
            TG2 = (r_T + r_G2) * np.vstack(
                [math.cos(theta_12), math.sin(theta_12)])
            G2G1 = np.vstack([r_T + r_G1, 0.]) - TG2
            d_G2G1_sq = G2G1[0]**2 + G2G1[1]**2
            d_G2G1 = math.sqrt(d_G2G1_sq)
            d_G1M = np.r_[l1 - d1, l1 + d1, l1 - d1, l1 + d1]
            d_G2M = np.r_[l2 - d2, l2 - d2, l2 + d2, l2 + d2]
            sgn = -1.
            cos_a = (d_G2G1_sq + d_G2M**2 - d_G1M**2) / (2 * d_G2G1 * d_G2M)
            if (np.abs(cos_a) > 1.).any():
                # I.e. triangle is impossible; happens when constraint 17 is
                # violated.
                # Return a value that will violate constraint 18 as well.
                return 2*r_T
            sin_a = sgn * np.sqrt(1 - cos_a**2)
            rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            TM = TG2 + d_G2M * np.einsum('ijk,jk->ik', rot_a, G2G1 / d_G2G1)
            # Return the maximum distance.
            return np.sqrt(TM[0]**2 + TM[1]**2).max()


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
            """Reset the properties."""
            self.r_T = properties[0] # Turntable radius
            self.r_G1 = properties[1] # Gear 1 radius
            self.r_G2 = properties[2] # Gear 2 radius
            self.theta_12 = properties[3] # Polar angle between gears
            self.d1 = properties[4] # Pivot-gear center distance for gear 1
            self.d2 = properties[5] # Pivot-gear center distance for gear 2
            self.l1 = properties[6] # Length of arm 1
            self.l2 = properties[7] # Length of arm 2

            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            gcd_1 = gcd(self.r_T, self.r_G1)
            gcd_2 = gcd(self.r_T, self.r_G2)
            gcd_ = gcd(self.r_G1 / gcd_1, self.r_G2 / gcd_2)
            nb_turns = (self.r_G1 / gcd_1) * (self.r_G2 / gcd_2) / gcd_
            if not self.d1:
                # Degenerate case.
                nb_turns /= self.r_G1 / (gcd_1*gcd_)
            if not self.d2:
                # Degenerate case.
                nb_turns /= self.r_G2 / (gcd_2*gcd_)
            if self.per_turn:
                nb_samples = nb_turns * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            interval_length = nb_turns * 2 * math.pi
            # Property range
            t_range = np.linspace(0., interval_length, nb_samples)
            # Pivots positions
            # 1
            theta1 = - (self.r_T / self.r_G1) * t_range
            TG1 = np.vstack([self.r_T + self.r_G1, 0.])
            G1P1 = self.d1 * np.vstack([np.cos(theta1), np.sin(theta1)])
            TP1 = TG1 + G1P1
            # 2
            theta2 = - (self.r_T / self.r_G2) * t_range
            TG2 = (self.r_T + self.r_G2) * np.vstack(
                [math.cos(self.theta_12), math.sin(self.theta_12)])
            G2P2 = self.d2 * np.vstack([np.cos(theta2), np.sin(theta2)])
            TP2 = TG2 + G2P2
            # Vector between pivots
            P2P1 = TP1 - TP2
            d21_sq = P2P1[0]**2 + P2P1[1]**2
            d21 = np.sqrt(d21_sq)
            # Angle between arm 2 and P2P1 (law of cosines)
            sgn = -1.
            cos_a = (d21_sq + self.l2**2 - self.l1**2)  / (2 * d21 * self.l2)
            sin_a = sgn * np.sqrt(1 - cos_a**2)
            # Tracer in referential R_0
            rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            TM = TP2 + self.l2 * np.einsum('ijk,jk->ik', rot_a, P2P1 / d21)
            # Tracer in referential R_T
            cos_ = np.cos(t_range)
            sin_ = np.sin(t_range)
            # /!\ Rotation of -theta
            # FIXME: find out which one it is
            rot = np.array([[cos_, -sin_], [sin_, cos_]])
            curve = np.einsum('ijk,jk->ik', rot, TM)

            # Assign trajectories.
            self.assembly['pivot_1'] = TP1
            self.assembly['pivot_2'] = TP2
            self.assembly['pen'] = TM

            return curve

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert (0 <= pid < 8)
            if pid == 0:
                self.r_T = value
            elif pid == 1:
                self.r_G1 = value
            elif pid == 2:
                self.r_G2 = value
            elif pid == 3:
                self.theta_12 = value
            elif pid == 4:
                self.d1 = value
            elif pid == 5:
                self.d2 = value
            elif pid == 6:
                self.l1 = value
            else:
                self.l2 = value


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
