#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ball-kicking leg

@author: Robin Roussel
"""
import math
import numpy as np
import scipy.optimize as opt
from tinyik import Actuator, ConjugateGradientOptimizer

from ._mecha import Mechanism, DrawingMechanism
#from utils import skipends, farey


class Kicker(DrawingMechanism):
    """Ball-kicking leg."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 0
        nb_cprops = 6
        max_nb_turns = 1 # Arbitrary value
        _prop_constraint_map = {
            0: (0, 2, 4, 5),
            1: (1, 2, 4, 5),
            2: (2, 4, 5),
            3: (4, 5),
            4: (4, 5),
            5: (3, )
            }

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data."""
            try:
                return cstr[cls]
            except KeyError:
                cstr[cls] = (
                    lambda p: p[0],            # 00: r1 >= 0
                    lambda p: p[1],            # 01 r2 >= 0
                    lambda p: p[2] - p[0] - p[1] - cls.eps,
                                               # 02: d_G1G2 > r1+r2
                    lambda p: p[5],            # 03: d >= 0
                    # Sufficient non-singularity conditions.
                    lambda p: p[3] + p[4] - p[0] - p[1] - p[2] - cls.eps,
                                               # 04: l1+l2 > d_G1G2+r1+r2
                    lambda p: p[2] - np.abs(p[0]-p[1]) - np.abs(p[3]-p[4]) - cls.eps
                                               # 05: d_G1G2 - |r1-r2| > |l1-l2|
                    # Warning: this last condition works because both gears
                    # turn at the _same_speed_ with the same _phase_.
                    # The left member is min(d_P1P2).
                    # In the general case it could be d_G1G2 - (r1+r2), but
                    # using this as an upper bound for |l1-l2| is quite
                    # conservative.
                    )
                return cstr[cls]

        @classmethod
        def get_bounds(cls, prop, pid):
            """Get the bounds of the property prop[pid]."""
            assert(0 <= pid < len(prop))

            if pid == 5:
                return 0., np.inf

            prop = list(prop)
            cs = cls.get_constraints()
            cs = [cs[i] for i in cls._prop_constraint_map[pid]]
            def get_cons_vec(x):
                prop[pid] = x
                return np.hstack([c(prop) for c in cs])

            if pid == 0 or pid == 1:
                max_ = opt.fmin_cobyla(
                    lambda x: -x, prop[pid], cons=get_cons_vec,
                    disp=0, catol=cls.eps)
                return 0., max_
            if pid == 3 or pid == 4:
                min_ = opt.fmin_cobyla(
                    lambda x: x, prop[pid], cons=get_cons_vec,
                    disp=0, catol=cls.eps)
                return min_, np.inf
            else:
                prop = np.column_stack((prop, prop))
                min_, max_ = opt.fmin_cobyla(
                    lambda x: x[0]-x[1], prop[pid], cons=get_cons_vec,
                    disp=0, catol=cls.eps)
                return min_, max_

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
            self._reset_linkage()

            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def _reset_linkage(self):
            r1, r2, d_G1G2, l1, l2, d = self.props

            halflen = (l1 + d)
            thigh = [-halflen, 0., 0.]
            calf = [0., -halflen, 0.]
            foot = [-.2*halflen,  0., 0.]
            # Quick experiments suggest that the CG optimizer is best here.
            self.leg = Actuator(['z', thigh, 'z', calf, 'z', foot],
                                optimizer=ConjugateGradientOptimizer())
            # TODO: let origin be movable by the user.
            self.leg._origin = np.array([[d_G1G2*1.2], [halflen]])
            # The angles output by tinyik are relative from one link to the
            # next, and for two consecutive links, relative to the original
            # angle between them; therefore we need to keep track of the
            # original link-to-link angles.
            abs_angles = np.array(
                [0.] + [math.atan2(v[1], v[0]) for v in (thigh, calf, foot)]
                )
            self.leg._init_angles = abs_angles[1:] - abs_angles[:-1]
            self.leg._halflen = halflen

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert(0 <= pid < 7)
            self.props[pid] = value
            self._reset_linkage()

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            # Gears turn at the same speed: there's only 1 cycle.
            nb_turns = 1
            return 2*math.pi*nb_turns

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
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
            _, _, _, l1, _, d = self.props
            OG1, OG2, OP1, OP2, OC = self._compute_vectors(t)
            asb['gear_1']['pos'] = OG1
            asb['gear_2']['pos'] = OG2
            asb['pivot_1']['pos'] = OP1
            asb['pivot_2']['pos'] = OP2
            asb['pivot_12']['pos'] = OP1 + (OC-OP1) * l1 / (l1 + d)
            asb['connector']['pos'] = OC
            self._compute_IK(asb, OC)

        def _compute_IK(self, asb, pos):
            try:
                self.leg.ee = np.append(
                    (pos - self.leg._origin).ravel(), 0.)
            except np.linalg.LinAlgError:
                print("IK Error: Singular matrix")
            a1 = self.leg.angles[0] + self.leg._init_angles[0]
            a2 = a1 + self.leg.angles[1] + self.leg._init_angles[1]
            asb['hip']['pos'] = self.leg._origin
            asb['knee']['pos'] = self.leg._halflen * np.array([
                [math.cos(a1)], [math.sin(a1)]]) + asb['hip']['pos']
            asb['ankle']['pos'] = self.leg._halflen * np.array([
                [math.cos(a2)], [math.sin(a2)]]) + asb['knee']['pos']
            asb['end_effector']['pos'] = asb['ankle']['pos'] + (
                (pos - asb['ankle']['pos'])*1.5)

        def _compute_vectors(self, t):
            t = np.atleast_1d(-t)
            r1, r2, d_G1G2, l1, l2, d = self.props
            # Fixed points
            OG1 = np.array([[0.], [0.]])
            OG2 = np.array([[d_G1G2],[0.]])
            # Equations
            OP1 = r1 * np.vstack([np.cos(t), np.sin(t)])
            OP2 = r2 * np.vstack([np.cos(-t), np.sin(-t)]) + OG2
            P1P2 = OP2 - OP1
            d12_sq = P1P2[0]**2 + P1P2[1]**2
            d12 = np.sqrt(d12_sq)
            cos_a = (d12_sq + l1**2 - l2**2) / (2.*l1*d12)
            # Under the non-singularity constraint sin > 0 for all t
            sin_a = -np.sqrt(1. - cos_a**2)
            rot_a = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            OC = OP1 + (d+l1) * np.einsum('ijk,jk->ik', rot_a, P1P2 / d12)
            return OG1, OG2, OP1, OP2, OC


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
#        OC = self._simulator.simulate_cycle()
#        OE = []
#        for pos in OC.T:
#            self._simulator._compute_IK(self.assembly, pos.reshape(2, 1))
#            OE.append(self.assembly['end_effector']['pos'])
#        return np.hstack(OE)
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
            'connector': {'pos': None},
            'hip': {'pos': None},
            'knee': {'pos': None},
            'ankle': {'pos': None},
            'end_effector': {'pos': None}
            }
