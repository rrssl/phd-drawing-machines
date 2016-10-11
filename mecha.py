# -*- coding: utf-8 -*-
"""
Parametrized cyclic mechanisms

@author: Robin Roussel
"""

from fractions import gcd
import math
import numpy as np
import scipy.optimize as opt
import scipy.special as spec

from utils import skipends, farey
import curves as cu


class Mechanism:
    """Common base class for all cyclic mechanical assemblies."""


    class ConstraintSolver:
        """Class for handling design constraints.

        Ideally we would use a Constraint Problem Solver, but for now we're
        just going to redefine a custom subclass for each mechanism.

        This class provides both a list of constraints and a per-property
        method to evaluate bounds. Although there is a dual relationship
        between bounds and constraints, and we could write a generic method to
        convert one set into the other, rewriting both of them separately for
        each mechanism allows to present them in a way most suitable for each
        purpose (e.g. by filtering out redundant constraints, or avoiding
        costly bound computations).
        """
        eps = 1e-6
        nb_dprops = 0
        nb_cprops = 0

        @classmethod
        def get_constraints(cls, cstr={}):
            """Get the constraints data.

            Note the mutable default allowing to memoize the constraints.
            """
            raise NotImplementedError

        @classmethod
        def check_constraints(cls, props):
            """Check that the input properties comply with the constraints."""
            return all(cons(props) >= 0 for cons in cls.get_constraints())

        @classmethod
        def get_bounds(cls, prop, pid):
            """Get the bounds of the property prop[pid]."""
            raise NotImplementedError

        @classmethod
        def sample_feasible_domain(cls, grid_resol):
            """Sample the feasible domain."""
            raise NotImplementedError


    class Simulator:
        """Class for simulating the movement of the parts.

        In practice we can redefine a custom subclass with analytic formulas.
        """

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the class fields."""
            raise NotImplementedError

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            raise NotImplementedError

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            raise NotImplementedError


    def __init__(self, *props, verbose=False):
        self.constraint_solver = type(self).ConstraintSolver
        self._simulator = None
        self.structure_graph = None
        self.verbose = verbose

        self.reset(*props)

    def reset(self, *props):
        """Reset all properties at once, if they meet the constraints."""
        if self.constraint_solver.check_constraints(props):
            self.props = list(props)
            if self._simulator:
                self._simulator.reset(props)
            else:
                self._simulator = type(self).Simulator(props)
        else:
            if self.verbose:
                print("Error: invalid mechanism properties.", props)
                print("Constraints: ",
                      [cons(props)
                       for cons in self.constraint_solver.get_constraints()])

    def update_prop(self, pid, value, check=True):
        """Update the property referenced by the input index.

        Returns True if the new value complies with the constraints, False
        otherwise.
        """
        assert(0 <= pid < len(self.props))

        oldval = self.props[pid]
        self.props[pid] = value
        if (not check
            or self.constraint_solver.check_constraints(self.props)):
            self._simulator.update_prop(pid, value)
            return True
        else:
            if self.verbose:
                print("Error: invalid mechanism property.", pid, value)
                print("Constraints: ",
                      [cons(self.props)
                       for cons in self.constraint_solver.get_constraints()])
            self.props[pid] = oldval
            return False

    def get_prop_bounds(self, pid):
        """Get the bounds of the property referenced by the input index."""
        return self.constraint_solver.get_bounds(self.props, pid)



class DrawingMechanism(Mechanism):
    """Functional specialization: Mechanism which can draw a closed curve."""

    def get_curve(self, nb=2**6, per_turn=True):
        """Return an array of points sampled on the full curve.

        By default the number of points is not absolute, but rather "per turn"
        of the input driver. In this case, powers of 2 can make the
        computation faster.
        The curve parameter is evenly sampled, but the curve points are not
        necessarily evenly spaced.
        """
        raise NotImplementedError

    def get_point(self, t):
        """Return a specific curve point."""
        raise NotImplementedError

    def get_arc(self, start, stop, nb):
        """Return 'nb' curve points evenly sampled on a parameter interval.

        The curve parameter is evenly sampled, but the curve points are not
        necessarily evenly spaced.
        """
        raise NotImplementedError



class BaseSpirograph(DrawingMechanism):
    """Internal Spirograph with circular gears."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 1
        max_nb_turns = 20 # Arbitrary value

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
            if self.trocho:
                self.trocho.reset(properties)
            else:
                self.trocho = cu.Hypotrochoid(*properties)
            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            if self.trocho.d:
                gcd_ = gcd(self.trocho.R, self.trocho.r)
                nb_turns = self.trocho.r / gcd_
            else:
                # Degenerate case.
                nb_turns = 1
            if self.per_turn:
                nb_samples = nb_turns * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            interval_length = nb_turns * 2 * math.pi
            t_range = np.linspace(0., interval_length, nb_samples)

            return self.trocho.get_point(t_range)

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



class EllipticSpirograph(DrawingMechanism):
    """Spirograph with an elliptic gear."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 2
        max_nb_turns = 20 # Arbitrary value

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
        def sample_feasible_domain(cls, grid_resol=(10,10)):
            """Sample the feasible domain."""
            # Note: we could try a time-efficient version using numpy's
            # vectorization, progressively building a 4D tensor then
            # 'flattening' it to a Nx4 matrix.
            n_e2, n_d = grid_resol[-2], grid_resol[-1]
            for req, R in skipends(farey(cls.max_nb_turns)):

                emax2 = cls._get_e2max(R, req) - 2*cls.eps
                for e2 in np.linspace(0, emax2, n_e2):

                    dmax = cls._get_dmax(req, e2) - 2*cls.eps
                    for d in np.linspace(0, dmax, n_d):
                        yield R, req, e2, d

        @staticmethod
        def _get_e2max(R, req):
            """Get the upper bound of the squared eccentricity."""
            # Approximate this bound using an inversion of the Naive
            # Approximation of the ellipse circumference.
            emax2_approx = ((R - 4*req) + math.sqrt(R*(R + 8*req))) / (2*R)
            # Compute the exact bound.
            return opt.fsolve(
                lambda x: x + (req*math.pi / (2*spec.ellipe(x)*R)) ** 2 - 1,
                emax2_approx)[0]

        @staticmethod
        def _get_dmax(req, e2):
            """Get the upper bound of the pole distance."""
            return req * math.pi / (2 * spec.ellipe(e2)) # dmax = semimajor

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

        def simulate_cycle(self, reuse=True):
            """Simulate one cycle of the assembly's motion."""
            gcd_ = gcd(self.roul.n_obj.r, self.roul.m_obj.req)
            nb_turns = self.roul.n_obj.r / gcd_
            if not self.roul.T[0]:
                # Degenerate cases.
                if not self.roul.m_obj.e2:
                    nb_turns /= self.roul.m_obj.req
                elif not self.roul.m_obj.req % 2:
                    # Since (r,req) = 1, if req is even then r is odd.
                    nb_turns /= 2

            if self.per_turn:
                nb_samples = nb_turns * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            interval_length = nb_turns * 2 * math.pi
            curve = self.roul.get_range(0., interval_length, nb_samples, reuse)

            assert(not np.isnan(curve.min()))
            return curve

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



class SingleGearFixedFulcrumCDM(DrawingMechanism):
    """Cycloid Drawing Machine with the 'simple setup'."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 2
        nb_cprops = 4
        max_nb_turns = 25 # Arbitrary value

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
                lambda x: x, prop[pid], cons=cstrs, disp=0) + 2*cls.eps
            max_ = opt.fmin_cobyla(
                lambda x: -x, prop[pid], cons=cstrs, disp=0) - 2*cls.eps

            if pid == 0 or pid == 1:
                min_ = math.ceil(min_)
                max_ = math.floor(max_)

            return min_, max_

        @classmethod
        def sample_feasible_domain(cls, grid_resol=(10, 10, 10, 10)):
            """Sample the feasible domain.
            Works if the domain is convex (the necessary condition is a bit
            more complex).
            """
            n = grid_resol[-4:]
            p = [0, 0] +  4*[0.]
            for R_t, R_g in skipends(farey(cls.max_nb_turns)):

                p[:2] = R_t, R_g
                for d_f in np.linspace(p[0]+cls.eps, 2*cls.max_nb_turns, n[0]):

                    p[2] = d_f
                    for theta_g in np.linspace(0., np.pi, n[1]):

                        p[3] = theta_g
                        for d_p in np.linspace(
                            cls.eps, cls._get_FG(*p[:4]), n[2]):

                            p[4] = d_p
                            for d_s in np.linspace(
                                *cls.get_bounds(p, 5), num=n[3]):
                                p[5] = d_s
                                yield p.copy()

#            for R_g, R_t in skipends(farey(cls.max_nb_turns)):
#                pass # Copy sub-loops

        @staticmethod
        def _get_FG(R_t, R_g, d_f, theta_g):
            """Get the distance between the fulcrum and the gear center."""
            OG =  R_t + R_g
            return math.sqrt(d_f**2 + OG**2 - 2*d_f*OG*math.cos(theta_g))

        @classmethod
        def _get_OP2_max(cls, R_t, R_g, d_f, theta_g, d_p, d_s):
            """Get the maximum distance between the center and the penholder."""
            OG =  R_t + R_g
            FG = cls._get_FG(R_t, R_g, d_f, theta_g)
            alpha = math.atan2(d_s, FG)
            theta_fg = math.pi - math.atan2(OG*math.sin(theta_g),
                                            d_f - OG*math.cos(theta_g))
            return d_f**2 + d_p**2 + 2*d_f*d_p*math.cos(theta_fg-alpha)


    class Simulator(Mechanism.Simulator):
        """Class for simulating the movement of the parts."""

        def __init__(self, properties, nb_samples=2**6, per_turn=True):
            self.reset(properties, nb_samples, per_turn)

        def reset(self, properties, nb_samples=2**6, per_turn=True):
            """Reset the properties."""
            self.R_t = properties[0] # Turntable radius
            self.R_g = properties[1] # Gear radius
            self.C_f = np.array([properties[2], 0.]) # Fulcrum center
            self.theta_g = properties[3] # Gear polar angle
            self.d_p = properties[4] # Fulcrum-penholder distance
            self.d_s = properties[5] # Gear center - slider distance
            # Gear center
            self.C_g = (self.R_t + self.R_g) * np.array(
                [math.cos(self.theta_g), math.sin(self.theta_g)])

            self.nb_samples = nb_samples
            self.per_turn = per_turn

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            gcd_ = gcd(self.R_t, self.R_g)
            nb_turns = self.R_t / gcd_
            if not self.d_s:
                # Degenerate case.
                nb_turns /= self.R_g / gcd_
            if self.per_turn:
                nb_samples = nb_turns * self.nb_samples + 1
            else:
                nb_samples = self.nb_samples
            interval_length = nb_turns * 2 * math.pi
            # Property range
            t_range = np.linspace(0., interval_length, nb_samples)
            # Slider curve
            curve = (self.d_s * np.vstack([np.cos(t_range), np.sin(t_range)]) +
                     self.C_g.reshape((2, 1)))
            # Connecting rod vector
            curve -= self.C_f.reshape((2, 1))
            # Penholder curve
            curve *= self.d_p / np.linalg.norm(curve, axis=0)
            curve += self.C_f.reshape((2, 1))
            # Space rotation
            ratio = self.R_g / self.R_t
            cos = np.cos(t_range * ratio)
            sin = np.sin(t_range * ratio)
            rot = np.array([[cos, -sin], [sin, cos]])
            curve = np.einsum('ijk,jk->ik', rot, curve)

            return curve

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            assert (0 <= pid < 6)
            if pid == 0:
                self.R_t = value
            elif pid == 1:
                self.R_g = value
            elif pid == 2:
                self.C_f[0] = value
            elif pid == 3:
                self.theta_g = value
            elif pid == 4:
                self.d_p = value
            else:
                self.d_s = value
            if pid == 0 or pid == 1 or pid == 3:
                self.C_g = (self.R_t + self.R_g) * np.array(
                    [math.cos(self.theta_g), math.sin(self.theta_g)])

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



class HootNanny(DrawingMechanism):
    """Hoot-Nanny (Magic Designer), aka the HTMLSpirograph."""


    class ConstraintSolver(Mechanism.ConstraintSolver):
        """Class for handling design constraints."""
        nb_dprops = 3
        nb_cprops = 4
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
            rot = np.array([[cos_, sin_], [-sin_, cos_]])
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
