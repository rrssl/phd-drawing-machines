# -*- coding: utf-8 -*-
"""
Bases classes for parameterized cyclic mechanisms.

@author: Robin Roussel
"""

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
        eps = 1e-4 # Absolute tolerance for constraint violation.
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

        def compute_state(self, asb, t):
            """Compute the state of the assembly a time t."""
            raise NotImplementedError

        def simulate_cycle(self):
            """Simulate one cycle of the assembly's motion."""
            raise NotImplementedError

        def get_cycle_length(self):
            """Compute and return the interval length of one full cycle."""
            raise NotImplementedError

        def update_prop(self, pid, value):
            """Update the property referenced by the input index."""
            raise NotImplementedError


    def __init__(self, *props, verbose=False):
        self.constraint_solver = type(self).ConstraintSolver
        self._simulator = None
        self.assembly = self._create_assembly()
        self.verbose = verbose

        self.reset(*props)

    def reset(self, *props):
        """Reset all properties at once, if they meet the constraints."""
        if self.constraint_solver.check_constraints(props):
            self.props = list(props)
            try:
                self._simulator.reset(props)
            except AttributeError:
                self._simulator = type(self).Simulator(props)
            self._simulator.compute_state(self.assembly, 0.)
            return True
        else:
            if self.verbose:
                print("Error: invalid mechanism properties.", props)
                print("Constraints: ",
                      [cons(props)
                       for cons in self.constraint_solver.get_constraints()])
            return False

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
            self._simulator.compute_state(self.assembly, 0.)
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

    def set_state(self, t):
        """Set value of the time parameter."""
        self._simulator.compute_state(self.assembly, t)

    @staticmethod
    def _create_assembly():
        raise NotImplementedError



class DrawingMechanism(Mechanism):
    """Functional specialization: Mechanism that can draw a closed curve."""

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
