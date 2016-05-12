# -*- coding: utf-8 -*-
"""
Library of parametric curves.

@author: Robin Roussel
"""

from fractions import Fraction
import numpy as np
import scipy.optimize as opt
import scipy.special as spec

class Curve:
    """Base class for curves."""

    def __init__(self):
        pass

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        pass


class Hypotrochoid(Curve):
    """Hypotrochoid class (parameters R, r, d)."""

    def __init__(self, ext_gear_radius, int_gear_radius, tracer_dist):
        self.R = ext_gear_radius
        self.r = int_gear_radius
        self.d = tracer_dist

    @staticmethod
    def get_param_combinations(num_R_vals, num_d_vals):
        """Get an array of all possible parameter combinations."""
        combi = np.empty((0, 3))
        for R in range(1, num_R_vals + 1):
            for r in range(1, R):
                if Fraction(R, r).denominator == r: # Avoid repeating patterns
                    for d in np.linspace(0, r, num_d_vals + 1, endpoint=False):
                        if d != 0.: # Exlude d=0 and d=r
                            combi = np.vstack([combi, np.array([R, r, d])])
        return combi

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        R = self.R
        r = self.r
        d = self.d
        return np.vstack([(R - r) * np.cos(t) + d * np.cos(t * (R - r) / r),
                          (R - r) * np.sin(t) - d * np.sin(t * (R - r) / r)])


class Circle(Curve):
    """Parametric curve of a circle."""

    def __init__(self, radius):
        self.r = radius

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        return self.r * np.vstack([np.cos(t), np.sin(t)])

    def get_jac(self, t):
        """Get the [x'(t), y'(t)] jacobian(s)."""
        return self.r * np.vstack([-np.sin(t), np.cos(t)])

    def get_perimeter(self):
        """Get the full perimeter of the ellipse."""
        return 2 * np.pi * self.r

    def get_arclength(self, end):
        """Get the arc length(s) s(t)."""
        return self.r * end

    def get_arclength_der(self, end):
        """Get the derivative(s) of the arc length s'(t)."""
        return self.r


class Ellipse(Curve):
    """Parametric curve of an ellipse."""

    def __init__(self, semimajor, semiminor):
        self.a = semimajor
        self.b = semiminor
        # /!\ This scipy implementation uses the convention E(phi, m) with m
        # the elliptic parameter, which in our case needs to be e**2.
        self.e2 = 1 - semiminor * semiminor / (semimajor * semimajor)

        self.ellipe_val = spec.ellipe(self.e2)

    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
        return np.vstack([self.a * np.cos(t), self.b * np.sin(t)])

    def get_jac(self, t):
        """Get the [x'(t), y'(t)] jacobian(s)."""
        return np.vstack([self.a * -np.sin(t), self.b * np.cos(t)])

    def get_perimeter(self):
        """Get the full perimeter of the ellipse."""
        # Use the value of the complete elliptic integral of the 2nd kind.
        return 4 * self.a * self.ellipe_val

    def get_arclength(self, t):
        """Get the arc length(s) s(t)."""
        # Use the incomplete elliptic integral of the 2nd kind.
        # /!\ The "amplitude" phi of E(phi, k) as it is often found in the
        # litterature is NOT the same as our ellipse parameter here. To get the
        # arc length right we have to use s(t) = E(t + pi/2, e) - E(e).
        return self.a * (spec.ellipeinc(t + (np.pi / 2), self.e2) -
                         self.ellipe_val)

    def get_arclength_der(self, t):
        """Get the derivative(s) of the arc length s'(t)."""
        # Easy with the fundamental theorem of calculus.
        return self.a * np.sqrt(1 - self.e2 * np.cos(t) ** 2)
        
    def get_param_from_arclength(self, s):
        """Get the parameter t given s(t)."""
        obj_func = lambda t: self.get_arclength(t) - s
        obj_jac = lambda t: np.diag(self.get_arclength_der(t))
        # Initialize by approximating with a circle of equal perimeter.
        init_guess = 2 * np.pi * s / self.get_perimeter()
        return opt.fsolve(obj_func, init_guess, fprime=obj_jac)        


class RouletteEllipseInCircle(Curve):
    """Parametric roulette curve of an ellipse rolling inside a circle."""
    def __init__(self, radius, semimajor, semiminor, tracer_dist):
        self.R = radius
        self.a = semimajor
        self.b = semiminor
        self.T = np.array([[tracer_dist], [0.]])

        self.circle = Circle(radius)
        self.ellipse = Ellipse(semimajor, semiminor)

        if semimajor * semimajor / semiminor >= radius:
            print("Warning: The minimum curvature of the ellipse should be "
                  "greater than the curvature of the circle (min(K_e) = "
                  "{:.2f}, K_c = {:.2f}).".format(semiminor / semimajor ** 2,
                                                  1 / radius))

#        n_perim = self.circle.get_perimeter()
#        m_perim = self.ellipse.get_perimeter()
#        self.ratio = Fraction(n_perim / m_perim).limit_denominator(int(max_nb_turns))
    
    def get_point(self, t):
        """Get the [x(t), y(t)] point(s)."""
#        num = self.ratio.numerator
        # Get the parameter values for both curves.
        tn = t
        svals = tn * self.R
        tm = self.ellipse.get_param_from_arclength(svals)
        # Get the points' coordinates from the parameter values.
        self.n_pts = self.circle.get_point(tn)
        self.m_pts = self.ellipse.get_point(tm)
#        self.m_pts = np.hstack([np.tile(self.m_pts[:, :-1], (1, num)),
#                                   self.m_pts[:, 0].reshape(2,1)])
        # Compute pairs of jacobians.
        n_jacs = self.circle.get_jac(tn) / self.R # Constant speed: |ds/dt| = R
        m_jacs = self.ellipse.get_jac(tm) / self.ellipse.get_arclength_der(tm)
#        m_jacs = np.hstack([np.tile(m_jacs[:, :-1], (1, num)),
#                            m_jacs[:, 0].reshape(2,1)])
        # Compute the rotations between each pair of jacobians.
        norms = np.sqrt(m_jacs[0] ** 2 + m_jacs[1] ** 2) # ||n_jacs|| = 1
        cos = np.einsum('ij,ij->j', m_jacs, n_jacs) / norms # dot product
        sin = (m_jacs[0] * n_jacs[1] - m_jacs[1] * n_jacs[0]) / norms # cross
        self.rot = np.array([[cos, -sin],
                             [sin, cos]])
        # Apply the general roulette formula:
        # P = F + R(T-M)
        roulette = self.n_pts + np.einsum('ijk,jk->ik', 
                                          self.rot, self.T - self.m_pts)
        
        return roulette
        
    def update_tracer(self, tracer_dist):
        """Update the roulette to a different tracer position.
        
        This convenience function is here to avoid recomputing the same
        intermediary results when all that is need is to change the tracer
        position.        
        """
        self.T = np.array([[tracer_dist], [0.]])
        roulette = self.n_pts + np.einsum('ijk,jk->ik', 
                                          self.rot, self.T - self.m_pts)
        
        return roulette        
