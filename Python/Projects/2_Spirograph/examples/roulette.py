#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generation of a non-trivial roulette: ellipse rolling without slipping inside
a circle.

@author: Robin Roussel
"""
import context
from fractions import Fraction
import matplotlib.pyplot as plt
import matplotlib.patches as pat
import numpy as np
#import scipy.optimize as opt

from curves import Circle, Ellipse


def get_roulette_ellipse_inside_circle(r, a, b, tracer, nb_pts=81,
                                       max_nb_turns=30):
    """
    Calculate and show the roulette of an ellipse rolling inside a circle.

    Terms:
    -----
     - Fundamental (or orbital) period: smallest positive period; number of
    revolutions the center of the rolling object makes before the figure is
    completed.
     - Complementary period: number of revolutions around its own axis the
    rolling objects makes before the figure is completed.
    """
    # Define the curves.
    nmov = Circle(r)
    mov = Ellipse(a, b)
    # Check that the min curvature of contact_m is greater than the curvature
    # of  C_n; in the present case this amounts to test that a**2/b < r.
    if a * a / b >= r:
        print("Error: The minimum curvature of the ellipse must be "
              "greater than the curvature of the circle "
              "(min(K_e) = {:.2f}, K_c = {:.2f}).".format(b / a ** 2, 1 / r))
        return
    # Compute their respective total perimeter.
    n_perim = nmov.get_perimeter()
    m_perim = mov.get_perimeter()
    print("Ellipse perimeter: {:.10f}".format(m_perim))
    # Compute the ratio between the perimeters s.t. we get as close as possible
    # to the exact value, while keeping under the maximum allowed number
    # self-rotations of the mov curve.
    ratio = Fraction(n_perim / m_perim).limit_denominator(int(max_nb_turns))
    num = ratio.numerator # complementary period
#    den = ratio.denominator # fundamental period
    print("Ratio error: {}".format(abs(n_perim / m_perim - ratio)))
    # Sample the arc length parameter. It runs over an interval of length
    # l such that, given the approximated ratio n / d,
    # l = n * m_perim or l = d * n_perim.
    # (The choice does not really matter, abs(error) will be the same.)
    # Since it doesn't have to be sampled homogeneously, the simplest thing is
    # to sample the ellipse parameter, get the corresponding arc lengths,
    # tile the list for as many ellipse rotations as we need, and deduce the
    # circle parameter samples from it.
    m_tvals = np.linspace(0., 2 * np.pi, nb_pts)
    svals = mov.get_arclength(m_tvals)
    svals = np.hstack([svals[:-1] + i * m_perim for i in range(num)] +
                      [m_perim * num])
    # Alternatively, if we wanted to sample the ellipse length homogeneously,
    # we would first need to create the corresponding sequence of lengths,
    # and then use a simple optimization to get the ellipse parameter values
    # from it. Here's how:
#    # /!\ When using np.linspace() we want to make sure that there is exactly
#    # nb_pts points per self-revolution of the rolling curve.
#    svals = np.linspace(0., num * m_perim, 1 + num * (nb_pts - 1))
#    # Compute the parameter values for the ellipse. Personal tests showed
#    # that the vectorized resolution is about 10 times faster for nb_pts=50,
#    # to balance of course with the memory overhead.
#    obj_func = lambda t: mov.get_arclength(t) - svals[:nb_pts]
#    obj_jac = lambda t: np.diag(mov.get_arclength_der(t))
#    # Initialize by approximating with a circle of equal perimeter.
#    init_guess = 2 * np.pi * svals[:nb_pts] / m_perim
#    m_tvals = opt.fsolve(obj_func, init_guess, fprime=obj_jac)
    n_tvals = svals / r
    # Get the points' coordinates from the parameter values.
    n_points = nmov.get_point(n_tvals)
    m_points = mov.get_point(m_tvals)
    m_points = np.hstack([np.tile(m_points[:, :-1], (1, num)),
                          m_points[:, 0].reshape(2,1)])
    # Compute pairs of jacobians.
    # Note: the jacobian must be taken wrt the arc length s and not wrt the
    # parameter t. For this we use the chain rule.
    # In practice this is not really necessary if dt/ds is a scalar, since it
    # only affects the norm of the jacobians. But it is more rigorous to do
    # it anyway.
    n_jacs = nmov.get_jac(n_tvals) / r # Circle => constant speed: |ds/dt| = r
    m_jacs = mov.get_jac(m_tvals) / mov.get_arclength_der(m_tvals)
    m_jacs = np.hstack([np.tile(m_jacs[:, :-1], (1, num)),
                        m_jacs[:, 0].reshape(2,1)])
    # Compute the rotations from each mov jacobian to each nmov jacobian.
    norms = np.sqrt(m_jacs[0] ** 2 + m_jacs[1] ** 2) # ||n_jacs|| = 1
    cos = np.einsum('ij,ij->j', m_jacs, n_jacs) / norms # dot product
    sin = (m_jacs[0] * n_jacs[1] - m_jacs[1] * n_jacs[0]) / norms # cross
    rot = np.array([[cos, -sin],
                    [sin, cos]])
    # Apply the general roulette formula:
    # P = F + R(T-M)
    tracer = tracer.reshape((2,1))
    roul = n_points + np.einsum('ijk,jk->ik', rot, tracer - m_points)




    # Display.
    plt.ioff()
    plt.figure()
    ax = plt.subplot(211)

    plt.plot(n_points[0], n_points[1],
             m_points[0, :nb_pts], m_points[1, :nb_pts])
    plt.plot(roul[0], roul[1])

    global time
    time = 0.
    timer = plt.gcf().canvas.new_timer(interval=50)
    ell_centers = n_points + np.einsum('ijk,jk->ik', rot, - m_points)
    ell = pat.Ellipse(ell_centers[:, 0], 2*a, 2*b, 0., fill=False, color='g')
    tra = pat.Circle(roul[:, 0], 0.05, color='g', fill=False)
    contact_n = pat.Circle(n_points[:, 0], 0.05, color='b')
    contact_m = pat.Circle(m_points[:, 0], 0.05, color='g')

    ax.add_artist(ell)
    ax.add_artist(tra)
    ax.add_artist(contact_n)
    ax.add_artist(contact_m)

    plt.margins(0.1)
    ax.set_aspect('equal')


    ax = plt.subplot(212)
    plt.plot(cos)
    plt.plot(sin)
    plt.plot(np.arctan2(sin, cos))

    marker = pat.Circle((0., sin[0] / cos[0]), 0.1, color='r')
    ax.add_artist(marker)

    def animate():

        global time
        idx = time % cos.shape[0]

        angle = np.arctan2(sin[idx], cos[idx])
        ell.angle = angle * 180 / np.pi
        ell.center = ell_centers[:, idx]
        tra.center = roul[:, idx]
        contact_n.center = n_points[:, idx]
        contact_m.center = m_points[:, idx]

        marker.center = [idx, angle]

        plt.gcf().canvas.draw()
        time += 1.

    timer.add_callback(animate)
    timer.start()

    plt.show()


def main():
    """Entry point."""
    get_roulette_ellipse_inside_circle(2.6154075096 * 4 / 3, 1.5, 1.1,
                                       np.array([1.1, 0.]))


if __name__ == "__main__":
    main()
