#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the different types of circular spur gears available.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import context
from gearprofile import Involute, Sinusoidal, Cycloidal, SinusoidalElliptic
from curves import Ellipse


def main():
    """Entry point."""
    radext = 3
    radint = 2
    radial_pitch = 10
    # Use lambdas instead of intances since you cannot add the same object to
    # mutliple Axes instances.
    get_primext = lambda: pat.Circle((0, 0), radext, color='purple',
                                     linestyle='dashed', fill=False)
    get_primint = lambda: pat.Circle((radext - radint, 0), radint, color='red',
                                     linestyle='dashed', fill=False)

    # First gear type: involute
    ax = plt.subplot(221)
    gearext = Involute(radext, radial_pitch*radext, internal=True)
    profext = gearext.get_profile()
    plt.plot(profext[0], profext[1])

    gearint = Involute(radint, radial_pitch*radint)
    profint = gearint.get_profile()
    profint[0] += radext - radint
    plt.plot(profint[0], profint[1])

    ax.set_aspect('equal')
    ax.add_patch(get_primext())
    ax.add_patch(get_primint())


    # Second gear type: sinusoidal
    ax = plt.subplot(222)
    gearext = Sinusoidal(radext, radial_pitch*radext, tooth_radius=0.1)
    profext = gearext.get_profile()
    plt.plot(profext[0], profext[1])

    gearint = Sinusoidal(radint, radial_pitch*radint, tooth_radius=0.1)
    profint = gearint.get_profile()
    profint[0] += radext - radint
    plt.plot(profint[0], profint[1])

    ax.set_aspect('equal')
    ax.add_patch(get_primext())
    ax.add_patch(get_primint())


    # Third gear type: cycloidal
    ax = plt.subplot(223)
    gearext = Cycloidal(radext, radial_pitch*radext)
    profext = gearext.get_profile()
    plt.plot(profext[0], profext[1])

    gearint = Cycloidal(radint, radial_pitch*radint)
    profint = gearint.get_profile()
    profint[0] += radext - radint
    plt.plot(profint[0], profint[1])

    ax.set_aspect('equal')
    ax.add_patch(get_primext())
    ax.add_patch(get_primint())


    # Fourth gear type: elliptic
    get_primint = lambda a,b: pat.Ellipse(
        (radext - a, 0), 2*a, 2*b, color='red', linestyle='dashed', fill=False)
    # Compute the ellipse parameters s.t. its perimeter is equivalent to a 
    # circle of radius 'radint'.
    ecc = 0.6
    semimajor, semiminor = Ellipse.convert_reduced_to_semiaxes(radint, ecc**2)

    ax = plt.subplot(224)
#    ax = plt.subplot(111)
    gearext = Sinusoidal(radext, radial_pitch*radext, tooth_radius=0.1)
    profext = gearext.get_profile()
    plt.plot(profext[0], profext[1])

    gearint = SinusoidalElliptic(semimajor, semiminor, radial_pitch*radint,
                                 tooth_radius=0.1)
    profint = gearint.get_profile()
    profint[0] += radext - semimajor
    plt.plot(profint[0], profint[1])

    ax.set_aspect('equal')
    ax.add_patch(get_primext())
    ax.add_patch(get_primint(semimajor, semiminor))




    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
