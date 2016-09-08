#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example showing the different types of circular spur gears available.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
import matplotlib.patches as pat

import context
import gearprofile as gear
from curves import Ellipse2


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
    ax = plt.subplot(231)
    gearext = gear.Involute(radext, radial_pitch*radext, internal=True)
    profext = gearext.get_profile()
    plt.plot(*profext)

    gearint = gear.Involute(radint, radial_pitch*radint)
    profint = gearint.get_profile()
    profint[0] += radext - radint
    plt.plot(*profint)

    ax.set_aspect('equal')
    ax.set_title("Involute")
    ax.add_patch(get_primext())
    ax.add_patch(get_primint())


    # Second gear type: sinusoidal
    ax = plt.subplot(232)
    gearext = gear.Sinusoidal(radext, radial_pitch*radext, tooth_radius=0.1)
    profext = gearext.get_profile()
    plt.plot(*profext)

    gearint = gear.Sinusoidal(radint, radial_pitch*radint, tooth_radius=0.1)
    profint = gearint.get_profile()
    profint[0] += radext - radint
    plt.plot(*profint)

    ax.set_aspect('equal')
    ax.set_title("Sinusoidal")
    ax.add_patch(get_primext())
    ax.add_patch(get_primint())


    # Third gear type: cycloidal
    ax = plt.subplot(233)
    gearext = gear.Cycloidal(radext, radial_pitch*radext)
    profext = gearext.get_profile()
    plt.plot(*profext)

    gearint = gear.Cycloidal(radint, radial_pitch*radint)
    profint = gearint.get_profile()
    profint[0] += radext - radint
    plt.plot(*profint)

    ax.set_aspect('equal')
    ax.set_title("Cycloidal")
    ax.add_patch(get_primext())
    ax.add_patch(get_primint())


    # Fifth gear type: elliptical involute
    get_primint = lambda a,b: pat.Ellipse(
        (radext - a, 0), 2*a, 2*b, color='red', linestyle='dashed', fill=False)

    ecc = 0.6
    ellip = Ellipse2(radint, ecc**2)
    semimajor, semiminor = ellip.a, ellip.b

    ax = plt.subplot(234)
    gearext = gear.InvoluteElliptical(radext, 0., radial_pitch*radext,
                                      internal=True)
    profext = gearext.get_profile()
    plt.plot(*profext)

    gearint = gear.InvoluteElliptical(radint, ecc**2, radial_pitch*radint)
    profint = gearint.get_profile()
    profint[0] += radext - semimajor
    plt.plot(*profint)

    ax.set_aspect('equal')
    ax.set_title("Elliptical involute")
    ax.add_patch(get_primext())
    ax.add_patch(get_primint(semimajor, semiminor))


    # Fourth gear type: elliptical sinusoidal
    ax = plt.subplot(235)
    gearext = gear.Sinusoidal(radext, radial_pitch*radext, tooth_radius=0.1)
    profext = gearext.get_profile()
    plt.plot(*profext)

    gearint = gear.SinusoidalElliptical(
        semimajor, semiminor, radial_pitch*radint, tooth_radius=0.1)
    profint = gearint.get_profile()
    profint[0] += radext - semimajor
    plt.plot(*profint)

    ax.set_aspect('equal')
    ax.set_title("Elliptical sinusoidal")
    ax.add_patch(get_primext())
    ax.add_patch(get_primint(semimajor, semiminor))




    plt.gcf().canvas.set_window_title("Gear profiles")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
