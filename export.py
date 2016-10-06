#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export script.

@author: Robin Roussel
"""
import math
import svgwrite as svg
from gearprofile import Involute, Sinusoidal, Cycloidal, InvoluteElliptical

def get_svg_context(filename, dims_cm):
    """Returns the SVG context."""
    width = str(dims_cm[0])
    height = str(dims_cm[1])

    return svg.Drawing(filename, profile='tiny',
                       size=(width + 'cm', height + 'cm'),
                       viewBox='0 0 ' + width + ' ' + height)


def export_pinion_svg(gear, filename, holes=None):
    """Export pinion to SVG."""
    profile = gear.get_profile()
    profile = profile.T - profile.min(axis=1)
    dims = profile.max(axis=0)

    margin = 0.1
    profile += dims * margin * 0.5
    dims *= 1 + margin

    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=0.01))
    cut.add(cont.polyline(points=profile))
    # Holes
    for hole in holes:
        cut.add(cont.circle(center=dims * 0.5 + hole, r=0.3))
#    # Markings
#    write = cont.add(cont.g(stroke='black', stroke_width=0.01, font_size=0.5))
#    for hole in holes:
#        rad = "{:.0f}".format(math.sqrt(hole[0] * hole[0] + hole[1] * hole[1]))
#        write.add(cont.text(text=rad, insert=dims * 0.5 + hole - (0.2, 0.5)))

    cont.save()


def export_internal_gear_svg(gear, filename):
    """Export internal gear to SVG."""
    profile = gear.get_profile()

    profile = profile.T - profile.min(axis=1)
    dims = profile.max(axis=0)

    margin = 0.2
    profile += dims * margin * 0.5
#    dims *= 1 + margin

    cont = get_svg_context(filename, dims*(1+margin))
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=0.01))
    cut.add(cont.polyline(points=profile))
    # External boundary
    cut.add(cont.rect(insert=dims*margin*0.25, size=dims*(1+margin/2)))
    # Holes for fixations.
    cut.add(cont.circle(center=dims*margin*0.5, r=0.2))
    cut.add(cont.circle(center=dims*margin*0.5+[dims[0], 0.], r=0.2))
    cut.add(cont.circle(center=dims*margin*0.5+[0., dims[1]], r=0.2))
    cut.add(cont.circle(center=dims*margin*0.5+dims, r=0.2))

    cont.save()


def main():
    """Entry point."""
#    rad_pinion = 5
#    holes = [(i, 0.) for i in range(1, rad_pinion)]
#    rad_internal = 11
#    circular_pitch = 10
#
#    export_pinion_svg(
#        Involute(rad_pinion, rad_pinion*circular_pitch),
#        'svg/involute_pinion.svg', holes)
#    export_internal_gear_svg(
#        Involute(rad_internal, rad_internal*circular_pitch, internal=True),
#        'svg/involute_internal.svg')
#
#    export_pinion_svg(
#        Sinusoidal(rad_pinion, rad_pinion*circular_pitch, tooth_radius=0.1),
#        'svg/sinusoidal_pinion.svg', holes)
#    export_internal_gear_svg(
#        Sinusoidal(rad_internal, rad_internal*circular_pitch, tooth_radius=0.1),
#        'svg/sinusoidal_internal.svg')
#
#    export_pinion_svg(
#        Cycloidal(rad_pinion, rad_pinion*circular_pitch),
#        'svg/cycloidal_pinion.svg', holes)
#    export_internal_gear_svg(
#        Cycloidal(rad_internal, rad_internal*circular_pitch),
#        'svg/cycloidal_internal.svg')

    rad_pinion = 6
    rad_internal = 10
    circular_pitch = 10

    export_internal_gear_svg(
        InvoluteElliptical(rad_internal, 0., rad_internal*circular_pitch,
                           internal=True),
        'svg/involute_elliptical_fixed.svg')
    holes = [(2*1., 0.)]
    export_pinion_svg(
        InvoluteElliptical(rad_pinion, .2, rad_pinion*circular_pitch),
        'svg/involute_elliptical_moving_fixpoi_1.svg', holes)
    holes = [(2*1.232, 0.)]
    export_pinion_svg(
        InvoluteElliptical(rad_pinion, .429, rad_pinion*circular_pitch),
        'svg/involute_elliptical_moving_fixpoi_2.svg', holes)
    holes = [(2*.4755, 0.)]
    export_pinion_svg(
        InvoluteElliptical(rad_pinion, .31, rad_pinion*circular_pitch),
        'svg/involute_elliptical_moving_fixisectangle_1.svg', holes)
    holes = [(2*.615, 0.)]
    export_pinion_svg(
        InvoluteElliptical(rad_pinion, .4005, rad_pinion*circular_pitch),
        'svg/involute_elliptical_moving_fixisectangle_2.svg', holes)


if __name__ == "__main__":
    main()
