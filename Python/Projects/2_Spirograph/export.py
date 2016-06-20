#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export script.

@author: Robin Roussel
"""
import math
import svgwrite as svg
from gearprofile import InvoluteGear, SinusoidalGear, CycloidalGear

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
        cut.add(cont.circle(center=dims * 0.5 + hole, r=0.2))
    # Markings
    write = cont.add(cont.g(stroke='black', stroke_width=0.01, font_size=0.5))
    for hole in holes:
        rad = "{:.0f}".format(math.sqrt(hole[0] * hole[0] + hole[1] * hole[1]))
        write.add(cont.text(text=rad, insert=dims * 0.5 + hole - (0.2, 0.5)))

    cont.save()    


def export_internal_gear_svg(gear, filename):
    """Export internal gear to SVG."""
    profile = gear.get_profile()

    profile = profile.T - profile.min(axis=1)
    dims = profile.max(axis=0)

    margin = 0.2
    profile += dims * margin * 0.5
    dims *= 1 + margin
    
    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=0.01))
    cut.add(cont.polyline(points=profile))
    # External boundary
    cut.add(cont.circle(center=dims*0.5, r=gear.pitch_radius*(1+margin)))
    
    cont.save()


def main():
    """Entry point."""
    rad_pinion = 5
    holes = [(i, 0.) for i in range(1, rad_pinion)]
    rad_internal = 11
    circular_pitch = 10

    export_pinion_svg(
        InvoluteGear(rad_pinion, rad_pinion*circular_pitch), 
        'svg/involute_pinion.svg', holes)
    export_internal_gear_svg(
        InvoluteGear(rad_internal, rad_internal*circular_pitch, internal=True), 
        'svg/involute_internal.svg')

    export_pinion_svg(
        SinusoidalGear(rad_pinion, rad_pinion*circular_pitch, tooth_radius=0.1), 
        'svg/sinusoidal_pinion.svg', holes)
    export_internal_gear_svg(
        SinusoidalGear(rad_internal, rad_internal*circular_pitch, tooth_radius=0.1), 
        'svg/sinusoidal_internal.svg')

    export_pinion_svg(
        CycloidalGear(rad_pinion, rad_pinion*circular_pitch), 
        'svg/cycloidal_pinion.svg', holes)
    export_internal_gear_svg(
        CycloidalGear(rad_internal, rad_internal*circular_pitch), 
        'svg/cycloidal_internal.svg')


if __name__ == "__main__":
    main()
