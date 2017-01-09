#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export script.

@author: Robin Roussel
"""
import math
import numpy as np
import svgwrite as svg

from gearprofile import Involute, Sinusoidal, Cycloidal, InvoluteElliptical


EXPORT_SPIRO = False
EXPORT_ELLIP_SPIRO = False
EXPORT_HOOT = True


def get_svg_context(filename, dims_cm):
    """Returns the SVG context."""
    width = str(dims_cm[0])
    height = str(dims_cm[1])

    return svg.Drawing(filename, profile='tiny',
                       size=(width + 'cm', height + 'cm'),
                       viewBox='0 0 ' + width + ' ' + height)


def export_gear_svg(gear, filename, holes=[], hole_size=.3,
                    center_hole_size=None):
    """Export gear to SVG."""
    profile = gear.get_profile()
    profile = profile.T - profile.min(axis=1)
    dims = profile.max(axis=0)

    margin = .1
    profile += dims * margin / 2
    dims *= 1 + margin

    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.01))
    cut.add(cont.polyline(points=profile))
    # Holes
    if center_hole_size is not None:
        cut.add(cont.circle(center=dims/2, r=center_hole_size/2))
    for hole in holes:
        cut.add(cont.circle(center=dims/2 + hole, r=hole_size/2))
#    # Markings
#    write = cont.add(cont.g(stroke='black', stroke_width=0.01, font_size=0.5))
#    for hole in holes:
#        rad = "{:.0f}".format(math.sqrt(hole[0] * hole[0] + hole[1] * hole[1]))
#        write.add(cont.text(text=rad, insert=dims * 0.5 + hole - (0.2, 0.5)))

    cont.save()


def export_internal_ring_svg(gear, filename, hole_size=.3):
    """Export internal ring to SVG."""
    profile = gear.get_profile()

    profile = profile.T - profile.min(axis=1)
    dims = profile.max(axis=0)

    margin = .2
    profile += dims * margin / 2
#    dims *= 1 + margin

    cont = get_svg_context(filename, dims*(1+margin))
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.01))
    cut.add(cont.polyline(points=profile))
    # External boundary
    cut.add(cont.rect(insert=dims*margin/4, size=dims*(1+margin/2)))
    # Holes for fixations.
    cut.add(cont.circle(center=dims*margin/2, r=hole_size/2))
    cut.add(cont.circle(center=dims*margin/2 + [dims[0], 0.], r=hole_size/2))
    cut.add(cont.circle(center=dims*margin/2 + [0., dims[1]], r=hole_size/2))
    cut.add(cont.circle(center=dims*margin/2 + dims, r=hole_size/2))

    cont.save()


def export_hoot_base_svg(rad_turntable, rad_gear1, rad_gear2, gear_angle,
                         filename):
    hole_size = .6
    margin = .1
    dims = 2*(rad_turntable*(1 + 2*margin) + np.array([rad_gear1, rad_gear2]))
    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.01))
    # Base contour
    xy_b = [rad_turntable * margin, rad_turntable * margin]
    size = dims - 2*rad_turntable*margin
    cut.add(cont.rect(insert=xy_b, size=size, rx=hole_size, ry=hole_size))
    # Turntable hole
    xy_tt = [rad_turntable * (1 + 2*margin), rad_turntable * (1 + 2*margin)]
    cut.add(cont.circle(center=xy_tt, r=hole_size/2))
    # Gear 1 hole
    xy_g1 = [xy_tt[0] + rad_turntable*(1 - margin) + rad_gear1,
             xy_tt[1] - hole_size/2]
    size = [rad_turntable*2*margin, hole_size]
    cut.add(cont.rect(insert=xy_g1, size=size, rx=hole_size/2, ry=hole_size))
    # Gear 2 hole
    xy_g2 = [xy_tt[0] + rad_turntable*(1 - margin) + rad_gear2,
             xy_tt[1] - hole_size/2]
    g2 = cont.rect(insert=xy_g2, size=size, rx=hole_size/2, ry=hole_size)
    g2.rotate(gear_angle*180/math.pi, xy_tt)
    cut.add(g2)
    # Holes for supports
    hole_size = .5
    xy = [rad_turntable*2*margin, rad_turntable*2*margin]
    cut.add(cont.circle(center=xy, r=hole_size/2))
    xy = [dims[0] - rad_turntable*2*margin, xy[1]]
    cut.add(cont.circle(center=xy, r=hole_size/2))
    xy = dims - rad_turntable*2*margin
    cut.add(cont.circle(center=xy, r=hole_size/2))
    xy = [rad_turntable*2*margin, xy[1]]
    cut.add(cont.circle(center=xy, r=hole_size/2))

    cont.save()


def export_toothless_ring_svg(rad_int, rad_ext, filename):
    margin = .1
    dims = 2*rad_ext*(1+margin)*np.array([1, 1])
    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.01))

    cut.add(cont.circle(center=dims/2, r=rad_int))
    cut.add(cont.circle(center=dims/2, r=rad_ext))

    cont.save()


def export_hoot_arm_svg(length, rad1, rad2, filename):
    margin = .1
    rmax = max(rad1, rad2)
    dims = np.array([length + 2*(rad1 + rad2), 4*rmax])
    cont = get_svg_context(filename, (1 + 2*margin)*dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.01))

    xy = margin * dims
    cut.add(cont.rect(insert=xy, size=dims, rx=rmax, ry=rmax))
    xy += [2*rad1, 2*rmax]
    cut.add(cont.circle(center=xy, r=rad1))
    xy[0] += length
    cut.add(cont.circle(center=xy, r=rad2))

    cont.save()


def main():
    """Entry point."""
    if EXPORT_SPIRO:
        rad_gear = 2.5
        rad_ring = 5.5
        circular_pitch = 15
        holes = [(i*.5, 0.) for i in range(1, 2*rad_gear)]

        export_gear_svg(
            Involute(rad_gear, rad_gear*circular_pitch),
            'svg/spiro/involute_pinion.svg', holes)
        export_internal_ring_svg(
            Involute(rad_ring, rad_ring*circular_pitch, internal=True),
            'svg/spiro/involute_internal.svg')

        export_gear_svg(
            Sinusoidal(rad_gear, rad_gear*circular_pitch, tooth_radius=.1),
            'svg/spiro/sinusoidal_pinion.svg', holes)
        export_internal_ring_svg(
            Sinusoidal(rad_ring, rad_ring*circular_pitch, tooth_radius=.1),
            'svg/spiro/sinusoidal_internal.svg')

        export_gear_svg(
            Cycloidal(rad_gear, rad_gear*circular_pitch),
            'svg/spiro/cycloidal_pinion.svg', holes)
        export_internal_ring_svg(
            Cycloidal(rad_ring, rad_ring*circular_pitch),
            'svg/spiro/cycloidal_internal.svg')

    if EXPORT_ELLIP_SPIRO:
        rad_gear = 3
        rad_ring = 5
        circular_pitch = 15

        export_internal_ring_svg(
            InvoluteElliptical(rad_ring, 0., rad_ring*circular_pitch,
                               internal=True),
            'svg/ellip_spiro/involute_elliptical_fixed.svg')
        holes = [(1., 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .2, rad_gear*circular_pitch),
            'svg/ellip_spiro/involute_elliptical_moving_fixpoi_1.svg', holes)
        holes = [(1.232, 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .429, rad_gear*circular_pitch),
            'svg/ellip_spiro/involute_elliptical_moving_fixpoi_2.svg', holes)
        holes = [(.4755, 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .31, rad_gear*circular_pitch),
            'svg/ellip_spiro/involute_elliptical_moving_fixisectangle_1.svg',
            holes)
        holes = [(.615, 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .4005, rad_gear*circular_pitch),
            'svg/ellip_spiro/involute_elliptical_moving_fixisectangle_2.svg',
            holes)

    if EXPORT_HOOT:
        scale = 1
        rad_turntable = 10*scale
        rad_gear1 = 4*scale
        rad_gear2 = 2*scale
        gear_angle = .9*scale
        dist1 = 2.5*scale
        dist2 = 1.5*scale
        length1 = 10.*scale
        length2 = 9.95*scale
        circular_pitch = 12/scale
        rad_pivot = .41
        rad_pen = .5
        # Base
        export_hoot_base_svg(
            rad_turntable, rad_gear1, rad_gear2, gear_angle,
            'svg/hoot_nanny/base.svg')
        # Turntable

        export_gear_svg(
            Involute(rad_turntable, int(rad_turntable*circular_pitch)),
            'svg/hoot_nanny/turntable_gear.svg',
            center_hole_size=2*rad_pivot)
        export_toothless_ring_svg(
            1.3/2, rad_turntable*.97,
            'svg/hoot_nanny/turntable_ring.svg')
        # Gears
        holes = [(dist1, 0.)]
        export_gear_svg(
            Involute(rad_gear1, int(rad_gear1*circular_pitch)),
            'svg/hoot_nanny/gear1.svg',
            holes, center_hole_size=2*rad_pivot)
        holes = [(dist2, 0.)]
        export_gear_svg(
            Involute(rad_gear2, int(rad_gear2*circular_pitch)),
            'svg/hoot_nanny/gear2.svg',
            holes, center_hole_size=2*rad_pivot)
        # Arms
        export_hoot_arm_svg(
            length1, rad_pivot, rad_pen, 'svg/hoot_nanny/arm1.svg')
        export_hoot_arm_svg(
            length2, rad_pivot, rad_pen, 'svg/hoot_nanny/arm2.svg')
        # Bunch of custom rings
        export_toothless_ring_svg(
            rad_pen, 2*rad_pen,
            'svg/hoot_nanny/pen_ring.svg')
        export_toothless_ring_svg(
            rad_pivot, 2*rad_pivot,
            'svg/hoot_nanny/pivot_ring.svg')


if __name__ == "__main__":
    main()
