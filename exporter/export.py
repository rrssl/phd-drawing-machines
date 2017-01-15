#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export script.

@author: Robin Roussel
"""
import datetime, os
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
                         rad_dgear, square_hole_size, filename):
    hole_size = .6
    margin = .1 * rad_turntable
    TG2 = (rad_turntable + rad_gear2) * np.array([math.cos(gear_angle),
                                                  math.sin(gear_angle)])
    dims = 4*margin + rad_turntable + np.array(
        [2*rad_gear1 + abs(min(-rad_turntable, TG2[0]-rad_gear2)),
         2*rad_dgear + max (rad_turntable, TG2[1]+rad_gear2)])
    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.01))
    # Base contour
    xy_b = [margin, margin]
    size = dims - 2*margin
    cut.add(cont.rect(insert=xy_b, size=size, rx=hole_size, ry=hole_size))
    # Turntable hole
    tt_c = [2*margin + abs(min(-rad_turntable, TG2[0]-rad_gear2)),
            2*margin + 2*rad_dgear + rad_turntable]
    size = [square_hole_size, square_hole_size]
    xy_tt = [tt_c[0] - size[0]/2, tt_c[1] - size[1]/2]
    cut.add(cont.rect(insert=xy_tt, size=size))
    # Gear 1 hole
    xy_g1 = [tt_c[0] + rad_turntable + rad_gear1 - margin,
             tt_c[1] - hole_size/2]
    size = [2*margin, hole_size]
    cut.add(cont.rect(insert=xy_g1, size=size, rx=hole_size/2, ry=hole_size))
    # Gear 2 hole
    xy_g2 = [tt_c[0] + rad_turntable + rad_gear2 - margin,
             tt_c[1] - hole_size/2]
    g2 = cont.rect(insert=xy_g2, size=size, rx=hole_size/2, ry=hole_size)
    g2.rotate(gear_angle*180/math.pi, tt_c)
    cut.add(g2)
    # Driving gear hole
    xy_dg = [tt_c[0] + rad_turntable + rad_dgear - margin,
             tt_c[1] - hole_size/2]
    dg = cont.rect(insert=xy_dg, size=size, rx=hole_size/2, ry=hole_size)
    dg.rotate(-90, tt_c)
    cut.add(dg)
    # Holes for supports
    hole_size = .5
    xy = [2*margin, 2*margin]
    cut.add(cont.circle(center=xy, r=hole_size/2))
    xy = [dims[0] - 2*margin, xy[1]]
    cut.add(cont.circle(center=xy, r=hole_size/2))
    xy = dims - 2*margin
    cut.add(cont.circle(center=xy, r=hole_size/2))
    xy = [2*margin, xy[1]]
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


def export_hoot(props, scale, base, name):
    base = base.format("hoot_nanny")
    os.makedirs(base)
    # TODO shorten varaible names, use numpy vector
    rad_turntable = props[0]*scale
    rad_gear1 = props[1]*scale
    rad_gear2 = props[2]*scale
    gear_angle = props[3]
    dist1 = props[4]*scale
    dist2 = props[5]*scale
    length1 = props[6]*scale
    length2 = props[7]*scale
    circular_pitch = 12/scale
    rad_dgear = .333 * rad_turntable # Radius of the driving gear
    # The 3 following dimensions come from physical measurments.
    # They are not affected by the scale factor.
    tt_hole_size = 4.02
    base_sq_hole_size = 1.991
    rad_pivot = .405
    rad_penholder = .705
    # Base
    export_hoot_base_svg(
        rad_turntable, rad_gear1, rad_gear2, gear_angle, rad_dgear,
        base_sq_hole_size, base+name.format("base"))
    # Turntable
    export_gear_svg(
        Involute(rad_turntable, int(rad_turntable*circular_pitch)),
        base+name.format("turntable_gear"),
        center_hole_size=tt_hole_size)
    export_toothless_ring_svg(
        rad_turntable*.90, rad_turntable*.95,
        base+name.format("turntable_ring"))
    # Gears
    holes = [(dist1, 0.)]
    export_gear_svg(
        Involute(rad_gear1, int(rad_gear1*circular_pitch)),
        base+name.format("gear1"),
        holes, center_hole_size=2*rad_pivot)
    holes = [(dist2, 0.)]
    export_gear_svg(
        Involute(rad_gear2, int(rad_gear2*circular_pitch)),
        base+name.format("gear2"),
        holes, center_hole_size=2*rad_pivot)
    holes = [(rad_dgear*.8, 0.)]
    export_gear_svg(
        Involute(rad_dgear, int(rad_dgear*circular_pitch)),
        base+name.format("dgear"),
        holes, center_hole_size=2*rad_pivot)
    # Arms
    export_hoot_arm_svg(
        length1, rad_pivot, rad_penholder, base+name.format("arm1"))
    export_hoot_arm_svg(
        length2, rad_pivot, rad_penholder, base+name.format("arm2"))
    # Bunch of custom rings
#    export_toothless_ring_svg(
#        rad_penholder, 2*rad_penholder,
#        base+name.format("pen_ring"))
#    export_toothless_ring_svg(
#        rad_pivot, 2*rad_pivot,
#        base+name.format("pivot_ring"))


def main():
    base = "svg/{}/" + datetime.datetime.now().strftime("%Y%m%d-%H:%M") + "/"
    name = "{}.svg"


    if EXPORT_SPIRO:
        base = base.format("spiro")
        os.makedirs(base)
        rad_gear = 2.5
        rad_ring = 5.5
        circular_pitch = 15
        holes = [(i*.5, 0.) for i in range(1, 2*int(rad_gear))]

        export_gear_svg(
            Involute(rad_gear, int(rad_gear*circular_pitch)),
            base+name.format("involute_pinion"), holes)
        export_internal_ring_svg(
            Involute(rad_ring, int(rad_ring*circular_pitch), internal=True),
            base+name.format("involute_internal"))

        export_gear_svg(
            Sinusoidal(rad_gear, int(rad_gear*circular_pitch), tooth_radius=.1),
            base+name.format("sinusoidal_pinion"), holes)
        export_internal_ring_svg(
            Sinusoidal(rad_ring, int(rad_ring*circular_pitch), tooth_radius=.1),
            base+name.format("sinusoidal_internal"))

        export_gear_svg(
            Cycloidal(rad_gear, int(rad_gear*circular_pitch)),
            base+name.format("cycloidal_pinion"), holes)
        export_internal_ring_svg(
            Cycloidal(rad_ring, int(rad_ring*circular_pitch)),
            base+name.format("cycloidal_internal"))

    if EXPORT_ELLIP_SPIRO:
        base = base.format("ellip_spiro")
        os.makedirs(base)
        rad_gear = 3
        rad_ring = 5
        circular_pitch = 15

        export_internal_ring_svg(
            InvoluteElliptical(rad_ring, 0., rad_ring*circular_pitch,
                               internal=True),
            base+name.format("involute_elliptical_fixed"))
        holes = [(1., 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .2, rad_gear*circular_pitch),
            base+name.format("involute_elliptical_moving_fixpoi_1"), holes)
        holes = [(1.232, 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .429, rad_gear*circular_pitch),
            base+name.format("involute_elliptical_moving_fixpoi_2"), holes)
        holes = [(.4755, 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .31, rad_gear*circular_pitch),
            base+name.format("involute_elliptical_moving_fixisectangle_1"),
            holes)
        holes = [(.615, 0.)]
        export_gear_svg(
            InvoluteElliptical(rad_gear, .4005, rad_gear*circular_pitch),
            base+name.format("involute_elliptical_moving_fixisectangle_2"),
            holes)

    if EXPORT_HOOT:
#        export_hoot((10, 4, 2, .9, 2.5, 1.5, 10, 9.95),
#                    1., base, name)
#        export_hoot((15, 14, 3, 1.3316335903703507, 3.7699182132015086, 2.0515806228896993, 32.6563894503349, 16.11314752952731),
#                    .5, base, name)
        export_hoot((15, 15, 11, 1.4828358173458094, 6.662438560329804, 4.188996353258297, 34.33010940225151, 30.81963260425178),
                    .5, base, name)


if __name__ == "__main__":
    main()
