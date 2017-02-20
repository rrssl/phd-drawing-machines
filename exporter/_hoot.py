#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporting the Hoot-Nanny

@author: Robin Roussel
"""
import os
import math
import numpy as np

from _base import (get_svg_context, export_gear_svg, export_toothless_ring_svg,
                   STROKE_WIDTH)
from gearprofile import Involute

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
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.005))
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


def export_hoot_arm_svg(length, rad1, rad2, filename):
    margin = .1
    rmax = max(rad1, rad2)
    dims = np.array([length + 2*(rad1 + rad2), 4*rmax])
    cont = get_svg_context(filename, (1 + 2*margin)*dims)
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))

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
    # TODO shorten variable names, use numpy vector
    rad_turntable = props[0]*scale
    rad_gear1 = props[1]*scale
    rad_gear2 = props[2]*scale
    gear_angle = props[3]
    dist1 = props[4]*scale
    dist2 = props[5]*scale
    length1 = props[6]*scale
    length2 = props[7]*scale
    circular_pitch = 12/scale
    rad_dgear = .5 * rad_turntable # Radius of the driving gear
    # The 4 following dimensions come from physical measurements.
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
