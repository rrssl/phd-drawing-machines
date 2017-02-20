# -*- coding: utf-8 -*-
"""
Exporting the Thing

@author: Robin Roussel
"""
import os
import math
import numpy as np

import _context
from _base import (get_svg_context, export_gear_svg, export_toothless_ring_svg,
                   add_support_line, add_support_holes_line,
                   STROKE_WIDTH)
from gearprofile import Involute
from mecha import Thing


def export_thing_base_svg(asb, rad_tt, tt_hole_size, gears_hole_size, filename):
    base_thickness = .3
    support_base_height = .5
    margin = .1 * rad_tt
    nb_gears = sum(1 for name in asb.keys() if name.startswith('gear'))
    gpos = np.hstack([asb['gear_{}'.format(i)]['pos'] for i in range(nb_gears)])
    gpos = np.hstack([gpos, asb['dgear']['pos']])
    xmin, ymin = np.abs(gpos.min(axis=1))
    xmax, ymax = gpos.max(axis=1)
    dims = 8*margin + np.array([xmax+xmin, ymax+ymin])
    dims_ext = dims + [0, 4*(margin+base_thickness+support_base_height)]

    cont = get_svg_context(filename, dims_ext)
    cut = cont.add(cont.g(fill='none', stroke='red', stroke_width=.005))
    # Base contour
    xy_b = [margin, margin]
    size = dims - 2*margin
    cut.add(cont.rect(insert=xy_b, size=size, rx=margin, ry=margin))
    # Supports
    nb = 13
    #  -------
    # |   1   |
    # | 3   4 |
    # |   2   |
    #  -------
    add_support_holes_line(cut, cont,
                           (3*margin, dims[1]-2*margin-base_thickness),
                           (dims[0]-6*margin, base_thickness*.95), nb, axis='x')
    add_support_holes_line(cut, cont,
                           (3*margin, 2*margin),
                           (dims[0]-6*margin, base_thickness*.95), nb, axis='x')
    add_support_holes_line(cut, cont,
                           (2*margin, 3*margin),
                           (base_thickness*.95, dims[1]-6*margin), nb, axis='y')
    add_support_holes_line(cut, cont,
                           (dims[0]-2*margin-base_thickness, 3*margin),
                           (base_thickness*.95, dims[1]-6*margin), nb, axis='y')
    tot = support_base_height + base_thickness
    add_support_line(
        cut, cont, [margin, dims[0]+2*margin], dims[0]-6*margin,
        support_base_height, base_thickness, nb)
    add_support_line(
        cut, cont, [margin, dims[0]+3*margin+tot], dims[0]-6*margin,
        support_base_height, base_thickness, nb)
    add_support_line(
        cut, cont, [margin, dims[0]+4*margin+2*tot], dims[1]-6*margin,
        support_base_height, base_thickness, nb)
    add_support_line(
        cut, cont, [margin, dims[0]+5*margin+3*tot], dims[1]-6*margin,
        support_base_height, base_thickness, nb)
    # Turntable hole
    tt_c = 4*margin + np.array([xmin, ymin])
    size = np.array([tt_hole_size, tt_hole_size])
    xy_tt = tt_c - size / 2
    cut.add(cont.rect(insert=xy_tt, size=size))
    # Gears holes
    size = np.array([gears_hole_size*1.1, gears_hole_size])
    for xy in gpos.T:
        angle = np.arctan2(xy[1], xy[0])
        xy_g = tt_c + xy - size / 2
        g = cont.rect(insert=xy_g, size=size, rx=size[1]/2, ry=size[1])
        g.rotate(angle*180/math.pi, xy_g+size/2)
        cut.add(g)

    cont.save()


def export_thing_arm_svg(notch_x, notch_length, notch_depth, rail_width,
                         pivot_hole_size, arm_length, filename):
    dims = np.array([arm_length + 3*pivot_hole_size,
                     rail_width + 4*notch_depth])
    margin = .1 * dims
    cont = get_svg_context(filename, dims+2*margin)
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))
    # Outer contour
    xy = margin
    cut.add(cont.rect(insert=xy, size=dims, rx=margin[1], ry=margin[1]))
    # Pivot hole
    xy += [pivot_hole_size, rail_width/2 + 2*notch_depth]
    cut.add(cont.circle(center=xy, r=pivot_hole_size/2))
    # Rail
    xy += [pivot_hole_size, -rail_width/2]
    size = [arm_length, rail_width]
    cut.add(cont.rect(insert=xy, size=size, rx=rail_width/2, ry=rail_width/2))
    # Notches
    xy += [notch_x - notch_length/2 - pivot_hole_size, -2*notch_depth]
    size = [notch_length, notch_depth]
    cut.add(cont.rect(insert=xy, size=size))
    xy += [0, dims[1]-notch_depth]
    cut.add(cont.rect(insert=xy, size=size))

    cont.save()


def export_thing_end_arm_svg(length, sq_hole_width, sq_hole_angle, hole_rad,
                             filename):
    sq_hole_diag = sq_hole_width * math.sqrt(2) / 2
    dmax = max(hole_rad, sq_hole_diag)
    dims = np.array([length + 2*(hole_rad + sq_hole_diag), 4*dmax])
    margin = .1 * dims
    cont = get_svg_context(filename, dims + 2*margin)
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))

    xy = margin
    cut.add(cont.rect(insert=xy, size=dims, rx=dmax, ry=dmax))
    xy += [2*hole_rad, 2*dmax]
    cut.add(cont.circle(center=xy, r=hole_rad))
    xy[0] += length
    size = [sq_hole_width, sq_hole_width]
    r = cont.rect(insert=xy-sq_hole_width/2, size=size)
    r.rotate(sq_hole_angle*180/math.pi, xy)
    cut.add(r)

    cont.save()


def export_thing(props, scale, base, name):
    base = base.format("thing")
    os.makedirs(base)
    # Drawing machine dimensions
    m = Thing(*props)
    asb = m.assembly
    for coord in asb.values():
        if coord.get('pos') is not None:
            coord['pos'] *= scale
    rad_tt = scale
    rad_dgear = .5 * rad_tt # Radius of the driving gear
    ang_dgear = np.pi * 5 / 4
    asb['dgear'] = {
        'pos': (rad_tt + rad_dgear) * np.vstack([math.cos(ang_dgear),
                                                 math.sin(ang_dgear)])}
    nb_gears = len(props)
    radii = 1. / np.arange(2, 2+nb_gears)[::-1]
    radii[2] = 6/7
    radii *= scale
    amplitudes = np.asarray(props) * scale
    circular_pitch = 2*scale
    # The following dimensions come from physical measurements.
    # They are not affected by the scale factor.
    base_tt_hole_size = 1.991
    base_gears_hole_size = 1.2
    rad_tt_pivot = 4.01
    rad_gear_pivot = .405
    rad_pivot_slider = .31
    notch_length = .6
    notch_depth = .19
    rad_arm_pivot = .4
    end_arm_sq_hole_width = .9
    rad_penholder = .705
    # Base
    export_thing_base_svg(asb, rad_tt, base_tt_hole_size, base_gears_hole_size,
                          base+name.format("base"))
    # Turntable
    export_gear_svg(
        Involute(rad_tt, int(rad_tt*circular_pitch)),
        base+name.format("turntable_gear"),
        center_hole_size=rad_tt_pivot)
    export_toothless_ring_svg(
        rad_tt*.90, rad_tt*.95, base+name.format("turntable_ring"))
    # Gears
    for i, (rad, amp) in enumerate(zip(radii, amplitudes)):
        holes = [(amp, 0.)]
        gname = base + name.format("gear_{}".format(i+1))
        export_gear_svg(Involute(rad, int(rad*circular_pitch)), gname, holes,
                        center_hole_size=2*rad_gear_pivot)
    holes = [(.8* rad_dgear, 0.)]
    gname = base + name.format("dgear")
    export_gear_svg(Involute(rad_dgear, int(rad_dgear*circular_pitch)), gname,
                    holes, center_hole_size=2*rad_gear_pivot)
    # Arms
    m.set_state(np.linspace(0., m._simulator.get_cycle_length(), 2**10))
    asb = m.assembly
    for coord in asb.values():
        if coord.get('pos') is not None:
            coord['pos'] *= scale
    for i in range(nb_gears-1):
        # Notch position
        if i == 0:
            vec = asb['joint_0']['pos'][:, 0] - asb['pivot_0']['pos'][:, 0]
        else:
            vec = (asb['joint_{}'.format(i)]['pos'][:, 0]
                   - asb['joint_{}'.format(i-1)]['pos'][:, 0])
        notch_x = math.sqrt(vec[0]**2 + vec[1]**2)
        # Arm length
        if i == 0:
            vec = asb['pivot_1']['pos'] - asb['pivot_0']['pos']
        else:
            vec = (asb['pivot_{}'.format(i+1)]['pos']
                   - asb['joint_{}'.format(i)]['pos'])
        dist = np.sqrt(vec[0]**2 + vec[1]**2)
        arm_length = dist.max() * 1.1 # Safety

        aname = base + name.format("arm_{}".format(i+1))
        export_thing_arm_svg(notch_x, notch_length, notch_depth,
                             2*rad_pivot_slider, 2*rad_arm_pivot, arm_length,
                             aname)
    # End arm
    vec1 = (asb['pen-holder']['pos'][:, 0]
            - asb['joint_{}'.format(nb_gears - 2)]['pos'][:, 0])
    length1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    vec2 = (asb['pivot_{}'.format(nb_gears - 1)]['pos'][:, 0]
            - asb['joint_{}'.format(nb_gears - 2)]['pos'][:, 0])
    length2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    angle = math.acos(vec1.dot(vec2) / (length1 * length2))
    print(angle*180/np.pi)
    export_thing_end_arm_svg(length1, end_arm_sq_hole_width, angle,
                             rad_penholder, base+name.format("end_arm"))
