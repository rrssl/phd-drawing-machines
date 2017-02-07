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
import _context
import mecha

# From https://www.w3.org/TR/SVG/coords.html#Units
# Convert 0.1px (required by the laser cutter) to cm ( = our user unit)
STROKE_WIDTH = 0.1 / 35.43307


EXPORT_SPIRO = False
EXPORT_ELLIP_SPIRO = False
EXPORT_HOOT = False
EXPORT_THING = True


def get_svg_context(filename, dims_cm):
    """Returns the SVG context."""
    width = str(dims_cm[0])
    height = str(dims_cm[1])
    # Define viewBox with the same dimensions as 'size' to get unit in cm by
    # default.
    return svg.Drawing(filename, size=(width + 'cm', height + 'cm'),
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
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))
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
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))
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


def export_toothless_ring_svg(rad_int, rad_ext, filename):
    margin = .1
    dims = 2*rad_ext*(1+margin)*np.array([1, 1])
    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))

    cut.add(cont.circle(center=dims/2, r=rad_int))
    cut.add(cont.circle(center=dims/2, r=rad_ext))

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


def add_support_holes_line(g, cont, xy, size, nb, axis):
    x, y = xy
    w, h = size
    if axis == 'x':
        xlist, step = np.linspace(x, x+w, nb+2, endpoint=False, retstep=True)
        for xi in xlist[::2]:
            g.add(cont.rect(insert=(xi, y), size=(step, h)))
    if axis == 'y':
        ylist, step = np.linspace(y, y+h, nb+2, endpoint=False, retstep=True)
        for yi in ylist[::2]:
            g.add(cont.rect(insert=(x, yi), size=(w, step)))

def add_support_line(g, cont, xy, l, h1, h2, nb):
    x, y = xy
    points = [[x, y]]
    xlist, step = np.linspace(x, x+l, nb+2, endpoint=False, retstep=True)
    for xi in xlist[::2]:
        points.append([xi, y+h1])
        points.append([xi, y+h1+h2])
        points.append([xi+step, y+h1+h2])
        points.append([xi+step, y+h1])
    points.append([x+l, y])
    points.append([x, y])

    g.add(cont.polyline(points=points))


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
    m = mecha.Thing(*props)
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
    # The 4 following dimensions come from physical measurements.
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


def main():
    base = "svg/{}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + "/"
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
        export_hoot((15, 14, 3, 1.3316335903703507, 3.7699182132015086, 2.0515806228896993, 32.6563894503349, 16.11314752952731),
                    .5, base, name)
#        export_hoot((15, 15, 11, 1.4828358173458094, 6.662438560329804, 4.188996353258297, 34.33010940225151, 30.81963260425178),
#                    .5, base, name)

    if EXPORT_THING:
        export_thing((0.09191176470588247, 0.1663602941176472, 0.08226102941176472, 0.020220588235294157, 0.38419117647058854),
                     8, base, name)


if __name__ == "__main__":
    main()
