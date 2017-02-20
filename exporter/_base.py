# -*- coding: utf-8 -*-
"""
Base svg profile functions

@author: Robin Roussel
"""
import numpy as np
import svgwrite as svg

# From https://www.w3.org/TR/SVG/coords.html#Units
# Convert 0.1px (required by the laser cutter) to cm ( = our user unit)
STROKE_WIDTH = 0.1 / 35.43307

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

def export_toothless_ring_svg(rad_int, rad_ext, filename):
    margin = .1
    dims = 2*rad_ext*(1+margin)*np.array([1, 1])
    cont = get_svg_context(filename, dims)
    cut = cont.add(cont.g(fill='none', stroke='red',
                          stroke_width=STROKE_WIDTH))

    cut.add(cont.circle(center=dims/2, r=rad_int))
    cut.add(cont.circle(center=dims/2, r=rad_ext))

    cont.save()

