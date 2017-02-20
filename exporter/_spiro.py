# -*- coding: utf-8 -*-
"""
Exporting the Spirographs

@author: Robin Roussel
"""
import os

from _base import export_gear_svg, export_internal_ring_svg
from gearprofile import Involute, Sinusoidal, Cycloidal, InvoluteElliptical

def export_spiro(base, name):
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

def export_ellip_spiro(base, name):
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