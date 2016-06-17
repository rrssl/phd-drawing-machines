#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export script.

@author: Robin Roussel
"""
import svgwrite as svg
from gearprofile import InvoluteGear


def main():
    """Entry point."""
    gear = InvoluteGear(5, 20)
    profile = gear.get_profile()

    profile = profile.T - profile.min(axis=1)
    dims = profile.max(axis=0)

    margin = 0.1
    profile += dims * margin * 0.5
    dims *= 1 + margin
    
    width = str(dims[0])
    height = str(dims[1])

    dwg = svg.Drawing('test.svg', profile='tiny', 
                      size=(width + 'cm', height + 'cm'),
                      viewBox='0 0 ' + width + ' ' + height)
    group = dwg.add(dwg.g(fill='none', stroke='red', stroke_width=0.05))
    group.add(dwg.polyline(points=profile))
    group.add(dwg.circle(center=dims * 0.5 + (2, 0), r=0.2))
    dwg.save()


if __name__ == "__main__":
    main()
