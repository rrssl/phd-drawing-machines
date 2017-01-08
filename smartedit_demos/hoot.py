#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration demos with the Hoot-Nanny.

@author: Robin Roussel
"""
from matplotlib.lines import Line2D
import numpy as np

import context
from mecha import HootNanny
from smartedit_demos import ManyDimsDemo
from poitrackers import get_corresp_krvmax


class FixPosHoot(ManyDimsDemo):
    """Find the subspace where the PoIs coincide.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)


    def get_features(self, curves, params, poi):
        return poi

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's position is fixed by the "
                        "user.")


class FixLineHoot(ManyDimsDemo):
    """Find the subspace where the PoI lies on the same line.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curves, params, poi):
        return np.arctan2(poi[1], poi[0])

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The line on which the point of interest lies is fixed "
                        "by the user.\n")
        # Draw the constraint axis.
        end = self.ref_poi * 10
        line = Line2D([0., end[0]], [0., end[1]], linewidth=2, color='gold',
                      linestyle='dashed')
        frame.add_line(line)


def main():
    """Entry point."""
    if 0:
        from _config import fixposhoot_data as data
        app = FixPosHoot(**data)
    elif 1:
        from _config import fixlinehoot_data as data
        app = FixLineHoot(**data)
    app.run()

if __name__ == "__main__":
    main()
