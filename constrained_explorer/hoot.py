#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration demos with the Hoot-Nanny.

@author: Robin Roussel
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

import context
from curveproc import compute_curvature
from mecha import HootNanny
from smartedit_demos import ManyDimsDemo, _get_inwards_normal
from poitrackers import get_corresp_krvmax, get_corresp_isect


class FixPosHoot(ManyDimsDemo):
    """Find the subspace where the PoIs coincide.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)


    def get_features(self, curve, param, poi):
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

    def get_features(self, curve, param, poi):
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


class FixKrvHoot(ManyDimsDemo):
    """Find the subspace where the curvature at the PoI is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        return compute_curvature(curve)[param]

    ### VIEW

    def redraw(self):
        """Redraw dynamic elements."""
        super().redraw()
        # Mutliply radius by 2 to make the circle more visible.
        rk = 2. / compute_curvature(self.new_crv)[self.new_par]
        normal = _get_inwards_normal(self.new_crv, self.new_par) * rk
        self.new_osc_plt.center = self.new_poi + normal
        self.new_osc_plt.radius = rk

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's curvature is fixed by the "
                        "user.\n")
        # Ref osculating circle.
        # Mutliply radius by 2 to make the circle more visible.
        ref_rk = 2. / compute_curvature(self.ref_crv)[self.ref_par]
        normal = _get_inwards_normal(self.ref_crv, self.ref_par) * ref_rk
        self.ref_osc_plt = Circle(
            self.ref_poi+normal, ref_rk, color='k', alpha=.5, lw=1, fill=False,
            ls='--', label="Ref. osc. circle")
        frame.add_patch(self.ref_osc_plt)
        # New osculating circle.
        self.new_osc_plt = Circle((0,0), 0, color='b', alpha=.7, lw=2,
                              fill=False, ls='--', label="New osc. circle")
        frame.add_patch(self.new_osc_plt)


class FixIsectAngleHoot(ManyDimsDemo):
    """Find the subspace where the intersection angle is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_isect(ref_crv, ref_par, curves, loc_size=6)

    def get_features(self, curve, param, poi):
        if poi is None or not poi.size:
            feats= np.full(2, 1e6)
        else:
            curve = curve[:, :-1] # Remove last point
            n = curve.shape[1]
            param = np.asarray(param)
            v = curve[:, (param+1)%n] - curve[:, param%n]
            v /= np.linalg.norm(v, axis=0)
            feats = v[:, 1] - v[:, 0]
        return feats

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The angle at the intersection point is fixed by the "
                        "user.\n")


class FixDistHoot(ManyDimsDemo):
    """Find the subspace where the distance between two PoIs is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        diff = poi[:, 1] - poi[:, 0]
        return diff[0]**2 + diff[1]**2

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The distance between the two PoIs is fixed "
                        "by the user.\n")
#        # Draw the constraint axis.
#        end = self.ref_poi * 10
#        line = Line2D([0., end[0]], [0., end[1]], linewidth=2, color='gold',
#                      linestyle='dashed')
#        frame.add_line(line)


class FixDistKrvHoot(ManyDimsDemo):
    """Find the subspace where the distance between two PoIs is constant and
    the curvature at the PoIs is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(HootNanny, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        krv = compute_curvature(curve)[param]
        diffpos = (poi[:, 1] - poi[:, 0]) / poi[:, 0]
        diffkrv = (krv[1] - krv[0]) / krv[0]
        return diffpos[0]**2 + diffpos[1]**2 + diffkrv**2

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The distance between the two PoIs and the "
                        "difference\nbetween their curvatures "
                        "are fixed by the user.")
#        # Draw the constraint axis.
#        end = self.ref_poi * 10
#        line = Line2D([0., end[0]], [0., end[1]], linewidth=2, color='gold',
#                      linestyle='dashed')
#        frame.add_line(line)


def main():
    """Entry point."""
    if 1:
        from _config import fixposhoot_data as data
        app = FixPosHoot(**data)
    elif 0:
        from _config import fixlinehoot_data as data
        app = FixLineHoot(**data)
    elif 0:
        from _config import fixkrvhoot_data as data
        app = FixKrvHoot(**data)
    elif 0:
        from _config import fixisectanglehoot_data as data
        app = FixIsectAngleHoot(**data)
    elif 0:
        from _config import fixdisthoot_data as data
        app = FixDistHoot(**data)
    elif 0:
        from _config import fixdistkrvhoot_data as data
        app = FixDistKrvHoot(**data)
    app.run()

if __name__ == "__main__":
    main()
