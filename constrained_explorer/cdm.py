#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration demos with the Cycloid Drawing Machine.

@author: Robin Roussel
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np

import context
from curveproc import compute_curvature
from mecha import SingleGearFixedFulcrumCDM
from smartedit_demos import ManyDimsDemo, _get_inwards_normal
from poitrackers import get_corresp_krvmax, get_corresp_isect


class FixPosCDM(ManyDimsDemo):
    """Find the subspace where the PoIs coincide.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(SingleGearFixedFulcrumCDM, *args, **kwargs)

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
                        "user.\n")


class FixKrvCDM(ManyDimsDemo):
    """Find the subspace where the curve has the same curvature at the PoIs.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(SingleGearFixedFulcrumCDM, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        return compute_curvature(curve)[param]
    #    return np.arctan2(poi[1], poi[0])
    #    return np.r_[compute_curvature(curve)[param] * 3e-2, np.arctan2(poi[1], poi[0])]
    #    return np.r_[compute_curvature(curve)[param], np.arctan2(poi[1], poi[0])]

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


class FixLineCDM(ManyDimsDemo):
    """Find the subspace where the PoI always lies on a radial line.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(SingleGearFixedFulcrumCDM, *args, **kwargs)

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


class FixKrvLineCDM(ManyDimsDemo):
    """Find the subspace where the PoI has the same curvature and lies on the
    same radial line.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(SingleGearFixedFulcrumCDM, *args, **kwargs)

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        return np.r_[compute_curvature(curve)[param] * 3e-2, np.arctan2(poi[1], poi[0])]
    #    return np.r_[compute_curvature(curve)[param], np.arctan2(poi[1], poi[0])]

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The curvature at the point of interest \nAND the line "
                        "on which it lies are both fixed by the user.\n")
        # Draw the constraint axis.
        end = self.ref_poi * 10
        line = Line2D([0., end[0]], [0., end[1]], linewidth=2, color='gold',
                      linestyle='dashed')
        frame.add_line(line)


class FixIsectAngleCDM(ManyDimsDemo):
    """Find the subspace where the intersection angle is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(SingleGearFixedFulcrumCDM, *args, **kwargs)

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
                        "The angle at the intersection point is fixed by the"
                        "user.\n")


def main():
    """Entry point."""
    if 0:
        from _config import fixposcdm_data as data
        app = FixPosCDM(**data)
    elif 1:
        from _config import fixkrvcdm_data as data
        app = FixKrvCDM(**data)
    elif 0:
        from _config import fixlinecdm_data as data
        app = FixLineCDM(**data)
    elif 0:
        from _config import fixkrvlinecdm_data as data
        SingleGearFixedFulcrumCDM.ConstraintSolver.max_nb_turns = 12
        app = FixKrvLineCDM(**data)
    elif 0:
        from _config import fixisectanglecdm_data as data
        app = FixIsectAngleCDM(**data)
    app.run()

if __name__ == "__main__":
    main()