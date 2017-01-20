#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration demos with the Thing.

@author: Robin Roussel
"""
from matplotlib.patches import Circle
import context
from curveproc import compute_curvature
from smartedit_demos import ManyDimsDemo, _get_inwards_normal
from poitrackers import get_corresp_krvmax


class FixPosThing(ManyDimsDemo):
    """Find the subspace where the PoIs coincide.

    We use index value as an approx. of parameter value (discretized curve).
    """

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


class FixKrvThing(ManyDimsDemo):
    """Find the subspace where the curvature at the PoI is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        return compute_curvature(curve)[param]

    ### VIEW

#    def redraw(self):
#        """Redraw dynamic elements."""
#        super().redraw()
#        # Mutliply radius by 2 to make the circle more visible.
#        rk = 2. / compute_curvature(self.new_crv)[self.new_par]
#        normal = _get_inwards_normal(self.new_crv, self.new_par) * rk
#        self.new_osc_plt.center = self.new_poi + normal
#        self.new_osc_plt.radius = rk

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's curvature is fixed by the "
                        "user.\n")
#        # Ref osculating circle.
#        # Mutliply radius by 2 to make the circle more visible.
#        ref_rk = 2. / compute_curvature(self.ref_crv)[self.ref_par]
#        normal = _get_inwards_normal(self.ref_crv, self.ref_par) * ref_rk
#        self.ref_osc_plt = Circle(
#            self.ref_poi+normal, ref_rk, color='k', alpha=.5, lw=1, fill=False,
#            ls='--', label="Ref. osc. circle")
#        frame.add_patch(self.ref_osc_plt)
#        # New osculating circle.
#        self.new_osc_plt = Circle((0,0), 0, color='b', alpha=.7, lw=2,
#                              fill=False, ls='--', label="New osc. circle")
#        frame.add_patch(self.new_osc_plt)


def main():
    """Entry point."""
    if 0:
        from _config import fixposthing_data as data
        app = FixPosThing(**data)
    if 1:
        from _config import fixkrvthing_data as data
        app = FixKrvThing(**data)
    app.run()

if __name__ == "__main__":
    main()
