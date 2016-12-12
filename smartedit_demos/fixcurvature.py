# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The curvature at the corresponding PoIs is constant."
Meaning that the invariant 'feature' of the PoIs here is the curvature.

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs have the same parameter value."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff t1 = t2.
This simplifies the correspondence tracking for this simple demonstration;
however there is no loss of generality.

Lastly this criterion allows us to use index value as a proxy for parameter
value.

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import context
from mecha import EllipticSpirograph
from curveproc import compute_curvature
from smartedit_demos import TwoDimsDemo


class FixKrvDemo(TwoDimsDemo):
    """Find the curvature-invariant subspace."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (5, 3)
        self.cont_prop = (.2, 1.)
        self.pts_per_dim = 15
        self.keep_ratio = .05
        self.deg_invar_poly = 2
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
        self.labels = ["$e^2$", "$d$"]
        self.nb_crv_pts = 2**6
        # Reference curve and parameter.
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
        self.ref_par = 0
        self.ref_poi, self.ref_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
        # New curve and parameter(s).
        self.new_crv = None
        self.new_poi = None
        # Solution space.
        self.samples = None
        self.scores = None
        self.phi = None
        self.invar_space_bnds = None
        # Optimal solution.
        self.opt_path = None
        self.phi_opt = None

        self.compute_invar_space()

        self.init_draw()

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        cor_poi = [crv[:, ref_par] for crv in curves]
        cor_par = [ref_par] * len(curves)

        return cor_poi, cor_par

    def get_features(self, curve, param, poi):
        return compute_curvature(curve)[param]

    ### VIEW

    def redraw(self):
        """Redraw dynamic elements."""
        super().redraw()
        rk = 1 / compute_curvature(self.new_crv)[self.ref_par]
        self.new_osc.center = self.new_poi - (rk, 0)
        self.new_osc.radius = rk

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's curvature is fixed by the "
                        "user.\n")
        # Ref osculating circle.
        ref_rk = 1 / compute_curvature(self.ref_crv)[self.ref_par]
        osc = Circle(self.ref_poi-(ref_rk, 0.), ref_rk, color='k', alpha=.5,
                     fill=False, ls='dotted', label="Ref. osc. circle")
        frame.add_patch(osc)
        # New osculating circle.
        self.new_osc = Circle((0,0), 0, color='b', alpha=.5,
                              fill=False, ls='dotted', label="New osc. circle")
        frame.add_patch(self.new_osc)


def main():
    """Entry point."""
    plt.ioff()

    FixKrvDemo()

    plt.show()

if __name__ == "__main__":
    main()
