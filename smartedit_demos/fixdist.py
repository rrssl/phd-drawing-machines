# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "Two PoIs in the curve are always at the same distance."
But in practice we implement the sufficient condition:
    "Two PoIs in the curve are always at the same relative position."
Meaning that the invariant 'feature' here is the difference between the PoIs'
position.

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs are the closest PoIs of the same type."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

Lastly we use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
#import math
import matplotlib.pyplot as plt

import context
from mecha import EllipticSpirograph
from smartedit_demos import TwoDimsDemo
from poitrackers import get_corresp_krvmax


class FixDistDemo(TwoDimsDemo):
    """Find the subspace where the PoIs are at the same distance."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (4, 3)
#        self.cont_prop = (.31, .48) # Nonzero dist between PoIs
#        self.cont_prop = (.3, 1.) # Nonzero dist between PoIs
#        self.cont_prop = (.3, .692) # Quasi zero dist
        self.cont_prop = (.1, .2) # Quasi zero dist

        self.pts_per_dim = 20
        self.keep_ratio = .05
        self.deg_invar_poly = 2
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
        self.labels = ["$e^2$", "$d$"]
        self.nb_crv_pts = 2**6
        # Reference curve and parameter.
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
        self.ref_par = (150, 117)
#        self.ref_par = (53, 267)
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
        return get_corresp_krvmax(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        return poi[:, 1] - poi[:, 0]

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The distance between points of interest is fixed by "
                        "the user.\n")


def main():
    """Entry point."""
    plt.ioff()

    FixDistDemo()

    plt.show()

if __name__ == "__main__":
    main()
