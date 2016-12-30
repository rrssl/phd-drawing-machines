# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The position of the corresponding PoIs is constant."
Except this time, the PoI is an intersection point.

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs are the closest PoIs of the same type."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

Lastly we use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
#import math
from itertools import compress
import matplotlib.pyplot as plt
import numpy as np

import context
from mecha import EllipticSpirograph
from smartedit_demos import TwoDimsDemo, get_dist
from poitrackers import get_corresp_isect


class FixIsectPosDemo(TwoDimsDemo):
    """Find the subspace where the PoIs coincide."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (5, 3)
        self.cont_prop = (.31, .48) # Quasi zero angle between segments
        self.pts_per_dim = 17
        self.keep_ratio = .05
        self.deg_invar_poly = 3
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
        self.labels = ["$e^2$", "$d$"]
        self.nb_crv_pts = 2**6
        # Reference curve and parameter.
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
#        self.ref_par = (11, 117)
        self.ref_par = (53, 267)
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
        return get_corresp_isect(ref_crv, ref_par, curves)

    def get_features(self, curve, param, poi):
        feats = np.array([1e6, 1e6]) if poi is None else poi
        return feats

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's position is fixed by the "
                        "user.\n")


def main():
    """Entry point."""
    app = FixIsectPosDemo()
    app.run()


if __name__ == "__main__":
    main()
