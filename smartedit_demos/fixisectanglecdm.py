# -*- coding: utf-8 -*-
"""
Finding the property subspace of a geometric invariant in curve space.

 -- The curve invariant is the intersection angle.

 -- Corresponding PoIs have the same curve parameter.
This simplifies the correspondence tracking for this simple demonstration;
however there is no loss of generality.

 -- We use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
import numpy as np

import context
from mecha import SingleGearFixedFulcrumCDM
from smartedit_demos import ManyDimsDemo
from poitrackers import get_corresp_isect


class FixIsectAngleCDM(ManyDimsDemo):
    """Find the subspace where the intersection angle is constant."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (2, 2)
        self.cont_prop = (3.5, 2.9, 3.8, 1.7)
        self.pts_per_dim = 5
        self.keep_ratio = .05
        self.nbhood_size = .1
        self.ndim_invar_space = 2
        self.mecha = SingleGearFixedFulcrumCDM(*self.disc_prop+self.cont_prop)
#        self.nb = 2**5
        self.labels = ["$d_f$", r"$ \theta_g$", "$d_p$", "$d_s$"]
        self.nb_crv_pts = 2**6
        # Reference curve and parameter(s).
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)

        self.ref_par = (44, 61)
        self.ref_poi, self.ref_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
        print(self.ref_par)
        # New curve and parameter(s).
        self.new_crv = None
        self.new_poi = None
        # Solution space.
        self.phi = None
        self.phi_inv = None
        self.pca = None
        self.new_cont_prop = None
        self.invar_space_bnds = None
        self.compute_invar_space()

        self.init_draw()

        # Controller
        self.slider_active = False
        self.fig.canvas.mpl_connect('button_release_event',
                                    self.on_button_release)

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
    app = FixIsectAngleCDM()
    app.run()


if __name__ == "__main__":
    main()
