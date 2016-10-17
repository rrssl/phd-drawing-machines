# -*- coding: utf-8 -*-
"""
Finding the property subspace of a geometric invariant in curve space.

 -- The curve invariant is the position of the PoI.

 -- Corresponding PoIs have the same curve parameter.
This simplifies the correspondence tracking for this simple demonstration;
however there is no loss of generality.

 -- We use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
import matplotlib.pyplot as plt

import context
from mecha import HootNanny
from smartedit_demos import ManyDimsDemo
from poitrackers import get_corresp_krvmax


class FixPosHoot(ManyDimsDemo):
    """Find the subspace where the PoIs coincide."""

    def __init__(self):
        # Initial parameters.
#        self.disc_prop = (9, 5, 3)
#        self.cont_prop = (.96, .75, 1.5, 9.9, 7.9)
        self.disc_prop = (10, 4, 2)
        self.cont_prop = (.69, 2.52, .86, 10.22, 9.56)
        self.pts_per_dim = 5
        self.keep_ratio = .05
        self.nbhood_size = .1
        self.ndim_invar_space = 3
        self.mecha = HootNanny(*self.disc_prop+self.cont_prop)
        self.nb_crv_pts = 2**7
#        self.nb = 2**5
        self.labels = [r"$ \theta_{12}$", "$d_1$", "$d_2$", "$l_1$", "$l_2$"]
        # Reference curve and parameter(s).
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
        self.ref_par = 100
        self.ref_poi, self.ref_par = self.get_corresp(
            self.ref_crv, self.ref_par, [self.ref_crv])
        self.ref_poi, self.ref_par = self.ref_poi[0], self.ref_par[0]
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


def main():
    """Entry point."""
    plt.ioff()

    FixPosHoot()

    plt.show()


if __name__ == "__main__":
    main()
