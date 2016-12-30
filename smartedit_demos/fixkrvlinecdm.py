# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The curvature at the PoI and the line on which it lies are constant."

Moreover, the correspondance between PoIs is defined as follows:
    "Corresponding PoIs are the closest PoIs of the same type."
E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
respectively parametrized by t1 and t2,
    r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

Lastly we use index value as an approx. of parameter value (discretized curve).

@author: Robin Roussel
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

import context
from curveproc import compute_curvature
from mecha import SingleGearFixedFulcrumCDM
from smartedit_demos import ManyDimsDemo
from poitrackers import get_corresp_krvmax

SingleGearFixedFulcrumCDM.ConstraintSolver.max_nb_turns = 12


class FixKrvLineCDM(ManyDimsDemo):
    """Find the subspace where the PoI has the same curvature and lies on the
    same radial line."""

    def __init__(self):
        # Initial parameters.
        self.disc_prop = (12, 4)
        self.cont_prop = (13.9, 3., 17.5, 3.3)
#        self.disc_prop = (12, 3)
#        self.cont_prop = (13., 2.3, 14., 2.5)
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
        self.ref_par = 0
#        self.ref_par = (98, 183) # Intersection
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


def main():
    """Entry point."""
    app = FixKrvLineCDM()
    app.run()

if __name__ == "__main__":
    main()
