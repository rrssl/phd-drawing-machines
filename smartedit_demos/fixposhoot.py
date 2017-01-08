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
import context
from mecha import HootNanny as DrawingMachine
from smartedit_demos import ManyDimsDemo
from poitrackers import get_corresp_krvmax


class FixPosHoot(ManyDimsDemo):
    """Find the subspace where the PoIs coincide."""

    def __init__(self, props, init_poi_id, pts_per_dim=5,
                 keep_ratio=.05, nbhood_size=.1, ndim_invar_space=2,
                 nb_crv_pts=2**6):
        # Initial parameters.
        nb_dprops = DrawingMachine.ConstraintSolver.nb_dprops
        self.disc_prop = props[:nb_dprops]
        self.cont_prop = props[nb_dprops:]
        self.pts_per_dim = pts_per_dim
        self.keep_ratio = keep_ratio
        self.nbhood_size = nbhood_size
        self.ndim_invar_space = ndim_invar_space
        self.mecha = DrawingMachine(*props)
        self.nb_crv_pts = nb_crv_pts
        self.labels = DrawingMachine.param_names[nb_dprops:]
        # Reference curve and parameter(s).
        self.ref_crv = self.mecha.get_curve(self.nb_crv_pts)
        self.ref_par = init_poi_id
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
    from _config import fixposhoot_data as data
    app = FixPosHoot(**data)
    app.run()

if __name__ == "__main__":
    main()
