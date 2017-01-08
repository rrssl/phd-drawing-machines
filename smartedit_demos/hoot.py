#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration demos with the Hoot-Nanny.

@author: Robin Roussel
"""
from matplotlib.lines import Line2D
import numpy as np

import context
from mecha import HootNanny as DrawingMachine
from smartedit_demos import ManyDimsDemo
from poitrackers import get_corresp_krvmax


class FixPosHoot(ManyDimsDemo):
    """Find the subspace where the PoIs coincide.

    We use index value as an approx. of parameter value (discretized curve).
    """

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


class FixLineHoot(ManyDimsDemo):
    """Find the subspace where the PoI lies on the same line.

    We use index value as an approx. of parameter value (discretized curve).
    """

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


def main():
    """Entry point."""
    if 1:
        from _config import fixposhoot_data as data
        app = FixPosHoot(**data)
    elif 0:
        from _config import fixlinehoot_data as data
        app = FixLineHoot(**data)
    app.run()

if __name__ == "__main__":
    main()
