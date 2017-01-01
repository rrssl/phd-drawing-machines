# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The position of the corresponding PoIs is constant."
Meaning that the invariant 'feature' of the PoIs here is the (x,y) coordinates:
i.e. for this simple example, curve space and feature space coincide.

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
import context
from smartedit_demos import TwoDimsDemo
from mecha import EllipticSpirograph


class FixPosDemo(TwoDimsDemo):
    """Find the position-invariant subspace."""

    def __init__(self, disc_prop, cont_prop, pts_per_dim=20, keep_ratio=.05,
                 deg_invar_poly=2, nb_crv_pts=2**6):
        # Initial parameters.
        self.disc_prop = disc_prop
        self.cont_prop = cont_prop
        self.pts_per_dim = pts_per_dim
        self.keep_ratio = keep_ratio
        self.deg_invar_poly = deg_invar_poly
        self.mecha = EllipticSpirograph(*self.disc_prop+self.cont_prop)
        self.labels = ["$e^2$", "$d$"]
        self.nb_crv_pts = nb_crv_pts
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
        return poi

#    def get_optimal_path(self):
#        """Return the invariant space computed optimally."""
#        bnd_e2 = self.mecha.get_prop_bounds(2)
#        e2 = np.linspace(bnd_e2[0], bnd_e2[1], self.num_e2_vals)
#        # Sol: r - a(e2) + d = x_ref with 2aE(e2) = pi*req
#        x_ref = self.ref_crv[0, self.ref_par]
#        r, req = self.disc_prop
#        a = math.pi * req / (2 * ellipe(e2))
#        d = x_ref - r + a
#
#        return e2, d

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's position is fixed by the "
                        "user.\n")


def main():
    """Entry point."""
    from _config import fixpos_data as data
    app = FixPosDemo(**data)
    app.run()

if __name__ == "__main__":
    main()
