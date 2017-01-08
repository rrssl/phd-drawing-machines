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
#import context
from smartedit_demos import TwoDimsDemo


class FixPosDemo(TwoDimsDemo):
    """Find the position-invariant subspace."""

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
