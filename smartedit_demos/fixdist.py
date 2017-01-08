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
import context
from smartedit_demos import TwoDimsDemo
from poitrackers import get_corresp_krvmax


class FixDistDemo(TwoDimsDemo):
    """Find the subspace where the PoIs are at the same distance."""

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
    from _config import fixdist_data as data
    app = FixDistDemo(**data)
    app.run()


if __name__ == "__main__":
    main()
