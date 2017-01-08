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
import numpy as np

#import context
from smartedit_demos import TwoDimsDemo
from poitrackers import get_corresp_isect


class FixIsectPosDemo(TwoDimsDemo):
    """Find the subspace where the PoIs coincide."""

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
    from _config import fixisectpos_data as data
    app = FixIsectPosDemo(**data)
    app.run()


if __name__ == "__main__":
    main()
