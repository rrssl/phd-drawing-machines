# -*- coding: utf-8 -*-
"""
Finding the 'kernel' in property space of a curve invariant in feature space.

Here the curve invariant is the following:
    "The intersection angle is constant."

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


class FixIsectAngle(TwoDimsDemo):
    """Find the subspace where the PoIs coincide."""

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        return get_corresp_isect(ref_crv, ref_par, curves, loc_size=10)

    def get_features(self, curve, param, poi):
        if param is None or None in param:
#            feats = 1e6
            feats= np.full(2, 1e6)
        else:
            curve = curve[:, :-1] # Remove last point
            n = curve.shape[1]
            param = np.asarray(param)
            v = curve[:, (param+1)%n] - curve[:, param%n]
#            v = crv[:, (par+1)%n] - crv[:, (par-1)%n]
            v /= np.linalg.norm(v, axis=0)
            feats = v[:, 1] - v[:, 0]
#            feats.append(v[0, 0] * v[1, 1] - v[1, 0] * v[0, 1])
        return feats

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The angle at the point of intersection is fixed by "
                        "the user.\n")


def main():
    """Entry point."""
    from _config import fixisectangle_data as data
    app = FixIsectAngle(**data)
    app.run()


if __name__ == "__main__":
    main()
