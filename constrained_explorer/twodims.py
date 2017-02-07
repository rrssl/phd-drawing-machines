#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration in the special case of a bi-dimensional parameter
space. This allows us to visualize this space easily.

@author: Robin Roussel
"""
import numpy as np
from matplotlib.patches import Circle

import _context
from curveproc import compute_curvature
from _base import TwoDimsDemo
from poitrackers import get_corresp_krvmax, get_corresp_isect


class FixPosDemo(TwoDimsDemo):
    """Find the position-invariant subspace.

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
    """

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


class FixKrvDemo(TwoDimsDemo):
    """Find the curvature-invariant subspace.

    Here the curve invariant is the following:
        "The curvature at the corresponding PoIs is constant."
    Meaning that the invariant 'feature' of the PoIs here is the curvature.

    Moreover, the correspondance between PoIs is defined as follows:
        "Corresponding PoIs have the same parameter value."
    E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
    respectively parametrized by t1 and t2,
        r1(t1) === r2(t2) iff t1 = t2.
    This simplifies the correspondence tracking for this simple demonstration;
    however there is no loss of generality.

    Lastly this criterion allows us to use index value as a proxy for parameter
    value.
    """

    ### MODEL

    def get_corresp(self, ref_crv, ref_par, curves):
        cor_poi = [crv[:, ref_par] for crv in curves]
        cor_par = [ref_par] * len(curves)

        return cor_poi, cor_par

    def get_features(self, curve, param, poi):
        return compute_curvature(curve)[param]

    ### VIEW

    def redraw(self):
        """Redraw dynamic elements."""
        super().redraw()
        rk = 1 / compute_curvature(self.new_crv)[self.ref_par]
        self.new_osc.center = self.new_poi - (rk, 0)
        self.new_osc.radius = rk

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's curvature is fixed by the "
                        "user.\n")
        # Ref osculating circle.
        ref_rk = 1 / compute_curvature(self.ref_crv)[self.ref_par]
        osc = Circle(self.ref_poi-(ref_rk, 0.), ref_rk, color='k', alpha=.5,
                     fill=False, ls='dotted', label="Ref. osc. circle")
        frame.add_patch(osc)
        # New osculating circle.
        self.new_osc = Circle((0,0), 0, color='b', alpha=.5,
                              fill=False, ls='dotted', label="New osc. circle")
        frame.add_patch(self.new_osc)


class FixDistDemo(TwoDimsDemo):
    """Find the subspace where the PoIs are at the same distance.

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
    """

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


class FixIsectPosDemo(TwoDimsDemo):
    """Find the subspace where the PoIs coincide.

    Here the curve invariant is the following:
        "The position of the corresponding PoIs is constant."
    Except this time, the PoI is an intersection point.

    Moreover, the correspondance between PoIs is defined as follows:
        "Corresponding PoIs are the closest PoIs of the same type."
    E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
    respectively parametrized by t1 and t2,
        r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

    Lastly we use index value as an approx. of parameter value (discretized curve).
    """

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


class FixIsectAngle(TwoDimsDemo):
    """Find the subspace where the PoIs coincide.

    Here the curve invariant is the following:
        "The intersection angle is constant."

    Moreover, the correspondance between PoIs is defined as follows:
        "Corresponding PoIs are the closest PoIs of the same type."
    E.g. for two parametric curves r1 and r2 (i.e. two points in property space),
    respectively parametrized by t1 and t2,
        r1(t1) === r2(t2) iff type(r1(t1)) = type(r2(t2)) and t2 = argmin|t1 - t|.

    Lastly we use index value as an approx. of parameter value (discretized curve).
    """

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
    if 0:
        from _config import fixpos_data as data
        app = FixPosDemo(**data)
    elif 0:
        from _config import fixkrv_data as data
        app = FixKrvDemo(**data)
    elif 1:
        from _config import fixdist_data as data
        app = FixDistDemo(**data)
    elif 0:
        from _config import fixisectpos_data as data
        app = FixIsectPosDemo(**data)
    elif 1:
        from _config import fixisectangle_data as data
        app = FixIsectAngle(**data)

    app.run()

if __name__ == "__main__":
    main()
