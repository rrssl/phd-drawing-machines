#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exploration with in multidimensional spaces.

@author: Robin Roussel
"""
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

import _context
from curveproc import compute_curvature
from _base import ManyDimsDemo, _get_inwards_normal


class FixPos(ManyDimsDemo):
    """Find the subspace where the PoIs coincide.

    We use index value as an approx. of parameter value (discretized curve).
    """

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's position is fixed by the "
                        "user.")


class FixLine(ManyDimsDemo):
    """Find the subspace where the PoI lies on the same line.

    We use index value as an approx. of parameter value (discretized curve).
    """
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


class FixKrv(ManyDimsDemo):
    """Find the subspace where the curvature at the PoI is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    ### VIEW

    def redraw(self):
        """Redraw dynamic elements."""
        super().redraw()
        # Mutliply radius by 2 to make the circle more visible.
        rk = 2. / compute_curvature(self.new_crv)[self.new_par]
        normal = _get_inwards_normal(self.new_crv, self.new_par) * rk
        self.new_osc_plt.center = self.new_poi + normal
        self.new_osc_plt.radius = rk

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's curvature is fixed by the "
                        "user.\n")
        # Ref osculating circle.
        # Mutliply radius by 2 to make the circle more visible.
        ref_rk = 2. / compute_curvature(self.ref_crv)[self.ref_par]
        normal = _get_inwards_normal(self.ref_crv, self.ref_par) * ref_rk
        self.ref_osc_plt = Circle(
            self.ref_poi+normal, ref_rk, color='k', alpha=.5, lw=1, fill=False,
            ls='--', label="Ref. osc. circle")
        frame.add_patch(self.ref_osc_plt)
        # New osculating circle.
        self.new_osc_plt = Circle((0,0), 0, color='b', alpha=.7, lw=2,
                              fill=False, ls='--', label="New osc. circle")
        frame.add_patch(self.new_osc_plt)


class FixKrvNoDisplay(ManyDimsDemo):
    """Find the subspace where the curvature at the PoI is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """

    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The point of interest's curvature is fixed by the "
                        "user.\n")


class FixKrvLine(ManyDimsDemo):
    """Find the subspace where the PoI has the same curvature and lies on the
    same radial line.

    We use index value as an approx. of parameter value (discretized curve).
    """
#    def redraw(self):
#        """Redraw dynamic elements."""
#        super().redraw()
#        # Mutliply radius by 2 to make the circle more visible.
#        rk = 2. / compute_curvature(self.new_crv)[self.new_par]
#        normal = _get_inwards_normal(self.new_crv, self.new_par) * rk
#        self.new_osc_plt.center = self.new_poi + normal
#        self.new_osc_plt.radius = rk

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
#        # Ref osculating circle.
#        # Mutliply radius by 2 to make the circle more visible.
#        ref_rk = 2. / compute_curvature(self.ref_crv)[self.ref_par]
#        normal = _get_inwards_normal(self.ref_crv, self.ref_par) * ref_rk
#        self.ref_osc_plt = Circle(
#            self.ref_poi+normal, ref_rk, color='k', alpha=.5, lw=1, fill=False,
#            ls='--', label="Ref. osc. circle")
#        frame.add_patch(self.ref_osc_plt)
#        # New osculating circle.
#        self.new_osc_plt = Circle((0,0), 0, color='b', alpha=.7, lw=2,
#                              fill=False, ls='--', label="New osc. circle")
#        frame.add_patch(self.new_osc_plt)


class FixIsectAngle(ManyDimsDemo):
    """Find the subspace where the intersection angle is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The angle at the intersection point is fixed by the "
                        "user.\n")


class FixDist(ManyDimsDemo):
    """Find the subspace where the distance between two PoIs is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The distance between the two PoIs is fixed "
                        "by the user.\n")
#        # Draw the constraint axis.
#        end = self.ref_poi * 10
#        line = Line2D([0., end[0]], [0., end[1]], linewidth=2, color='gold',
#                      linestyle='dashed')
#        frame.add_line(line)


class FixDistKrv(ManyDimsDemo):
    """Find the subspace where the distance between two PoIs is constant and
    the curvature at the PoIs is constant.

    We use index value as an approx. of parameter value (discretized curve).
    """
    ### VIEW

    def draw_curve_space(self, frame):
        """Draw the curve."""
        super().draw_curve_space(frame)
        frame.set_title("Curve space (visible in the UI).\n"
                        "The distance between the two PoIs and the "
                        "difference\nbetween their curvatures "
                        "are fixed by the user.")
#        # Draw the constraint axis.
#        end = self.ref_poi * 10
#        line = Line2D([0., end[0]], [0., end[1]], linewidth=2, color='gold',
#                      linestyle='dashed')
#        frame.add_line(line)


def main():
    """Entry point."""
    # CDM
    if 0:
        from _config import fixposcdm_data as data
        app = FixPos(**data)
    elif 0:
        from _config import fixkrvcdm_data as data
        app = FixKrv(**data)
    elif 0:
        from _config import fixlinecdm_data as data
        app = FixLine(**data)
    elif 0:
        from _config import fixkrvlinecdm_data as data
        data['mecha_type'].ConstraintSolver.max_nb_turns = 12
        app = FixKrvLine(**data)
    elif 0:
        from _config import fixisectanglecdm_data as data
        app = FixIsectAngle(**data)
    # Hoot-Nanny
    elif 0:
        from _config import fixposhoot_data as data
        app = FixPos(**data)
    elif 0:
        from _config import fixlinehoot_data as data
        app = FixLine(**data)
    elif 0:
        from _config import fixkrvhoot_data as data
        app = FixKrv(**data)
    elif 0:
        from _config import fixisectanglehoot_data as data
        app = FixIsectAngle(**data)
    elif 0:
        from _config import fixdisthoot_data as data
        app = FixDist(**data)
    elif 0:
        from _config import fixdistkrvhoot_data as data
        app = FixDistKrv(**data)
    # Kicker
    elif 1:
        from _config import fixposkick_data as data
        app = FixPos(**data)
    # Thing
    elif 0:
        from _config import fixposthing_data as data
        app = FixPos(**data)
    elif 0:
        from _config import fixkrvthing_data as data
        app = FixKrvNoDisplay(**data)
    app.run()

if __name__ == "__main__":
    main()
