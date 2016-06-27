# -*- coding: utf-8 -*-
"""
Module for creating involute spur gears.

@author: Robin Roussel
"""
import math
import numpy as np

import curves as cu


class GearProfile:
    points_per_tooth = 20

class Involute(GearProfile):
    """Circular involute spur gear profile.

    Conventional values for the pressure angle: 14.5°, 20°, 25°.
    """

    def __init__(self, pitch_radius, nb_teeth, pressure_angle=20,
                 internal=False):
        self.pitch_radius = pitch_radius
        self.nb_teeth = nb_teeth
        self.pressure_angle = pressure_angle
        self.internal = internal

#        if nb_teeth > 40:
#            print("Warning: a number of teeth > 40 will not give valid results.")

    def get_addendum(self):
        """Return the addendum.

        Radial distance between the pitch circle and the top of the teeth.
        For spur gears it is (by convention?) the inverse of the diametral
        pitch.
        If the gear is internal, addendum and dedendum are switched.
        """
        if self.internal:
            if self.pressure_angle == 14.5:
                return 1.157 / self.get_diametral_pitch()
            else:
                return 1.25 / self.get_diametral_pitch()
        else:
            return 1 / self.get_diametral_pitch()

    def get_addendum_circle_radius(self):
        """Get the radius of the outer circle."""
        return self.pitch_radius + self.get_addendum()

    def get_base_circle_radius(self):
        """Get the radius of the base circle.

        radius of the circle from which the the involute curve is generated.
        """
        return self.pitch_radius * math.cos(self.pressure_angle * math.pi / 180)

    def get_clearance(self):
        """Return the clearance.

        Difference between the dedendum and the addendum.
        """
        return self.get_dedendum() - self.get_addendum()

    def get_dedendum(self):
        """Get the dedendum.

        Radial distance between the pitch circle and the bottom of the teeth.
        By convention, with P the diametral pitch:
         - pressure angle = 14.5°: b = 1.157 / P
         - pressure angle = 20° or 25°: b = 1.25 / P
        Apparently the 1.157 is an approximation of 1 + Pi/20, but the
        rationale behind this remains mysterious.
        If the gear is internal, addendum and dedendum are switched.
        """
        if not self.internal:
            if self.pressure_angle == 14.5:
                return 1.157 / self.get_diametral_pitch()
            else:
                return 1.25 / self.get_diametral_pitch()
        else:
            return 1 / self.get_diametral_pitch()

    def get_dedendum_circle_radius(self):
        """Get the radius of the outer circle."""
        return self.pitch_radius - self.get_dedendum()

    def get_diametral_pitch(self):
        """Get the diametral pitch.

        Ratio of the number of teeth to the pitch diameter.
        """
        return self.nb_teeth / (2 * self.pitch_radius)

    def get_profile(self):
        """Create a circular gear profile."""
        # Get one tooth minus the last point.
        tooth = self.get_tooth()[:, :-1]
        # Symmetrize it.
        period_angle = 2 * math.pi / self.nb_teeth
        cos = math.cos(period_angle)
        sin = math.sin(period_angle)
        rot = np.array([[cos, -sin],
                        [sin, cos]])
        teeth = [tooth]
        for i in range(1, self.nb_teeth):
            teeth.append(rot.dot(teeth[i - 1]))
        teeth.append(tooth[:, 0].reshape((2, 1)))

        return np.hstack(teeth)

    def get_tooth(self):
        """Get a tooth of the gear profile."""
        # The tooth pattern is composed of 4 parts: the tooth floor, the base,
        # the circle involute, and the top.
        # Between the floor and the base we can possibly add a fillet.

        # Define the principal variables.
        phi = self.pressure_angle * math.pi / 180
        ra = self.get_addendum_circle_radius()
        rb = self.get_base_circle_radius()
        rd = self.get_dedendum_circle_radius()
        # Define the principal involute arguments.
        involute_base_arg = 0 if rd < rb else math.sqrt((rd / rb) ** 2 - 1)
        involute_prim_arg = math.tan(phi)
        involute_top_arg = math.sqrt((ra / rb) ** 2 - 1)
        # Define the principal angles.
        #
        # Angle from one point on a tooth to the corresponding point on an
        # adjacent tooth (i.e. 'period' of the profile).
        period_angle = 2 * math.pi / self.nb_teeth
        # Angle between the base of the tooth and the intersection of the
        # involute with the primitive.
        base_prim_angle = involute_prim_arg - phi
        # Angle between the base and the top of the tooth.
        base_top_angle = involute_top_arg - math.atan(involute_top_arg)
        # Complementary angle.
        prim_top_angle = base_top_angle - base_prim_angle

        # Compute the arguments.
        points_per_arc = 5
        tooth_base_args = np.linspace(- period_angle / 2,
                                      - (period_angle / 4 + base_prim_angle),
                                      points_per_arc)
        involute_args = np.linspace(involute_base_arg,
                                    involute_top_arg, 2*points_per_arc)[:-1]
        tooth_top_args = np.linspace(- period_angle / 4 + prim_top_angle,
                                     0., points_per_arc)

        # Compute the half-tooth.
        # Root
        root = rd * np.vstack([np.cos(tooth_base_args),
                               np.sin(tooth_base_args)])
        # Involute
        phase = - (period_angle / 4 + base_prim_angle)
        invol = cu.CircleInvolute(rb, phase).get_point(involute_args)
        # Top
        top = ra * np.vstack([np.cos(tooth_top_args),
                              np.sin(tooth_top_args)])
        halftooth = np.hstack([root, invol, top])

        # Mirror it.
        mirror = halftooth[:, :-1].copy()
        mirror[1] *= -1

        return np.hstack([halftooth, mirror[:, ::-1]])


class Sinusoidal(GearProfile):
    """Circular sinusoidal gear profile."""

    def __init__(self, pitch_radius, nb_teeth, tooth_radius):
        self.pitch_radius = pitch_radius
        self.nb_teeth = nb_teeth
        self.tooth_radius = tooth_radius

    def get_profile(self):
        """Create a circular gear profile."""
        R = self.pitch_radius
        N = self.nb_teeth
        r = self.tooth_radius
        t = np.linspace(0, 2 * math.pi, GearProfile.points_per_tooth * N)
        return (R + r * np.cos(t * N)) * np.vstack([np.cos(t), np.sin(t)])


class Cycloidal(GearProfile):
    """Circular cycloidal gear profile."""

    def __init__(self, pitch_radius, nb_teeth):
        self.pitch_radius = pitch_radius
        self.nb_teeth = nb_teeth

    def get_profile(self):
        """Create a circular gear profile."""
        # Get one tooth minus the last point.
        tooth = self.get_tooth()[:, :-1]
        # Symmetrize it.
        period_angle = 2 * math.pi / self.nb_teeth
        cos = math.cos(period_angle)
        sin = math.sin(period_angle)
        rot = np.array([[cos, -sin],
                        [sin, cos]])
        teeth = [tooth]
        for i in range(1, self.nb_teeth):
            teeth.append(rot.dot(teeth[i - 1]))
        teeth.append(tooth[:, 0].reshape((2, 1)))

        return np.hstack(teeth)

    def get_tooth(self):
        """Get a tooth of the gear profile."""
        tooth_radius = 0.5 * self.pitch_radius / self.nb_teeth
        hypo = cu.Hypotrochoid(self.pitch_radius, tooth_radius, tooth_radius)
        epi = cu.Epitrochoid(self.pitch_radius, tooth_radius, tooth_radius)
        # Compute the arguments.
        period_angle = 2 * math.pi / self.nb_teeth
        te = np.linspace(0, period_angle / 2, GearProfile.points_per_tooth)
        th = np.linspace(period_angle / 2, period_angle, GearProfile.points_per_tooth)
        # Compute the tooth.
        tooth = np.hstack([epi.get_point(te)[:, :-1], hypo.get_point(th)])
        # Rotate it to be coherent with the other gear profiles.
        cos = math.cos(- period_angle / 4)
        sin = math.sin(- period_angle / 4)
        rot = np.array([[cos, -sin],
                        [sin, cos]])

        return rot.dot(tooth)


class SinusoidalElliptic(GearProfile):
    """Sinusoidal elliptic gear profile."""

    def __init__(self, pitch_semimajor, pitch_semiminor, nb_teeth, tooth_radius):
        self.pitch_semimajor = pitch_semimajor
        self.pitch_semiminor = pitch_semiminor
        self.nb_teeth = nb_teeth
        self.tooth_radius = tooth_radius

    def get_profile(self):
        """Create a circular gear profile."""
        shape = cu.Ellipse(self.pitch_semimajor, self.pitch_semiminor)
        N = self.nb_teeth
        r = self.tooth_radius
        t = np.linspace(0, 2 * math.pi, GearProfile.points_per_tooth * N)
#        theta = np.arctan((b / a) * np.tan(t))

        primitive = shape.get_point(t)
        profile = shape.get_normal(t)
        profile /= np.linalg.norm(profile, axis=0)
        length_ratios = shape.get_arclength(t) / shape.get_perimeter()
        profile *= r * np.cos(length_ratios * N * 2 * math.pi)

        return primitive + profile
