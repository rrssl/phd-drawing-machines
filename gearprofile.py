# -*- coding: utf-8 -*-
"""
Module for creating involute spur gears.

@author: Robin Roussel
"""
import math
import numpy as np
import scipy.optimize as opt

import curves as cu


class GearProfile:
    points_per_tooth = 2**4

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
        # Note: the actual formula for the profile is
        # r*cos(2*pi*N*arclength(t)/perimeter) = r*cos(2*pi*N*r*t/(2*pi*r))
        #                                      = r*cos(N*t)
        return (R + r * np.cos(N * t)) * np.vstack([np.cos(t), np.sin(t)])


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
        th = np.linspace(period_angle / 2, period_angle,
                         GearProfile.points_per_tooth)
        # Compute the tooth.
        tooth = np.hstack([epi.get_point(te)[:, :-1], hypo.get_point(th)])
        # Rotate it to be coherent with the other gear profiles.
        cos = math.cos(- period_angle / 4)
        sin = math.sin(- period_angle / 4)
        rot = np.array([[cos, -sin],
                        [sin, cos]])

        return rot.dot(tooth)


class InvoluteElliptical(GearProfile):
    """Elliptical involute spur gear profile."""

    def __init__(self, pitch_req, pitch_e2, nb_teeth, pressure_angle=20,
                 internal=False):
        self.req = pitch_req
        self.e2 = pitch_e2
        self.nb_teeth = nb_teeth
        self.phi_n = pressure_angle * math.pi / 180
        self.internal = internal

        self.pitch = cu.Ellipse2(self.req, self.e2)
        m = 2 * self.req / nb_teeth # module
        self.A0 = m * (1, 2)[self.internal] # Addendum
        self.B0 = m * math.pi / 4
        # Radius of the shaper cutter
        # (Arbitrarily chosen, but seems to work for number of teeth >= 20.)
        self.rs = m * 12

        self.cos_n = math.cos(self.phi_n)
        self.sin_n = math.sin(self.phi_n)
        self.tan_n = self.sin_n / self.cos_n

        self.rho_f = m / 2  # radius of curvature of the fillet
        self.theta_f = ((math.pi/2) - self.phi_n) # angular spread of the filet
        self.cos_f = math.cos(self.theta_f)
        self.sin_f = math.sin(self.theta_f)

    def get_corner_ctr(self, phi_c, sgn):
        """Get the center of the rounded corner of the shaper cutter at current
        position.
        """
        x = sgn*(self.B0 + self.A0*self.tan_n + self.rho_f*self.sin_f) + self.rs*phi_c
        y = -self.A0 + self.rho_f*self.cos_f + self.rs
        return x, y

    def get_lp(self, phi_c, sgn):
        """Get lp, an intermediary parameter."""
        return -(sgn*self.rs*phi_c + self.B0)*self.sin_n - self.A0/self.cos_n

    def get_work_pt(self, phi_c, sgn):
        """Get the coordinates of the working point in the local ref. of the
        cutting rack.
        """
        lp = self.get_lp(phi_c, sgn)
        x = sgn * (self.B0 + self.A0*self.tan_n - lp*self.sin_n) + self.rs*phi_c
        y = lp*self.cos_n - self.A0 + self.rs
        return x, y

    def get_meshing_eq(self, phi_c, phi_s, sgn):
        """Get the value of the equation of meshing between the cutter and the
        gear.
        A value of 0 means that the two elements are meshing correctly.
        """
        A1, B1 = self.get_work_pt(phi_c, sgn)
        left = (sgn*self.rs) * np.cos(self.phi_n - sgn * (phi_s - phi_c))
        right = (sgn*B1*self.cos_n - A1*self.sin_n)
        return left - right

#def get_phi_s_bounds(sgn):
#    phi_c_bnds = -sgn * (B0 + np.array([1, 3])*A0 / (cos_n*sin_n)) / rs
#    return np.array([
#        opt.fsolve(lambda x: get_meshing_eq(val, x, sgn), 0.)[0]
#        for val in phi_c_bnds]) * rs
##    t_bnds = ellipse.get_arclength_inv(s_bnds)
##    phi_bnds = np.arctan2(np.sin(t_bnds) * math.sqrt(1 - e**2), np.cos(t_bnds))
##    print(phi_bnds)

    def get_profile_side(self, sgn):
        """Returns working regions from one side of each tooth, and fillets
        from the opposite side.
         -- sgn > 0: left-side working regions, right-side fillets,
         -- sgn < 0: right-side working regions, left-side fillets.
        """
        sgn = int(sgn)
        working = []
        fillet = []

        for tid in range(self.nb_teeth):
            # phi_s
            s = np.linspace(
                2*math.pi*self.req*tid/self.nb_teeth,
                2*math.pi*self.req*(tid+1)/self.nb_teeth,
                GearProfile.points_per_tooth)
            phi_s = (s - s[(-1, 0)[sgn > 0]]) / self.rs
            # phi
            t = self.pitch.get_arclength_inv(s)
            phi = np.arctan2(np.sin(t) * math.sqrt(1 - self.e2), np.cos(t))
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            # gamma
            den = np.sqrt(1 + cos_phi**2 * self.e2 * (self.e2 - 2))
            cos_gamma = sin_phi / den
            sin_gamma = cos_phi*(1 - self.e2) / den
            gamma = np.arctan2(sin_gamma, cos_gamma)
            # phi_c
            phi_c = opt.fsolve(
                lambda x: self.get_meshing_eq(x, phi_s, sgn), phi_s)

            r = np.linalg.norm(self.pitch.get_point(t), axis=0)
            cos_ = np.cos(phi_s - gamma - phi_c)
            sin_ = np.sin(phi_s - gamma - phi_c)

            # Working point in rack ref
            xy = self.get_work_pt(phi_c, sgn)
            # Working point in fixed ref
            xy = np.vstack(
                [ xy[0]*cos_ + xy[1]*sin_ + r*cos_phi + self.rs*sin_gamma,
                 -xy[0]*sin_ + xy[1]*cos_ - r*sin_phi - self.rs*cos_gamma]
                )

    #        lp = self.get_lp(phi_c, sgn)
    #        valid = np.logical_and(lp >= 0., lp <= 2*self.A0/self.cos_n)
    #        working.append(xy[:, valid])
            working.append(xy)

            if not self.internal:
                # Corner center in rack ref
                xy = self.get_corner_ctr(phi_c, -sgn)
                # Corner center in fixed ref = Primary trochoid
                # (We reverse it because if we go up the tooth on one side,
                # we will go down the fillet on the other, and vice-versa.)
                xy = np.vstack(
                    [ xy[0]*cos_ + xy[1]*sin_ + r*cos_phi + self.rs*sin_gamma,
                     -xy[0]*sin_ + xy[1]*cos_ - r*sin_phi - self.rs*cos_gamma]
                    )[:, ::-1]
                # Secondary trochoid
                xy_prime = np.vstack([np.gradient(xy[0]), np.gradient(xy[1])])
                norm = np.sqrt(xy_prime[0]**2 + xy_prime[1]**2)
                xy_prime /= norm
                xy[0] += self.rho_f * xy_prime[1]
                xy[1] -= self.rho_f * xy_prime[0]
                # Filter values.
                ref = xy_prime[:, (-1, 0)[sgn > 0]]
                cosines = ref[0]*xy_prime[0, :] + ref[1]*xy_prime[1, :]
                xy = xy[:, cosines > 0]

                fillet.append(xy)

        if sgn > 0:
            working = working[1:] + working[:1]

        return working, fillet

    def get_profile(self):
        """Create a circular gear profile."""
        working_l, fillet_r = self.get_profile_side(1)
        working_r, fillet_l = self.get_profile_side(-1)
        working = [np.hstack([l, r]) for l, r in zip(working_l, working_r)]
        if not self.internal:
            fillets = [np.hstack([r, l])
                       for r, l in zip(fillet_r, fillet_l)]
            profile = [np.hstack([f, w]) for f, w in zip(fillets, working)]
        else:
            profile = working
        profile = np.hstack(profile)
        profile = np.hstack([profile, profile[:, [0]]])

        return profile


class SinusoidalElliptical(GearProfile):
    """Sinusoidal elliptical gear profile."""

    def __init__(self, pitch_semimajor, pitch_semiminor, nb_teeth, tooth_radius):
        self.pitch_semimajor = pitch_semimajor
        self.pitch_semiminor = pitch_semiminor
        self.nb_teeth = nb_teeth
        self.tooth_radius = tooth_radius

    def get_profile(self):
        """Create an elliptical gear profile."""
        shape = cu.Ellipse(self.pitch_semimajor, self.pitch_semiminor)
        N = self.nb_teeth
        r = self.tooth_radius
        t = np.linspace(0, 2 * math.pi, GearProfile.points_per_tooth * N)

        primitive = shape.get_point(t)
        profile = shape.get_normal(t)
        profile /= np.linalg.norm(profile, axis=0)
        length_ratios = shape.get_arclength(t) / shape.get_perimeter()
        profile *= r * np.cos(2 * math.pi * N * length_ratios)

        return primitive + profile
