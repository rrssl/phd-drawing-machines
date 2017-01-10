# -*- coding: utf-8 -*-
"""
Ellipticalal involute gear profile.

In case of pointy teeth: reduce addendum or increase shaper radius.

@author: Robin Roussel
"""
import math
import matplotlib.pyplot as plt

import numpy as np
import scipy.optimize as opt

import context
from curves import Ellipse2

class InvoluteElliptical:
    """Elliptical involute spur gear profile."""

    def __init__(self, pitch_req, pitch_e2, nb_teeth, pressure_angle=20,
                 internal=False):
        self.req = pitch_req
        self.e2 = pitch_e2
        self.nb_teeth = nb_teeth
        self.phi_n = pressure_angle * math.pi / 180
        self.internal = internal

        self.pitch = Ellipse2(self.req, self.e2)
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
                2*math.pi*self.req*(tid+1)/self.nb_teeth, 2**4)
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


def turn_it_like_a_sock(curve, ellipse):
    def dist(t):
        return np.linalg.norm(curve - ellipse.get_point(t))
    # Find orthogonal projection on the ellipse.
    t_init = np.arctan2(curve[1], curve[0] * math.sqrt(1 - ellipse.e2))
    t = opt.minimize(dist, t_init).x
    proj_vect = curve - ellipse.get_point(t)
    # Apply symmetry
    new_curve = curve - 2*proj_vect

    return new_curve


if __name__ == "__main__":
    e = .6
    req = 10.
    nb_teeth = 42

    plt.figure(1)
    ax = plt.subplot(111)

    gearint = InvoluteElliptical(req, e**2, nb_teeth)
    profint = gearint.get_profile()

    pitchint = gearint.pitch.get_range(0, 2*math.pi, 2**7)


    gearext = InvoluteElliptical(1.5*req, 0.03, int(1.5*nb_teeth), internal=True)
    profext = gearext.get_profile()
#    profext[0] *= -1
#    profext = turn_it_like_a_sock(profext, gearext.pitch)

    pitchext = gearext.pitch.get_range(0, 2*math.pi, 2**7)

    shift = gearext.pitch.a - gearint.pitch.a
    profint[0] += shift
    pitchint[0] += shift

    ax.plot(*profext)
    ax.plot(*pitchext, linestyle='dashed')

    ax.plot(*profint, label="Profile")
    ax.plot(*pitchint, label="Pitch curve", linestyle='dashed')
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.ioff()
    plt.show()

