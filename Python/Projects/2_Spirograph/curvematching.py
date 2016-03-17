#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# To avoid numpy errors with PyLint: --extension-pkg-whitelist=numpy

"""
Curve matching algorithms

@author: Robin Roussel
"""
import numpy as np
import scipy.signal as sig
import scipy.interpolate as interp
#import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
try:
    import cv2
except ImportError:
    CV2_IMPORTED = False
else:
    CV2_IMPORTED = True

from curves import Hypotrochoid
import curveplotlib as cplt
import curvedistances as cdist
if CV2_IMPORTED: import curveimproc as cimp
    

TEST_IMG_INPUT = False


if CV2_IMPORTED:
    def test_distance_field(cand_curve, target_curve):
        """Test the distance field descriptor."""
        # Show the candidate curve embedded in the distance field of the target
        # curve.
        df = cdist.DistanceField()
        desc = df.get_desc(target_curve)
        adapted_cand_curve = cimp.fitInBox(cand_curve, desc.shape)
        cplt.imshow(desc, adapted_cand_curve)
        plt.title('Candidate curve in the distance field of the target curve.')
        
        # Compute the DF-distance.
        df_dist = df.get_dist(adapted_cand_curve, target_curve)
        print("DF-distance: {}".format(df_dist))


    def test_hu_moments(cand_curve, target_curve, cand_img, ref_img):
        """Test the Hu moments descriptor."""
        # Compute the Hu moments of the full curve.
        hm = cdist.HuMoments()
        mom = hm.get_desc(target_curve)
        print(mom)

        # Compute the distance.
        hu_dist = hm.get_dist(target_curve, cand_curve)
        print("Distance between Hu Moments: {}".format(hu_dist))

        # Display the moments.
        plt.title("Absolute value of the Hu moments of the target curve.")
        plt.bar(np.arange(len(mom)) + .6, abs(mom), log=True)

        # Compute the external contours.
        ref_ext_ctr = cimp.getExtContour(ref_img)
        cand_ext_ctr = cimp.getExtContour(cand_img)

        # Display the external contours.
        plt.gcf().add_subplot(121)
        cplt.imshow(ref_ext_ctr)
        plt.title("Target's external contour.")
        plt.gcf().add_subplot(122)
        cplt.imshow(cand_ext_ctr)
        plt.title("Candidate's external contour.")

        # Compute the Hu moments of the target external contour.
        mom = hm.get_desc(ref_ext_ctr)

        # Compute the distance.
        hu_dist = hm.get_dist(ref_ext_ctr, cand_ext_ctr)
        print("Distance between external Hu Moments: {}".format(hu_dist))

        plt.title("Absolute value of the Hu moments of the target curve.")
        plt.bar(np.arange(len(mom)) + .6, abs(mom), log=True)

        # Compute the internal contours.
        ref_int_ctr = cimp.getIntContour(ref_img)
        cand_int_ctr = cimp.getIntContour(cand_img)

        # Display the internal contours.
        plt.gcf().add_subplot(121)
        cplt.imshow(ref_int_ctr)
        plt.title("Target's internal contour.")
        plt.gcf().add_subplot(122)
        cplt.imshow(cand_int_ctr)
        plt.title("Candidate's internal contour.")

        # Compute the Hu moments of the target internal contour.
        mom = hm.get_desc(ref_int_ctr)

        # Compute the distance.
        hu_dist = hm.get_dist(ref_int_ctr, cand_int_ctr)
        print("Distance between internal Hu Moments: {}".format(hu_dist))

        # Display the moments.
        plt.title("Absolute value of the Hu moments of the target curve.")
        plt.bar(np.arange(len(mom)) + .6, abs(mom), log=True)


def test_curvature_features(curve, r, R, sampling_rate):
    """Test the curvature features descriptor."""
    # Compute Fourier transform.
    cf = cdist.CurvatureFeatures(sampling_rate)
    cvt = cf.compute_curvature(curve)
    fourier = np.fft.rfft(cvt)
    power = abs(fourier) * abs(fourier)
    freq = np.fft.rfftfreq(curve.shape[-1])
    # Find power peaks and keep the ones >= 1% of the main peak.
    argrelmax = sig.argrelmax(power)[0]
    argrelmax = argrelmax[power[argrelmax] >= 0.01 * power[0]]
    maxima = np.vstack([freq[argrelmax], power[argrelmax]])
    # Test the main frequency (the other are just harmonics).
    f1 = maxima[0, 0]
    print("Main frequence: {}".format(f1))
    theta_period = 1 / (f1  * sampling_rate)
    print("Corresponding main theta period: {}".format(theta_period))
    print("phi_(max, max) = 2pi * r / R = {}".format(2 * np.pi * r / R))

    # Show the curvature and Fourier peaks.
    plt.gcf().add_subplot(311)
    plt.title('Curvature plot.')
    plt.plot(cvt)
    plt.xlim(xmax=len(cvt)-1)
    plt.gcf().add_subplot(312)
    plt.title('Curvature on the trajectory.')
    cplt.cvtshow(curve, cvt)
    plt.gcf().add_subplot(313)
    plt.title('Fourier transform.')
    plt.plot(freq, power)
    plt.scatter(maxima[0], maxima[1], c='r')
    plt.xlim(freq[0], freq[-1])
    plt.ylim(ymin=0)

    base_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches((base_size[0], base_size[1] * 1.5),
                              forward=True)


class DistanceProperties:
    """Class testing the mathematical properties of a distance."""

    def __init__(self, distance, display=False, epsilon=1e-3):
        self.distance = distance
        self.epsilon = epsilon
        self.display = display

        self.current_plot_id = 230

    def test_identity(self, curve):
        """Test the identity of the member distance on a curve."""
        score = self.distance(curve, curve)
        return score < self.epsilon

    def test_positivity(self, curve, other_curve):
        """Test the positivity of the member distance for 2 given curves."""
        score = self.distance(curve, other_curve)

        if self.display:
            self.disp_binary_rel(score, 'Positivity', curve, other_curve)

        return score > 0

    def test_trans_invariance(self, curve):
        """Test the translation invariance of the member distance on a curve."""
        t = curve.max() / 2
        tr_curve = curve + t
        score = self.distance(curve, tr_curve)

        if self.display:
            self.disp_binary_rel(score, 'Trans. invariance', curve, tr_curve)

        return score < self.epsilon

    def test_rot_invariance(self, curve):
        """Test the rotation invariance of the member distance on a curve."""
        # Since the curve can be point-symmetric, we sample several angles.
        for i in range(1, 6):
            angle = np.pi / (2*i)
            R = np.array([[np.cos(angle), -np.sin(angle)],
                          [np.sin(angle), np.cos(angle)]])
            rot_curve = R.dot(curve)
            score = self.distance(curve, rot_curve)
            if score >= self.epsilon:
                res = False
                break
        else:
            res = True

        if self.display:
            self.disp_binary_rel(score, 'Rot. invariance', curve, rot_curve)

        return res

    def test_scale_invariance(self, curve):
        """Test the scale invariance of the member distance on a curve."""
        S = 0.5
        sca_curve = S * curve
        score = self.distance(curve, sca_curve)

        if self.display:
            self.disp_binary_rel(score, 'Scale invariance', curve, sca_curve)

        return score < self.epsilon

    def test_reflection_invariance(self, curve):
        """Test the reflection invariance of the member distance on a curve."""
        S = -1.0
        ref_curve = S * curve
        score = self.distance(curve, ref_curve)

        if self.display:
            self.disp_binary_rel(score, 'Ref. invariance', curve, ref_curve)

        return score < self.epsilon

    def test_triangle_ineq(self, curve):
        """Test the triangle inequality for the member distance on a curve."""
        half_len = curve.shape[1] / 2
        curve_part_1 = curve[:, :half_len]
        curve_part_2 = curve[:, half_len:]
        score = (self.distance(curve_part_1, curve_part_2) +
                 self.distance(curve_part_2, curve) -
                 self.distance(curve, curve_part_2))

        if self.display:
            self.current_plot_id += 1
            plt.gcf().add_subplot(self.current_plot_id)
            cplt.plot(curve)
            cplt.plot(curve_part_1, 'g+')
            cplt.plot(curve_part_2, 'r+')
            plt.gca().set_aspect('equal')
            plt.title("Triangle inequality\n"
                      "d12 + d23 - d13 = {:.2f}".format(score))

        return score >= 0

    def test_symmetry(self, curve, other_curve):
        """Test the symmetry of the member distance for 2 given curves."""
        score = abs(self.distance(curve, other_curve) -
                    self.distance(other_curve, curve))
        return score < self.epsilon

    def compute_dist_properties(self, curve, other_curve):
        """Compute the properties of the member distance for 2 given curves."""
        if np.array_equal(curve, other_curve):
            raise Exception("The input curves must be different.")
            
        if self.display:
            plt.figure()

        id_res = self.test_identity(curve)
        pos_res = self.test_positivity(curve, other_curve)
        trinv_res = self.test_trans_invariance(curve)
        rotinv_res = self.test_rot_invariance(curve)
        scinv_res = self.test_scale_invariance(curve)
        refinv_res = self.test_reflection_invariance(curve)
        tri_res = self.test_triangle_ineq(curve)
        sym_res = self.test_symmetry(curve, other_curve)

        # Output results.
        print("Distance properties:")
        print(" - Identity: {}".format(id_res))
        print(" - Positivity: {}".format(pos_res))
        print(" - Translation invariance: {}".format(trinv_res))
        print(" - Rotation invariance: {}".format(rotinv_res))
        print(" - Scale invariance: {}".format(scinv_res))
        print(" - Reflection invariance: {}".format(refinv_res))
        print(" - Triangle inequality: {}".format(tri_res))
        print(" - Symmetry: {}".format(sym_res))

    def disp_binary_rel(self, score, name, curve1, curve2):
        """Plot the transformation associated to a binary relation test."""
        self.current_plot_id += 1
        plt.gcf().add_subplot(self.current_plot_id)
        cplt.plot(curve1)
        cplt.plot(curve2)
        plt.gca().set_aspect('equal')
        plt.title("{}\nd = {:.2f}".format(name, score))


def test_curve_retrieval(target_curve, distance):
    """Test the retrieving capacities of a distance."""
    # Compute combinations.
    num_R_vals = 8
    num_d_vals = 14
    combi = Hypotrochoid.get_param_combinations(num_R_vals, num_d_vals)
    # Compare each candidate curve.
    min_dist = np.inf
    argmin = (-1, -1, -1)
    points_per_turn = 50    
    for c in combi:
        # Generate the curve.
        cand_curve = get_curve(c, c[1], points_per_turn)
        # Compute the distance.
        dist = distance(cand_curve, target_curve)
#        print(dist, c)
        if dist == 0.:            
            argmin = c
            break
        elif dist < min_dist:
#            print("******************************************")
            min_dist = dist
            argmin = c
    
    return argmin

def plot_distance(target_curve, distance, ):
    dist_list = []
    d_r_ratio_list = []
    r_R_ratio_list = []
    # Compute combinations.
    num_R_vals = 8
    num_d_vals = 14
    combi = Hypotrochoid.get_param_combinations(num_R_vals, num_d_vals)
    # Compare each candidate curve.
    points_per_turn = 50  
    for c in combi:
        # Generate the curve.
        cand_curve = get_curve(c, c[1], points_per_turn)
        # Compute the distance.
        dist = distance(cand_curve, target_curve)

        dist_list.append(dist)
        d_r_ratio_list.append(c[2] / c[1])
        r_R_ratio_list.append(c[1] / c[0])     
    
    X = np.array(d_r_ratio_list)
    Y = np.array(r_R_ratio_list)
    Z = np.array(dist_list)
    # Define grid.
    xi = np.linspace(X.min(), X.max(), 100)
    yi = np.linspace(Y.min(), Y.max(), 100)
    # Grid the data.        
    xi, yi = np.meshgrid(xi, yi)
    zi = interp.griddata((X, Y), Z, (xi, yi), method='cubic')
    # Plot the data.
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xi, yi, zi, rstride=3, cstride=3, cmap=plt.cm.winter,
                    norm=colors.LogNorm(vmin=zi.min(), vmax=zi.max()),
                    linewidth=0, antialiased=True)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_zlim(0., zi.max())
    ax.set_xlabel('d / r')
    ax.set_ylabel('r / R')
    ax.set_zlabel('Distance')
    plt.title("Plot of the distance as a function\n"
              "of the candidate hypotrochoid's parameters.")    

def get_curve(params, nb_turns, nb_samples_per_turn):
    """Generate the curve (hypotrochoid) of given parameters."""
    nb_samples = nb_turns * nb_samples_per_turn
    interval_length = nb_turns * 2 * np.pi
    theta_range = np.linspace(0., interval_length, nb_samples)
    hypo = Hypotrochoid(theta_range, *params)
    return np.vstack([hypo.getX(), hypo.getY()])

def main():
    plt.ioff()

    # Get the reference curve image.
    if TEST_IMG_INPUT:
        if CV2_IMPORTED:
            ref_img = cv2.imread('userinput_5_3_1.5.png', 0)
        else:
            print("Error loading the image: OpenCV module not imported.")
    else:
        params = (5., 3., 1.5)
        nb_turns = 3
        samples_per_turn = 50
        ref_curve = get_curve(params, nb_turns, samples_per_turn)

        # Generate the image.
        if CV2_IMPORTED:
            shp = (512, 512)
            ref_img = cimp.getim(ref_curve, shp)

#    if CV2_IMPORTED:
#        imshow(ref_img)

    # Get the candidate curve.
    params = (5., 3., 2.)
    nb_turns = 3
    samples_per_turn = 50
    cand_curve = get_curve(params, nb_turns, samples_per_turn)

    if CV2_IMPORTED:
        shp = ref_img.shape
        cand_img = cimp.getim(cand_curve, shp)

    # Not used here but could be useful.
#    points = cand_curve.T
#    cand_curve_length = sum(la.norm(points[:-1] - points[1:], axis=1))


    ### 1. DISTANCE FIELD ###

    if CV2_IMPORTED:
        print("Distance field.")
        
        # Test the code.
#        test_distance_field(cand_curve, ref_curve)

        # Test the distance properties.
        df = cdist.DistanceField()
        df_props = DistanceProperties(df.get_dist, display=False)
        df_props.compute_dist_properties(cand_curve, ref_curve)
        
        # Test curve retrieval.
        print("Target arguments: 5, 3, 1.5")
        retrieved_args = test_curve_retrieval(ref_curve, df.get_dist)
        print("Retrieved arguments: {}".format(retrieved_args))
#        plot_distance(ref_curve, df.get_dist)
        
        print('\n')
        
    ### 2. CURVATURE'S FEATURES ###
    
    print("Curvature's features.")
    
    sampling_rate = samples_per_turn / (2 * np.pi)
    # Test the code.
    test_curvature_features(cand_curve, params[1], params[0], sampling_rate)
    
    # Test the distance properties.
    cf = cdist.CurvatureFeatures(sampling_rate)
    cf_props = DistanceProperties(cf.get_dist, display=True)
    cf_props.compute_dist_properties(cand_curve, ref_curve)
    
    # Test curve retrieval.
    print("Target arguments: 5, 3, 1.5")
#    descriptor = cf.get_desc(ref_curve)
#    print("Descriptor: {}".format(descriptor))
    retrieved_args = test_curve_retrieval(ref_curve, cf.get_dist)
    print("Retrieved arguments: {}".format(retrieved_args))
    plot_distance(ref_curve, cf.get_dist)
    
    print('\n')

    ### 3. HU MOMENTS ###

#    if CV2_IMPORTED:
#        # Test the code.
#        test_hu_moments(cand_curve, ref_curve, cand_img, ref_img)
#        # Test the distance properties.
#        hm = cdist.HuMoments()
#        hm_props = DistanceProperties(hm.get_dist, display=True)
#        hm_props.compute_dist_properties(cand_curve, ref_curve)

    ### 4. ZERNIKE MOMENTS ###

    plt.show()


if __name__ == "__main__":
    main()
