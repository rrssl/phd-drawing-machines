# -*- coding: utf-8 -*-
"""
Tools for the benchmarking of curve matching techniques.

@author: Robin Roussel
"""

import random
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

import curvegen as cg
import curveplotlib as cplt
import curveproc as cpr


class DistanceProperties:
    """Class testing the mathematical properties of a dissimilarity measure."""

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
        

class Robustness:
    """Evaluator of the robustness of a curve recognition technique."""

    def __init__(self, curve_matcher, distorsion=0.02, nb_tests=30, 
                 epsilon=1.0):
        self.matcher = curve_matcher
        self.distorsion = distorsion
        self.nb_tests = nb_tests
        self.epsilon = epsilon
        
        
    def get_success_rate(self):
        """Compute the success rate."""
        # Compute the pool of candidate curves.
        combi = cg.get_param_combinations()
        size = combi.shape[0]
        # Run the tests.
        nb_successes = 0
        target_ids = np.random.choice(np.arange(size), self.nb_tests, 
                                      replace=False)
        for tid in target_ids:
            # Compute and distort the target curve.
            target = cg.get_curve(combi[tid])
            distorted = cpr.get_hand_drawn(target, amplitude=self.distorsion)
            # Run the curve matching algorithm on the target.
            matched = self.matcher(distorted, combi)[0]
            # Compute difference of errors in the parameter space and append it.
#            print(combi[tid], matched)
#            self.show_distorsion(target)
#            plt.show()
            success = la.norm(combi[tid] - matched) < self.epsilon
            nb_successes += success
        return nb_successes / self.nb_tests
        
    def show_distorsion(self, curve):
        """Temporary function to show the effect of the distorsion."""
        plt.figure()
        cplt.plot(curve, ':')
        hand_curve = cpr.get_hand_drawn(curve, amplitude=self.distorsion)
        cplt.plot(hand_curve, 'r-', linewidth=2)        
        plt.gca().set_aspect('equal')        
        plt.margins(0.1)


class Complexity:
    """Evaluator of the complexity of a matching technique."""

    def __init__(self, distance, display=False, epsilon=1e-3):
        self.distance = distance
        self.epsilon = epsilon
        self.display = display


class RelativeParametricError:
    """Evaluator of the relative parametric error of a matching technique."""

    def __init__(self, curve_matcher, nb_tests=20):
        self.matcher = curve_matcher
        self.nb_tests = nb_tests
    
    @staticmethod
    def get_min_dist(target_point, cand_points):
        """Return the minimal distance to the target in parameter space."""
        d = ((cand_points - target_point) ** 2).sum(axis=1)
        return np.sqrt(d.min())
#        e_norm_min = np.inf
#        for cand_pt in cand_points:
#            e_norm = la.norm(target_point - cand_pt)
#            if e_norm < e_norm_min:
#                e_norm_min = e_norm
#        return e_norm_min
        
    def get_error(self):
        """Compute the relative parametric error."""
        # Compute the pool of candidate curves.
        combi = cg.get_param_combinations()
        size = combi.shape[0]
        # Run the tests.
        rel_errors = []
        for i in range(self.nb_tests):
            # Extract a random target curve from the candidate pool.
            target_id = random.randrange(0, size)
            target = combi[target_id]
            # Find the closest candidate curve in the parameter space, compute
            # its error, append it to the list
            candidates = np.delete(combi, target_id, axis=0)
            min_param_error = self.get_min_dist(target, candidates)
            # Run the curve matching algorithm on the target.
            matched = self.matcher(cg.get_curve(target), candidates)
            # Compute difference of errors in the parameter space and append it.
            param_error = la.norm(target - matched)
            rel_errors.append(1 - min_param_error / param_error)

        rel_errors = np.array(rel_errors)
    #    print(e_diffs)
        return rel_errors.mean(), np.median(rel_errors), rel_errors.std()

#def get_precision_recall(curve_matcher):
#    nb_tests = 20
#    points_per_turn = 50
#    # Curves from the same class as the target have their parameters in a
#    # sphere of this radius around the target parameters.
#    target_class_param_radius = 0.6
#    # Curves are classified in the same class as the target if their
#    # dissimilarity measure to the target is below the one between the
#    # closest curve and the target times the following factor.
#    classifier_threshold_factor = 1.1
#
#    # Compute the pool of candidate curves.
#    combi = Hypotrochoid.get_param_combinations()
#    size = combi.shape[0]
#    # Run the tests.
#    precisions = []
#    recalls = []
#    for i in range(nb_tests):
#        # Extract a target curve from the candidate pool.
#        target_id = random.randrange(0, size)
#        target = combi[target_id]
#        target_curve = get_curve(target, target[1], points_per_turn)
#        # Classify each candidate wrt the target.
#        candidates = np.delete(combi, target_id, axis=0)
##        target_ = np.array([target[1] / target[0], target[2] / target[1]])
#        relevant = np.zeros(candidates.shape[0], dtype=bool)
#        e_norm_min = np.inf
#        argmin = []
#        for j, cand in enumerate(candidates):
##            cand_ = np.array([cand[1] / cand[0], cand[2] / cand[1]])
#            e_norm = la.norm(target - cand)
#            relevant[j] = e_norm < target_class_param_radius
#            if e_norm < e_norm_min:
#                e_norm_min = e_norm
#                argmin = cand
#        # Run the classification algorithm on the target.
#        closest_curve = get_curve(argmin, argmin[1], points_per_turn)
#        classifier_threshold = (classifier_threshold_factor *
#                                curve_matcher(closest_curve, target_curve))
#        retrieved = classify_curve(target_curve, candidates, curve_matcher,
#                                   classifier_threshold)
#        intersection = retrieved * relevant
##        print(classifier_threshold)
##        print(retrieved.sum())
##        print(relevant.sum())
#        # Compute the precision and recall and append them.
#        prec = intersection.sum() / retrieved.sum()
#        rec = intersection.sum() / relevant.sum()
#        precisions.append(prec)
#        recalls.append(rec)
#
#    precisions = np.array(precisions)
#    recalls = np.array(recalls)
#    return precisions.mean(), recalls.mean()
