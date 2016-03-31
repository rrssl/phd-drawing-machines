# -*- coding: utf-8 -*-
"""
Library of functions to evaluate different curve matching techniques.

@author: Robin Roussel
"""

import random
import numpy as np
import scipy.linalg as la
from curves import Hypotrochoid

def get_curve(params, nb_turns, nb_samples_per_turn):
    """Generate the curve (hypotrochoid) of given parameters."""
    nb_samples = nb_turns * nb_samples_per_turn
    interval_length = nb_turns * 2 * np.pi
    theta_range = np.linspace(0., interval_length, nb_samples)
    hypo = Hypotrochoid(theta_range, *params)
    return np.vstack([hypo.getX(), hypo.getY()])

def match_curve(target_curve, cand_params, curve_matcher):
    """Find the candidate best matching the input curve."""
    # Compare each candidate curve.
    min_dist = np.inf
    argmin = np.empty(3)
    points_per_turn = 50
    for cand in cand_params:
        # Generate the curve.
        cand_curve = get_curve(cand, cand[1], points_per_turn)
        # Compute the distance.
        dist = curve_matcher(cand_curve, target_curve)
#        print(dist, c)
        if dist == 0.:
            argmin = cand
            break
        elif dist < min_dist:
#            print("******************************************")
            min_dist = dist
            argmin = cand

    return argmin
    
def classify_curve(target_curve, cand_params, curve_matcher, threshold):
    """Find the candidates in the same class as the input curve."""
    # Compare each candidate curve.
    belongs = np.zeros(cand_params.shape[0], dtype=bool)
    points_per_turn = 50
    for i, cand in enumerate(cand_params):
        # Generate the curve.
        cand_curve = get_curve(cand, cand[1], points_per_turn)
        # Compute the distance.
        dist = curve_matcher(cand_curve, target_curve)
#        print(dist, c)
        belongs[i] = dist <= threshold

    return belongs
    

def get_relative_error(curve_matcher):
    nb_tests = 20
    num_R_vals = 10
    num_d_vals = 20
    points_per_turn = 50
    
    # Compute the pool of candidate curves.
    combi = Hypotrochoid.get_param_combinations(num_R_vals, num_d_vals)
    size = combi.shape[0]
    # Run the tests.
    rel_errors = []
    for i in range(nb_tests):        
        # Extract a target curve from the candidate pool.
        target_id = random.randrange(0, size)
        target = combi[target_id]
        target_curve = get_curve(target, target[1], points_per_turn)     
        # Find the closest candidate curve in the parameter space, compute its
        # error, append it to the list
        candidates = np.delete(combi, target_id, axis=0)
#        target_ = np.array([target[1] / target[0], target[2] / target[1]])
        e_norm_min = np.inf
        for cand in candidates:
#            cand_ = np.array([cand[1] / cand[0], cand[2] / cand[1]])
            e_norm = la.norm(target - cand)
            if e_norm < e_norm_min:
                e_norm_min = e_norm
#                cand_temp = cand
#        print(target, cand_temp, e_norm_min)
        # Run the curve matching algorithm on the target.
        matched = match_curve(target_curve, candidates, curve_matcher)
        # Compute difference of errors in the parameter space and append it.
#        matched_ = np.array([matched[1] / matched[0], matched[2] / matched[1]])
        e_match_norm = la.norm(target - matched)
        rel_errors.append(1 - e_norm_min / e_match_norm)
        
    rel_errors = np.array(rel_errors)
#    print(e_diffs)
    return rel_errors.mean(), np.median(rel_errors), rel_errors.std()

def get_precision_recall(curve_matcher):
    nb_tests = 20
    num_R_vals = 10
    num_d_vals = 15
    points_per_turn = 50
    # Curves from the same class as the target have their parameters in a 
    # sphere of this radius around the target parameters.
    target_class_param_radius = 0.6
    # Curves are classified in the same class as the target if their 
    # dissimilarity measure to the target is below the one between the 
    # closest curve and the target times the following factor.
    classifier_threshold_factor = 1.1
    
    # Compute the pool of candidate curves.
    combi = Hypotrochoid.get_param_combinations(num_R_vals, num_d_vals)
    size = combi.shape[0]
    # Run the tests.
    precisions = []
    recalls = []
    for i in range(nb_tests):
        # Extract a target curve from the candidate pool.
        target_id = random.randrange(0, size)
        target = combi[target_id]
        target_curve = get_curve(target, target[1], points_per_turn)     
        # Classify each candidate wrt the target.
        candidates = np.delete(combi, target_id, axis=0)
#        target_ = np.array([target[1] / target[0], target[2] / target[1]])
        relevant = np.zeros(candidates.shape[0], dtype=bool)
        e_norm_min = np.inf
        argmin = []
        for j, cand in enumerate(candidates):
#            cand_ = np.array([cand[1] / cand[0], cand[2] / cand[1]])
            e_norm = la.norm(target - cand)
            relevant[j] = e_norm < target_class_param_radius
            if e_norm < e_norm_min:
                e_norm_min = e_norm
                argmin = cand
        # Run the classification algorithm on the target.
        closest_curve = get_curve(argmin, argmin[1], points_per_turn)
        classifier_threshold = (classifier_threshold_factor * 
                                curve_matcher(closest_curve, target_curve))
        retrieved = classify_curve(target_curve, candidates, curve_matcher,
                                   classifier_threshold)
        intersection = retrieved * relevant
#        print(classifier_threshold)
#        print(retrieved.sum())
#        print(relevant.sum())
        # Compute the precision and recall and append them.
        prec = intersection.sum() / retrieved.sum()        
        rec = intersection.sum() / relevant.sum()    
        precisions.append(prec)
        recalls.append(rec)
        
    precisions = np.array(precisions)    
    recalls = np.array(recalls)    
    return precisions.mean(), recalls.mean()     
    