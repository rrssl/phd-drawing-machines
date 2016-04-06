#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# To avoid numpy errors with PyLint: --extension-pkg-whitelist=numpy

"""
Entry file.

@author: Robin Roussel
"""
import numpy as np
import scipy.interpolate as interp
#import numpy.linalg as la
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

try:
    import cv2
except ImportError:
    CV2_IMPORTED = False
else:
    CV2_IMPORTED = True

import curvedistances as cdist
import curvegen as cg
if CV2_IMPORTED:
    import curveimproc as cimp
import curvematching as cmat
import curveplotlib as cplt
import matcheval as mev


TEST_IMG_INPUT = False


def test_curve_retrieval(target_curve, distance):
    """Test the retrieving capacities of a distance."""
    num_retrieved = 5
    # Compute combinations.
    cand_params = cg.get_param_combinations()
    # Compare each candidate curve.
    distances = np.array([distance(cg.get_curve(cand), target_curve)
                          for cand in cand_params])
    # Collect the closest curves.
    sortid = np.argsort(distances)
    mindist = distances[sortid][:num_retrieved]
    argmin = cand_params[sortid, :][:num_retrieved, :]

    return argmin, mindist


def show_curve_retrieval(target_curve, optimized_curve, retrieved_curves, 
                         distances, title):
    """Show the curve retrieval results for a given distance."""
    nb_retrieved = retrieved_curves.shape[0]    
    plt.figure(figsize=(16, 9))
    plt.suptitle(title)
    plot_grid_size = (3, nb_retrieved)
    
    # Show the target.
    frame = plt.subplot2grid(
        plot_grid_size, (0, 0), rowspan=2, colspan=nb_retrieved,
        title="Target curve. {}".format(np.array([5, 3, 1.5])))
    cplt.plot(target_curve, 'r-', label="Target")
    cplt.plot(optimized_curve, 'b--', label="Optimized")
    plt.legend(loc='best')
    frame.set_aspect('equal')
    
    # Show the retrieved curves.
    for i, c in enumerate(retrieved_curves):
        frame = plt.subplot2grid(
            plot_grid_size, (2, i), sharex=frame, sharey=frame,
            title="d = {:.3f}\n{}".format(distances[i], c))
        curve_pts = cg.get_curve(c)
        curve_pts = curve_pts * abs(target_curve).max() / abs(curve_pts).max()
        cplt.plot(curve_pts)
        frame.set_aspect('equal')
        
    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, hspace=0.5)
    plt.autoscale()
    plt.margins(0.1)


def plot_distance(target_curve, distance, name):
    """Plot a given distance to a given target curve."""
    # Compute combinations.
    combi = cg.get_param_combinations()
    D = np.empty(combi.shape[0])
    # Compare each candidate curve.
    for i, c in enumerate(combi):
        # Generate the curve.
        cand_curve = cg.get_curve(c)
        # Compute the distance.
        D[i] = distance(cand_curve, target_curve)
#        print(dist, c)

    X = combi[:, 2] / combi[:, 1]
    Y = combi[:, 1] / combi[:, 0]
    # Define grid.
    xi = np.linspace(X.min(), X.max(), 50)
    yi = np.linspace(Y.min(), Y.max(), 50)
    # Grid the data.
    xi, yi = np.meshgrid(xi, yi)
    zi = interp.griddata((X, Y), D, (xi, yi), method='linear')
    if zi.min() < 0:
        zi = zi - 2 * zi.min()
    # Plot the data.
    ax = plt.figure().gca(projection='3d')
    ax.plot_surface(xi, yi, zi, rstride=1, cstride=1, cmap=plt.cm.winter,
                    norm=colors.LogNorm(vmin=zi.min(), vmax=zi.max()),
                    linewidth=0.1, antialiased=True)
    ax.scatter(1.5/3, 3/5)
    ax.set_xlim(0., 1.)
    ax.set_ylim(0., 1.)
    ax.set_zlim(0., zi.max())
    ax.set_xlabel('d / r')
    ax.set_ylabel('r / R')
    ax.set_zlabel('D')
    plt.title("Plot of the " + name + " dissimilarity measure\n"
              "as a function of the candidate hypotrochoid's parameters.")


def main():
    """Entry point."""
    np.set_printoptions(precision=5, suppress=True)
    plt.ioff()

    # Get the reference curve image.
    if TEST_IMG_INPUT:
        if CV2_IMPORTED:
            ref_img = cv2.imread('userinput_5_3_1.5.png', 0)
        else:
            print("Error loading the image: OpenCV module not imported.")
    else:
        ref_params = (5., 3., 1.502)
        ref_curve = cg.get_curve(ref_params)

#        # Generate the image.
#        if CV2_IMPORTED:
#            shp = (512, 512)
#            ref_img = cimp.getim(ref_curve, shp)

    if TEST_IMG_INPUT:
        if CV2_IMPORTED:
            cplt.imshow(ref_img)

    # Get a candidate curve.
    cand_params = (5., 3., 2.5)
#    params = (2., 1., 0.5) # Ellipse
    samples_per_turn = 50
    sampling_rate = samples_per_turn / (2 * np.pi)
    cand_curve = cg.get_curve(cand_params, nb_samples_per_turn=samples_per_turn)

#    if CV2_IMPORTED:
#        shp = ref_img.shape
#        cand_img = cimp.getim(cand_curve, shp)

    combi = cg.get_param_combinations()

    distances = {
#        "Distance field.": cdist.DistanceField(),
#        "Curvature's features.": cdist.CurvatureFeatures(sampling_rate),
#        "Hu moments.": cdist.HuMoments(contour_method=None),
#        "Zernike moments.": cdist.ZernikeMoments(radius=128, degree=6,
#                                                 contour_method=None,
#                                                 filled_contour=False),
        "Perceptual features.": cdist.PerceptualFeatures()
        }

    for name, dist in distances.items():
        print(name)

        # Test distance properties.
#        df_props = mev.DistanceProperties(dist.get_dist, display=True)
#        df_props.compute_dist_properties(cand_curve, ref_curve)

        # Test curve matching.
        dist_func = dist.get_dist
        print("Target arguments: {}".format(ref_params))
        matcher = cmat.CurveMatcher(dist_func)
#        retrieved_args = []
        retrieved_args = matcher(ref_curve, combi)
        print("Retrieved arguments: {}".format(retrieved_args))
        
        # Test curve optimization.
        optimizer = cmat.CurveOptimizer(dist_func, retrieved_args, ref_curve)
        optimized_args = np.hstack([retrieved_args[:2], 
                                     optimizer.optimize(display=True).x])
        print("Optimized arguments: {}".format(optimized_args))

        # Plot the curve dissimilarity measure.
#        plot_distance(ref_curve, dist_func, name)

        # Get the relative parametric error.
#        error_eval = mev.RelativeParametricError(matcher)
#        error_dist = error_eval.get_error()
#        print("Relative parametric error"
#              " mean, median and std: {}".format(error_dist))

#        # Get the precision and recall.
#        prec_rec = mev.get_precision_recall(dist_func)
#        print("Precision and recall: {}".format(prec_rec))

        # Show curve retrieval.
        show_curve_retrieval(ref_curve, cg.get_curve(optimized_args),
                             *test_curve_retrieval(ref_curve, dist_func),
                             title=name)
        print('\n')
        
        plt.show()

if __name__ == "__main__":
    main()
