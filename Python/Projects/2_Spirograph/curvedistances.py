# -*- coding: utf-8 -*-
"""
Library of curve distances.

@author: Robin Roussel
"""

import numpy as np
import numpy.linalg as la
import scipy.signal as sig
from matplotlib.mlab import PCA

try:
    import cv2
except ImportError:
    CV2_IMPORTED = False
else:
    CV2_IMPORTED = True

if CV2_IMPORTED: import curveimproc as cimp

class CurveDistance:
    """Base class for shape distances."""

    def __init__(self, use_cache=True):
        # If true, the target descriptor will be saved and re-used if the
        # target is the same. Useful for curve retrieval.
        self.use_cache = use_cache

    def get_target_desc(self, target_curve):
        """Get the target descriptor (possibly cached to save computations)."""
        if self.use_cache:
            try:
                if np.array_equal(self.cached_target, target_curve):
                    target_desc = self.cached_target_desc
                else:
                    self.cached_target = target_curve.copy()
                    target_desc = self.get_desc(target_curve)
                    self.cached_target_desc = target_desc
            except AttributeError:
                self.cached_target = target_curve.copy()
                target_desc = self.get_desc(target_curve)
                self.cached_target_desc = target_desc
        else:
            target_desc = self.get_desc(target_curve)

        return target_desc

    @staticmethod
    def normalize_pose(curve):
        # Center the curve.
        curve_average = curve.mean(axis=1)
        centered_curve = curve - curve_average.reshape(2, 1)
        # Compute PCA.
        results = PCA(centered_curve)
        # TODO compare with other methods (sklearn, linalg.eig(cov()), OpenCV, etc.)
        # TODO check "using eigenvalues is better for high-dimensional data
        # and fewer samples, whereas using Singular value decomposition is
        # better if you have more samples than dimensions."
        # NOTE avoid using external libraries when they're not needed!

    def get_desc(self, curve):
        pass

    def get_dist(self, cand_curve, target_curve):
        pass


if CV2_IMPORTED:
    class DistanceField(CurveDistance):
        """Shape distance based on the distance transform of the target."""

        def __init__(self, mask_size=5, resol=(512, 512), use_cache=True):
            super().__init__(use_cache)
            self.mask_size = mask_size
            self.resol = resol
            self.diag_ratio_dist_normalization = 0.05

        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if curve.shape[0] == 2:
                img = cimp.getim(curve, self.resol)
            else:
                img = curve
            return cv2.distanceTransform(~img, cv2.DIST_L2, self.mask_size)

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            df = self.get_target_desc(target_curve)
            # Adapt the candidate curve to the distance field.
            adapted_cand_curve = cimp.fit_in_box(cand_curve, df.shape)
            # Compute distance normalization factor.
            diag = (df.shape[0] * df.shape[0] + df.shape[1] * df.shape[1])**0.5
            normalization_factor = self.diag_ratio_dist_normalization * diag
            # Compute the average normalized distance.
            nb_samples = adapted_cand_curve.shape[1]
            return sum(df[int(y), int(x)] / normalization_factor
                       for x, y in adapted_cand_curve.T) / nb_samples

    # Bug: these constants are not defined in the OpenCV Python namespace.
    cv2.CV_CONTOURS_MATCH_I1 = 1
    cv2.CV_CONTOURS_MATCH_I2 = 2
    cv2.CV_CONTOURS_MATCH_I3 = 3

    USE_NO_CONTOUR = 0
    USE_EXT_CONTOUR = 1
    USE_INT_CONTOUR = 2
    USE_INTEXT_CONTOUR = 3

    class HuMoments(CurveDistance):
        """Shape distance based on the Hu moments of the shapes."""
        
        # TODO: reduce the number of options.
        # (1 type + 1 type x (1 contour method + 3 contours methods x 2 filled or not))
        # x 3 histogram matching methods = 24 possibilites!

        def __init__(self, 
                     use_image=True, 
                     contour_method=USE_NO_CONTOUR,
                     filled_contour=True,
                     hist_match_method=cv2.CV_CONTOURS_MATCH_I2, 
                     use_cache=True):
            # TODO: use cache to store contours.
            super().__init__(use_cache)
            self.use_image = use_image
            self.contour_method = contour_method
            self.filled_contour = filled_contour,
            self.hist_match_method = hist_match_method

        def adapt_curve(self, curve):
            if self.use_image:
                if curve.shape[0] == 2:
                    shp = (512, 512)
                    curve = cimp.getim(curve, shp)
                filled = self.filled
    
                if self.contour_method == USE_NO_CONTOUR:
                    return curve
                elif self.contour_method == USE_EXT_CONTOUR:
                    return cimp.get_ext_contour(curve, filled)
                elif self.contour_method == USE_INT_CONTOUR:
                    return cimp.get_int_contour(curve, filled)
                elif self.contour_method == USE_INTEXT_CONTOUR:
                    if filled:
                        return (cimp.get_ext_contour(curve, filled) -
                                cimp.get_int_contour(curve, filled))
                    else:
                        return (cimp.get_ext_contour(curve, filled) +
                                cimp.get_int_contour(curve, filled))
                else:
                    return curve
            else:
                return curve

        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            curve = self.adapt_curve(curve)
            m = cv2.moments(curve)
            return cv2.HuMoments(m)

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            cand_curve = self.adapt_curve(cand_curve)
            target_curve = self.adapt_curve(target_curve)
            return cv2.matchShapes(cand_curve, target_curve,
                                   self.hist_match_method, 0)

#class SphericalDensity:
#    """Shape distance based on the histogram of the spherical mapping."""
#    pass

class ZernikeMoments(CurveDistance):
    """Shape distance based on the Zernike moments of the shapes."""

    def __init__(self):
        pass

    def get_desc(self, curve):
        """Get the descriptor on the input curve."""
        pass

    def get_dist(self, cand_curve, target_curve):
        """Get the distance between two curves."""
        pass

class CurvatureFeatures(CurveDistance):
    """Shape distance based on the curvature of the shapes."""

    def __init__(self, sampling_rate, closed_curve=True, use_cache=True):
        super().__init__(use_cache)
        self.sampling_rate = sampling_rate
        self.closed_curve = closed_curve

    def compute_curvature(self, curve):
        """Compute the curvature along the input curve."""
        if self.closed_curve:
            # Extrapolate so that the curvature at the boundaries is correct.
            curve = np.hstack([curve[:, -3:-1], curve, curve[:, 1:3]])

        dx_dt = np.gradient(curve[0])
        dy_dt = np.gradient(curve[1])
        d2x_dt2 = np.gradient(dx_dt)
        d2y_dt2 = np.gradient(dy_dt)
        curvature = (np.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) /
                     (dx_dt * dx_dt + dy_dt * dy_dt)**1.5)

        if self.closed_curve:
            curvature = curvature[2:-2]

        return curvature

    def get_desc(self, curve):
        """Get the descriptor on the input curve."""
        nb_samples = curve.shape[-1]
        cvt = self.compute_curvature(curve)

        # Compute the Fourier power spectrum.
        fourier = np.fft.rfft(cvt)
        power = abs(fourier) * abs(fourier)
        freq = np.fft.rfftfreq(nb_samples)
        # Find power peaks and keep the ones >= 1% of the main peak.
        argrelmax = sig.argrelmax(power)[0]
        if argrelmax.size:
            argrelmax = argrelmax[power[argrelmax] >= (0.01 * power[0])]
        if argrelmax.size:
            # Test the main frequency (the other are just harmonics).
            f1 = freq[argrelmax[0]]
            main_period = 1 / (f1  * self.sampling_rate)
        else:
            main_period = 0
        # Compute the curvature average and std.
        avg = cvt.mean()
        std = cvt.std()
        return np.array([main_period, avg, std])

    def get_dist(self, cand_curve, target_curve):
        """Get the distance between two curves."""
        target_desc = self.get_target_desc(target_curve)
        return la.norm(self.get_desc(cand_curve) - target_desc)
