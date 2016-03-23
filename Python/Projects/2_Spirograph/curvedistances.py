# -*- coding: utf-8 -*-
"""
Library of curve distances.

@author: Robin Roussel
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.stats as st

try:
    import cv2
except ImportError:
    CV2_IMPORTED = False
else:
    CV2_IMPORTED = True
    
try:
    import mahotas as mh
except ImportError:
    MH_IMPORTED = False
else:
    MH_IMPORTED = True

if CV2_IMPORTED: 
    import curveimproc as cimp
    
    
def compute_curvature(curve, is_closed=True):
    """Compute the curvature along the input curve."""
    if is_closed:
        # Extrapolate so that the curvature at the boundaries is correct.
        curve = np.hstack([curve[:, -3:-1], curve, curve[:, 1:3]])

    dx_dt = np.gradient(curve[0])
    dy_dt = np.gradient(curve[1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (np.abs(dx_dt * d2y_dt2 - d2x_dt2 * dy_dt) /
                 (dx_dt * dx_dt + dy_dt * dy_dt)**1.5)

    if is_closed:
        curvature = curvature[2:-2]

    return curvature

class CurveDistance:
    """Base class for shape distances."""

    def __init__(self, use_cache=True, normalize=True):
        # If true, the target descriptor will be saved and re-used if the
        # target is the same. Useful for curve retrieval.
        self.use_cache = use_cache
        
        self.normalize = normalize

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
        """Normalize the curve's position, scale, orientation and skewness."""
        # Center and (uniformly) rescale the curve.
        mean = curve.mean(axis=1)
        try:
            std_inv = 1 / la.norm(curve.std(axis=1))
        except ZeroDivisionError:
            std_inv = 1.
        curve = (curve - mean.reshape(2, 1)) * std_inv        
        
        # Compute the SVD (numerically more stable than eig(cov), and more 
        # efficient when nb_samples >> nb_dims).
        u, _, _ = la.svd(curve)
        # Rotate the curve.
        curve = u.dot(curve)
        
        # Normalize mirror reflection to get positive skewness (i.e. most of
        # the mass is on the positive side of each axis).
        curve = curve * np.sign(st.skew(curve, axis=1)).reshape(2, 1)
        
        return curve

    def get_desc(self, curve):
        pass

    def get_dist(self, cand_curve, target_curve):
        pass


if CV2_IMPORTED:
    class DistanceField(CurveDistance):
        """Shape distance based on the distance transform of the target."""

        def __init__(self, mask_size=5, resol=(512, 512), *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.mask_size = mask_size
            self.resol = resol
            self.diag_ratio_dist_normalization = 0.05

        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if curve.shape[0] == 2:
                if self.normalize:
                    curve = self.normalize_pose(curve)
                img = cimp.getim(curve, self.resol)
            else:
                img = curve
            return cv2.distanceTransform(~img, cv2.DIST_L2, self.mask_size)

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            df = self.get_target_desc(target_curve)
            # Adapt the candidate curve to the distance field.
            if cand_curve.shape[0] == 2:
                if self.normalize:
                    cand_curve = self.normalize_pose(cand_curve)
            cand_curve = cimp.fit_in_box(cand_curve, df.shape)
            # Compute distance normalization factor.
            diag = (df.shape[0] * df.shape[0] + df.shape[1] * df.shape[1])**0.5
            normalization_factor = self.diag_ratio_dist_normalization * diag
            # Compute the average normalized distance.
            nb_samples = cand_curve.shape[1]
            return sum(df[int(y), int(x)] / normalization_factor
                       for x, y in cand_curve.T) / nb_samples

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
        # (1 contour method + 3 contours methods x 2 filled or not)
        # x 3 (or more) histogram matching methods = 21 (or more) possibilites!

        def __init__(self, 
                     contour_method=USE_NO_CONTOUR,
                     filled_contour=True,
                     hist_match_method=cv2.CV_CONTOURS_MATCH_I2, 
                     use_cache=True):
            # TODO: use cache to store contours.
            super().__init__(use_cache)
            self.contour_method = contour_method
            self.filled_contour = filled_contour
            self.hist_match_method = hist_match_method

        def adapt_curve(self, curve):
            """Transform the curve according to the options."""
            if curve.shape[0] == 2:
                if self.normalize:
                    curve = self.normalize_pose(curve)
                # NOTE: we always use the method with raster images, because 
                # self-intersecting curves can give undefined moments (due to 
                # the 1st moment, m00 (the area), being computed as nearly
                # zero, and then being used to normalize the other moments).
                shp = (512, 512)
                curve = cimp.getim(curve, shp)
            filled_contour = self.filled_contour

            if self.contour_method == USE_NO_CONTOUR:
                return curve
            elif self.contour_method == USE_EXT_CONTOUR:
                return cimp.get_ext_contour(curve, filled_contour)
            elif self.contour_method == USE_INT_CONTOUR:
                return cimp.get_int_contour(curve, filled_contour)
            elif self.contour_method == USE_INTEXT_CONTOUR:
                if filled_contour:
                    return (cimp.get_ext_contour(curve, filled_contour) -
                            cimp.get_int_contour(curve, filled_contour))
                else:
                    return (cimp.get_ext_contour(curve, filled_contour) +
                            cimp.get_int_contour(curve, filled_contour))
            else:
                return curve

        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            curve = self.adapt_curve(curve)
            m = cv2.HuMoments(cv2.moments(curve))
            nonzero = np.abs(m) > 1e-5
            m[nonzero] = np.sign(m[nonzero]) * np.log10(np.abs(m[nonzero]))
            if not np.isfinite(m).all():
                print('infinite desc in Hu moms: ', m)
            
            return m

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            target_desc = self.get_target_desc(target_curve)
            cand_desc = self.get_desc(cand_curve)            
            nonzero = (np.abs(cand_desc) > 1e-5) * (np.abs(target_desc) > 1e-5)
            return la.norm((cand_desc - target_desc)[nonzero], 1)
#            cand_curve = self.adapt_curve(cand_curve)
#            target_curve = self.adapt_curve(target_curve)
#            return cv2.matchShapes(cand_curve, target_curve,
#                                   self.hist_match_method, 0)
            
    class PerceptualFeatures(CurveDistance):
        """Shape distance based on perceptually intuituve features."""
        
        def __init__(self, closed_curve, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed_curve = closed_curve
            
        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if self.normalize:
                curve = self.normalize_pose(curve)
            shp = (512, 512)
            curve_img = cimp.getim(curve, shp)
                
            # Feature 1: contour areas ratio
            int_contour = cimp.get_int_contour(curve_img, filled=False)
            int_area = cv2.contourArea(int_contour)
            ext_contour = cimp.get_ext_contour(curve_img, filled=True)
            ext_area = cv2.contourArea(ext_contour)
            area_ratio = int_area / ext_area
            # Feature 2: number of curvature maxima
            cvt = compute_curvature(curve, self.closed_curve)
            argrelmax = sig.argrelmax(cvt)[0]
            nb_max = argrelmax.size
            # Feature 3: average curvature maxima
            if nb_max:
                avg_max = np.mean(cvt[argrelmax])
            else:
                avg_max = np.mean(cvt)
            
            return np.array([area_ratio, nb_max, avg_max])
    
        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            target_desc = self.get_target_desc(target_curve)
            cand_desc = self.get_desc(cand_curve)
            return la.norm(target_desc - cand_desc)        
            

#class SphericalDensity:
#    """Shape distance based on the histogram of the spherical mapping."""
#    pass

if MH_IMPORTED:
    class ZernikeMoments(CurveDistance):
        """Shape distance based on the Zernike moments of the shapes."""
    
        def __init__(self, radius, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.radius = radius
    
        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if curve.shape[0] == 2:
                if self.normalize:
                    curve = self.normalize_pose(curve)
                shp = (512, 512)
                curve = cimp.getim(curve, shp)
            return mh.features.zernike_moments(curve, radius=self.radius,
                                               degree=8)
    
        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            target_desc = self.get_target_desc(target_curve)
            cand_desc = self.get_desc(cand_curve)
            return la.norm(target_desc - cand_desc, 1)

class CurvatureFeatures(CurveDistance):
    """Shape distance based on the curvature of the shapes."""

    def __init__(self, sampling_rate, closed_curve=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sampling_rate = sampling_rate
        self.closed_curve = closed_curve

    def get_desc(self, curve):
        """Get the descriptor on the input curve."""
        if self.normalize:
            curve = self.normalize_pose(curve)
        nb_samples = curve.shape[-1]
        cvt = compute_curvature(curve, self.closed_curve)

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
