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

import curveproc as cpr


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
        if curve.size == 0:
            return curve
            
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
    EXT_CONTOUR = 1
    INT_CONTOUR = 2
    INTEXT_CONTOUR = 3


    def get_contour(curve_img, contour_type=EXT_CONTOUR, filled=False):
        """Return a contour of the input shape."""
        if contour_type == EXT_CONTOUR:
            return cimp.get_ext_contour(curve_img, filled)
        elif contour_type == INT_CONTOUR:
            return cimp.get_int_contour(curve_img, filled)
        elif contour_type == INTEXT_CONTOUR:
            if filled:
                return (cimp.get_ext_contour(curve_img, filled) -
                        cimp.get_int_contour(curve_img, filled))
            else:
                return (cimp.get_ext_contour(curve_img, filled) +
                        cimp.get_int_contour(curve_img, filled))


    class DistanceField(CurveDistance):
        """Shape distance based on the distance transform of the target."""

        def __init__(self, mask_size=5, resol=(512,512), *args, **kwargs):
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
            # Unoptimized version: 
            # return sum(df[int(y), int(x)] / normalization_factor
            #            for x, y in cand_curve.T) / nb_samples
            return (df[tuple(cand_curve[::-1].astype(np.intp))].sum() / 
                    (nb_samples * normalization_factor))

    # Bug: these constants are not defined in the OpenCV Python namespace.
    cv2.CV_CONTOURS_MATCH_I1 = 1
    cv2.CV_CONTOURS_MATCH_I2 = 2
    cv2.CV_CONTOURS_MATCH_I3 = 3

    class HuMoments(CurveDistance):
        """Shape distance based on the Hu moments of the shapes."""

        # TODO: reduce the number of options.
        # (1 contour method + 3 contours methods x 2 filled or not)
        # x 3 (or more) histogram matching methods = 21 (or more) possibilites!

        def __init__(self,
                     contour_method=None,
                     filled_contour=False,
                     hist_match_method=cv2.CV_CONTOURS_MATCH_I2,
                     threshold=1e-15,
                     *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.contour_method = contour_method
            self.filled_contour = filled_contour
            self.hist_match_method = hist_match_method
            self.threshold = threshold

        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if curve.shape[0] == 2:
                if self.normalize:
                    curve = self.normalize_pose(curve)
                # NOTE: we always use the method with raster images, because
                # self-intersecting curves can give undefined moments (due to
                # the 1st moment, m00 (the area), being computed as nearly
                # zero, and then being used to normalize the other moments).
                shp = (512, 512)
                curve = cimp.getim(curve, shp)
            if self.contour_method is not None:
                curve = get_contour(curve, self.contour_method,
                                    self.filled_contour)
            m = cv2.HuMoments(cv2.moments(curve))
            nonzero = np.abs(m) > self.threshold
            m[nonzero] = np.sign(m[nonzero]) * np.log10(np.abs(m[nonzero]))
            if not np.isfinite(m).all():
                print('infinite desc in Hu moms: ', m)

            return m

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            target_desc = self.get_target_desc(target_curve)
            cand_desc = self.get_desc(cand_curve)
            nonzero = ((np.abs(cand_desc) > self.threshold) *
                       (np.abs(target_desc) > self.threshold))
            return la.norm((cand_desc - target_desc)[nonzero])
#            cand_curve = self.adapt_curve(cand_curve)
#            target_curve = self.adapt_curve(target_curve)
#            return cv2.matchShapes(cand_curve, target_curve,
#                                   self.hist_match_method, 0)

    class PerceptualFeatures(CurveDistance):
        """Shape distance based on perceptually intuituve features."""

        def __init__(self, closed_curve=True, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed_curve = closed_curve

        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if self.normalize:
                curve = self.normalize_pose(curve)
            shp = (512, 512)
            curve_img = cimp.getim(curve, shp)

            # Feature 1: contour areas ratio
            int_area = cimp.get_int_contour(curve_img, filled=True).sum()
            ext_area = cimp.get_ext_contour(curve_img, filled=True).sum()
            # Feature 2: number of curvature maxima / length of curve
            cvt = cpr.compute_curvature(curve, self.closed_curve)
            argrelmax = sig.argrelmax(cvt)[0]
            nb_max = argrelmax.size
#            # /!\ scipy.linalg.norm currently does not accept the 'axis' arg.
#            length = sum(np.linalg.norm(curve.T[:-1] - curve.T[1:], axis=1))
            # Feature 3: average curvature maxima
            if nb_max:
                avg_max = np.mean(cvt[argrelmax])
            else:
                avg_max = np.mean(cvt)

            return np.array([int_area / ext_area,
                             nb_max,
                             np.log(avg_max)])

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            target_desc = self.get_target_desc(target_curve)
            cand_desc = self.get_desc(cand_curve)
            return la.norm(target_desc - cand_desc)


if MH_IMPORTED:
    class ZernikeMoments(CurveDistance):
        """Shape distance based on the Zernike moments of the shapes."""

        def __init__(self, radius=128, degree=8, contour_method=None,
                     filled_contour=False,  *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.radius = radius
            self.degree = degree
            self.contour_method = contour_method
            self.filled_contour = filled_contour


        def get_desc(self, curve):
            """Get the descriptor on the input curve."""
            if curve.shape[0] == 2:
                if self.normalize:
                    curve = self.normalize_pose(curve)
                shp = (512, 512)
                curve = cimp.getim(curve, shp)
            if self.contour_method is not None:
                curve = get_contour(curve, self.contour_method,
                                    self.filled_contour)
            return mh.features.zernike_moments(curve, radius=self.radius,
                                               degree=self.degree)

        def get_dist(self, cand_curve, target_curve):
            """Get the distance between two curves."""
            target_desc = self.get_target_desc(target_curve)
            cand_desc = self.get_desc(cand_curve)
            return la.norm(target_desc - cand_desc)

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
        cvt = cpr.compute_curvature(curve, self.closed_curve)

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


#class SphericalDensity:
#    """Shape distance based on the histogram of the spherical mapping."""
#    pass
