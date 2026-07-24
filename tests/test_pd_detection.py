import unittest

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt

from src.denoise import median_gaussian_filter
from src.pd_detection import detect_pd_candidate


def legacy_detector(profile, start_distance=50, peak_distance=15):
    derivative = np.gradient(profile)
    absolute = np.abs(derivative)
    assessed = absolute.copy()
    assessed[:start_distance] = 0
    peaks, properties = find_peaks(assessed, height=0, distance=peak_distance)
    if len(peaks) >= 2:
        ranked = np.argsort(properties["peak_heights"])[::-1]
        return (
            int(peaks[ranked[0]]),
            float(properties["peak_heights"][ranked[0]]),
            int(peaks[ranked[1]]),
            float(properties["peak_heights"][ranked[1]]),
            False,
        )
    if len(peaks) == 1:
        return int(peaks[0]), float(properties["peak_heights"][0]), int(peaks[0]), 0.0, False
    position = int(np.argmax(assessed))
    return position, float(absolute[position]), position, 0.0, True


class ProductionPDDetectionTests(unittest.TestCase):
    def test_common_detector_matches_previous_inline_algorithm(self):
        rng = np.random.default_rng(19)
        profile = np.r_[np.zeros(100), np.ones(100) * 500]
        profile = profile + rng.normal(0, 40, profile.size)
        expected = legacy_detector(profile)
        actual = detect_pd_candidate(profile)

        self.assertEqual(actual.primary.position, expected[0])
        self.assertAlmostEqual(actual.primary.height, expected[1])
        self.assertEqual(actual.secondary.position, expected[2])
        self.assertAlmostEqual(actual.secondary.height, expected[3])
        self.assertEqual(actual.used_argmax_fallback, expected[4])
        np.testing.assert_allclose(actual.derivative, np.gradient(profile))

    def test_flat_blank_preserves_argmax_fallback(self):
        result = detect_pd_candidate(np.ones(100), start_distance=10)
        self.assertTrue(result.used_argmax_fallback)
        self.assertEqual(result.primary.position, 0)
        self.assertEqual(result.primary.height, 0)

    def test_median_gaussian_matches_existing_scipy_sequence(self):
        profile = np.array([10, 9, 100, 11, 12, 13, 14], dtype=float)
        expected = gaussian_filter1d(medfilt(profile, kernel_size=5), sigma=2)
        actual = median_gaussian_filter(profile, kernel_size=5, sigma=2)
        np.testing.assert_allclose(actual, expected)


if __name__ == "__main__":
    unittest.main()
