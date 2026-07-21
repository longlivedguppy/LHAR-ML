import unittest

import numpy as np

from src.mrf import huber_mrf_denoise, robust_normalize


class HuberMRFTests(unittest.TestCase):
    def test_constant_profile_is_unchanged(self):
        profile = np.full(64, 12345.0)
        result = huber_mrf_denoise(profile, regularization=1.0)
        np.testing.assert_allclose(result.profile, profile, atol=1e-8)
        self.assertTrue(result.converged)

    def test_zero_regularization_is_identity(self):
        profile = np.array([1.0, 4.0, 2.0, 8.0])
        result = huber_mrf_denoise(profile, regularization=0.0)
        np.testing.assert_allclose(result.profile, profile)

    def test_huber_reduces_noise_without_moving_step(self):
        random = np.random.default_rng(7)
        clean = np.r_[np.zeros(100), np.ones(100)]
        noisy = clean + random.normal(0.0, 0.08, clean.size)
        result = huber_mrf_denoise(noisy, regularization=1.0)
        self.assertLess(np.std(result.profile[:80]), np.std(noisy[:80]))
        raw_position = int(np.argmax(np.abs(np.gradient(noisy))))
        filtered_position = int(np.argmax(np.abs(np.gradient(result.profile))))
        self.assertLessEqual(abs(filtered_position - raw_position), 2)

    def test_input_validation(self):
        with self.assertRaises(ValueError):
            robust_normalize(np.empty(0))
        with self.assertRaises(ValueError):
            huber_mrf_denoise(np.ones(3), regularization=-1.0)


if __name__ == "__main__":
    unittest.main()
