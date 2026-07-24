import unittest

import numpy as np
import pandas as pd

from experiments.pd_detection_limit.config import Phase1Config
from experiments.pd_detection_limit.generate_signal import generate_synthetic_profile
from experiments.pd_detection_limit.metrics import calculate_trial_error, summarize_trials
from experiments.pd_detection_limit.noise_models import gaussian_random_noise
from experiments.pd_detection_limit.run_benchmark import run_trial


def small_config():
    return Phase1Config(
        phase=1,
        profile_length=101,
        pd50_true=50.0,
        baseline=30000.0,
        background_slope=0.0,
        single_amplitude=2000.0,
        single_width=4.0,
        single_noise_std=100.0,
        amplitudes=(0.0, 2000.0),
        widths=(4.0,),
        noise_stds=(100.0,),
        n_trials=2,
        base_seed=11,
        tolerance_px=5.0,
        start_distance=10,
        peak_distance=15,
        median_kernel=5,
        gaussian_sigma=5.0,
        mrf_regularization=1.0,
    )


class SyntheticPDLimitTests(unittest.TestCase):
    def test_sigmoid_midpoint_is_ground_truth_half_amplitude(self):
        profile = generate_synthetic_profile(
            length=101,
            baseline=30000,
            amplitude=2000,
            pd50_true=50,
            width=4,
        )
        self.assertAlmostEqual(profile.sigmoid[50], 0.5)
        self.assertAlmostEqual(profile.clean_profile[50], 31000)

    def test_gaussian_noise_is_seed_reproducible(self):
        first = gaussian_random_noise(20, 3.0, np.random.default_rng(7))
        second = gaussian_random_noise(20, 3.0, np.random.default_rng(7))
        np.testing.assert_array_equal(first, second)

    def test_blank_has_false_positive_but_no_truth_error(self):
        metrics = calculate_trial_error(50, 42, signal_present=False, tolerance_px=5)
        self.assertTrue(metrics["false_positive"])
        self.assertFalse(metrics["correct_detection"])
        self.assertTrue(np.isnan(metrics["signed_error_px"]))
        self.assertTrue(np.isnan(metrics["absolute_error_px"]))

    def test_all_methods_share_reproducible_noisy_input(self):
        config = small_config()
        first = run_trial(config, amplitude=2000, width=4, noise_std=100, seed=17)
        second = run_trial(config, amplitude=2000, width=4, noise_std=100, seed=17)
        np.testing.assert_array_equal(first.synthetic.noisy_profile, second.synthetic.noisy_profile)
        self.assertEqual(set(first.profiles), {"raw", "median_gaussian", "huber_mrf"})
        self.assertEqual(len(first.rows), 3)
        rows_by_method = {row["denoise_method"]: row for row in first.rows}
        self.assertTrue(np.isnan(rows_by_method["raw"]["mrf_regularization"]))
        self.assertTrue(np.isnan(rows_by_method["median_gaussian"]["mrf_regularization"]))
        self.assertEqual(rows_by_method["huber_mrf"]["mrf_regularization"], 1.0)

    def test_summary_separates_blank_fpr_from_signal_accuracy(self):
        config = small_config()
        rows = []
        for seed in (11, 12):
            rows.extend(run_trial(config, amplitude=0, width=4, noise_std=100, seed=seed).rows)
            rows.extend(run_trial(config, amplitude=2000, width=4, noise_std=100, seed=seed).rows)
        summary = summarize_trials(pd.DataFrame(rows))
        blank = summary[summary["amplitude"] == 0]
        signal = summary[summary["amplitude"] != 0]
        self.assertTrue(blank["mae_px"].isna().all())
        self.assertTrue((blank["false_positive_rate"] == 1.0).all())
        self.assertTrue(signal["false_positive_rate"].isna().all())
        self.assertTrue(signal["correct_detection_rate"].notna().all())


if __name__ == "__main__":
    unittest.main()
