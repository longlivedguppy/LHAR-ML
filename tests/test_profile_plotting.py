import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

import numpy as np
from PIL import Image

from compare_mrf_profile import (
    Peak,
    build_denoise_tag,
    build_line_count_tag,
    build_output_directory,
    calculate_derivative_ylim,
    calculate_intensity_ylim,
    calculate_peak_prominence_threshold,
    extract_parallel_profiles,
    load_averaged_vertical_profile,
    parallel_line_offsets,
)


class ProfilePlottingTests(unittest.TestCase):
    def test_denoise_tags_include_non_default_parameters(self):
        self.assertEqual(build_denoise_tag("median"), "median")
        self.assertEqual(build_denoise_tag("median", median_kernel=3), "median_k3")
        self.assertEqual(build_denoise_tag("gaussian", gaussian_sigma=1), "gaussian_sigma_1")
        self.assertEqual(
            build_denoise_tag("median_gaussian", median_kernel=3, gaussian_sigma=1),
            "median_gaussian_k3_sigma_1",
        )
        self.assertEqual(build_denoise_tag("mrf", regularization=0.1), "mrf_lambda_0p1")
        self.assertEqual(build_denoise_tag("mrf", regularization=10), "mrf_lambda_10")

    def test_derivative_limits_ignore_excluded_outlier(self):
        derivative = np.r_[1000.0, np.ones(99) * 10.0]
        peak = Peak(position=50, height=10.0, prominence=5.0, width=2.0)
        lower, upper = calculate_derivative_ylim(derivative, [peak], start=1, target_fraction=0.35)
        self.assertEqual(lower, 0.0)
        self.assertAlmostEqual(10.0 / upper, 0.35)

    def test_intensity_limits_place_values_in_upper_region(self):
        raw = np.array([100.0, 150.0, 200.0])
        denoised = np.array([110.0, 150.0, 190.0])
        lower, upper = calculate_intensity_ylim(raw, denoised, target_fraction=0.45)
        axis_range = upper - lower
        self.assertAlmostEqual((200.0 - 100.0) / axis_range, 0.45)
        self.assertGreater((100.0 - lower) / axis_range, 0.5)

    def test_output_directory_appends_method_tag_after_existing_hierarchy(self):
        path = build_output_directory(
            Path("plots"),
            "sampleA",
            Path("angle_000.tiff"),
            1000,
            "mrf_lambda_10",
        )
        self.assertEqual(path, Path("plots/sampleA/angle_000/x1000/mrf_lambda_10"))

    def test_parallel_offsets_are_symmetric_for_odd_and_even_counts(self):
        np.testing.assert_array_equal(parallel_line_offsets(5), [-2, -1, 0, 1, 2])
        np.testing.assert_array_equal(
            parallel_line_offsets(10),
            [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        )

    def test_even_line_mean_uses_subpixel_symmetric_sampling(self):
        image_array = np.tile(np.arange(12, dtype=np.uint16) * 10, (4, 1))
        image_array += np.arange(4, dtype=np.uint16)[:, None]
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "profile.png"
            Image.fromarray(image_array).save(image_path)
            profiles = extract_parallel_profiles(image_path, 5, 3, 0, line_count=10)
            averaged = load_averaged_vertical_profile(image_path, 5, 3, 0, line_count=10)

        self.assertEqual(profiles.shape, (10, 4))
        np.testing.assert_allclose(averaged, [53, 52, 51, 50])

    def test_parallel_sampling_fails_if_requested_strip_crosses_image_boundary(self):
        image_array = np.zeros((4, 12), dtype=np.uint8)
        with TemporaryDirectory() as directory:
            image_path = Path(directory) / "profile.png"
            Image.fromarray(image_array).save(image_path)
            with self.assertRaisesRegex(ValueError, "cannot sample 20 lines"):
                extract_parallel_profiles(image_path, 5, 3, 0, line_count=20)

    def test_output_directory_separates_line_counts(self):
        path = build_output_directory(
            Path("plots"),
            "sampleA",
            Path("angle_000.tiff"),
            1000,
            "mrf_lambda_10",
            20,
        )
        self.assertEqual(
            path,
            Path("plots/sampleA/angle_000/x1000/mrf_lambda_10/line_20"),
        )
        self.assertEqual(build_line_count_tag(20), "line_20")

    def test_line_spacing_controls_sampling_width_and_supports_100_lines(self):
        twenty_wide = parallel_line_offsets(20, line_spacing=4)
        hundred = parallel_line_offsets(100)
        self.assertAlmostEqual(twenty_wide[0], -38.0)
        self.assertAlmostEqual(twenty_wide[-1], 38.0)
        self.assertAlmostEqual(twenty_wide[-1] - twenty_wide[0], 76.0)
        self.assertAlmostEqual(hundred[0], -49.5)
        self.assertAlmostEqual(hundred[-1], 49.5)
        self.assertEqual(build_line_count_tag(20, 4), "line_20_spacing_4")

    def test_fixed_prominence_is_identical_for_different_noise_levels(self):
        low_noise = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        high_noise = low_noise * 100
        self.assertEqual(
            calculate_peak_prominence_threshold(low_noise, 0, 5.0, fixed_prominence=100),
            100,
        )
        self.assertEqual(
            calculate_peak_prominence_threshold(high_noise, 0, 5.0, fixed_prominence=100),
            100,
        )

    def test_output_directory_separates_spacing_and_fixed_prominence(self):
        path = build_output_directory(
            Path("plots"),
            "sampleA",
            Path("angle_000.tiff"),
            1000,
            "mrf_lambda_10",
            20,
            4,
            100,
        )
        self.assertEqual(
            path,
            Path(
                "plots/sampleA/angle_000/x1000/mrf_lambda_10/"
                "line_20_spacing_4/peak_prominence_100"
            ),
        )


if __name__ == "__main__":
    unittest.main()
