"""Run the isolated single-line Huber MRF comparison for every TIFF in a dataset."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from compare_mrf_profile import (
    DEFAULT_PLOT_METHOD,
    DEFAULT_PLOT_REGULARIZATION,
    DEFAULT_REGULARIZATIONS,
    PLOT_METHODS,
    calculate_metrics,
    comparison_plot_multi_method,
    find_significant_peaks,
    gaussian_filter,
    load_vertical_line,
    median_filter,
    select_plot_profile,
    simple_plot_single_method,
)
from src.mrf import huber_mrf_denoise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="20260512")
    parser.add_argument("--x", type=int, default=1000)
    parser.add_argument("--y-start", type=int, default=1400)
    parser.add_argument("--y-end", type=int, default=200)
    parser.add_argument("--start-distance", type=int, default=50)
    parser.add_argument("--median-kernel", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=5.0)
    parser.add_argument("--regularization", nargs="+", type=float, default=DEFAULT_REGULARIZATIONS)
    parser.add_argument("--plot-method", choices=PLOT_METHODS, default=DEFAULT_PLOT_METHOD)
    parser.add_argument("--plot-regularization", type=float, default=DEFAULT_PLOT_REGULARIZATION)
    parser.add_argument(
        "--comparison-plots",
        action="store_true",
        help="also save the legacy five-panel multi-method plots",
    )
    parser.add_argument("--peak-noise-factor", type=float, default=5.0)
    parser.add_argument("--output-root", type=Path, default=Path("mrf_comparison_batch"))
    parser.add_argument("--skip-individual-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    images = sorted_dataset_images(Path("data") / "raw" / args.dataset)
    if not images:
        raise FileNotFoundError(f"no TIFF images found for {args.dataset}")

    output_dir = args.output_root / args.dataset / f"x{args.x}"
    individual_dir = output_dir / "individual_plots"
    comparison_dir = output_dir / "comparison_plots"
    profile_dir = output_dir / "profiles"
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir.mkdir(exist_ok=True)
    if args.comparison_plots:
        comparison_dir.mkdir(exist_ok=True)
    profile_dir.mkdir(exist_ok=True)

    all_profiles: dict[str, list[np.ndarray]] = {}
    aggregate_metrics: list[dict[str, object]] = []
    run_records: list[dict[str, object]] = []

    for angle, image_index, image_path in tqdm(images, desc="MRF batch", unit="image"):
        raw = load_vertical_line(image_path, args.x, args.y_start, args.y_end)
        median = median_filter(raw, args.median_kernel)
        gaussian = gaussian_filter(raw, args.gaussian_sigma)
        median_gaussian = gaussian_filter(median, args.gaussian_sigma)
        profiles = {"Raw": raw, "Median+Gaussian": median_gaussian}
        mrf_diagnostics: dict[str, dict[str, object]] = {}
        regularizations = list(args.regularization)
        if args.plot_method == "mrf" and args.plot_regularization not in regularizations:
            regularizations.append(args.plot_regularization)
        mrf_results = {}
        for regularization in regularizations:
            label = f"Huber MRF lambda={regularization:g}"
            result = huber_mrf_denoise(raw, regularization=regularization)
            profiles[label] = result.profile
            mrf_results[label] = result
            mrf_diagnostics[label] = {
                "lambda": regularization,
                "huber_delta_normalized": result.huber_delta,
                "iterations": result.iterations,
                "converged": result.converged,
                "relative_change": result.relative_change,
                "robust_center": result.robust_scale.center,
                "robust_scale": result.robust_scale.scale,
            }

        derivatives = {name: np.gradient(profile) for name, profile in profiles.items()}
        peaks = {
            name: find_significant_peaks(np.abs(derivative), args.start_distance, args.peak_noise_factor)
            for name, derivative in derivatives.items()
        }
        metrics = calculate_metrics(profiles, derivatives, peaks, args.start_distance)
        for row in metrics:
            aggregate_metrics.append({"angle_deg": angle, "image_index": image_index, "image": image_path.name, **row})
        for name, profile in profiles.items():
            all_profiles.setdefault(name, []).append(profile)

        save_compact_profiles(profile_dir / f"angle_{angle:03d}.csv", profiles, derivatives)
        if not args.skip_individual_plots:
            denoised_profile, denoise_label, title_method, analysis_name = select_plot_profile(
                args.plot_method,
                args.plot_regularization,
                median,
                gaussian,
                median_gaussian,
                mrf_results,
            )
            denoised_derivative = np.gradient(denoised_profile)
            selected_peaks = peaks.get(analysis_name)
            if selected_peaks is None:
                selected_peaks = find_significant_peaks(
                    np.abs(denoised_derivative), args.start_distance, args.peak_noise_factor
                )
            simple_plot_single_method(
                individual_dir / f"angle_{angle:03d}.png",
                raw,
                denoised_profile,
                denoised_derivative,
                selected_peaks,
                args.start_distance,
                denoise_label,
                title_method,
                image_path.stem,
            )
            if args.comparison_plots:
                comparison_plot_multi_method(
                    comparison_dir / f"angle_{angle:03d}.png",
                    profiles,
                    derivatives,
                    peaks,
                    args.start_distance,
                )
        run_records.append(
            {
                "angle_deg": angle,
                "image_index": image_index,
                "image": str(image_path),
                "mrf": mrf_diagnostics,
                "candidate_counts": {name: len(method_peaks) for name, method_peaks in peaks.items()},
            }
        )

    matrices = {name: np.vstack(profile_list) for name, profile_list in all_profiles.items()}
    angles = np.array([item[0] for item in images])
    save_metrics(output_dir / "metrics_all_images.csv", aggregate_metrics)
    save_profile_archive(output_dir / "profiles_all_images.npz", angles, matrices)
    summary = summarize_metrics(aggregate_metrics)
    save_metrics(output_dir / "summary_by_method.csv", summary)
    comparison = compare_with_median_gaussian(aggregate_metrics)
    save_metrics(output_dir / "comparison_to_median_gaussian.csv", comparison)
    if args.comparison_plots:
        save_heatmaps(output_dir / "all_angles_profile_heatmaps.png", angles, matrices, args.start_distance)
        save_derivative_heatmaps(output_dir / "all_angles_abs_derivative_heatmaps.png", angles, matrices, args.start_distance)
        save_summary_plot(output_dir / "summary_metrics_by_angle.png", aggregate_metrics)

    metadata = {
        "scope": "all TIFFs, one vertical line per image; candidates are not PD classifications",
        "dataset": args.dataset,
        "image_count": len(images),
        "angles_deg": angles.tolist(),
        "line": {"x": args.x, "y_start": args.y_start, "y_end": args.y_end},
        "median_gaussian": {"median_kernel": args.median_kernel, "gaussian_sigma": args.gaussian_sigma},
        "mrf_regularization": list(args.regularization),
        "normal_plot": {
            "method": args.plot_method,
            "mrf_regularization": args.plot_regularization if args.plot_method == "mrf" else None,
            "legacy_comparison_plots": args.comparison_plots,
        },
        "start_distance": args.start_distance,
        "peak_noise_factor": args.peak_noise_factor,
        "argmax_fallback": False,
        "runs": run_records,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print_batch_summary(output_dir, summary)


def sorted_dataset_images(dataset_dir: Path) -> list[tuple[int, int, Path]]:
    records = []
    for path in (*dataset_dir.glob("*.tif"), *dataset_dir.glob("*.tiff")):
        match = re.search(r"_\s*(\d+)_\s*(\d+)$", path.stem)
        if match:
            records.append((int(match.group(2)), int(match.group(1)), path))
    return sorted(records, key=lambda item: (item[0], item[1]))


def save_compact_profiles(path: Path, profiles: dict[str, np.ndarray], derivatives: dict[str, np.ndarray]) -> None:
    names = list(profiles)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance", *names, *(f"derivative:{name}" for name in names)])
        for index in range(len(next(iter(profiles.values())))):
            writer.writerow([index, *(profiles[name][index] for name in names), *(derivatives[name][index] for name in names)])


def save_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_profile_archive(path: Path, angles: np.ndarray, matrices: dict[str, np.ndarray]) -> None:
    arrays = {"angles_deg": angles}
    for name, matrix in matrices.items():
        key = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
        arrays[key] = matrix
    np.savez_compressed(path, **arrays)


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    lower = np.percentile(matrix, 5, axis=1, keepdims=True)
    upper = np.percentile(matrix, 95, axis=1, keepdims=True)
    return np.clip((matrix - lower) / np.maximum(upper - lower, 1e-12), 0, 1)


def save_heatmaps(path: Path, angles: np.ndarray, matrices: dict[str, np.ndarray], start: int) -> None:
    figure, axes = plt.subplots(len(matrices), 1, figsize=(15, 3.2 * len(matrices)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (name, matrix) in zip(axes, matrices.items()):
        image = axis.imshow(row_normalize(matrix), aspect="auto", cmap="viridis", extent=[0, matrix.shape[1] - 1, angles[-1], angles[0]])
        axis.axvspan(0, start, color="lightgray", alpha=0.35)
        axis.set_title(name)
        axis.set_ylabel("Angle (deg)")
        figure.colorbar(image, ax=axis, label="Row-wise robust normalized intensity")
    axes[-1].set_xlabel("Distance from ROI start (pixels)")
    figure.suptitle("All-angle single-line profiles (visual comparison only)", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_derivative_heatmaps(path: Path, angles: np.ndarray, matrices: dict[str, np.ndarray], start: int) -> None:
    figure, axes = plt.subplots(len(matrices), 1, figsize=(15, 3.2 * len(matrices)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, (name, matrix) in zip(axes, matrices.items()):
        absolute = np.abs(np.gradient(matrix, axis=1))
        display_max = max(float(np.percentile(absolute[:, start:], 99)), 1e-12)
        image = axis.imshow(absolute, aspect="auto", cmap="magma", vmin=0, vmax=display_max, extent=[0, matrix.shape[1] - 1, angles[-1], angles[0]])
        axis.axvspan(0, start, color="lightgray", alpha=0.35)
        axis.set_title(name)
        axis.set_ylabel("Angle (deg)")
        figure.colorbar(image, ax=axis, label="Absolute derivative")
    axes[-1].set_xlabel("Distance from ROI start (pixels)")
    figure.suptitle("All-angle absolute derivatives (99th-percentile display scale per method)", fontsize=15)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def summarize_metrics(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    summary = []
    for method in methods:
        selected = [row for row in rows if row["method"] == method]
        noise = np.array([float(row["derivative_noise_mad"]) for row in selected])
        rmse = np.array([float(row["rmse_from_raw"]) for row in selected])
        candidate_count = np.array([int(row["significant_peak_count"]) for row in selected])
        positions = [float(row["major_candidate_position"]) for row in selected if row["major_candidate_position"] != ""]
        summary.append(
            {
                "method": method,
                "images": len(selected),
                "noise_mad_median": float(np.median(noise)),
                "noise_mad_mean": float(np.mean(noise)),
                "rmse_from_raw_median": float(np.median(rmse)),
                "images_with_candidate": int(np.count_nonzero(candidate_count)),
                "candidate_rate": float(np.count_nonzero(candidate_count) / len(selected)),
                "candidate_position_median": "" if not positions else float(np.median(positions)),
                "candidate_position_std": "" if not positions else float(np.std(positions)),
                "total_candidates": int(candidate_count.sum()),
                "interpretation": "candidate_only_not_PD",
            }
        )
    return summary


def compare_with_median_gaussian(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    reference = {
        int(row["angle_deg"]): float(row["major_candidate_position"])
        for row in rows
        if row["method"] == "Median+Gaussian" and row["major_candidate_position"] != ""
    }
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    comparison = []
    for method in methods:
        differences = []
        for row in rows:
            if row["method"] != method or row["major_candidate_position"] == "":
                continue
            angle = int(row["angle_deg"])
            if angle in reference:
                differences.append(abs(float(row["major_candidate_position"]) - reference[angle]))
        values = np.asarray(differences, dtype=float)
        comparison.append(
            {
                "method": method,
                "comparable_images": int(values.size),
                "median_absolute_candidate_shift_px": "" if not values.size else float(np.median(values)),
                "mean_absolute_candidate_shift_px": "" if not values.size else float(np.mean(values)),
                "within_5px": int(np.count_nonzero(values <= 5)),
                "within_10px": int(np.count_nonzero(values <= 10)),
                "within_50px": int(np.count_nonzero(values <= 50)),
                "over_50px": int(np.count_nonzero(values > 50)),
                "interpretation": "agreement_with_filter_candidate_not_PD_accuracy",
            }
        )
    return comparison


def save_summary_plot(path: Path, rows: list[dict[str, object]]) -> None:
    methods = list(dict.fromkeys(str(row["method"]) for row in rows))
    figure, axes = plt.subplots(2, 1, figsize=(15, 10), constrained_layout=True)
    for method in methods:
        selected = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["angle_deg"]))
        angles = [int(row["angle_deg"]) for row in selected]
        noise = [float(row["derivative_noise_mad"]) for row in selected]
        positions = [np.nan if row["major_candidate_position"] == "" else float(row["major_candidate_position"]) for row in selected]
        axes[0].plot(angles, noise, marker=".", linewidth=1, label=method)
        axes[1].plot(angles, positions, marker=".", linewidth=1, label=method)
    axes[0].set_title("Derivative noise MAD by angle")
    axes[0].set_ylabel("MAD (gray value / pixel)")
    axes[1].set_title("Significant candidate position by angle (not a PD classification)")
    axes[1].set_ylabel("Distance (pixels)")
    for axis in axes:
        axis.set_xlabel("Angle (deg)", fontsize=14)
        axis.tick_params(axis="both", labelsize=12)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=12, framealpha=0.95)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def print_batch_summary(output_dir: Path, summary: list[dict[str, object]]) -> None:
    print(f"Batch output: {output_dir}")
    for row in summary:
        print(f"- {row['method']}: median noise MAD={row['noise_mad_median']:.4g}, candidate images={row['images_with_candidate']}/{row['images']}")
    print("All candidates remain descriptive only and are not classified as PD boundaries.")


if __name__ == "__main__":
    main()
