"""Compare one raw LHAR image line with Median+Gaussian and Huber MRF.

This is a deliberately isolated preliminary experiment. It does not import or
write to the production pipeline's processed/output directories and does not
classify any derivative peak as a PD boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("mrf_comparison") / ".matplotlib").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, medfilt, peak_widths

from src.mrf import MRFResult, huber_mrf_denoise


DEFAULT_DATASET = "260115-1-20um50cyc"
DEFAULT_REGULARIZATIONS = (0.1, 1.0, 10.0)


@dataclass(frozen=True)
class Peak:
    position: int
    height: float
    prominence: float
    width: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="one TIFF image; default is the 0-degree image in the 50-cycle dataset")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset used when --image is omitted")
    parser.add_argument("--x", type=int, default=878, help="single vertical line x-coordinate")
    parser.add_argument("--y-start", type=int, default=1400, help="first vertical coordinate (inclusive)")
    parser.add_argument("--y-end", type=int, default=200, help="last vertical coordinate (inclusive)")
    parser.add_argument("--start-distance", type=int, default=50, help="exclude this many initial profile samples from peak assessment")
    parser.add_argument("--median-kernel", type=int, default=5)
    parser.add_argument("--gaussian-sigma", type=float, default=5.0)
    parser.add_argument("--regularization", nargs="+", type=float, default=DEFAULT_REGULARIZATIONS, metavar="LAMBDA")
    parser.add_argument("--output-root", type=Path, default=Path("mrf_comparison"))
    parser.add_argument("--peak-noise-factor", type=float, default=5.0, help="minimum prominence as a multiple of derivative MAD noise")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    image_path = args.image or select_zero_degree_image(Path("data") / "raw" / args.dataset)
    raw_profile = load_vertical_line(image_path, args.x, args.y_start, args.y_end)
    median_gaussian = gaussian_filter(median_filter(raw_profile, args.median_kernel), args.gaussian_sigma)

    profiles: dict[str, np.ndarray] = {"Raw": raw_profile, "Median+Gaussian": median_gaussian}
    mrf_results: dict[str, MRFResult] = {}
    for regularization in args.regularization:
        label = f"Huber MRF lambda={regularization:g}"
        result = huber_mrf_denoise(raw_profile, regularization=regularization)
        profiles[label] = result.profile
        mrf_results[label] = result

    derivatives = {name: np.gradient(values) for name, values in profiles.items()}
    peaks = {
        name: find_significant_peaks(np.abs(derivative), args.start_distance, args.peak_noise_factor)
        for name, derivative in derivatives.items()
    }
    metrics = calculate_metrics(profiles, derivatives, peaks, args.start_distance)

    output_dir = build_output_directory(args.output_root, args.dataset, image_path, args.x)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_profiles(output_dir / "profiles.csv", profiles, derivatives)
    save_metrics(output_dir / "metrics.csv", metrics)
    save_peaks(output_dir / "significant_peaks.csv", peaks, derivatives["Raw"], args.start_distance)
    save_figure(output_dir / "comparison_overview.png", profiles, derivatives, peaks, args.start_distance)
    save_metadata(output_dir / "run_metadata.json", args, image_path, raw_profile, mrf_results, peaks)
    save_summary(output_dir / "summary.txt", metrics, peaks, mrf_results)
    print_summary(output_dir, metrics, peaks, mrf_results)


def select_zero_degree_image(dataset_dir: Path) -> Path:
    candidates = sorted((*dataset_dir.glob("*.tif"), *dataset_dir.glob("*.tiff")))
    if not candidates:
        raise FileNotFoundError(f"no TIFF images found in {dataset_dir}")
    zero_degree = [path for path in candidates if re.search(r"_\s*0_\s*0$", path.stem)]
    return zero_degree[0] if zero_degree else candidates[0]


def load_vertical_line(image_path: Path, x: int, y_start: int, y_end: int) -> np.ndarray:
    with Image.open(image_path) as image:
        array = np.asarray(image)
    if array.ndim == 3:
        array = array.astype(float).mean(axis=2)
    if array.ndim != 2:
        raise ValueError(f"expected a 2D image, got shape {array.shape}")
    if not 0 <= x < array.shape[1]:
        raise ValueError(f"x={x} is outside image width {array.shape[1]}")
    if not (0 <= y_start < array.shape[0] and 0 <= y_end < array.shape[0]):
        raise ValueError(f"y range {y_start}..{y_end} is outside image height {array.shape[0]}")
    step = 1 if y_start <= y_end else -1
    return array[np.arange(y_start, y_end + step, step), x].astype(float)


def median_filter(profile: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("median kernel must be a positive odd integer")
    return medfilt(profile, kernel_size=kernel_size)


def gaussian_filter(profile: np.ndarray, sigma: float, truncate: float = 4.0) -> np.ndarray:
    if sigma <= 0:
        raise ValueError("gaussian sigma must be positive")
    return gaussian_filter1d(profile, sigma=sigma, truncate=truncate)


def find_significant_peaks(values: np.ndarray, start: int, noise_factor: float) -> list[Peak]:
    """Return thresholded local maxima; never fall back to argmax."""
    if values.size < 3 or start >= values.size - 1:
        return []
    noise = derivative_noise_scale(values[start:])
    minimum_prominence = max(noise_factor * noise, np.finfo(float).eps)
    search_values = values[start:]
    local, properties = find_peaks(search_values, prominence=minimum_prominence)
    if local.size == 0:
        return []
    widths = peak_widths(search_values, local, rel_height=0.5)[0]
    found = [
        Peak(
            position=int(index + start),
            height=float(search_values[index]),
            prominence=float(properties["prominences"][rank]),
            width=float(widths[rank]),
        )
        for rank, index in enumerate(local)
    ]
    return sorted(found, key=lambda peak: peak.prominence, reverse=True)


def derivative_noise_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    return 1.4826 * float(np.median(np.abs(values - median)))


def calculate_metrics(
    profiles: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
    peaks: dict[str, list[Peak]],
    start: int,
) -> list[dict[str, object]]:
    raw = profiles["Raw"]
    reference_peaks = peaks["Median+Gaussian"]
    reference_position = reference_peaks[0].position if reference_peaks else None
    rows: list[dict[str, object]] = []
    for name, profile in profiles.items():
        residual = profile - raw
        derivative = derivatives[name]
        significant = peaks[name]
        main_peak = significant[0] if significant else None
        absolute_derivative = np.abs(derivative)
        numeric_max_position = start + int(np.argmax(absolute_derivative[start:]))
        artificial_suspects = sum(
            not raw_support(np.abs(derivatives["Raw"]), peak.position, start)[0]
            for peak in significant
        )
        rows.append(
            {
                "method": name,
                "rmse_from_raw": float(np.sqrt(np.mean(residual**2))),
                "mae_from_raw": float(np.mean(np.abs(residual))),
                "residual_std": float(np.std(residual)),
                "derivative_noise_mad": derivative_noise_scale(derivative[start:]),
                "max_abs_derivative": float(absolute_derivative[numeric_max_position]),
                "max_abs_derivative_position": numeric_max_position,
                "major_candidate_position": "" if main_peak is None else main_peak.position,
                "major_candidate_shift_from_median_gaussian": "" if main_peak is None or reference_position is None else main_peak.position - reference_position,
                "main_peak_prominence": "" if main_peak is None else main_peak.prominence,
                "main_peak_width": "" if main_peak is None else main_peak.width,
                "significant_peak_count": len(significant),
                "artificial_peak_suspect_count": artificial_suspects,
                "assessment": "undetectable" if main_peak is None else "candidate_only_not_PD",
            }
        )
    return rows


def save_profiles(path: Path, profiles: dict[str, np.ndarray], derivatives: dict[str, np.ndarray]) -> None:
    names = list(profiles)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["distance", *names, *(f"derivative:{name}" for name in names), *(f"abs_derivative:{name}" for name in names)])
        for index in range(len(next(iter(profiles.values())))):
            writer.writerow([index, *(profiles[name][index] for name in names), *(derivatives[name][index] for name in names), *(abs(derivatives[name][index]) for name in names)])


def save_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_peaks(path: Path, peaks: dict[str, list[Peak]], raw_derivative: np.ndarray, start: int) -> None:
    raw_absolute = np.abs(raw_derivative)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "rank_by_prominence", "position", "height", "prominence", "width", "raw_local_peak_support", "raw_support_distance", "artificial_peak_suspect", "interpretation"])
        for method, method_peaks in peaks.items():
            for rank, peak in enumerate(method_peaks, 1):
                supported, distance = raw_support(raw_absolute, peak.position, start)
                writer.writerow([method, rank, peak.position, peak.height, peak.prominence, peak.width, supported, distance, not supported, "candidate_only_not_PD"])


def raw_support(raw_absolute_derivative: np.ndarray, position: int, start: int, tolerance: int = 3) -> tuple[bool, object]:
    """Check whether a filtered candidate is backed by a nearby Raw local maximum."""
    lower = max(start, position - tolerance, 1)
    upper = min(raw_absolute_derivative.size - 2, position + tolerance)
    if lower > upper:
        return False, ""
    local_positions = [
        index
        for index in range(lower, upper + 1)
        if raw_absolute_derivative[index] > raw_absolute_derivative[index - 1]
        and raw_absolute_derivative[index] >= raw_absolute_derivative[index + 1]
    ]
    if not local_positions:
        return False, ""
    distance = min(abs(position - index) for index in local_positions)
    return True, distance


def save_figure(
    path: Path,
    profiles: dict[str, np.ndarray],
    derivatives: dict[str, np.ndarray],
    peaks: dict[str, list[Peak]],
    start: int,
) -> None:
    distance = np.arange(len(next(iter(profiles.values()))))
    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
    color_by_name = dict(zip(profiles, colors))
    figure, axes = plt.subplots(5, 1, figsize=(15, 22), constrained_layout=True)

    for name, profile in profiles.items():
        axes[0].plot(distance, profile, label=name, color=color_by_name[name], linewidth=1.4)
        axes[1].plot(distance, derivatives[name], label=name, color=color_by_name[name], linewidth=1.2)
        axes[2].plot(distance, np.abs(derivatives[name]), label=name, color=color_by_name[name], linewidth=1.2)
    axes[0].set_title("Single-line intensity profiles")
    axes[1].set_title("First derivatives")
    axes[2].set_title("Absolute first derivatives (markers are candidates, not PD labels)")

    for name, method_peaks in peaks.items():
        if method_peaks:
            peak = method_peaks[0]
            axes[2].scatter(peak.position, peak.height, color=color_by_name[name], marker="o", s=45)

    all_positions = [method_peaks[0].position for method_peaks in peaks.values() if method_peaks]
    if all_positions:
        zoom_left = max(start, min(all_positions) - 50)
        zoom_right = min(len(distance) - 1, max(all_positions) + 50)
        for name in profiles:
            axes[3].plot(distance, np.abs(derivatives[name]), label=name, color=color_by_name[name], linewidth=1.4)
        axes[3].set_xlim(zoom_left, zoom_right)
        axes[3].set_title("Major derivative-candidate neighborhood")
    else:
        axes[3].text(0.5, 0.5, "No significant derivative candidate detected", ha="center", va="center", transform=axes[3].transAxes)
        axes[3].set_title("Major derivative-candidate neighborhood")

    raw = profiles["Raw"]
    for name, profile in profiles.items():
        if name != "Raw":
            axes[4].plot(distance, profile - raw, label=f"{name} - Raw", color=color_by_name[name], linewidth=1.2)
    axes[4].set_title("Residuals from Raw")

    for axis in axes:
        axis.axvspan(0, start, color="lightgray", alpha=0.35, label="excluded from peak assessment")
        axis.set_xlabel("Distance from ROI start (pixels)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("Gray value")
    axes[1].set_ylabel("Gray value / pixel")
    axes[2].set_ylabel("|Gray value / pixel|")
    axes[3].set_ylabel("|Gray value / pixel|")
    axes[4].set_ylabel("Gray-value residual")
    figure.suptitle("Preliminary single-line comparison: Raw vs Median+Gaussian vs Huber MRF", fontsize=16)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_metadata(
    path: Path,
    args: argparse.Namespace,
    image_path: Path,
    raw_profile: np.ndarray,
    mrf_results: dict[str, MRFResult],
    peaks: dict[str, list[Peak]],
) -> None:
    payload = {
        "scope": "preliminary single-image single-line comparison; no PD classification",
        "image": str(image_path),
        "image_line": {"x": args.x, "y_start": args.y_start, "y_end": args.y_end, "samples": int(raw_profile.size)},
        "median_gaussian": {"median_kernel": args.median_kernel, "gaussian_sigma": args.gaussian_sigma},
        "peak_assessment": {"start_distance": args.start_distance, "minimum_prominence_noise_factor": args.peak_noise_factor, "argmax_fallback": False},
        "mrf": {
            label: {
                "regularization": result.regularization,
                "huber_delta_normalized": result.huber_delta,
                "robust_center": result.robust_scale.center,
                "robust_scale_p95_minus_p5": result.robust_scale.scale,
                "iterations": result.iterations,
                "converged": result.converged,
                "relative_change": result.relative_change,
            }
            for label, result in mrf_results.items()
        },
        "significant_peaks": {name: [asdict(peak) for peak in method_peaks] for name, method_peaks in peaks.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_summary(path: Path, metrics: list[dict[str, object]], peaks: dict[str, list[Peak]], mrf_results: dict[str, MRFResult]) -> None:
    lines = [
        "Preliminary Huber MRF comparison",
        "No detected derivative candidate in this report is automatically a PD boundary.",
        "",
    ]
    for row in metrics:
        position = row["major_candidate_position"] if row["major_candidate_position"] != "" else "undetectable"
        lines.append(
            f"{row['method']}: noise_MAD={row['derivative_noise_mad']:.6g}, "
            f"main_candidate_position={position}, significant_peaks={row['significant_peak_count']}, "
            f"RMSE_from_raw={row['rmse_from_raw']:.6g}"
        )
    lines.extend(["", "MRF convergence:"])
    for label, result in mrf_results.items():
        lines.append(f"{label}: delta={result.huber_delta:.6g}, iterations={result.iterations}, converged={result.converged}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_summary(output_dir: Path, metrics: list[dict[str, object]], peaks: dict[str, list[Peak]], mrf_results: dict[str, MRFResult]) -> None:
    print(f"Comparison output: {output_dir}")
    for row in metrics:
        position = row["major_candidate_position"] if row["major_candidate_position"] != "" else "undetectable"
        print(f"- {row['method']}: derivative noise MAD={row['derivative_noise_mad']:.4g}, main candidate={position}, peaks={row['significant_peak_count']}")
    for label, result in mrf_results.items():
        print(f"- {label}: delta={result.huber_delta:.4g}, iterations={result.iterations}, converged={result.converged}")
    print("Candidates are not classified as PD boundaries; no argmax fallback was used.")


def build_output_directory(root: Path, dataset: str, image_path: Path, x: int) -> Path:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", image_path.stem).strip("_")
    return root / dataset / safe_stem / f"x{x}"


if __name__ == "__main__":
    main()
