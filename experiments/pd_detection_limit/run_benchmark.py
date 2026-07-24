"""Run Phase 1 synthetic PD detection-limit experiments.

Use ``single`` first for one diagnostic profile, then ``sweep`` for the
parameter grid and Monte Carlo aggregation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_DIR / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.pd_detection_limit.config import Phase1Config, load_phase1_config
from experiments.pd_detection_limit.generate_signal import SyntheticProfile, generate_synthetic_profile
from experiments.pd_detection_limit.metrics import calculate_trial_error, summarize_trials
from experiments.pd_detection_limit.noise_models import gaussian_random_noise
from src.denoise import median_gaussian_filter
from src.mrf import MRFResult, huber_mrf_denoise
from src.pd_detection import PDDetectionResult, detect_pd_candidate


METHOD_LABELS = {
    "raw": "Raw",
    "median_gaussian": "Median+Gaussian",
    "huber_mrf": "Huber MRF",
}
METHOD_COLORS = {
    "raw": "tab:gray",
    "median_gaussian": "tab:red",
    "huber_mrf": "tab:blue",
}


@dataclass(frozen=True)
class TrialArtifacts:
    """Detailed arrays retained only when a diagnostic output is needed."""

    synthetic: SyntheticProfile
    profiles: dict[str, np.ndarray]
    detections: dict[str, PDDetectionResult]
    rows: list[dict[str, object]]
    mrf_result: MRFResult


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("single", "sweep"),
        default="single",
        help="run one diagnostic trial or the full parameter-grid Monte Carlo",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_DIR / "configs" / "phase1.json",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="override the configured Monte Carlo count (use 5 for a smoke run)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="explicit run directory; otherwise a deterministic trial/run tag is used",
    )
    parser.add_argument("--no-plots", action="store_true", help="write CSV/JSON only")
    return parser.parse_args()


def run_trial(
    config: Phase1Config,
    *,
    amplitude: float,
    width: float,
    noise_std: float,
    seed: int,
) -> TrialArtifacts:
    """Generate one noisy input and assess all methods on exactly that input."""
    rng = np.random.default_rng(seed)
    noise = gaussian_random_noise(config.profile_length, noise_std, rng)
    synthetic = generate_synthetic_profile(
        length=config.profile_length,
        baseline=config.baseline,
        amplitude=amplitude,
        pd50_true=config.pd50_true,
        width=width,
        background_slope=config.background_slope,
        noise=noise,
    )

    median_gaussian = median_gaussian_filter(
        synthetic.noisy_profile,
        kernel_size=config.median_kernel,
        sigma=config.gaussian_sigma,
    )
    mrf_result = huber_mrf_denoise(
        synthetic.noisy_profile,
        regularization=config.mrf_regularization,
    )
    profiles = {
        "raw": synthetic.noisy_profile,
        "median_gaussian": median_gaussian,
        "huber_mrf": mrf_result.profile,
    }
    detections = {
        method: detect_pd_candidate(
            profile,
            start_distance=config.start_distance,
            peak_distance=config.peak_distance,
        )
        for method, profile in profiles.items()
    }

    signal_present = amplitude != 0.0
    rows: list[dict[str, object]] = []
    for method, detection in detections.items():
        pd50_estimated = float(detection.primary.position)
        error = calculate_trial_error(
            config.pd50_true,
            pd50_estimated,
            signal_present,
            config.tolerance_px,
        )
        assessed = detection.absolute_derivative.copy()
        assessed[: config.start_distance] = 0
        numeric_max_position = int(np.argmax(assessed))
        is_mrf = method == "huber_mrf"
        rows.append(
            {
                "phase": config.phase,
                "random_seed": int(seed),
                "signal_present": bool(signal_present),
                "pd50_true": float(config.pd50_true),
                "amplitude": float(amplitude),
                "width": float(width),
                "baseline": float(config.baseline),
                "background_slope": float(config.background_slope),
                "noise_std": float(noise_std),
                "measured_noise_id": "",
                "measured_noise_scale": 0.0,
                "denoise_method": method,
                "median_kernel": int(config.median_kernel),
                "gaussian_sigma": float(config.gaussian_sigma),
                "mrf_regularization": float(config.mrf_regularization) if is_mrf else np.nan,
                "mrf_huber_delta": float(mrf_result.huber_delta) if is_mrf else np.nan,
                "mrf_iterations": int(mrf_result.iterations) if is_mrf else np.nan,
                "mrf_converged": bool(mrf_result.converged) if is_mrf else "",
                "pd50_estimated": pd50_estimated,
                **error,
                "max_abs_derivative": float(assessed[numeric_max_position]),
                "max_abs_derivative_position": numeric_max_position,
                "max_peak_position": int(detection.primary.position),
                "max_peak_height": float(detection.primary.height),
                "second_peak_position": int(detection.secondary.position),
                "second_peak_height": float(detection.secondary.height),
                "detection_used_argmax_fallback": bool(detection.used_argmax_fallback),
                "tolerance_px": float(config.tolerance_px),
                "start_distance": int(config.start_distance),
                "peak_distance": int(config.peak_distance),
            }
        )
    return TrialArtifacts(synthetic, profiles, detections, rows, mrf_result)


def save_trial_profiles(path: Path, artifacts: TrialArtifacts) -> None:
    """Save all profiles and derivatives for one inspectable trial."""
    columns: dict[str, np.ndarray] = {
        "x": artifacts.synthetic.x,
        "clean_ground_truth": artifacts.synthetic.clean_profile,
        "noise": artifacts.synthetic.noise,
        "noisy_input": artifacts.synthetic.noisy_profile,
    }
    for method, profile in artifacts.profiles.items():
        detection = artifacts.detections[method]
        columns[f"profile_{method}"] = profile
        columns[f"derivative_{method}"] = detection.derivative
        columns[f"abs_derivative_{method}"] = detection.absolute_derivative
    pd.DataFrame(columns).to_csv(path, index=False)


def save_method_detection_plot(
    path: Path,
    artifacts: TrialArtifacts,
    method: str,
) -> None:
    """Save one readable input/denoise/derivative/PD-prediction figure."""
    row = next(item for item in artifacts.rows if item["denoise_method"] == method)
    signal_present = bool(row["signal_present"])
    pd50_true = float(row["pd50_true"])
    prediction = artifacts.detections[method].primary
    detection = artifacts.detections[method]
    x = artifacts.synthetic.x
    method_label = METHOD_LABELS[method]
    if method == "huber_mrf":
        method_label += f" (lambda={artifacts.mrf_result.regularization:g})"

    figure, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    axes[0].plot(
        x,
        artifacts.synthetic.clean_profile,
        color="black",
        linewidth=2.0,
        label="Noise-free synthetic profile",
    )
    axes[0].plot(
        x,
        artifacts.synthetic.noisy_profile,
        color="0.65",
        linewidth=1.0,
        alpha=0.75,
        label="Synthetic profile + Gaussian noise",
    )
    if method != "raw":
        axes[0].plot(
            x,
            artifacts.profiles[method],
            color=METHOD_COLORS[method],
            linewidth=1.8,
            label=method_label,
        )
    if signal_present:
        axes[0].axvline(pd50_true, color="black", linestyle="--", linewidth=2, label="True PD50")
    axes[0].axvline(
        prediction.position,
        color=METHOD_COLORS[method],
        linestyle=":",
        linewidth=2,
        label=f"PD prediction ({method_label}): {prediction.position} px",
    )
    axes[0].set_ylabel("Gray Value")
    axes[0].set_title(f"Phase 1: {method_label} PD detection")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=10)

    axes[1].plot(
        x,
        detection.absolute_derivative,
        color=METHOD_COLORS[method],
        linewidth=1.4,
        label=f"Absolute derivative ({method_label})",
    )
    axes[1].scatter(
        prediction.position,
        prediction.height,
        color=METHOD_COLORS[method],
        s=55,
        zorder=3,
        label=f"PD prediction: {prediction.position} px",
    )
    if signal_present:
        axes[1].axvline(pd50_true, color="black", linestyle="--", linewidth=2, label="True PD50")
    axes[1].set_xlabel("Position (pixels)")
    axes[1].set_ylabel("Absolute first derivative")
    axes[1].grid(alpha=0.25)
    axes[1].legend(ncol=2, fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def save_trial_figures(directory: Path, artifacts: TrialArtifacts) -> None:
    """Save one figure per method so lines and predictions do not overlap."""
    directory.mkdir(parents=True, exist_ok=True)
    filenames = {
        "raw": "01_raw_detection.png",
        "median_gaussian": "02_median_gaussian_detection.png",
        "huber_mrf": "03_huber_mrf_detection.png",
    }
    for method, filename in filenames.items():
        save_method_detection_plot(directory / filename, artifacts, method)


def save_heatmaps(output_dir: Path, summary: pd.DataFrame, config: Phase1Config) -> None:
    """Save method-aligned amplitude/width maps for each Gaussian noise level."""
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(METHOD_LABELS)
    signal_summary = summary[summary["amplitude"] != 0]
    amplitudes = [value for value in config.amplitudes if value != 0]
    widths = list(config.widths)

    for noise_std in config.noise_stds:
        noise_rows = signal_summary[np.isclose(signal_summary["noise_std"], noise_std)]
        for metric, label, fixed_limits in (
            ("correct_detection_rate", "Correct detection rate", (0.0, 1.0)),
            ("mae_px", "MAE (pixels)", None),
        ):
            finite = noise_rows[metric].to_numpy(dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0:
                continue
            vmin, vmax = fixed_limits or (0.0, float(np.max(finite)))
            if np.isclose(vmin, vmax):
                vmax = vmin + 1.0

            figure, axes = plt.subplots(1, len(methods), figsize=(15, 4.5), sharex=True, sharey=True)
            images = []
            for axis, method in zip(np.atleast_1d(axes), methods):
                selected = noise_rows[noise_rows["denoise_method"] == method]
                grid = selected.pivot(index="width", columns="amplitude", values=metric)
                grid = grid.reindex(index=widths, columns=amplitudes)
                image = axis.imshow(grid.to_numpy(dtype=float), origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap="viridis")
                images.append(image)
                axis.set_title(METHOD_LABELS[method])
                axis.set_xticks(range(len(amplitudes)), [f"{value:g}" for value in amplitudes], rotation=45)
                axis.set_yticks(range(len(widths)), [f"{value:g}" for value in widths])
                axis.set_xlabel("Amplitude (Gray Value)")
            np.atleast_1d(axes)[0].set_ylabel("Sigmoid width (pixels)")
            figure.suptitle(f"{label}; Gaussian noise std={noise_std:g}")
            figure.subplots_adjust(left=0.07, right=0.88, bottom=0.2, top=0.82, wspace=0.12)
            colorbar_axis = figure.add_axes([0.91, 0.2, 0.015, 0.62])
            figure.colorbar(images[0], cax=colorbar_axis, label=label)
            safe_noise = f"{noise_std:g}".replace(".", "p")
            figure.savefig(output_dir / f"heatmap_{metric}_noise_{safe_noise}.png", dpi=200)
            plt.close(figure)


def save_metadata(
    path: Path,
    config: Phase1Config,
    mode: str,
    effective_n_trials: int,
) -> None:
    payload = {
        "mode": mode,
        "effective_n_trials": effective_n_trials,
        "config": config.to_dict(),
        "interpretation": "algorithmic_detection_limit_not_optical_measurement_limit",
        "blank_semantics": "amplitude_zero_has_no_true_PD; production_fallback_output_is_counted_as_false_positive",
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def format_tag_number(value: float) -> str:
    """Format a numeric parameter for deterministic output-directory names."""
    return f"{float(value):.12g}".replace("-", "m").replace(".", "p")


def default_output_directory(config: Phase1Config, mode: str, n_trials: int) -> Path:
    base = PACKAGE_DIR / "outputs" / "phase1" / mode
    if mode == "single":
        tag = (
            f"trial_seed_{config.base_seed}"
            f"_amp_{format_tag_number(config.single_amplitude)}"
            f"_width_{format_tag_number(config.single_width)}"
            f"_noise_{format_tag_number(config.single_noise_std)}"
            f"_mrf_lambda_{format_tag_number(config.mrf_regularization)}"
        )
    else:
        tag = (
            f"run_seed_{config.base_seed}_trials_{n_trials}"
            f"_mrf_lambda_{format_tag_number(config.mrf_regularization)}"
        )
    return base / tag


def run_single(config: Phase1Config, output_dir: Path, make_plots: bool) -> None:
    artifacts = run_trial(
        config,
        amplitude=config.single_amplitude,
        width=config.single_width,
        noise_std=config.single_noise_std,
        seed=config.base_seed,
    )
    trials = pd.DataFrame(artifacts.rows)
    summary = summarize_trials(trials)
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in (data_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    trials.to_csv(data_dir / "trial_method_results.csv", index=False)
    summary.to_csv(data_dir / "method_summary.csv", index=False)
    save_trial_profiles(data_dir / "profiles.csv", artifacts)
    if make_plots:
        save_trial_figures(figures_dir, artifacts)
    save_metadata(metadata_dir / "run_metadata.json", config, "single", 1)


def run_sweep(
    config: Phase1Config,
    output_dir: Path,
    n_trials: int,
    make_plots: bool,
) -> None:
    all_rows: list[dict[str, object]] = []
    diagnostic: TrialArtifacts | None = None
    conditions = itertools.product(config.amplitudes, config.widths, config.noise_stds)
    for amplitude, width, noise_std in conditions:
        for trial_index in range(n_trials):
            artifacts = run_trial(
                config,
                amplitude=amplitude,
                width=width,
                noise_std=noise_std,
                seed=config.base_seed + trial_index,
            )
            all_rows.extend(artifacts.rows)
            if diagnostic is None and amplitude != 0:
                diagnostic = artifacts

    trials = pd.DataFrame(all_rows)
    summary = summarize_trials(trials)
    data_dir = output_dir / "data"
    figures_dir = output_dir / "figures"
    metadata_dir = output_dir / "metadata"
    for directory in (data_dir, metadata_dir):
        directory.mkdir(parents=True, exist_ok=True)
    trials.to_csv(data_dir / "trial_method_results.csv", index=False)
    summary.to_csv(data_dir / "condition_method_summary.csv", index=False)
    if make_plots:
        if diagnostic is not None:
            save_trial_profiles(data_dir / "example_signal_trial_profiles.csv", diagnostic)
            save_trial_figures(figures_dir / "example_signal_trial", diagnostic)
        save_heatmaps(figures_dir / "heatmaps", summary, config)
    save_metadata(metadata_dir / "run_metadata.json", config, "sweep", n_trials)


def main() -> None:
    args = parse_arguments()
    config = load_phase1_config(args.config)
    if args.n_trials is not None and args.n_trials < 1:
        raise ValueError("--n-trials must be at least one")
    n_trials = args.n_trials if args.n_trials is not None else config.n_trials
    output_dir = args.output_dir or default_output_directory(config, args.mode, n_trials)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "single":
        run_single(config, output_dir, make_plots=not args.no_plots)
    else:
        run_sweep(config, output_dir, n_trials, make_plots=not args.no_plots)
    print(f"Phase 1 {args.mode} results: {output_dir}")


if __name__ == "__main__":
    main()
