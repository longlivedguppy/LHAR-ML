"""Trial-level and aggregate PD-detection-limit metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


DEFAULT_GROUP_COLUMNS = (
    "phase",
    "amplitude",
    "width",
    "baseline",
    "background_slope",
    "noise_std",
    "measured_noise_id",
    "measured_noise_scale",
    "denoise_method",
    "mrf_regularization",
)


def calculate_trial_error(
    pd50_true: float,
    pd50_estimated: float | None,
    signal_present: bool,
    tolerance_px: float,
) -> dict[str, object]:
    """Calculate truth error while keeping blank trials semantically separate."""
    detected = pd50_estimated is not None and np.isfinite(pd50_estimated)
    if not signal_present:
        return {
            "detected": bool(detected),
            "signed_error_px": np.nan,
            "absolute_error_px": np.nan,
            "correct_detection": False,
            "false_positive": bool(detected),
        }
    if not detected:
        return {
            "detected": False,
            "signed_error_px": np.nan,
            "absolute_error_px": np.nan,
            "correct_detection": False,
            "false_positive": False,
        }

    signed_error = float(pd50_estimated - pd50_true)
    absolute_error = abs(signed_error)
    return {
        "detected": True,
        "signed_error_px": signed_error,
        "absolute_error_px": absolute_error,
        "correct_detection": bool(absolute_error <= tolerance_px),
        "false_positive": False,
    }


def summarize_trials(
    trials: pd.DataFrame,
    group_columns: Sequence[str] = DEFAULT_GROUP_COLUMNS,
) -> pd.DataFrame:
    """Aggregate Monte Carlo trials into accuracy and false-positive metrics."""
    if trials.empty:
        raise ValueError("trials must not be empty")
    missing = set(group_columns).difference(trials.columns)
    if missing:
        raise ValueError(f"missing grouping columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    grouped = trials.groupby(list(group_columns), dropna=False, sort=True)
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        result = dict(zip(group_columns, keys))
        signal_present = bool(group["signal_present"].iloc[0])
        estimated = group.loc[group["detected"], "pd50_estimated"].astype(float)
        result.update(
            {
                "n_trials": int(len(group)),
                "n_detected": int(group["detected"].sum()),
                "mae_px": (
                    float(group["absolute_error_px"].mean()) if signal_present else np.nan
                ),
                "signed_bias_px": (
                    float(group["signed_error_px"].mean()) if signal_present else np.nan
                ),
                "std_estimated_pd_px": (
                    float(np.std(estimated.to_numpy(), ddof=0))
                    if signal_present and not estimated.empty
                    else np.nan
                ),
                "correct_detection_rate": (
                    float(group["correct_detection"].mean()) if signal_present else np.nan
                ),
                "false_positive_rate": (
                    float(group["false_positive"].mean()) if not signal_present else np.nan
                ),
                "mean_max_peak_height": float(group["max_peak_height"].mean()),
                "argmax_fallback_rate": float(group["detection_used_argmax_fallback"].mean()),
            }
        )
        rows.append(result)
    return pd.DataFrame(rows)
