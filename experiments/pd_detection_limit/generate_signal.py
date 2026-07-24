"""Ground-truth synthetic 1D PD-profile generation only."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit


@dataclass(frozen=True)
class SyntheticProfile:
    """Clean truth, supplied noise, and their observed sum."""

    x: np.ndarray
    sigmoid: np.ndarray
    clean_profile: np.ndarray
    noise: np.ndarray
    noisy_profile: np.ndarray


def generate_synthetic_profile(
    *,
    length: int,
    baseline: float,
    amplitude: float,
    pd50_true: float,
    width: float,
    background_slope: float = 0.0,
    noise: np.ndarray | None = None,
) -> SyntheticProfile:
    """Generate baseline + linear background + sigmoid + supplied noise.

    The sign of ``amplitude`` naturally permits a future falling front. This
    function does not denoise, differentiate, or detect a PD candidate.
    """
    if length < 1:
        raise ValueError("length must be positive")
    if width <= 0 or not np.isfinite(width):
        raise ValueError("width must be a positive finite value")
    if not 0 <= pd50_true <= length - 1:
        raise ValueError("pd50_true must lie inside the profile")

    x = np.arange(length, dtype=float)
    sigmoid = expit((x - pd50_true) / width)
    clean_profile = baseline + amplitude * sigmoid + background_slope * x

    if noise is None:
        noise_values = np.zeros(length, dtype=float)
    else:
        noise_values = np.asarray(noise, dtype=float)
        if noise_values.shape != (length,):
            raise ValueError(f"noise must have shape ({length},)")
        if not np.all(np.isfinite(noise_values)):
            raise ValueError("noise contains NaN or infinite values")

    return SyntheticProfile(
        x=x,
        sigmoid=sigmoid,
        clean_profile=clean_profile,
        noise=noise_values,
        noisy_profile=clean_profile + noise_values,
    )
