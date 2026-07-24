"""Noise inputs for synthetic PD profiles.

Only Gaussian noise is generated in Phase 1. Measured profiles must be loaded
from supplied data; this module intentionally has no measured-noise simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def gaussian_random_noise(
    length: int,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw zero-mean Gaussian noise from the supplied reproducible generator."""
    if length < 1:
        raise ValueError("length must be positive")
    if noise_std < 0:
        raise ValueError("noise_std must be non-negative")
    return rng.normal(loc=0.0, scale=noise_std, size=length)


def load_measured_noise_profile(path: Path, expected_length: int | None = None) -> np.ndarray:
    """Load a future Phase 3/4 measured profile from ``.npy`` or one-column CSV.

    The values are returned as measured. Extraction, detrending, or a claim
    that they are pure noise belongs to the later Phase 3 design.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        values = np.load(path)
    elif suffix == ".csv":
        values = np.loadtxt(path, delimiter=",", ndmin=1)
    else:
        raise ValueError("measured profiles must be .npy or one-column .csv files")

    profile = np.asarray(values, dtype=float).squeeze()
    if profile.ndim != 1 or profile.size == 0:
        raise ValueError("measured profile must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(profile)):
        raise ValueError("measured profile contains NaN or infinite values")
    if expected_length is not None and profile.size != expected_length:
        raise ValueError(
            f"measured profile has length {profile.size}, expected {expected_length}"
        )
    return profile
