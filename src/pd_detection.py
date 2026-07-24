"""Reusable form of the current production 1D PD-candidate detector.

The detector deliberately preserves the behavior in ``main.py``. In
particular, it always returns a primary position: when ``find_peaks`` finds no
local maximum, the numerical maximum of the assessed absolute derivative is
used as a fallback. That behavior is important for blank-test false positives.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks


@dataclass(frozen=True)
class PeakCandidate:
    """One derivative-peak candidate."""

    position: int
    height: float


@dataclass(frozen=True)
class PDDetectionResult:
    """Derivative arrays and the two candidates retained by production."""

    derivative: np.ndarray
    absolute_derivative: np.ndarray
    primary: PeakCandidate
    secondary: PeakCandidate
    used_argmax_fallback: bool


def detect_pd_candidate(
    profile: np.ndarray,
    start_distance: int = 50,
    peak_distance: int = 15,
) -> PDDetectionResult:
    """Run the current gradient/absolute-gradient/peak-selection procedure.

    Peaks are ranked by height. If only one peak exists, the secondary
    candidate repeats its position with zero height, matching ``main.py``.
    If none exists, ``argmax`` supplies the primary candidate.
    """
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("profile must be a one-dimensional array with at least two values")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile contains NaN or infinite values")
    if start_distance < 0:
        raise ValueError("start_distance must be non-negative")
    if peak_distance < 1:
        raise ValueError("peak_distance must be at least one")

    derivative = np.gradient(values)
    absolute_derivative = np.abs(derivative)
    assessed = absolute_derivative.copy()
    assessed[:start_distance] = 0

    peaks, properties = find_peaks(assessed, height=0, distance=peak_distance)
    if len(peaks) >= 2:
        ranked = np.argsort(properties["peak_heights"])[::-1]
        primary_index = int(ranked[0])
        secondary_index = int(ranked[1])
        primary = PeakCandidate(
            position=int(peaks[primary_index]),
            height=float(properties["peak_heights"][primary_index]),
        )
        secondary = PeakCandidate(
            position=int(peaks[secondary_index]),
            height=float(properties["peak_heights"][secondary_index]),
        )
        used_fallback = False
    elif len(peaks) == 1:
        position = int(peaks[0])
        primary = PeakCandidate(position, float(properties["peak_heights"][0]))
        secondary = PeakCandidate(position, 0.0)
        used_fallback = False
    else:
        position = int(np.argmax(assessed))
        primary = PeakCandidate(position, float(absolute_derivative[position]))
        secondary = PeakCandidate(position, 0.0)
        used_fallback = True

    return PDDetectionResult(
        derivative=derivative,
        absolute_derivative=absolute_derivative,
        primary=primary,
        secondary=secondary,
        used_argmax_fallback=used_fallback,
    )
