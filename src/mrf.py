"""Edge-preserving one-dimensional Huber MRF denoising.

This module is intentionally independent from the production LHAR pipeline.
It contains no image loading, plotting, peak detection, or file output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustScale:
    """Parameters used to map a profile to a robust, dimensionless scale."""

    center: float
    scale: float


@dataclass(frozen=True)
class MRFResult:
    """Huber MRF result and diagnostics."""

    profile: np.ndarray
    normalized_profile: np.ndarray
    robust_scale: RobustScale
    huber_delta: float
    regularization: float
    iterations: int
    converged: bool
    relative_change: float


def robust_normalize(profile: np.ndarray, eps: float = 1e-12) -> tuple[np.ndarray, RobustScale]:
    """Normalize using the median and the 5--95 percentile range."""
    values = _as_profile(profile)
    center = float(np.median(values))
    scale = float(np.percentile(values, 95) - np.percentile(values, 5))
    if not np.isfinite(scale) or scale <= eps:
        mad = float(np.median(np.abs(values - center)))
        scale = max(1.4826 * mad, eps)
    return (values - center) / scale, RobustScale(center=center, scale=scale)


def estimate_huber_delta(normalized_profile: np.ndarray, minimum: float = 1e-4) -> float:
    """Estimate Huber's transition from the MAD of adjacent differences.

    The factor 1.345 is the conventional robust-efficiency choice. A positive
    floor keeps exactly flat or quantized profiles numerically well-defined.
    """
    values = _as_profile(normalized_profile)
    differences = np.diff(values)
    if differences.size == 0:
        return minimum
    median_difference = float(np.median(differences))
    mad = float(np.median(np.abs(differences - median_difference)))
    noise_sigma = 1.4826 * mad / np.sqrt(2.0)
    return max(1.345 * noise_sigma, minimum)


def huber_mrf_denoise(
    profile: np.ndarray,
    regularization: float,
    huber_delta: float | None = None,
    max_iterations: int = 300,
    tolerance: float = 1e-7,
    weight_epsilon: float = 1e-8,
) -> MRFResult:
    """Minimize a quadratic data term plus Huber first-difference penalty.

    The convex objective is solved by iteratively reweighted least squares.
    Every iteration solves a symmetric tridiagonal system in O(n). Input is
    robustly normalized internally and returned on its original intensity scale.
    """
    if regularization < 0:
        raise ValueError("regularization must be non-negative")
    if max_iterations < 1:
        raise ValueError("max_iterations must be at least one")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")

    normalized, scale = robust_normalize(profile)
    delta = estimate_huber_delta(normalized) if huber_delta is None else float(huber_delta)
    if not np.isfinite(delta) or delta <= 0:
        raise ValueError("huber_delta must be finite and positive")

    if normalized.size < 2 or regularization == 0:
        restored = normalized * scale.scale + scale.center
        return MRFResult(restored, normalized.copy(), scale, delta, regularization, 0, True, 0.0)

    estimate = normalized.copy()
    relative_change = np.inf
    converged = False

    for iteration in range(1, max_iterations + 1):
        differences = np.diff(estimate)
        weights = np.minimum(1.0, delta / np.maximum(np.abs(differences), weight_epsilon))
        edge_weights = regularization * weights

        diagonal = np.ones(estimate.size, dtype=float)
        diagonal[:-1] += edge_weights
        diagonal[1:] += edge_weights
        off_diagonal = -edge_weights
        updated = _solve_symmetric_tridiagonal(diagonal, off_diagonal, normalized)

        denominator = max(float(np.linalg.norm(estimate)), weight_epsilon)
        relative_change = float(np.linalg.norm(updated - estimate) / denominator)
        estimate = updated
        if relative_change <= tolerance:
            converged = True
            break

    restored = estimate * scale.scale + scale.center
    return MRFResult(
        profile=restored,
        normalized_profile=estimate,
        robust_scale=scale,
        huber_delta=delta,
        regularization=regularization,
        iterations=iteration,
        converged=converged,
        relative_change=relative_change,
    )


def _as_profile(profile: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=float)
    if values.ndim != 1:
        raise ValueError("profile must be one-dimensional")
    if values.size == 0:
        raise ValueError("profile must not be empty")
    if not np.all(np.isfinite(values)):
        raise ValueError("profile contains NaN or infinite values")
    return values


def _solve_symmetric_tridiagonal(diagonal: np.ndarray, off_diagonal: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve a positive-definite tridiagonal system using Thomas' method."""
    diagonal = np.asarray(diagonal, dtype=float).copy()
    upper = np.asarray(off_diagonal, dtype=float).copy()
    lower = upper.copy()
    solution = np.asarray(rhs, dtype=float).copy()

    for index in range(1, diagonal.size):
        factor = lower[index - 1] / diagonal[index - 1]
        diagonal[index] -= factor * upper[index - 1]
        solution[index] -= factor * solution[index - 1]

    solution[-1] /= diagonal[-1]
    for index in range(diagonal.size - 2, -1, -1):
        solution[index] = (solution[index] - upper[index] * solution[index + 1]) / diagonal[index]
    return solution
