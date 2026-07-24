"""Configuration loading and validation for the Phase 1 benchmark."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Phase1Config:
    """All parameters needed for a reproducible Phase 1 run."""

    phase: int
    profile_length: int
    pd50_true: float
    baseline: float
    background_slope: float
    single_amplitude: float
    single_width: float
    single_noise_std: float
    amplitudes: tuple[float, ...]
    widths: tuple[float, ...]
    noise_stds: tuple[float, ...]
    n_trials: int
    base_seed: int
    tolerance_px: float
    start_distance: int
    peak_distance: int
    median_kernel: int
    gaussian_sigma: float
    mrf_regularization: float

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "Phase1Config":
        converted = dict(values)
        for key in ("amplitudes", "widths", "noise_stds"):
            converted[key] = tuple(float(value) for value in converted[key])
        config = cls(**converted)
        config.validate()
        return config

    def validate(self) -> None:
        if self.phase != 1:
            raise ValueError("this runner currently supports phase=1 only")
        if self.profile_length < 3:
            raise ValueError("profile_length must be at least three")
        if not math.isfinite(self.pd50_true) or not 0 <= self.pd50_true <= self.profile_length - 1:
            raise ValueError("pd50_true must lie inside the profile")
        if not math.isfinite(self.baseline):
            raise ValueError("baseline must be finite")
        if not math.isfinite(self.background_slope) or self.background_slope != 0:
            raise ValueError("Phase 1 requires background_slope=0")
        if not math.isfinite(self.single_amplitude):
            raise ValueError("single_amplitude must be finite")
        if any(not math.isfinite(amplitude) for amplitude in self.amplitudes):
            raise ValueError("all amplitudes must be finite")
        if (
            not math.isfinite(self.single_width)
            or self.single_width <= 0
            or not self.widths
            or any(not math.isfinite(width) or width <= 0 for width in self.widths)
        ):
            raise ValueError("all sigmoid widths must be positive")
        if (
            not math.isfinite(self.single_noise_std)
            or self.single_noise_std < 0
            or not self.noise_stds
            or any(not math.isfinite(value) or value < 0 for value in self.noise_stds)
        ):
            raise ValueError("all noise standard deviations must be non-negative")
        if not self.amplitudes:
            raise ValueError("amplitudes must not be empty")
        if self.n_trials < 1:
            raise ValueError("n_trials must be at least one")
        if not math.isfinite(self.tolerance_px) or self.tolerance_px < 0:
            raise ValueError("tolerance_px must be non-negative")
        if not 0 <= self.start_distance < self.profile_length:
            raise ValueError("start_distance must lie inside the profile")
        if self.peak_distance < 1:
            raise ValueError("peak_distance must be at least one")
        if self.median_kernel < 1:
            raise ValueError("median_kernel must be positive")
        if not math.isfinite(self.gaussian_sigma) or self.gaussian_sigma <= 0:
            raise ValueError("gaussian_sigma must be positive")
        if not math.isfinite(self.mrf_regularization) or self.mrf_regularization < 0:
            raise ValueError("mrf_regularization must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_phase1_config(path: Path) -> Phase1Config:
    """Load a JSON config without introducing a YAML dependency."""
    with path.open(encoding="utf-8") as handle:
        values = json.load(handle)
    return Phase1Config.from_mapping(values)
