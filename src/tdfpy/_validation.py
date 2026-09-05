"""Shared validation before array operations and JIT kernels."""

from numbers import Integral, Real

import numpy as np


def nonnegative(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be finite and nonnegative.")


def integer(name: str, value: int, minimum: int | None = 0) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, Integral)
        or (minimum is not None and value < minimum)
    ):
        raise ValueError(
            f"{name} must be an integer"
            + (f" >= {minimum}." if minimum is not None else ".")
        )


def choice(name: str, value: str, allowed: tuple[str, ...]) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}.")


def arrays(*values: np.ndarray) -> None:
    if any(not isinstance(a, np.ndarray) or a.ndim != 1 for a in values):
        raise ValueError("Peak arrays must be one-dimensional NumPy arrays.")
    if len({a.size for a in values}) != 1:
        raise ValueError("Peak arrays must have equal lengths.")
    if any(not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a) for a in values):
        raise ValueError("Peak arrays must contain real numbers.")
    if any(not np.all(np.isfinite(a)) for a in values):
        raise ValueError("Peak arrays must contain finite values.")


def index_arrays(
    scan: np.ndarray, tof: np.ndarray, intensity: np.ndarray, num_scans: int
) -> None:
    arrays(scan, tof, intensity)
    integer("num_scans", num_scans)
    if not np.issubdtype(scan.dtype, np.integer) or not np.issubdtype(
        tof.dtype, np.integer
    ):
        raise ValueError("Scan and TOF indices must have integer dtypes.")
    if (
        np.any(scan < 0)
        or np.any(scan >= num_scans)
        or np.any(tof < 0)
        or np.any(intensity < 0)
    ):
        raise ValueError("Peak indices or intensities are outside their valid range.")


def bounds(name: str, value: tuple[float, float] | None) -> None:
    if value is None:
        return
    if len(value) != 2:
        raise ValueError(f"{name} must contain two bounds.")
    if not np.all(np.isfinite(value)) or value[0] < 0 or value[1] < value[0]:
        raise ValueError(f"{name} must have finite, nonnegative, ordered bounds.")


def merge_config(
    mz_tolerance: float,
    mz_tolerance_type: str,
    im_tolerance: float,
    im_tolerance_type: str,
    min_peaks: int,
    max_peaks: int | None,
    peak_noise_window: float,
    peak_noise_end_fraction: float,
) -> None:
    choice("mz_tolerance_type", mz_tolerance_type, ("ppm", "da"))
    choice("im_tolerance_type", im_tolerance_type, ("relative", "absolute"))
    for name, value in (
        ("mz_tolerance", mz_tolerance),
        ("im_tolerance", im_tolerance),
        ("peak_noise_window", peak_noise_window),
        ("peak_noise_end_fraction", peak_noise_end_fraction),
    ):
        nonnegative(name, value)
    if mz_tolerance_type == "ppm" and mz_tolerance > 1e6:
        raise ValueError("mz_tolerance in ppm must not exceed 1,000,000.")
    if peak_noise_end_fraction > 1:
        raise ValueError("peak_noise_end_fraction must not exceed 1.")
    integer("min_peaks", min_peaks)
    if max_peaks is not None:
        integer("max_peaks", max_peaks, minimum=None)
