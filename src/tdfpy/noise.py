"""
Noise estimation utilities for TDF data processing.
"""

from typing import Literal

import numpy as np


def _estimate_noise_mad(intensity_array: np.ndarray) -> float:
    """Estimate noise using Median Absolute Deviation."""
    median = np.median(intensity_array)
    mad = np.median(np.abs(intensity_array - median))
    # Noise threshold: median + k * MAD (k=3 to 5 is typical)
    return float(median + 3 * 1.4826 * mad)  # 1.4826 makes MAD consistent with std


def _estimate_noise_histogram(intensity_array: np.ndarray) -> float:
    """Estimate noise using histogram mode."""
    hist, bin_edges = np.histogram(intensity_array, bins=100)
    noise_bin_idx = int(np.argmax(hist))
    noise_mode = (bin_edges[noise_bin_idx] + bin_edges[noise_bin_idx + 1]) / 2

    # Estimate std of noise from histogram width
    half_max = hist[noise_bin_idx] / 2
    left_idx = noise_bin_idx
    while left_idx > 0 and hist[left_idx] > half_max:
        left_idx -= 1
    right_idx = noise_bin_idx
    while right_idx < len(hist) - 1 and hist[right_idx] > half_max:
        right_idx += 1

    noise_std = (bin_edges[right_idx] - bin_edges[left_idx]) / 2.355  # FWHM to std
    return float(noise_mode + 3 * noise_std)


def _estimate_noise_baseline(intensity_array: np.ndarray) -> float:
    """Estimate noise using bottom quantile statistics."""
    bottom_25 = np.percentile(intensity_array, 25)
    noise_intensities = intensity_array[intensity_array <= bottom_25]
    noise_mean = np.mean(noise_intensities)
    noise_std = np.std(noise_intensities)
    return float(noise_mean + 3 * noise_std)


def _estimate_noise_iterative_median(intensity_array: np.ndarray) -> float:
    """Estimate noise using iterative median filtering."""
    current = intensity_array.copy()
    for _ in range(3):
        median = np.median(current)
        mad = np.median(np.abs(current - median))
        threshold = median + 2 * 1.4826 * mad
        current = current[current <= threshold]
        if len(current) < 100:  # Safety check
            break
    return float(np.median(current) + 3 * np.std(current))


def estimate_noise_level(
    intensity_array: np.ndarray,
    method: Literal["mad", "percentile", "histogram", "baseline", "iterative_median"] | float = "mad",
) -> float:
    """
    Estimate noise level using various methods.

    Parameters:
    -----------
    intensity_array : np.ndarray
        Array of intensity values
    method : str or float or int
        Method to use: one of 'mad', 'percentile', 'histogram', 'baseline', 'iterative_median',
        or a numeric value (float/int) to be used directly as the noise level.

    Returns:
    --------
    float : Estimated noise level threshold
    """
    # If a numeric value is provided, use it directly as the noise level.
    if isinstance(method, (int, float)):
        return float(method)
    if method == "mad":
        return _estimate_noise_mad(intensity_array)
    if method == "percentile":
        return float(np.percentile(intensity_array, 75))
    if method == "histogram":
        return _estimate_noise_histogram(intensity_array)
    if method == "baseline":
        return _estimate_noise_baseline(intensity_array)
    if method == "iterative_median":
        return _estimate_noise_iterative_median(intensity_array)
    raise ValueError(f"Unknown noise estimation method: {method}")
