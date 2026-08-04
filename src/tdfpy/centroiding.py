"""
Higher-level Pythonic API for working with MS1 spectrum data from Bruker timsTOF files.

This module provides a cleaner interface using NamedTuples and convenience functions
for reading centroided MS1 spectra with peak clustering/centroiding algorithms.
"""

import logging
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd  # type: ignore

from .noise import NoiseSpec, coerce_filters
from .pipeline import (
    Centroider,
    MergePeaksCentroider,
    Smooth,
    apply_noise,
    convert,
    exclude_region,
    read_spectrum,
    subset_scans,
)
from .regions import ChargeStateRegion
from .tdf import PandasTdf
from .timsdata import TimsData

# Try to import Numba for JIT-accelerated implementation
try:
    from numba import njit as _njit  # ty: ignore[unresolved-import]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

logger = logging.getLogger(__name__)


def batch_iterator(input_list: list[Any], batch_size: int):
    for i in range(0, len(input_list), batch_size):
        yield input_list[i : i + batch_size]


class Peak(NamedTuple):
    """Represents a single mass spec peak.

    Attributes:
        mz: Mass-to-charge ratio
        intensity: Peak intensity (area)
        ion_mobility: Ion mobility value - either 1/K0 (reciprocal reduced mobility)
                     or CCS (collision cross section in Ų) depending on the
                     ion_mobility_type parameter used during extraction
    """

    mz: float
    intensity: float
    ion_mobility: float


if _HAS_NUMBA:
    @_njit(cache=True)
    def _merge_peaks_numba_kernel(
        mz_sorted, intensity_sorted, im_sorted, intensity_order,
        mz_tol_factor, mz_tol_abs, mob_tol_factor, mob_tol_abs,
        mz_is_ppm, im_is_relative, min_peaks, max_peaks,
        peak_noise_filter, peak_noise_window, peak_noise_end_fraction,
    ):
        n = len(mz_sorted)
        out_mz = np.empty(n, dtype=np.float64)
        out_intensity = np.empty(n, dtype=np.float64)
        out_im = np.empty(n, dtype=np.float64)
        used = np.zeros(n, dtype=np.bool_)
        count = 0
        noise_inv_window = 1.0 / peak_noise_window if peak_noise_window > 0.0 else 0.0
        noise_one_minus_end = 1.0 - peak_noise_end_fraction
        for order_idx in range(len(intensity_order)):
            peak_idx = intensity_order[order_idx]
            if used[peak_idx]:
                continue
            mz_peak = mz_sorted[peak_idx]
            im_peak = im_sorted[peak_idx]
            if mz_is_ppm:
                mz_tol = mz_peak * mz_tol_factor
            else:
                mz_tol = mz_tol_abs
            left_mz = mz_peak - mz_tol
            right_mz = mz_peak + mz_tol
            left_idx = np.searchsorted(mz_sorted, left_mz)
            right_idx = np.searchsorted(mz_sorted, right_mz, side='right')

            # Dynamic IM region growing: start at seed, expand bounds outward
            # one step at a time until no unused peak is within im_tolerance of
            # the current lower or upper boundary.
            # eps guards against floating-point edge cases where the gap is
            # mathematically equal to the tolerance but represented as slightly
            # larger (e.g. 1.01 - 1.0 > 0.01 in IEEE 754).
            im_lo = im_peak
            im_hi = im_peak
            changed = True
            while changed:
                changed = False
                for i in range(left_idx, right_idx):
                    if used[i]:
                        continue
                    im_i = im_sorted[i]
                    if im_i < im_lo:
                        expand = im_lo * mob_tol_factor if im_is_relative else mob_tol_abs
                        if im_lo - im_i <= expand * 1.0000001:
                            im_lo = im_i
                            changed = True
                    elif im_i > im_hi:
                        expand = im_hi * mob_tol_factor if im_is_relative else mob_tol_abs
                        if im_i - im_hi <= expand * 1.0000001:
                            im_hi = im_i
                            changed = True

            # Collect all unused peaks within the grown IM window
            total_int = 0.0
            weighted_mz = 0.0
            weighted_im = 0.0
            num_nearby = 0
            for i in range(left_idx, right_idx):
                if not used[i] and im_lo <= im_sorted[i] <= im_hi:
                    w = intensity_sorted[i]
                    total_int += w
                    weighted_mz += mz_sorted[i] * w
                    weighted_im += im_sorted[i] * w
                    num_nearby += 1
            if min_peaks > 0 and num_nearby < min_peaks:
                used[peak_idx] = True
                continue
            for i in range(left_idx, right_idx):
                if not used[i] and im_lo <= im_sorted[i] <= im_hi:
                    used[i] = True
            if total_int > 0.0:
                out_mz[count] = weighted_mz / total_int
                out_intensity[count] = total_int
                out_im[count] = weighted_im / total_int
            else:
                out_mz[count] = mz_peak
                out_intensity[count] = intensity_sorted[peak_idx]
                out_im[count] = im_peak
            count += 1

            # Peak-satellite noise filter: within ±peak_noise_window Da of the
            # anchor m/z and inside the centroid's IM window, suppress raw
            # points whose intensity falls below a linear ramp that decays from
            # the anchor's raw intensity at d=0 to (anchor * end_fraction) at
            # d=window. Marking them used prevents them from seeding their own
            # centroids. Real peaks above the ramp survive.
            if peak_noise_filter and peak_noise_window > 0.0:
                anchor_int = intensity_sorted[peak_idx]
                noise_left_mz = mz_peak - peak_noise_window
                noise_right_mz = mz_peak + peak_noise_window
                noise_left_idx = np.searchsorted(mz_sorted, noise_left_mz)
                noise_right_idx = np.searchsorted(mz_sorted, noise_right_mz, side='right')
                for i in range(noise_left_idx, noise_right_idx):
                    if used[i]:
                        continue
                    im_i = im_sorted[i]
                    if im_i < im_lo or im_i > im_hi:
                        continue
                    d = mz_sorted[i] - mz_peak
                    if d < 0.0:
                        d = -d
                    threshold = anchor_int * (1.0 - d * noise_inv_window * noise_one_minus_end)
                    if intensity_sorted[i] < threshold:
                        used[i] = True

            if max_peaks != -1 and count >= max_peaks:
                break
        return out_mz[:count], out_intensity[:count], out_im[:count]


def _merge_peaks_numba(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.1,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    peak_noise_filter: bool = False,
    peak_noise_window: float = 0.1,
    peak_noise_end_fraction: float = 0.1,
) -> np.ndarray:
    """Numba JIT-accelerated implementation of merge_peaks."""
    if len(mz_array) == 0:
        return np.empty((0, 3), dtype=np.float64)
    mz_is_ppm = 1 if mz_tolerance_type == "ppm" else 0
    im_is_relative = 1 if im_tolerance_type == "relative" else 0
    mz_tol_factor = mz_tolerance / 1e6 if mz_is_ppm else 0.0
    mz_tol_abs = 0.0 if mz_is_ppm else mz_tolerance
    mob_tol_factor = im_tolerance if im_is_relative else 0.0
    mob_tol_abs = 0.0 if im_is_relative else im_tolerance
    # A non-positive (or None) max_peaks means "no limit" — must match the
    # pure-Python kernel, which treats a falsy max_peaks as unlimited.
    _max_peaks = -1 if (max_peaks is None or max_peaks <= 0) else int(max_peaks)
    sort_idx = np.argsort(mz_array)
    mz_s = np.ascontiguousarray(mz_array[sort_idx], dtype=np.float64)
    int_s = np.ascontiguousarray(intensity_array[sort_idx], dtype=np.float64)
    im_s = np.ascontiguousarray(ion_mobility_array[sort_idx], dtype=np.float64)
    intensity_order = np.ascontiguousarray(np.argsort(int_s)[::-1].astype(np.int64))
    out_mz, out_int, out_im = _merge_peaks_numba_kernel(
        mz_s, int_s, im_s, intensity_order,
        mz_tol_factor, mz_tol_abs, mob_tol_factor, mob_tol_abs,
        mz_is_ppm, im_is_relative, min_peaks, _max_peaks,
        1 if peak_noise_filter else 0,
        float(peak_noise_window),
        float(peak_noise_end_fraction),
    )
    return np.column_stack([out_mz, out_int, out_im])


def merge_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.1,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    peak_noise_filter: bool = False,
    peak_noise_window: float = 0.1,
    peak_noise_end_fraction: float = 0.1,
    use_numba: bool = True,
) -> np.ndarray:
    """Centroid profile-like peaks using m/z and ion mobility tolerances.

    This function implements a greedy clustering algorithm that centroids raw peaks
    (similar to profile mode data) within specified m/z and ion mobility windows.
    Peaks are processed in descending order of intensity, and nearby peaks are
    combined using intensity-weighted averaging to produce centroided peaks.

    Args:
        mz_array: Array of m/z values from raw/profile-like data
        intensity_array: Array of intensity values
        ion_mobility_array: Array of ion mobility values (1/K0 or CCS)
        mz_tolerance: Tolerance for m/z matching during centroiding
        mz_tolerance_type: Type of m/z tolerance - "ppm" or "da" (daltons)
        im_tolerance: Tolerance for ion mobility matching during centroiding
        im_tolerance_type: Type of ion mobility tolerance - "relative" or "absolute"
        min_peaks: Minimum number of nearby raw peaks required to form a centroid.
                  Set to 0 or 1 to keep all peaks (no filtering).
        max_peaks: Maximum number of centroided peaks to return (keeps highest intensity)
        peak_noise_filter: If True, after each centroid is formed suppress raw
            points within ±``peak_noise_window`` Da of the anchor m/z and inside
            the centroid's IM window whose intensity falls below a linear
            threshold that decays from the **anchor point's raw intensity** at
            zero distance to ``anchor * peak_noise_end_fraction`` at the window
            edge. This kills TOF satellite/ringing noise around bright peaks
            without eliminating nearby real peaks that exceed the ramp.
            Comparison is point-to-point against the raw anchor intensity (not
            the summed centroid). Defaults to ``False``.
        peak_noise_window: Half-width in Da on each side of the anchor m/z over
            which the peak-noise ramp is applied. Defaults to ``0.1`` Da.
        peak_noise_end_fraction: Fraction of the anchor's raw intensity used as
            the suppression threshold at ``peak_noise_window`` distance.
            Defaults to ``0.1`` (10%).

    Returns:
        np.ndarray: Array of shape (N, 3) containing centroided peaks.
                   Columns are: [mz, intensity, ion_mobility]

    Example:
        ```python
        mz = np.array([100.0, 100.001, 200.0])
        intensity = np.array([1000.0, 500.0, 2000.0])
        im = np.array([0.8, 0.8, 0.9])
        peaks = merge_peaks(mz, intensity, im, mz_tolerance=10, mz_tolerance_type="ppm")
        ```
    """
    # Use Numba implementation if available
    if _HAS_NUMBA and use_numba:
        return _merge_peaks_numba(
            mz_array, intensity_array, ion_mobility_array,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            im_tolerance=im_tolerance,
            im_tolerance_type=im_tolerance_type,
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            peak_noise_filter=peak_noise_filter,
            peak_noise_window=peak_noise_window,
            peak_noise_end_fraction=peak_noise_end_fraction,
        )

    # Fallback to Python implementation
    return _merge_peaks_python(
        mz_array,
        intensity_array,
        ion_mobility_array,
        mz_tolerance,
        mz_tolerance_type,
        im_tolerance,
        im_tolerance_type,
        min_peaks,
        max_peaks,
        peak_noise_filter,
        peak_noise_window,
        peak_noise_end_fraction,
    )


def _merge_peaks_python(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.1,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    peak_noise_filter: bool = False,
    peak_noise_window: float = 0.1,
    peak_noise_end_fraction: float = 0.1,
) -> np.ndarray:
    """Python implementation of merge_peaks (fallback when Numba unavailable)."""
    logger.debug(
        "Centroiding %d raw peaks with mz_tol=%s %s, im_tol=%s %s, min_peaks=%d, max_peaks=%s",
        len(mz_array),
        mz_tolerance,
        mz_tolerance_type,
        im_tolerance,
        im_tolerance_type,
        min_peaks,
        max_peaks,
    )

    if len(mz_array) == 0:
        logger.debug("No raw peaks to centroid, returning empty array")
        return np.empty((0, 3), dtype=np.float64)

    # Pre-compute tolerances
    if mz_tolerance_type == "ppm":
        mz_tol_factor = mz_tolerance / 1e6
        mz_tol_abs = 0.0
    else:
        mz_tol_abs = mz_tolerance
        mz_tol_factor = 0.0

    if im_tolerance_type == "relative":
        mobility_tol_factor = im_tolerance
        mobility_tol_abs = 0.0
    else:
        mobility_tol_abs = im_tolerance
        mobility_tol_factor = 0.0

    # Sort by mz for binary search
    sort_idx = np.argsort(mz_array)
    mz_array = mz_array[sort_idx]
    intensity_array = intensity_array[sort_idx]
    ion_mobility_array = ion_mobility_array[sort_idx]
    logger.debug("Sorted %d peaks by m/z", len(mz_array))

    # Sort by intensity for greedy clustering
    intensity_order = np.argsort(intensity_array)[::-1]
    logger.debug("Created intensity-ordered index for greedy clustering")

    # Use boolean mask for tracking used peaks
    used_mask = np.zeros(len(mz_array), dtype=bool)
    merged_mz_list: list[float] = []
    merged_int_list: list[float] = []
    merged_mob_list: list[float] = []

    for peak_idx in intensity_order:
        if used_mask[peak_idx]:
            continue

        # Extract values (avoid redundant float conversions)
        mz_peak = mz_array[peak_idx]
        intensity_peak = intensity_array[peak_idx]
        mobility_peak = ion_mobility_array[peak_idx]

        # Calculate tolerances
        mz_tol = mz_peak * mz_tol_factor if mz_tolerance_type == "ppm" else mz_tol_abs

        # Binary search for mz range
        left_mz = mz_peak - mz_tol
        right_mz = mz_peak + mz_tol
        left_idx = int(np.searchsorted(mz_array, left_mz, side="left"))
        right_idx = int(np.searchsorted(mz_array, right_mz, side="right"))

        # Slice the mz window
        mobility_window = ion_mobility_array[left_idx:right_idx]
        intensity_window = intensity_array[left_idx:right_idx]
        mz_window = mz_array[left_idx:right_idx]
        used_window = used_mask[left_idx:right_idx]

        # Dynamic IM region growing: start bounds at seed, expand outward until
        # no unused peak is within im_tolerance of the current boundary.
        # The 1.0000001 factor guards against floating-point edge cases where the
        # gap is mathematically equal to the tolerance but represented as slightly
        # larger (e.g. 1.01 - 1.0 > 0.01 in IEEE 754).
        im_lo = float(mobility_peak)
        im_hi = float(mobility_peak)
        changed = True
        while changed:
            changed = False
            for i in range(len(mobility_window)):
                if used_window[i]:
                    continue
                im_i = float(mobility_window[i])
                if im_i < im_lo:
                    expand = im_lo * mobility_tol_factor if im_tolerance_type == "relative" else mobility_tol_abs
                    if im_lo - im_i <= expand * 1.0000001:
                        im_lo = im_i
                        changed = True
                elif im_i > im_hi:
                    expand = im_hi * mobility_tol_factor if im_tolerance_type == "relative" else mobility_tol_abs
                    if im_i - im_hi <= expand * 1.0000001:
                        im_hi = im_i
                        changed = True

        nearby_mask = (
            (mobility_window >= im_lo) & (mobility_window <= im_hi) & ~used_window
        )

        # Get nearby intensities (need this for multiple operations)
        nearby_intensities = intensity_window[nearby_mask]
        num_nearby = len(nearby_intensities)

        # Check minimum peaks requirement
        if min_peaks > 0 and num_nearby < min_peaks:
            used_mask[peak_idx] = True
            continue

        if num_nearby == 0:
            merged_mz_list.append(float(mz_peak))
            merged_int_list.append(float(intensity_peak))
            merged_mob_list.append(float(mobility_peak))
            used_mask[peak_idx] = True
            continue

        # Centroid peaks using intensity-weighted average. Guard against an
        # all-zero-intensity cluster (which would divide by zero and emit NaN
        # m/z and 1/K0); fall back to the seed peak, matching the numba kernel.
        nearby_mz = mz_window[nearby_mask]
        nearby_mobility = mobility_window[nearby_mask]
        total_intensity = np.sum(nearby_intensities)
        if total_intensity > 0.0:
            merged_mz = np.dot(nearby_mz, nearby_intensities) / total_intensity
            merged_mobility = (
                np.dot(nearby_mobility, nearby_intensities) / total_intensity
            )
        else:
            merged_mz = mz_peak
            merged_mobility = mobility_peak

        merged_mz_list.append(float(merged_mz))
        merged_int_list.append(float(total_intensity))
        merged_mob_list.append(float(merged_mobility))

        # Mark as used (convert local indices to global)
        global_nearby_idx = np.where(nearby_mask)[0] + left_idx
        used_mask[global_nearby_idx] = True

        # Peak-satellite noise filter: see Numba kernel for rationale. Within
        # ±peak_noise_window Da of the anchor m/z and inside the centroid's IM
        # window, suppress raw points whose intensity falls below a linear ramp
        # decaying from the anchor's raw intensity at d=0 to
        # (anchor * end_fraction) at d=window. Comparison is point-to-point
        # against the raw anchor intensity.
        if peak_noise_filter and peak_noise_window > 0.0:
            anchor_int = float(intensity_peak)
            noise_left_idx = int(
                np.searchsorted(mz_array, mz_peak - peak_noise_window, side="left")
            )
            noise_right_idx = int(
                np.searchsorted(mz_array, mz_peak + peak_noise_window, side="right")
            )
            noise_mz = mz_array[noise_left_idx:noise_right_idx]
            noise_int = intensity_array[noise_left_idx:noise_right_idx]
            noise_im = ion_mobility_array[noise_left_idx:noise_right_idx]
            noise_used = used_mask[noise_left_idx:noise_right_idx]
            d = np.abs(noise_mz - mz_peak)
            threshold = anchor_int * (
                1.0 - (d / peak_noise_window) * (1.0 - peak_noise_end_fraction)
            )
            suppress = (
                ~noise_used
                & (noise_im >= im_lo)
                & (noise_im <= im_hi)
                & (noise_int < threshold)
            )
            if suppress.any():
                global_suppress_idx = np.where(suppress)[0] + noise_left_idx
                used_mask[global_suppress_idx] = True

        # None or any non-positive max_peaks means "no limit" (kept consistent
        # with the numba kernel, which normalises the same way).
        if max_peaks is not None and max_peaks > 0 and len(merged_mz_list) >= max_peaks:
            logger.debug(
                "Reached max_peaks limit of %d, stopping centroiding", max_peaks
            )
            break

    # Per-call summary stays at DEBUG: get_centroided_spectrum emits the
    # user-facing INFO summary, so logging INFO here too would double up on
    # every frame (and the numba path logs nothing, so this keeps both consistent).
    logger.debug(
        "Centroiding complete: %d raw peaks → %d centroided peaks (%.1f%% reduction)",
        len(mz_array),
        len(merged_mz_list),
        100 - len(merged_mz_list) / len(mz_array) * 100,
    )
    logger.debug(
        "Total raw peaks used in centroiding: %d/%d", np.sum(used_mask), len(mz_array)
    )

    if not merged_mz_list:
        return np.empty((0, 3), dtype=np.float64)

    return np.column_stack((merged_mz_list, merged_int_list, merged_mob_list))


def get_raw_peaks(
    td: TimsData,
    frame_id: int,
    *,
    scan_range: tuple[int, int] | None = None,
    exclude: ChargeStateRegion | None = None,
    smooth: Smooth | None = None,
    noise: NoiseSpec = None,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
) -> np.ndarray:
    """Return raw peaks for a frame as a ``(N, 3)`` ``[mz, intensity, ion_mobility]`` array.

    Thin orchestrator over the :mod:`tdfpy.pipeline` ops:

    1. :func:`pipeline.read_spectrum` — read raw integer-index peaks.
    2. :func:`pipeline.subset_scans` — restrict to scan range, if given.
    3. :func:`pipeline.exclude_region` — drop ``exclude`` region, if given.
    4. :func:`pipeline.smooth` — box-smooth intensities, if ``smooth`` given.
    5. :func:`pipeline.apply_noise` — apply the coerced noise filter pipeline.
    6. :func:`pipeline.convert` — convert to ``(mz, intensity, ion_mobility)``.

    For full control over ordering or to plug in custom steps, call the
    pipeline ops directly — each takes and returns a
    :class:`~tdfpy.pipeline.RawSpectrum`.

    Args:
        td: TimsData instance connected to the analysis directory.
        frame_id: Frame ID to read.
        scan_range: Optional half-open ``(begin, end)`` scan range. Restricts
            the spectrum to peaks in that scan window — used by
            :class:`~tdfpy.DiaWindow` and :class:`~tdfpy.PrmTransition` to
            scope to their isolation window. ``None`` (default) keeps the
            whole frame.
        exclude: Optional region of (m/z, 1/K0) space to drop wholesale — for
            timsTOF MS1 use :class:`~tdfpy.regions.ChargeStateRegion` to drop
            singly-charged contamination. Applied in integer-index space, so
            there's no per-peak unit conversion.
        smooth: Optional :class:`~tdfpy.pipeline.Smooth` config. Box-sums (or
            means) intensity over a small ``(scan, TOF index)`` window before
            noise filtering, amplifying genuine ion-mobility streaks. ``None``
            (default) skips smoothing.
        noise: One or more noise filters. Accepts an instance, a list/tuple
            of instances, the string shorthand (``"mad"`` / ``"percentile"``
            / ``"histogram"`` / ``"baseline"`` / ``"iterative_median"``), or
            a numeric absolute threshold — see
            :func:`tdfpy.noise.coerce_filters`. ``None`` (default) disables.
        ion_mobility_type: Ion mobility representation — ``"ook0"`` (1/K0),
            ``"ccs"``, or ``"voltage"``.
    """
    spectrum = read_spectrum(td, frame_id)
    if scan_range is not None:
        spectrum = subset_scans(
            spectrum, scan_num_begin=scan_range[0], scan_num_end=scan_range[1]
        )
    if exclude is not None:
        spectrum = exclude_region(spectrum, exclude, td=td, frame_id=frame_id)
    if smooth is not None:
        spectrum = smooth.apply(spectrum)
    filters = coerce_filters(noise)
    if filters:
        spectrum = apply_noise(spectrum, filters, td=td, frame_id=frame_id)
    return convert(spectrum, td, frame_id, ion_mobility_type=ion_mobility_type)


def get_centroided_spectrum(
    td: TimsData,
    frame_id: int,
    *,
    scan_range: tuple[int, int] | None = None,
    exclude: ChargeStateRegion | None = None,
    smooth: Smooth | None = None,
    noise: NoiseSpec = None,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    centroid: Centroider | None = None,
) -> np.ndarray:
    """Extract a centroided spectrum for a single frame.

    Thin orchestrator over the :mod:`tdfpy.pipeline` ops. Threads a
    :class:`~tdfpy.pipeline.RawSpectrum` through optional scan-range
    restriction, region exclusion, intensity smoothing, and noise filtering,
    then hands it to the centroider — which decides whether to operate in
    integer index space (e.g. :class:`~tdfpy.pipeline.WatershedCentroider`) or
    after float conversion (e.g. :class:`~tdfpy.pipeline.MergePeaksCentroider`).

    Default centroider is :class:`~tdfpy.pipeline.MergePeaksCentroider`. Pass
    ``smooth=Smooth(...)`` for a position-preserving box-sum/mean smoothing
    pre-step; the :class:`~tdfpy.pipeline.WatershedCentroider` additionally
    has its own seed-stabilising smoother via its ``smooth_*_half_width`` fields.

    Returns an ``(N, 3)`` array of ``[mz, intensity, ion_mobility]`` centroids.
    """
    spectrum = read_spectrum(td, frame_id)
    if scan_range is not None:
        spectrum = subset_scans(
            spectrum, scan_num_begin=scan_range[0], scan_num_end=scan_range[1]
        )
    if exclude is not None:
        spectrum = exclude_region(spectrum, exclude, td=td, frame_id=frame_id)
    if smooth is not None:
        spectrum = smooth.apply(spectrum)
    filters = coerce_filters(noise)
    if filters:
        spectrum = apply_noise(spectrum, filters, td=td, frame_id=frame_id)

    if spectrum.empty:
        # An empty frame is common on sparse acquisitions, so this is INFO, not
        # a warning. If a *noise filter* emptied a non-empty frame, apply_noise
        # has already logged a warning naming the responsible filter.
        logger.info(
            "Frame %d has 0 peaks after read/smooth/noise; returning empty spectrum.",
            frame_id,
        )
        return np.empty((0, 3), dtype=np.float64)

    centroider = centroid if centroid is not None else MergePeaksCentroider()
    centroids = centroider(
        spectrum, td, frame_id, ion_mobility_type=ion_mobility_type
    )
    logger.info(
        "Centroided frame %d: %d raw → %d centroids",
        frame_id, len(spectrum), len(centroids),
    )
    return centroids


#: Default m/z tolerance for :func:`get_mobility_collapsed_spectrum`, in ppm.
#: Chosen by sweeping against Bruker's native peak picker on the bundled
#: fixtures: it puts the peak count within ~4% of Bruker's while keeping 99% of
#: Bruker's total intensity within 10 ppm of one of our centroids.
COLLAPSED_MZ_TOLERANCE_PPM = 30.0


def get_mobility_collapsed_spectrum(
    td: TimsData,
    scan_ranges: Sequence[tuple[int, int, int]],
    *,
    mz_tolerance: float = COLLAPSED_MZ_TOLERANCE_PPM,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    use_numba: bool = True,
) -> np.ndarray:
    """Centroid a set of scan ranges with the mobility dimension summed away.

    This is the shape of spectrum Bruker's ``tims_read_pasef_msms`` and
    ``tims_extract_centroided_spectrum_for_frame`` return: intensities are
    summed over the mobility axis, leaving a plain m/z spectrum. A single ion
    smears across roughly 8-10 adjacent TOF bins, so the collapsed profile is
    then centroided by greedy intensity-ordered merging.

    Args:
        td: Open :class:`~tdfpy.timsdata.TimsData`.
        scan_ranges: ``(frame_id, scan_begin, scan_end)`` triples, summed
            together. A PASEF precursor is typically spread over several frames.
        mz_tolerance: Merge tolerance for the greedy centroider.
        mz_tolerance_type: ``"ppm"`` or ``"da"``.
        use_numba: Use the JIT-compiled merge kernel when available.

    Returns:
        An ``(N, 2)`` array of ``[mz, intensity]`` sorted by descending
        intensity, as :func:`merge_peaks` produces.

    Note:
        Results are close to Bruker's peak picker but not identical to it --
        Bruker's algorithm is proprietary and appears to smooth before picking.
        On the bundled fixtures the strong peaks agree to ~0.5 ppm and the total
        ion current to within 0.1%, with ~4% more peaks reported.
    """
    if not scan_ranges:
        return np.empty((0, 2), dtype=np.float64)

    # Sum intensities per TOF index across every scan of every range. TOF indices
    # are integers on a shared grid within a frame, so this is an exact rollup
    # with no binning error.
    totals: dict[int, int] = {}
    for frame_id, scan_begin, scan_end in scan_ranges:
        for tof_indices, intensities in td.readScans(frame_id, scan_begin, scan_end):
            for tof_index, intensity in zip(
                tof_indices.tolist(), intensities.tolist(), strict=True
            ):
                totals[tof_index] = totals.get(tof_index, 0) + intensity

    if not totals:
        return np.empty((0, 2), dtype=np.float64)

    tof_index_array = np.fromiter(totals.keys(), dtype=np.int64, count=len(totals))
    intensity_array = np.fromiter(
        totals.values(), dtype=np.float64, count=len(totals)
    )
    order = np.argsort(tof_index_array)
    tof_index_array = tof_index_array[order]
    intensity_array = intensity_array[order]

    # The TOF grid is per-frame; all ranges of one precursor share a calibration,
    # so converting with the first frame is exact.
    mz_array = td.indexToMz(scan_ranges[0][0], tof_index_array.astype(np.float64))

    peaks = merge_peaks(
        mz_array,
        intensity_array,
        np.zeros_like(mz_array),
        mz_tolerance=mz_tolerance,
        mz_tolerance_type=mz_tolerance_type,
        # Mobility is already summed away, so no peak may be split by it.
        im_tolerance=np.inf,
        im_tolerance_type="absolute",
        # Each merged peak is a real ion even when it occupies a single TOF bin;
        # requiring more would discard most of the spectrum.
        min_peaks=1,
        use_numba=use_numba,
    )
    return np.ascontiguousarray(peaks[:, :2])


def calculate_nmass(mz: float, charge: int) -> float:
    """Calculate neutral mass from m/z and charge state."""
    return mz * abs(charge) - charge * 1.007276466812  # Subtract charge * proton mass


def get_tdf_df(td: TimsData) -> pd.DataFrame:
    pd_tdf = PandasTdf(td.analysis_directory)

    merged_df = pd.merge(
        pd_tdf.precursors,
        pd_tdf.frames,
        left_on="Parent",
        right_on="Id",
        suffixes=("_Precursor", "_Frame"),
    )

    pasef_frame_msms_info_df = pd_tdf.pasef_frame_msms_info.drop(["Frame"], axis=1)

    # count the number of items in each group
    pasef_frame_msms_info_df["count"] = pasef_frame_msms_info_df.groupby("Precursor")[
        "Precursor"
    ].transform("count")

    # keep only the row for each group
    pasef_frame_msms_info_df = pasef_frame_msms_info_df.drop_duplicates(
        subset="Precursor", keep="first"
    )
    if len(pasef_frame_msms_info_df) != len(merged_df):
        raise ValueError(
            f"PASEF frame MS/MS info row count ({len(pasef_frame_msms_info_df)}) "
            f"does not match precursor/frame merge count ({len(merged_df)}). "
            f"This indicates a data integrity issue in the .tdf file."
        )

    merged_df = pd.merge(
        merged_df,
        pasef_frame_msms_info_df,
        left_on="Id_Precursor",
        right_on="Precursor",
        suffixes=("_Precursor", "_PasefFrameMsmsInfo"),
    ).drop("Precursor", axis=1)

    merged_df["NeutralMass"] = merged_df.apply(
        lambda row: calculate_nmass(row["MonoisotopicMz"], row["Charge"]),
        axis=1,
    )

    return merged_df
