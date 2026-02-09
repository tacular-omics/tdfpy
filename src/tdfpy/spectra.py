"""
Higher-level Pythonic API for working with MS1 spectrum data from Bruker timsTOF files.

This module provides a cleaner interface using NamedTuples and convenience functions
for reading centroided MS1 spectra with peak clustering/centroiding algorithms.
"""

from typing import Any, Literal, NamedTuple
from collections.abc import Generator
import logging

import numpy as np
import pandas as pd  # type: ignore

from .timsdata import TimsData, oneOverK0ToCCSforMz
from .noise import estimate_noise_level
from .pandas_tdf import PandasTdf

# Try to import Rust extension, fallback to Python implementation
try:
    from tdfpy._tdfpy_rust import merge_peaks as _merge_peaks_rust  # type: ignore[import]

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False  # type: ignore[assignment]


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


def merge_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    use_rust: bool = True,
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

    Returns:
        np.ndarray: Array of shape (N, 3) containing centroided peaks.
                   Columns are: [mz, intensity, ion_mobility]

    Example:
        >>> mz = np.array([100.0, 100.001, 200.0])
        >>> intensity = np.array([1000.0, 500.0, 2000.0])
        >>> im = np.array([0.8, 0.8, 0.9])
        >>> peaks = merge_peaks(mz, intensity, im, mz_tolerance=10, mz_tolerance_type="ppm")
    """
    # Use Rust implementation if available
    if _HAS_RUST and use_rust:
        merged_mz, merged_intensity, merged_mobility = _merge_peaks_rust(  # type: ignore[call-arg]
            mz_array,
            intensity_array,
            ion_mobility_array,
            mz_tolerance,
            mz_tolerance_type,
            im_tolerance,
            im_tolerance_type,
            min_peaks,
            max_peaks,
        )

        if (
            not isinstance(merged_mz, np.ndarray)
            or not isinstance(merged_intensity, np.ndarray)
            or not isinstance(merged_mobility, np.ndarray)
        ):
            raise RuntimeError(
                "Rust merge_peaks did not return numpy arrays as expected"
            )

        # Stack arrays into (N, 3) matrix
        return np.column_stack((merged_mz, merged_intensity, merged_mobility))

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
    )


def _merge_peaks_python(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
) -> np.ndarray:
    """Python implementation of merge_peaks (fallback when Rust extension unavailable)."""
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
        mobility_tol = (
            mobility_peak * mobility_tol_factor
            if im_tolerance_type == "relative"
            else mobility_tol_abs
        )

        # Binary search for mz range
        left_mz = mz_peak - mz_tol
        right_mz = mz_peak + mz_tol
        left_idx = int(np.searchsorted(mz_array, left_mz, side="left"))
        right_idx = int(np.searchsorted(mz_array, right_mz, side="right"))

        # Only check mobility in the mz window
        mobility_window = ion_mobility_array[left_idx:right_idx]
        intensity_window = intensity_array[left_idx:right_idx]
        mz_window = mz_array[left_idx:right_idx]
        used_window = used_mask[left_idx:right_idx]

        # Find nearby peaks in mobility dimension (combined operation)
        nearby_mask = (
            np.abs(mobility_window - mobility_peak) <= mobility_tol
        ) & ~used_window

        # Get nearby intensities (need this for multiple operations)
        nearby_intensities = intensity_window[nearby_mask]
        num_nearby = len(nearby_intensities)

        # Check minimum peaks requirement
        if min_peaks > 0 and num_nearby < min_peaks:
            # Not enough nearby raw peaks to form a centroid, skip
            used_mask[peak_idx] = True
            continue

        if num_nearby == 0:
            # Edge case: no nearby raw peaks (shouldn't happen but be safe)
            merged_mz_list.append(float(mz_peak))
            merged_int_list.append(float(intensity_peak))
            merged_mob_list.append(float(mobility_peak))
            used_mask[peak_idx] = True
            continue

        # Centroid peaks using intensity-weighted average
        # Reuse already-sliced arrays to avoid re-indexing
        nearby_mz = mz_window[nearby_mask]
        nearby_mobility = mobility_window[nearby_mask]
        total_intensity = np.sum(nearby_intensities)
        merged_mz = np.dot(nearby_mz, nearby_intensities) / total_intensity
        merged_mobility = np.dot(nearby_mobility, nearby_intensities) / total_intensity

        merged_mz_list.append(float(merged_mz))
        merged_int_list.append(float(total_intensity))
        merged_mob_list.append(float(merged_mobility))

        # Mark as used (convert local indices to global)
        global_nearby_idx = np.where(nearby_mask)[0] + left_idx
        used_mask[global_nearby_idx] = True

        if max_peaks and len(merged_mz_list) >= max_peaks:
            logger.debug(
                "Reached max_peaks limit of %d, stopping centroiding", max_peaks
            )
            break

    logger.info(
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


def get_centroided_ms1_spectrum(
    td: TimsData,
    frame_id: int,
    spectrum_index: int | None = None,
    ion_mobility_type: Literal["ook0", "ccs"] = "ook0",
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    noise_filter: None
    | (
        Literal["mad", "percentile", "histogram", "baseline", "iterative_median"]
        | float
        | int
    ) = None,
    use_rust: bool = True,
) -> np.ndarray:
    """Extract a centroided MS1 spectrum for a single frame.

    This function reads raw profile-like scans from the frame, converts indices to m/z values,
    collects all raw peaks with their ion mobility values, and applies peak centroiding
    based on m/z and ion mobility tolerances to produce a centroided spectrum.

    Args:
        td: TimsData instance connected to the analysis directory
        frame_id: Frame ID to extract
        spectrum_index: Optional index for this spectrum (defaults to frame_id)
        ion_mobility_type: Type of ion mobility to calculate and include for each peak
                          - "ook0": 1/K0 (reciprocal reduced mobility) [default]
                          - "ccs": Collision Cross Section in Ų (requires charge state estimation)
        mz_tolerance: Tolerance for m/z matching during centroiding
        mz_tolerance_type: Type of m/z tolerance - "ppm" or "da" (daltons)
        im_tolerance: Tolerance for ion mobility matching during centroiding
        im_tolerance_type: Type of ion mobility tolerance - "relative" or "absolute"
        min_peaks: Minimum number of nearby raw peaks required to form a centroid (0 or 1 keeps all)
        max_peaks: Maximum number of centroided peaks to return
        noise_filter: Noise filtering method to apply before centroiding. Options:
                     - None: No noise filtering (default)
                     - "mad": Median Absolute Deviation method
                     - "percentile": 75th percentile threshold
                     - "histogram": Histogram mode-based estimation
                     - "baseline": Bottom quartile statistics
                     - "iterative_median": Iterative median filtering
                     - float/int: Direct intensity threshold value

    Returns:
        np.ndarray: Array of shape (N, 3) containing centroided peaks.
                   Columns are: [mz, intensity, ion_mobility]

    Raises:
        ValueError: If the frame_id doesn't exist or is not an MS1 frame
        RuntimeError: If the TimsData connection is not open

    Example:
        >>> with timsdata_connect('path/to/data.d') as td:
        ...     # Get centroided spectrum with 1/K0 (default)
        ...     peaks = get_centroided_ms1_spectrum(td, frame_id=1)
        ...     print(f"Found {len(peaks)} centroided peaks")
        ...
        ...     # Get spectrum with CCS values
        ...     spectrum = get_centroided_ms1_spectrum(td, frame_id=1, ion_mobility_type="ccs")
        ...
        ...     # Custom centroiding tolerances
        ...     spectrum = get_centroided_ms1_spectrum(
        ...         td, frame_id=1, mz_tolerance=10, im_tolerance=0.1
        ...     )
        ...
        ...     # With noise filtering
        ...     spectrum = get_centroided_ms1_spectrum(
        ...         td, frame_id=1, noise_filter="mad"
        ...     )
        ...
        ...     # With custom noise threshold
        ...     spectrum = get_centroided_ms1_spectrum(
        ...         td, frame_id=1, noise_filter=1000.0
        ...     )
    """
    logger.debug(
        "Extracting MS1 spectrum for frame_id=%d, noise_filter=%s",
        frame_id,
        noise_filter,
    )

    if td.conn is None:
        logger.error("TimsData connection is not open")
        raise RuntimeError("TimsData connection is not open")

    # Get frame metadata from the database
    cursor = td.conn.cursor()
    cursor.execute(
        "SELECT Time, NumScans, MsMsType FROM Frames WHERE Id = ?", (frame_id,)
    )
    result = cursor.fetchone()

    if result is None:
        logger.error("Frame %d not found in database", frame_id)
        raise ValueError(f"Frame {frame_id} not found in database")

    retention_time_sec, num_scans, msms_type = result
    logger.debug(
        "Frame %d metadata: RT=%.2fs, NumScans=%d, MsMsType=%d",
        frame_id,
        retention_time_sec,
        num_scans,
        msms_type,
    )

    if msms_type != 0:
        logger.error("Frame %d is not an MS1 frame (MsMsType=%d)", frame_id, msms_type)
        raise ValueError(f"Frame {frame_id} is not an MS1 frame (MsMsType={msms_type})")

    retention_time_min = retention_time_sec / 60.0

    if num_scans == 0:
        logger.warning("Frame %d has 0 scans, returning empty spectrum", frame_id)
        return np.empty((0, 3), dtype=np.float64)

    # Pre-compute ion mobility values for each scan (always required)
    logger.debug(
        "Computing %s ion mobility values for %d scans", ion_mobility_type, num_scans
    )
    ion_mobility = td.scanNumToOneOverK0(frame_id, np.arange(0, num_scans))  # type: ignore[call-arg]

    # Read all scans at once
    logger.debug("Reading %d scans from frame %d", num_scans, frame_id)
    results = td.readScans(frame_id, 0, num_scans)

    # Pre-allocate arrays with estimated size
    total_peaks = sum(len(idx) for idx, _ in results)
    logger.debug(
        "Frame %d contains %d total raw peaks across %d scans",
        frame_id,
        total_peaks,
        num_scans,
    )

    if total_peaks == 0:
        logger.warning("Frame %d has 0 peaks, returning empty spectrum", frame_id)
        return np.empty((0, 3), dtype=np.float64)

    logger.debug("Pre-allocating arrays for %d peaks", total_peaks)
    mz_array = np.empty(total_peaks, dtype=np.float64)
    intensity_array = np.empty(total_peaks, dtype=np.float64)
    ion_mobility_array = np.empty(total_peaks, dtype=np.float64)

    # Collect all peaks from all scans
    offset = 0
    logger.debug("Starting scan iteration and m/z conversion (profile-like raw data)")
    for scan_index, (index_array, intensity_scan) in enumerate(results):
        n_peaks = len(index_array)
        if n_peaks == 0:
            continue

        # Convert indices to m/z in batch
        mz_values = td.indexToMz(frame_id, index_array)

        # Fill pre-allocated arrays
        mz_array[offset : offset + n_peaks] = mz_values
        intensity_array[offset : offset + n_peaks] = intensity_scan
        ion_mobility_array[offset : offset + n_peaks] = ion_mobility[scan_index]
        offset += n_peaks

    # Trim arrays to actual size
    mz_array = mz_array[:offset]
    intensity_array = intensity_array[:offset]
    ion_mobility_array = ion_mobility_array[:offset]
    logger.debug("Collected %d raw profile-like peaks from all scans", offset)

    # Apply noise filtering if requested
    if noise_filter is not None:
        logger.debug("Applying noise filter: %s", noise_filter)
        noise_threshold = estimate_noise_level(intensity_array, method=noise_filter)
        noise_mask = intensity_array >= noise_threshold

        mz_array = mz_array[noise_mask]
        intensity_array = intensity_array[noise_mask]
        ion_mobility_array = ion_mobility_array[noise_mask]

        filtered_count = offset - len(intensity_array)
        logger.info(
            "Noise filtering complete: removed %d peaks below threshold %.2f (%d → %d peaks, %.1f%% removed)",
            filtered_count,
            noise_threshold,
            offset,
            len(intensity_array),
            filtered_count / offset * 100,
        )

    # Convert to CCS if requested
    if ion_mobility_type == "ccs":
        logger.debug("Converting 1/K0 to CCS values (assuming charge +1)")
        # Import conversion function
        ccs_array = np.array(
            [
                oneOverK0ToCCSforMz(ook0, 1, mz)
                for ook0, mz in zip(ion_mobility_array, mz_array)
            ],
            dtype=np.float64,
        )
        ion_mobility_array = ccs_array
        logger.debug("Completed CCS conversion")

    # Apply peak centroiding
    logger.debug("Starting peak centroiding algorithm")
    peaks = merge_peaks(
        mz_array=mz_array,
        intensity_array=intensity_array,
        ion_mobility_array=ion_mobility_array,
        mz_tolerance=mz_tolerance,
        mz_tolerance_type=mz_tolerance_type,
        im_tolerance=im_tolerance,
        im_tolerance_type=im_tolerance_type,
        min_peaks=min_peaks,
        max_peaks=max_peaks,
        use_rust=use_rust,
    )

    # Apply max_peaks limit if specified (post-centroiding)
    if max_peaks and len(peaks) > max_peaks:
        logger.debug("Applying max_peaks filter: %d → %d", len(peaks), max_peaks)
        # Sort by intensity (column 1) and take top N
        # argsort is ascending, so we take from the end [::-1]
        sort_indices = np.argsort(peaks[:, 1])[::-1][:max_peaks]
        peaks = peaks[sort_indices]

    logger.info(
        "Extracted centroided MS1 spectrum: frame_id=%d, RT=%.2f min, centroided_peaks=%d, raw_peaks=%d, ion_mobility_type=%s",
        frame_id,
        retention_time_min,
        len(peaks),
        total_peaks,
        ion_mobility_type,
    )

    return peaks


def get_centroided_ms1_spectra(
    td: TimsData,
    frame_ids: list[int] | None = None,
    ion_mobility_type: Literal["ook0", "ccs"] = "ook0",
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: int | None = None,
    noise_filter: None
    | (
        Literal["mad", "percentile", "histogram", "baseline", "iterative_median"]
        | float
    ) = None,
    use_rust: bool = True,
) -> Generator[np.ndarray, None, None]:
    """Extract centroided MS1 spectra for multiple frames.

    Convenience function to extract multiple centroided MS1 spectra. If frame_ids is not
    specified, all MS1 frames in the file will be processed. Raw profile-like data is
    converted to centroided spectra using peak clustering.

    Args:
        td: TimsData instance connected to the analysis directory
        frame_ids: Optional list of frame IDs to extract. If None, extracts all MS1 frames.
        ion_mobility_type: Type of ion mobility to calculate and include for each peak
                          - "ook0": 1/K0 (reciprocal reduced mobility) [default]
                          - "ccs": Collision Cross Section in Ų (requires charge state estimation)
        mz_tolerance: Tolerance for m/z matching during centroiding
        mz_tolerance_type: Type of m/z tolerance - "ppm" or "da" (daltons)
        im_tolerance: Tolerance for ion mobility matching during centroiding
        im_tolerance_type: Type of ion mobility tolerance - "relative" or "absolute"
        min_peaks: Minimum number of nearby raw peaks required to form a centroid
        noise_filter: Noise filtering method to apply before centroiding. Options:
                     - None: No noise filtering (default)
                     - "mad": Median Absolute Deviation method
                     - "percentile": 75th percentile threshold
                     - "histogram": Histogram mode-based estimation
                     - "baseline": Bottom quartile statistics
                     - "iterative_median": Iterative median filtering
                     - float/int: Direct intensity threshold value

    Returns:
        Generator yielding np.ndarray objects of shape (N, 3), ordered by frame ID.
        Columns are: [mz, intensity, ion_mobility]

    Example:
        >>> with timsdata_connect('path/to/data.d') as td:
        ...     # Get all centroided MS1 spectra with 1/K0 (default)
        ...     for peaks in get_centroided_ms1_spectra(td):
        ...         print(f"Spectrum: {len(peaks)} centroided peaks")
        ...
        ...     # Get spectra with CCS values
        ...     for spectrum in get_centroided_ms1_spectra(td, ion_mobility_type="ccs"):
        ...         print(f"CCS spectrum: {spectrum.num_peaks} peaks")
        ...
        ...     # Custom centroiding tolerances
        ...     spectra = list(get_centroided_ms1_spectra(
        ...         td, mz_tolerance=10, im_tolerance=0.1
        ...     ))
        ...
        ...     # With noise filtering
        ...     for spectrum in get_centroided_ms1_spectra(td, noise_filter="mad"):
        ...         print(f"Noise-filtered spectrum: {spectrum.num_peaks} peaks")
    """
    logger.info(
        "Starting batch MS1 centroided spectrum extraction (frame_ids=%s, noise_filter=%s)",
        "all MS1" if frame_ids is None else f"{len(frame_ids)} specified",
        noise_filter,
    )

    if td.conn is None:
        logger.error("TimsData connection is not open")
        raise RuntimeError("TimsData connection is not open")

    cursor = td.conn.cursor()

    if frame_ids is None:
        # Get all MS1 frame IDs
        logger.debug("Querying database for all MS1 frame IDs")
        cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id")
        frame_ids = [row[0] for row in cursor.fetchall()]
        logger.info("Found %d MS1 frames to process", len(frame_ids))
    else:
        logger.debug("Processing %d user-specified frame IDs", len(frame_ids))

    successful_count = 0
    failed_count = 0

    for idx, frame_id in enumerate(frame_ids):
        if (idx + 1) % 100 == 0:
            logger.info("Progress: %d/%d frames processed", idx + 1, len(frame_ids))

        try:
            spectrum = get_centroided_ms1_spectrum(
                td,
                frame_id=frame_id,
                spectrum_index=idx,
                ion_mobility_type=ion_mobility_type,
                mz_tolerance=mz_tolerance,
                mz_tolerance_type=mz_tolerance_type,
                im_tolerance=im_tolerance,
                im_tolerance_type=im_tolerance_type,
                min_peaks=min_peaks,
                max_peaks=max_peaks,
                noise_filter=noise_filter,
                use_rust=use_rust,
            )
            successful_count += 1
            logger.debug(
                "Successfully extracted centroided spectrum %d/%d: frame_id=%d",
                idx + 1,
                len(frame_ids),
                frame_id,
            )
            yield spectrum
        except (ValueError, RuntimeError) as e:
            # Log warning but continue processing
            failed_count += 1
            logger.warning(
                "Failed to extract spectrum for frame %d (%d/%d): %s",
                frame_id,
                idx + 1,
                len(frame_ids),
                e,
            )
            continue

    logger.info(
        "Batch centroiding complete: %d successful, %d failed, %d total",
        successful_count,
        failed_count,
        len(frame_ids),
    )

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
    assert len(pasef_frame_msms_info_df) == len(merged_df)

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
