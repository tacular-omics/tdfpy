"""
Higher-level Pythonic API for working with MS1 spectrum data from Bruker timsTOF files.

This module provides a cleaner interface using NamedTuples and convenience functions
for reading centroided MS1 spectra with peak clustering/centroiding algorithms.
"""

from typing import Generator, List, Literal, NamedTuple, Optional, Union
import logging
import threading
import os
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .timsdata import TimsData, oneOverK0ToCCSforMz
from .noise import estimate_noise_level


logger = logging.getLogger(__name__)


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


class Ms1Spectrum(NamedTuple):
    """Represents a complete MS1 spectrum from a single frame.

    Attributes:
        spectrum_index: Sequential index of this spectrum
        frame_id: Original frame ID from the TDF file
        retention_time: Retention time in minutes
        num_peaks: Number of peaks in the spectrum
        peaks: List of Peak objects
        ion_mobility_type: Type of ion mobility values in peaks ('ook0' or 'ccs')
    """

    spectrum_index: int
    frame_id: int
    retention_time: float
    peaks: List[Peak]
    ion_mobility_type: Literal["ook0", "ccs"]

    @property
    def num_peaks(self) -> int:
        """Number of peaks in the spectrum."""
        return len(self.peaks)

    def __repr__(self) -> str:
        return (
            f"Ms1Spectrum(index={self.spectrum_index}, frame_id={self.frame_id}, "
            f"retention_time={self.retention_time:.2f} min, num_peaks={self.num_peaks}, "
            f"ion_mobility_type='{self.ion_mobility_type}')"
        )


def merge_peaks(
    mz_array: np.ndarray,
    intensity_array: np.ndarray,
    ion_mobility_array: np.ndarray,
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: Optional[int] = None,
) -> List[Peak]:
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
        List of centroided Peak objects

    Example:
        >>> mz = np.array([100.0, 100.001, 200.0])
        >>> intensity = np.array([1000.0, 500.0, 2000.0])
        >>> im = np.array([0.8, 0.8, 0.9])
        >>> peaks = merge_peaks(mz, intensity, im, mz_tolerance=10, mz_tolerance_type="ppm")
    """
    logger.debug(
        "Centroiding %d raw peaks with mz_tol=%s %s, im_tol=%s %s, min_peaks=%d, max_peaks=%s",
        len(mz_array), mz_tolerance, mz_tolerance_type, im_tolerance, im_tolerance_type, min_peaks, max_peaks
    )

    if len(mz_array) == 0:
        logger.debug("No raw peaks to centroid, returning empty list")
        return []

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

    # Sort by mz for binary search (NumPy operation, releases GIL)
    sort_idx = np.argsort(mz_array)
    mz_array = mz_array[sort_idx]
    intensity_array = intensity_array[sort_idx]
    ion_mobility_array = ion_mobility_array[sort_idx]
    logger.debug("Sorted %d peaks by m/z", len(mz_array))

    # Sort by intensity for greedy clustering (NumPy operation, releases GIL)
    intensity_order = np.argsort(intensity_array)[::-1]
    logger.debug("Created intensity-ordered index for greedy clustering")

    # Use boolean mask for tracking used peaks (NumPy array)
    used_mask = np.zeros(len(mz_array), dtype=bool)
    merged_peaks: List[Peak] = []

    # Pre-compute tolerance arrays for vectorization
    if mz_tolerance_type == "ppm":
        mz_tol_array = mz_array * mz_tol_factor
    else:
        mz_tol_array = np.full(len(mz_array), mz_tol_abs)
    
    if im_tolerance_type == "relative":
        mobility_tol_array = ion_mobility_array * mobility_tol_factor
    else:
        mobility_tol_array = np.full(len(ion_mobility_array), mobility_tol_abs)

    for peak_idx in intensity_order:
        if used_mask[peak_idx]:
            continue

        mz_peak = float(mz_array[peak_idx])
        intensity_peak = float(intensity_array[peak_idx])
        mobility_peak = float(ion_mobility_array[peak_idx])

        # Get tolerances from pre-computed arrays
        mz_tol = float(mz_tol_array[peak_idx])
        mobility_tol = float(mobility_tol_array[peak_idx])

        # Binary search for mz range (NumPy operation, releases GIL)
        left_mz = mz_peak - mz_tol
        right_mz = mz_peak + mz_tol
        left_idx = int(np.searchsorted(mz_array, left_mz, side="left"))
        right_idx = int(np.searchsorted(mz_array, right_mz, side="right"))

        # Only check mobility in the mz window (vectorized operations)
        mz_window_slice = slice(left_idx, right_idx)
        mobility_window = ion_mobility_array[mz_window_slice]
        intensity_window = intensity_array[mz_window_slice]
        mz_window = mz_array[mz_window_slice]
        used_window = used_mask[mz_window_slice]

        # Find nearby peaks in mobility dimension (vectorized)
        mobility_diff = np.abs(mobility_window - mobility_peak)
        nearby_mask = (mobility_diff <= mobility_tol) & ~used_window

        # Check minimum peaks requirement
        if min_peaks > 0 and np.sum(nearby_mask) < min_peaks:
            # Not enough nearby raw peaks to form a centroid, skip
            used_mask[peak_idx] = True
            continue

        if not np.any(nearby_mask):
            # Edge case: no nearby raw peaks (shouldn't happen but be safe)
            merged_peaks.append(
                Peak(mz=mz_peak, intensity=intensity_peak, ion_mobility=mobility_peak)
            )
            used_mask[peak_idx] = True
            continue

        # Centroid peaks using intensity-weighted average (vectorized NumPy)
        nearby_intensities = intensity_window[nearby_mask]
        merged_intensity = float(np.sum(nearby_intensities))
        merged_mz = float(
            np.average(mz_window[nearby_mask], weights=nearby_intensities)
        )
        merged_mobility = float(
            np.average(mobility_window[nearby_mask], weights=nearby_intensities)
        )

        merged_peaks.append(
            Peak(mz=merged_mz, intensity=merged_intensity, ion_mobility=merged_mobility)
        )

        # Mark as used (convert local indices to global, vectorized)
        global_nearby_idx = np.where(nearby_mask)[0] + left_idx
        used_mask[global_nearby_idx] = True

        if max_peaks and len(merged_peaks) >= max_peaks:
            logger.debug("Reached max_peaks limit of %d, stopping centroiding", max_peaks)
            break

    logger.info(
        "Centroiding complete: %d raw peaks → %d centroided peaks (%.1f%% reduction)",
        len(mz_array), len(merged_peaks), 100 - len(merged_peaks) / len(mz_array) * 100
    )
    logger.debug(
        "Total raw peaks used in centroiding: %d/%d", np.sum(used_mask), len(mz_array)
    )

    return merged_peaks


def get_centroided_ms1_spectrum(
    td: TimsData,
    frame_id: int,
    spectrum_index: Optional[int] = None,
    ion_mobility_type: Literal["ook0", "ccs"] = "ook0",
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: Optional[int] = None,
    noise_filter: Optional[
        Union[
            Literal["mad", "percentile", "histogram", "baseline", "iterative_median"],
            float,
            int,
        ]
    ] = None,
) -> Ms1Spectrum:
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
        Ms1Spectrum object containing centroided peaks and metadata

    Raises:
        ValueError: If the frame_id doesn't exist or is not an MS1 frame
        RuntimeError: If the TimsData connection is not open

    Example:
        >>> with timsdata_connect('path/to/data.d') as td:
        ...     # Get centroided spectrum with 1/K0 (default)
        ...     spectrum = get_centroided_ms1_spectrum(td, frame_id=1)
        ...     print(f"Found {spectrum.num_peaks} centroided peaks")
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
        "Extracting MS1 spectrum for frame_id=%d, noise_filter=%s", frame_id, noise_filter
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
        frame_id, retention_time_sec, num_scans, msms_type
    )

    if msms_type != 0:
        logger.error("Frame %d is not an MS1 frame (MsMsType=%d)", frame_id, msms_type)
        raise ValueError(f"Frame {frame_id} is not an MS1 frame (MsMsType={msms_type})")

    retention_time_min = retention_time_sec / 60.0

    if num_scans == 0:
        logger.warning("Frame %d has 0 scans, returning empty spectrum", frame_id)
        return Ms1Spectrum(
            spectrum_index=spectrum_index if spectrum_index is not None else frame_id,
            frame_id=frame_id,
            retention_time=retention_time_min,
            peaks=[],
            ion_mobility_type=ion_mobility_type,
        )

    # Pre-compute ion mobility values for each scan (always required)
    logger.debug(
        "Computing %s ion mobility values for %d scans", ion_mobility_type, num_scans
    )
    ion_mobility = td.scanNumToOneOverK0(frame_id, np.arange(0, num_scans))

    # Read all scans at once
    logger.debug("Reading %d scans from frame %d", num_scans, frame_id)
    results = td.readScans(frame_id, 0, num_scans)

    # Pre-allocate arrays with estimated size
    total_peaks = sum(len(idx) for idx, _ in results)
    logger.debug(
        "Frame %d contains %d total raw peaks across %d scans", frame_id, total_peaks, num_scans
    )

    if total_peaks == 0:
        logger.warning("Frame %d has 0 peaks, returning empty spectrum", frame_id)
        return Ms1Spectrum(
            spectrum_index=spectrum_index if spectrum_index is not None else frame_id,
            frame_id=frame_id,
            retention_time=retention_time_min,
            peaks=[],
            ion_mobility_type=ion_mobility_type,
        )

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
            filtered_count, noise_threshold, offset, len(intensity_array), filtered_count / offset * 100
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
    )

    # Apply max_peaks limit if specified
    if max_peaks and len(peaks) > max_peaks:
        logger.debug("Applying max_peaks filter: %d → %d", len(peaks), max_peaks)
        # Sort by intensity and take top N
        peaks = sorted(peaks, key=lambda p: p.intensity, reverse=True)[:max_peaks]

    logger.info(
        "Extracted centroided MS1 spectrum: frame_id=%d, RT=%.2f min, centroided_peaks=%d, raw_peaks=%d, ion_mobility_type=%s",
        frame_id, retention_time_min, len(peaks), total_peaks, ion_mobility_type
    )

    return Ms1Spectrum(
        spectrum_index=spectrum_index if spectrum_index is not None else frame_id,
        frame_id=frame_id,
        retention_time=retention_time_min,
        peaks=peaks,
        ion_mobility_type=ion_mobility_type,
    )


def get_centroided_ms1_spectra(
    td: TimsData,
    frame_ids: Optional[List[int]] = None,
    ion_mobility_type: Literal["ook0", "ccs"] = "ook0",
    mz_tolerance: float = 8.0,
    mz_tolerance_type: Literal["ppm", "da"] = "ppm",
    im_tolerance: float = 0.05,
    im_tolerance_type: Literal["relative", "absolute"] = "relative",
    min_peaks: int = 3,
    max_peaks: Optional[int] = None,
    noise_filter: Optional[
        Union[
            Literal["mad", "percentile", "histogram", "baseline", "iterative_median"],
            float,
        ]
    ] = None,
    num_workers: Optional[int] = None,
    ordered: bool = True,
) -> Generator[Ms1Spectrum, None, None]:
    """Extract centroided MS1 spectra for multiple frames using parallel processing.

    Convenience function to extract multiple centroided MS1 spectra using a producer-consumer
    architecture with threading. If frame_ids is not specified, all MS1 frames in the file
    will be processed. Raw profile-like data is converted to centroided spectra using peak
    clustering. Processing is parallelized across multiple worker threads since NumPy
    operations and DLL calls release the GIL.

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
        max_peaks: Maximum number of centroided peaks to return
        noise_filter: Noise filtering method to apply before centroiding. Options:
                     - None: No noise filtering (default)
                     - "mad": Median Absolute Deviation method
                     - "percentile": 75th percentile threshold
                     - "histogram": Histogram mode-based estimation
                     - "baseline": Bottom quartile statistics
                     - "iterative_median": Iterative median filtering
                     - float/int: Direct intensity threshold value
        num_workers: Number of worker threads for parallel processing. If None, uses
                    os.cpu_count() workers. Set to 1 for sequential processing.
        ordered: If True, yields spectra in frame_id order (may cause delays if frames
                complete out of order). If False, yields spectra as soon as they're ready
                (faster but unordered).

    Returns:
        Generator yielding Ms1Spectrum objects, ordered by frame ID if ordered=True

    Example:
        >>> with timsdata_connect('path/to/data.d') as td:
        ...     # Get all centroided MS1 spectra with 1/K0 (default, parallel)
        ...     for spectrum in get_centroided_ms1_spectra(td):
        ...         print(f"Spectrum {spectrum.spectrum_index}: {spectrum.num_peaks} centroided peaks")
        ...
        ...     # Get spectra with CCS values using 8 workers
        ...     for spectrum in get_centroided_ms1_spectra(td, ion_mobility_type="ccs", num_workers=8):
        ...         print(f"CCS spectrum: {spectrum.num_peaks} peaks")
        ...
        ...     # Sequential processing (no parallelism)
        ...     spectra = list(get_centroided_ms1_spectra(
        ...         td, mz_tolerance=10, im_tolerance=0.1, num_workers=1
        ...     ))
        ...
        ...     # With noise filtering and parallel processing
        ...     for spectrum in get_centroided_ms1_spectra(td, noise_filter="mad", num_workers=4):
        ...         print(f"Noise-filtered spectrum: {spectrum.num_peaks} peaks")
    """
    logger.info(
        "Starting batch MS1 centroided spectrum extraction (frame_ids=%s, noise_filter=%s)",
        'all MS1' if frame_ids is None else f'{len(frame_ids)} specified', noise_filter
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
    # Determine number of workers
    if num_workers is None:
        num_workers = os.cpu_count() or 4
        num_workers = min(num_workers, len(frame_ids))
    num_workers = max(1, num_workers)  # At least 1 worker
    
    logger.info("Using %d worker threads for parallel processing", num_workers)
    logger.info("Using %d worker threads for parallel processing", num_workers)

    # Sequential processing if only 1 worker
    if num_workers == 1:
        logger.debug("Sequential processing mode (num_workers=1)")
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
                )
                successful_count += 1
                logger.info(
                    "Successfully extracted centroided spectrum %d/%d: frame_id=%d", idx + 1, len(frame_ids), frame_id
                )
                yield spectrum
            except (ValueError, RuntimeError) as e:
                # Log warning but continue processing
                failed_count += 1
                logger.warning(
                    "Failed to extract spectrum for frame %d (%d/%d): %s", frame_id, idx + 1, len(frame_ids), e
                )
                continue

        logger.info(
            "Batch centroiding complete: %d successful, %d failed, %d total",
            successful_count, failed_count, len(frame_ids)
        )
        return

    # Parallel processing with producer-consumer architecture
    logger.debug("Parallel processing mode with %d workers", num_workers)
    
    # Pre-read all frame data in main thread (avoids SQLite threading issues)
    logger.debug("Pre-reading frame metadata and raw data in main thread")
    frame_data_list = []
    
    for idx, frame_id in enumerate(frame_ids):
        try:
            # Read frame metadata from database (main thread only)
            cursor = td.conn.cursor()
            cursor.execute(
                "SELECT Time, NumScans, MsMsType FROM Frames WHERE Id = ?", (frame_id,)
            )
            result = cursor.fetchone()
            
            if result is None:
                logger.warning("Frame %d not found in database, skipping", frame_id)
                continue
                
            retention_time_sec, num_scans, msms_type = result
            
            if msms_type != 0:
                logger.warning("Frame %d is not an MS1 frame, skipping", frame_id)
                continue
            
            if num_scans == 0:
                logger.warning("Frame %d has 0 scans, skipping", frame_id)
                continue
            
            # Read raw scan data (DLL calls release GIL)
            ion_mobility = td.scanNumToOneOverK0(frame_id, np.arange(0, num_scans))
            results = td.readScans(frame_id, 0, num_scans)
            
            # Collect all raw peaks
            total_peaks = sum(len(idx_arr) for idx_arr, _ in results)
            
            if total_peaks == 0:
                logger.warning("Frame %d has 0 peaks, skipping", frame_id)
                continue
            
            # Pre-allocate and populate arrays
            mz_array = np.empty(total_peaks, dtype=np.float64)
            intensity_array = np.empty(total_peaks, dtype=np.float64)
            ion_mobility_array = np.empty(total_peaks, dtype=np.float64)
            
            offset = 0
            for scan_index, (index_array, intensity_scan) in enumerate(results):
                n_peaks = len(index_array)
                if n_peaks == 0:
                    continue
                
                # Convert indices to m/z (DLL call, releases GIL)
                mz_values = td.indexToMz(frame_id, index_array)
                
                mz_array[offset : offset + n_peaks] = mz_values
                intensity_array[offset : offset + n_peaks] = intensity_scan
                ion_mobility_array[offset : offset + n_peaks] = ion_mobility[scan_index]
                offset += n_peaks
            
            # Trim arrays
            mz_array = mz_array[:offset]
            intensity_array = intensity_array[:offset]
            ion_mobility_array = ion_mobility_array[:offset]
            
            # Convert to CCS in main thread if requested (avoids DLL calls in workers)
            if ion_mobility_type == "ccs":
                ccs_array = np.empty_like(ion_mobility_array)
                for i in range(len(ion_mobility_array)):
                    ccs_array[i] = oneOverK0ToCCSforMz(ion_mobility_array[i], 1, mz_array[i])
                ion_mobility_array = ccs_array
            
            frame_data_list.append({
                'idx': idx,
                'frame_id': frame_id,
                'retention_time': retention_time_sec / 60.0,
                'mz_array': mz_array,
                'intensity_array': intensity_array,
                'ion_mobility_array': ion_mobility_array,
            })
            
            if (len(frame_data_list)) % 100 == 0:
                logger.info("Pre-read %d/%d frames", len(frame_data_list), len(frame_ids))
                
        except Exception as e:
            logger.warning("Failed to pre-read frame %d: %s", frame_id, e)
            continue
    
    logger.info("Pre-read complete: %d/%d frames loaded", len(frame_data_list), len(frame_ids))
    
    # Now process centroiding in parallel (CPU-intensive, no DB/DLL access)
    result_queue: Queue = Queue()
    stats = {"successful": 0, "failed": 0}
    
    def worker(task_queue: Queue) -> None:
        """Worker thread that performs centroiding on pre-loaded data."""
        while True:
            item = task_queue.get()
            if item is None:  # Sentinel to stop worker
                task_queue.task_done()
                break
            
            frame_data = item
            idx = frame_data['idx']
            frame_id = frame_data['frame_id']
            
            try:
                # Copy arrays to ensure independent memory per thread (avoids cache line bouncing)
                mz_array = frame_data['mz_array']
                intensity_array = frame_data['intensity_array']
                ion_mobility_array = frame_data['ion_mobility_array']
                
                # Apply noise filtering if requested (vectorized)
                if noise_filter is not None:
                    noise_threshold = estimate_noise_level(intensity_array, method=noise_filter)
                    noise_mask = intensity_array >= noise_threshold
                    mz_array = mz_array[noise_mask]
                    intensity_array = intensity_array[noise_mask]
                    ion_mobility_array = ion_mobility_array[noise_mask]
                
                # CCS conversion already done in pre-read phase, just do centroiding
                # Apply peak centroiding (CPU-intensive, pure NumPy)
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
                )
                
                spectrum = Ms1Spectrum(
                    spectrum_index=idx,
                    frame_id=frame_id,
                    retention_time=frame_data['retention_time'],
                    peaks=peaks,
                    ion_mobility_type=ion_mobility_type,
                )
                
                result_queue.put((idx, spectrum, None))
                stats["successful"] += 1
                if (stats["successful"] + stats["failed"]) % 100 == 0:
                    logger.info(
                        "Progress: %d/%d frames processed", 
                        stats["successful"] + stats["failed"], 
                        len(frame_data_list)
                    )
                
            except Exception as e:
                result_queue.put((idx, None, e))
                stats["failed"] += 1
                logger.warning("Failed to process frame %d: %s", frame_id, e)
            finally:
                task_queue.task_done()
    
    # Create task queue and start workers
    task_queue: Queue = Queue()
    workers = []
    
    for _ in range(num_workers):
        t = threading.Thread(target=worker, args=(task_queue,), daemon=True)
        t.start()
        workers.append(t)
    
    # Producer thread to feed tasks (runs in background)
    def producer() -> None:
        """Producer feeds pre-loaded frame data to workers."""
        for frame_data in frame_data_list:
            task_queue.put(frame_data)
        
        # Send sentinel values
        for _ in range(num_workers):
            task_queue.put(None)
    
    producer_thread = threading.Thread(target=producer, daemon=True)
    producer_thread.start()
    
    # Collect results (main thread)
    if ordered:
        # Ordered mode: buffer results and yield in frame_id order
        results_dict = {}
        next_idx = 0
        total_expected = len(frame_data_list)
        
        while next_idx < total_expected:
            idx, spectrum, error = result_queue.get()
            results_dict[idx] = (spectrum, error)
            
            # Yield results in order
            while next_idx in results_dict:
                spectrum, error = results_dict.pop(next_idx)
                if error is None:
                    yield spectrum
                next_idx += 1
    else:
        # Unordered mode: yield results as soon as they're ready (faster)
        completed = 0
        total_expected = len(frame_data_list)
        
        while completed < total_expected:
            idx, spectrum, error = result_queue.get()
            completed += 1
            if error is None:
                yield spectrum
    
    # Wait for completion
    producer_thread.join()
    task_queue.join()
    for t in workers:
        t.join()
    
    logger.info(
        "Batch centroiding complete: %d successful, %d failed, %d total",
        stats["successful"], stats["failed"], len(frame_data_list)
    )
