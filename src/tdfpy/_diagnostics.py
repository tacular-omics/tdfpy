"""Internal pipeline instrumentation for development. No stable public API."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .noise import NoiseSpec, coerce_filters
from .pipeline import (
    Centroider,
    MergePeaksCentroider,
    RawSpectrum,
    Smooth,
    _prepare_spectrum,
    read_spectrum,
)
from .regions import ChargeStateRegion
from .timsdata import FrameMetadata, TimsData


@dataclass(frozen=True)
class StageDiagnostics:
    """Counts and intensity after one processing stage.

    Original intensities are normalized to a 100 ms accumulation window.
    Once smoothing has run, subsequent totals use the modified intensities.
    """

    name: str
    num_peaks: int
    intensity_sum: float
    intensity_basis: Literal["original", "smoothed"]


@dataclass(frozen=True)
class ProcessingProvenance:
    """Serializable input identity, calibration, and configuration description.

    Custom operations are described by their repr, which their authors should
    make informative. This record describes execution, it is not a loader for
    arbitrary Python objects.
    """

    analysis_directory: str
    package_version: str
    numpy_version: str
    frame: FrameMetadata
    mz_calibration: tuple[float, ...]
    configuration: tuple[tuple[str, str], ...]
    ion_mobility_type: str


@dataclass(frozen=True, eq=False)
class ProcessedSpectrum:
    """Centroid array plus per-stage accounting and provenance."""

    peaks: np.ndarray
    stages: tuple[StageDiagnostics, ...]
    provenance: ProcessingProvenance


def _process(
    spectrum: RawSpectrum,
    td: TimsData,
    frame_id: int,
    *,
    scan_range: tuple[int, int] | None,
    exclude: ChargeStateRegion | None,
    smooth: Smooth | None,
    noise: NoiseSpec,
    centroid: Centroider | None,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"],
) -> ProcessedSpectrum:
    from . import __version__

    stages = []
    basis: Literal["original", "smoothed"] = "original"

    def observe(name: str, value: RawSpectrum) -> None:
        nonlocal basis
        if name == "smooth":
            basis = "smoothed"
        stages.append(
            StageDiagnostics(name, len(value), float(value.intensities.sum()), basis)
        )

    filters = coerce_filters(noise)
    cfg = centroid if centroid is not None else MergePeaksCentroider()
    prepared = _prepare_spectrum(
        spectrum,
        td,
        frame_id,
        scan_range=scan_range,
        exclude=exclude,
        smoothing=smooth,
        noise=filters,
        ion_mobility_type=ion_mobility_type,
        observe=observe,
    )
    peaks = (
        cfg(prepared, td, frame_id, ion_mobility_type=ion_mobility_type)
        if not prepared.empty
        else np.empty((0, 3), dtype=np.float64)
    )
    stages.append(
        StageDiagnostics("centroid", len(peaks), float(peaks[:, 1].sum()), basis)
    )
    configuration = [
        ("scan_range", repr(scan_range)),
        ("exclude", repr(exclude)),
        ("smooth", repr(smooth)),
        ("centroid", repr(cfg)),
    ]
    configuration.extend((f"noise[{i}]", repr(f)) for i, f in enumerate(filters))
    provenance = ProcessingProvenance(
        td.analysis_directory,
        __version__,
        np.__version__,
        td.frame_metadata(frame_id),
        td.mz_calibration_key(frame_id),
        tuple(configuration),
        ion_mobility_type,
    )
    return ProcessedSpectrum(peaks, tuple(stages), provenance)


def process_frame(
    td: TimsData,
    frame_id: int,
    *,
    scan_range: tuple[int, int] | None = None,
    exclude: ChargeStateRegion | None = None,
    smooth: Smooth | None = None,
    noise: NoiseSpec = None,
    centroid: Centroider | None = None,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
) -> ProcessedSpectrum:
    """Centroid a frame using the standard pipeline and record diagnostics.

    The result's peaks match get_centroided_spectrum with the same arguments.
    Existing array-returning entry points remain available. CCS assumes +1.
    """
    return _process(
        read_spectrum(td, frame_id),
        td,
        frame_id,
        scan_range=scan_range,
        exclude=exclude,
        smooth=smooth,
        noise=noise,
        centroid=centroid,
        ion_mobility_type=ion_mobility_type,
    )
