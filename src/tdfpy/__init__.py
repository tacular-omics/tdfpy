"""
Package for working with TDF (Bruker Data File) data.
"""

from .centroiding import (
    get_centroided_spectrum,
    get_mobility_collapsed_spectrum,
    get_raw_peaks,
    merge_peaks,
)
from .elems import (
    Calibration,
    DDAMs1Frame,
    DIAMs1Frame,
    DiaWindow,
    DiaWindowGroup,
    Frame,
    MetaData,
    PRMMs1Frame,
    PasefFrameMsmsInfo,
    Precursor,
    PrmTarget,
    PrmTransition,
)
from .lookup import (
    DiaWindowLookup,
    Ms1FrameLookup,
    PrecursorLookup,
    PrmTargetLookup,
    PrmTransitionLookup,
)
from .noise import (
    AbsoluteThreshold,
    BaselineThreshold,
    DiaMs1WindowGate,
    HistogramThreshold,
    HorizontalHaloFilter,
    IntensityThreshold,
    IterativeMedianThreshold,
    MadThreshold,
    NoiseFilter,
    NoiseSpec,
    PercentileThreshold,
    SelectionPolygonGate,
    VerticalNoiseFilter,
    coerce_filters,
)
from .pipeline import (
    Centroider,
    MergePeaksCentroider,
    RawSpectrum,
    WatershedCentroider,
    Smooth,
    apply_noise,
    box_smooth,
    centroid_peaks,
    convert,
    exclude_region,
    read_spectrum,
    smooth,
    subset_scans,
)
from .reader import (
    DDA,
    DIA,
    PRM,
    get_acquisition_type,
)
from .regions import ChargeStateRegion
from .slicer import slice_d_folder
from .tdf import PandasTdf
from .calibration import UnsupportedCalibrationError
from .timsdata import TimsData, UnsupportedTdfError, timsdata_connect
from .viz import plot_centroiding
from .processing import iter_window_spectra
from .timsdata import FrameMetadata
from .validation import ValidationIssue, ValidationReport, validate_acquisition

__version__ = "4.0.1"

__all__ = [
    "FrameMetadata",
    "iter_window_spectra",
    "ValidationIssue",
    "ValidationReport",
    "validate_acquisition",
    # I/O
    "PandasTdf",
    "TimsData",
    "timsdata_connect",
    "UnsupportedTdfError",
    "UnsupportedCalibrationError",
    "DDA",
    "DIA",
    "PRM",
    "get_acquisition_type",
    "slice_d_folder",
    # Frame elements
    "Frame",
    "DDAMs1Frame",
    "DIAMs1Frame",
    "PRMMs1Frame",
    "DiaWindow",
    "DiaWindowGroup",
    "Precursor",
    "PasefFrameMsmsInfo",
    "PrmTarget",
    "PrmTransition",
    "MetaData",
    "Calibration",
    "DiaWindowLookup",
    "Ms1FrameLookup",
    "PrecursorLookup",
    "PrmTargetLookup",
    "PrmTransitionLookup",
    # Convenience peak extraction
    "get_raw_peaks",
    "get_centroided_spectrum",
    "get_mobility_collapsed_spectrum",
    "merge_peaks",
    # Pipeline ops (power-user composable API)
    "RawSpectrum",
    "read_spectrum",
    "subset_scans",
    "exclude_region",
    "smooth",
    "box_smooth",
    "Smooth",
    "apply_noise",
    "convert",
    "centroid_peaks",
    # Centroiders
    "Centroider",
    "MergePeaksCentroider",
    "WatershedCentroider",
    # Region exclusion
    "ChargeStateRegion",
    # Noise filters
    "NoiseFilter",
    "NoiseSpec",
    "coerce_filters",
    "IntensityThreshold",
    "AbsoluteThreshold",
    "MadThreshold",
    "PercentileThreshold",
    "HistogramThreshold",
    "BaselineThreshold",
    "IterativeMedianThreshold",
    "VerticalNoiseFilter",
    "HorizontalHaloFilter",
    "SelectionPolygonGate",
    "DiaMs1WindowGate",
    # Visualization
    "plot_centroiding",
]
