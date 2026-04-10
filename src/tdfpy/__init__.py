"""
Package for working with TDF (Bruker Data File) data.
"""

from .centroiding import (
    get_centroided_spectrum,
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
from .reader import (
    DDA,
    DIA,
    PRM,
    get_acquisition_type,
)
from .slicer import slice_d_folder
from .viz import plot_centroiding
from .tdf import PandasTdf
from .timsdata import TimsData, timsdata_connect

__version__ = "1.2.0"

__all__ = [
    "PandasTdf",
    "TimsData",
    "timsdata_connect",
    "merge_peaks",
    "get_centroided_spectrum",
    "get_raw_peaks",
    "plot_centroiding",
    "DDA",
    "DIA",
    "PRM",
    "get_acquisition_type",
    "MetaData",
    "Calibration",
    "Frame",
    "DIAMs1Frame",
    "DiaWindow",
    "DiaWindowGroup",
    "DDAMs1Frame",
    "Precursor",
    "PasefFrameMsmsInfo",
    "DiaWindowLookup",
    "Ms1FrameLookup",
    "PrecursorLookup",
    "slice_d_folder",
    "PrmTarget",
    "PrmTransition",
    "PRMMs1Frame",
    "PrmTargetLookup",
    "PrmTransitionLookup",
]
