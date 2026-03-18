"""
Package for working with TDF (Bruker Data File) data.
"""

from .centroiding import (
    get_centroided_spectrum,
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
    PasefFrameMsmsInfo,
    Precursor,
)
from .lookup import DiaWindowLookup, Ms1FrameLookup, PrecursorLookup
from .reader import (
    DDA,
    DIA,
    PRM,
    get_acquisition_type,
)
from .tdf import PandasTdf
from .timsdata import TimsData, timsdata_connect

__version__ = "1.0.2"

__all__ = [
    "PandasTdf",
    "TimsData",
    "timsdata_connect",
    "merge_peaks",
    "get_centroided_spectrum",
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
]
