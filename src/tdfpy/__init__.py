"""
Package for working with TDF (Bruker Data File) data.
"""

from .centroiding import (
    get_centroided_spectrum,
    merge_peaks,
)
from .reader import (
    DDA,
    DIA,
    PRM,
)
from .elems import(
    Calibration,
    DDAMs1Frame,
    DIAMs1Frame,
    DiaWindow,
    DiaWindowGroup,
    MetaData,
    PasefFrameMsmsInfo,
    Precursor,
)

from .tdf import PandasTdf
from .timsdata import TimsData, timsdata_connect
from .lookup import DiaWindowLookup, Ms1FrameLookup, PrecursorLookup

__version__ = "1.0.0"

__all__ = [
    "PandasTdf",
    "TimsData",
    "timsdata_connect",
    "merge_peaks",
    "get_centroided_spectrum",
    "DDA",
    "DIA",
    "PRM",
    "MetaData",
    "Calibration",
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
