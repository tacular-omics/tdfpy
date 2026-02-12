"""
Package for working with TDF (Bruker Data File) data.

This package provides classes and functions for reading and manipulating TDF data using pandas DataFrames and
the TimsData format.

Modules:
- pandas_tdf: Contains the PandasTdf class for working with TDF data using pandas DataFrames.
- timsdata: Contains the TimsData class for working with TimsData format.
- spectra: Contains higher-level Pythonic API for working with MS1 spectrum data.
- noise: Contains noise estimation functions for intensity filtering.

Attributes:
- __version__ (str): The current version of the package.
"""

from .tdf import PandasTdf
from .timsdata import TimsData, timsdata_connect
from .centroiding import (
    merge_peaks,
    get_centroided_spectrum,
)
from .reader import (
    DDA,
    DIA,
    PRM,
    MetaData,
    Calibration,
    DIAMs1Frame,
    DiaWindow,
    DDAMs1Frame,
    Precursor,
    PasefFrameMsmsInfo,
)

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
    "DDAMs1Frame",
    "Precursor",
    "PasefFrameMsmsInfo",
]
