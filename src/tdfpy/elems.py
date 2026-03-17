import datetime
import warnings
from dataclasses import dataclass
from enum import Enum, StrEnum
from functools import partial
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from .centroiding import get_centroided_spectrum
from .timsdata import TimsData, oneOverK0ToCCSforMz


class MsMsType(Enum):
    MS1 = 0
    DDA_MS2 = 8
    DIA_MS2 = 9


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN = "unknown"
    MIXED = "mixed"

    @staticmethod
    def from_str(s: str) -> "Polarity":
        """Convert a string to a `Polarity` enum value.

        Args:
            s: Polarity string. Accepted values (case-insensitive):
                `"positive"` or `"+"` → `Polarity.POSITIVE`;
                `"negative"` or `"-"` → `Polarity.NEGATIVE`;
                `"unknown"` or `"?"` → `Polarity.UNKNOWN`;
                `"mixed"` or `"mix"` → `Polarity.MIXED`.

        Returns:
            The matching `Polarity` enum member.

        Raises:
            ValueError: If the string does not match any known polarity.
        """
        s = s.lower()
        if s in ("positive", "+"):
            return Polarity.POSITIVE
        elif s in ("negative", "-"):
            return Polarity.NEGATIVE
        elif s in ("unknown", "unkown", "?"):
            return Polarity.UNKNOWN
        elif s in ("mixed", "mix"):
            return Polarity.MIXED
        else:
            raise ValueError(f"Unknown polarity string: {s}")


@dataclass
class _TdfData:
    _timsdata: TimsData

    @property
    def timsdata(self) -> TimsData:
        if self._timsdata.handle is None:
            raise RuntimeError(
                "TimsData connection is closed. Keep DFolder instance alive or use context manager."
            )
        return self._timsdata


@dataclass
class PasefFrameMsmsInfo(_TdfData):
    """A single PASEF MS/MS isolation window within a parent frame.

    Each instance corresponds to one row in the `PasefFrameMsMsInfo` table: a
    contiguous range of mobility scans acquired with a specific isolation window and
    collision energy, linked to a precursor ion.

    | Field | Type | Description |
    |---|---|---|
    | `frame_id` | `int` | Parent MS1 frame ID |
    | `scan_num_begin` | `int` | First mobility scan (inclusive) |
    | `scan_num_end` | `int` | Last mobility scan (inclusive) |
    | `isolation_mz` | `float` | Isolation window center m/z |
    | `isolation_width` | `float` | Isolation window width in Th |
    | `collision_energy` | `float` | Collision energy in eV |
    | `precursor` | `int \\| None` | Associated precursor ID |
    | `rt` | `float` | Retention time in seconds (from parent frame) |
    | `polarity` | `Polarity` | Ion polarity |
    """

    frame_id: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float
    precursor: int | None
    rt: float
    polarity: Polarity

    @property
    def unique_id(self) -> tuple[int, int | None]:
        if self.precursor is None:
            warnings.warn(
                f"Precursor is None for frame {self.frame_id}. Unique ID will be (frame_id, None).",
                UserWarning,
                stacklevel=2,
            )
        return (self.frame_id, self.precursor)

    @property
    def peaks(self) -> np.ndarray:
        scan = self.timsdata.extractCentroidedSpectrumForFrame(
            self.frame_id, self.scan_num_begin, self.scan_num_end
        )
        if scan is None:
            raise ValueError(
                f"Could not extract spectrum for frame {self.frame_id} with scan range {self.scan_num_begin}-{self.scan_num_end}."
            )
        mz_array = np.array(scan[0])
        intensity_array = np.array(scan[1])
        peaks = np.stack((mz_array, intensity_array), axis=-1)
        return peaks

    @property
    def scan_num_range(self) -> tuple[int, int]:
        return (self.scan_num_begin, self.scan_num_end)

    @property
    def ook0_begin(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.frame_id, [self.scan_num_begin])[0]

    @property
    def ook0_end(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.frame_id, [self.scan_num_end])[0]

    @property
    def ook0_range(self) -> tuple[float, float]:
        return (self.ook0_begin, self.ook0_end)

    @property
    def ccs_begin(self) -> float:
        return oneOverK0ToCCSforMz(self.ook0_begin, 1, self.isolation_mz)

    @property
    def ccs_end(self) -> float:
        return oneOverK0ToCCSforMz(self.ook0_end, 1, self.isolation_mz)

    @property
    def ccs_range(self) -> tuple[float, float]:
        return (self.ccs_begin, self.ccs_end)

    @property
    def voltage_begin(self) -> float:
        return self.timsdata.scanNumToVoltage(self.frame_id, [self.scan_num_begin])[0]

    @property
    def voltage_end(self) -> float:
        return self.timsdata.scanNumToVoltage(self.frame_id, [self.scan_num_end])[0]

    @property
    def voltage_range(self) -> tuple[float, float]:
        return (self.voltage_begin, self.voltage_end)

    @property
    def mz_begin(self) -> float:
        return self.isolation_mz - self.isolation_width / 2

    @property
    def mz_end(self) -> float:
        return self.isolation_mz + self.isolation_width / 2

    @property
    def mz_range(self) -> tuple[float, float]:
        return (self.mz_begin, self.mz_end)


@dataclass
class Precursor(_TdfData):
    """A detected precursor ion from a DDA acquisition.

    Combines data from the `Precursors` table with its associated PASEF MS/MS
    scan windows (`PasefFrameMsmsInfo`). Provides direct access to ion mobility,
    CCS, and fragmentation spectra.

    | Field | Type | Description |
    |---|---|---|
    | `precursor_id` | `int` | Unique precursor ID |
    | `largest_peak_mz` | `float` | m/z of the most intense isotope peak |
    | `average_mz` | `float` | Intensity-weighted average m/z |
    | `monoisotopic_mz` | `float \\| None` | Monoisotopic m/z (if determined) |
    | `charge` | `int \\| None` | Charge state (if determined) |
    | `scan_number` | `int` | Mobility scan bin containing this precursor |
    | `intensity` | `float` | Summed precursor intensity |
    | `parent_frame` | `int` | MS1 frame ID |
    | `pasef_frame_msms_infos` | `tuple[PasefFrameMsmsInfo, ...]` | Associated PASEF MS/MS windows |
    | `rt` | `float` | Retention time in seconds |
    """

    precursor_id: int
    largest_peak_mz: float
    average_mz: float
    monoisotopic_mz: float | None
    charge: int | None
    scan_number: int
    intensity: float
    parent_frame: int
    pasef_frame_msms_infos: tuple[PasefFrameMsmsInfo, ...]
    rt: float

    @property
    def ook0(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.parent_frame, [self.scan_number])[
            0
        ]

    @property
    def ccs(self) -> float:
        return oneOverK0ToCCSforMz(
            self.ook0, self.charge or 1, self.monoisotopic_mz or self.largest_peak_mz
        )

    @property
    def voltage(self) -> float:
        return self.timsdata.scanNumToVoltage(self.parent_frame, [self.scan_number])[0]

    @property
    def peaks(self) -> npt.NDArray[np.float64]:
        prec_map: dict[int, Any] = self.timsdata.readPasefMsMs([self.precursor_id])
        if self.precursor_id not in prec_map:
            raise ValueError(f"Precursor ID {self.precursor_id} not found in TimsData.")
        scan = prec_map[self.precursor_id]
        mz_array = np.array(scan[0])
        intensity_array = np.array(scan[1])
        peaks = np.stack((mz_array, intensity_array), axis=-1)
        return peaks

    @property
    def pasef_peaks(self) -> list[np.ndarray]:
        return [pasef_info.peaks for pasef_info in self.pasef_frame_msms_infos]

    def _get_pasef_frame_single_value(self, attr: str) -> Any:
        values = {getattr(info, attr) for info in self.pasef_frame_msms_infos}
        if len(values) == 0:
            warnings.warn(
                f"No values found for attribute '{attr}' in pasef_frame_msms_infos. Returning None.",
                UserWarning,
                stacklevel=2,
            )
            return None
        if len(values) > 1:
            warnings.warn(
                f"Multiple values found for attribute '{attr}' in pasef_frame_msms_infos. Returning None.",
                UserWarning,
                stacklevel=2,
            )
            return None
        return values.pop()

    @property
    def scan_num_range(self) -> tuple[int, int] | None:
        return self._get_pasef_frame_single_value("scan_num_range")

    @property
    def ook0_range(self) -> tuple[float, float] | None:
        return self._get_pasef_frame_single_value("ook0_range")

    @property
    def ccs_range(self) -> tuple[float, float] | None:
        return self._get_pasef_frame_single_value("ccs_range")

    @property
    def voltage_range(self) -> tuple[float, float] | None:
        return self._get_pasef_frame_single_value("voltage_range")

    @property
    def mz_range(self) -> tuple[float, float] | None:
        return self._get_pasef_frame_single_value("mz_range")

    @property
    def polarity(self) -> Polarity:
        polarities = {info.polarity for info in self.pasef_frame_msms_infos}
        if len(polarities) == 0:
            warnings.warn(
                "No polarities found in pasef_frame_msms_infos. Returning 'unknown' for polarity.",
                UserWarning,
                stacklevel=2,
            )
            return Polarity.UNKNOWN
        if len(polarities) != 1:
            warnings.warn(
                "Multiple polarities found in pasef_frame_msms_infos. Returning 'mixed' for polarity.",
                UserWarning,
                stacklevel=2,
            )
            return Polarity.MIXED
        return polarities.pop()

    @property
    def collision_energy(self) -> float | None:
        return self._get_pasef_frame_single_value("collision_energy")


@dataclass
class DiaWindowGroup:
    """A DIA isolation window definition (shared across frames in the same group).

    Defines the m/z isolation range, mobility scan range, and collision energy
    for one window within a DIA window group. `DiaWindow` extends this with
    per-frame fields (`frame_id`, `rt`, `polarity`).

    | Field | Type | Description |
    |---|---|---|
    | `window_index` | `int` | Index of this window within its group |
    | `window_group` | `int` | Window group ID |
    | `scan_num_begin` | `int` | First mobility scan (inclusive) |
    | `scan_num_end` | `int` | Last mobility scan (inclusive) |
    | `isolation_mz` | `float` | Isolation window center m/z |
    | `isolation_width` | `float` | Isolation window width in Th |
    | `collision_energy` | `float` | Collision energy in eV |
    """

    window_index: int
    window_group: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float

    @property
    def scan_num_range(self) -> tuple[int, int]:
        return (self.scan_num_begin, self.scan_num_end)

    @property
    def mz_begin(self) -> float:
        return self.isolation_mz - self.isolation_width / 2

    @property
    def mz_end(self) -> float:
        return self.isolation_mz + self.isolation_width / 2

    @property
    def mz_range(self) -> tuple[float, float]:
        return (self.mz_begin, self.mz_end)


@dataclass
class Frame(_TdfData):
    """Base class for a single timsTOF acquisition frame.

    A frame represents one complete TIMS-MS acquisition cycle. All fields below
    are present on `DDAMs1Frame` and `DIAMs1Frame` through inheritance.
    """

    frame_id: int
    """Unique frame ID (1-based)."""
    time: float
    """Acquisition time in seconds."""
    polarity: Polarity
    """Ion polarity of the acquisition."""
    scan_mode: int
    """Scan mode integer from the TDF schema."""
    msms_type: int
    """MS/MS type (0 = MS1, 8 = DDA MS2, 9 = DIA MS2)."""
    tims_id: int | None
    """TIMS device ID, if present."""
    max_intensity: int
    """Maximum peak intensity across all scans in this frame."""
    summed_intensities: int
    """Sum of all peak intensities in this frame."""
    num_scans: int
    """Number of TIMS scans (mobility bins) in this frame."""
    num_peaks: int
    """Total number of peaks across all scans in this frame."""
    mz_calibration: int
    """Reference to the m/z calibration entry."""
    t1: float
    """TIMS calibration coefficient T1."""
    t2: float
    """TIMS calibration coefficient T2."""
    tims_calibration: int
    """Reference to the TIMS calibration entry."""
    property_group: int | None
    """Reference to the property group entry, if present."""
    accumulation_time: float
    """Ion accumulation time in milliseconds."""
    ramp_time: float
    """TIMS ramp time in milliseconds."""

    @property
    def peaks(self) -> list[npt.NDArray[np.float64]]:
        """Read raw peaks for this frame and return as list of (mz, intensity) arrays."""
        d: list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = (
            self.timsdata.readScans(self.frame_id, 0, self.num_scans)
        )
        mz_int_arrays = []
        for index_array, int_array in d:
            mz_array = self.timsdata.indexToMz(self.frame_id, index_array)
            mz_int_arrays.append(
                np.stack((mz_array, int_array), axis=-1).astype(np.float64)
            )
        return mz_int_arrays

    def centroid(
        self,
        mz_tolerance: float = 8,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        im_tolerance: float = 0.05,
        im_tolerance_type: Literal["relative", "absolute"] = "relative",
        min_peaks: int = 3,
        max_peaks: int | None = None,
        noise_filter=None,
        ion_mobility_type: Literal["ccs", "ook0", "voltage"] = "ook0",
    ) -> np.ndarray:
        """Centroid the spectrum for this frame using the specified parameters."""
        get_spectrum = partial(
            get_centroided_spectrum,
            self.timsdata,
            frame_id=self.frame_id,
            spectrum_index=None,
            ion_mobility_type=ion_mobility_type,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            im_tolerance=im_tolerance,
            im_tolerance_type=im_tolerance_type,
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            noise_filter=noise_filter,
        )

        try:
            return get_spectrum(use_rust=True)
        except Exception:
            warnings.warn(
                f"Rust centroiding failed for frame {self.frame_id}. Falling back to Python implementation.",
                UserWarning,
                stacklevel=2,
            )
            return get_spectrum(use_rust=False)


@dataclass
class DDAMs1Frame(Frame):
    """An MS1 frame from a DDA acquisition.

    Inherits all fields from `Frame`. The `precursors` field lists every
    precursor detected in this frame.
    """

    precursors: tuple[Precursor, ...]
    """All precursors detected in this MS1 frame."""


@dataclass
class DiaWindow(DiaWindowGroup, _TdfData):
    """A DIA isolation window bound to a specific frame.

    Extends `DiaWindowGroup` with per-frame context (`frame_id`, `rt`,
    `polarity`). Provides raw and centroided spectrum access and ion mobility
    conversion properties.
    """

    frame_id: int
    """Frame ID this window belongs to."""
    rt: float
    """Retention time of the parent frame in seconds."""
    polarity: Polarity
    """Ion polarity."""

    @property
    def peaks(self) -> list[npt.NDArray[np.float64]]:
        """Read raw peaks for this DIA window and return as list of (mz, intensity) arrays."""
        d: list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]] = (
            self.timsdata.readScans(
                self.frame_id, self.scan_num_begin, self.scan_num_end
            )
        )
        mz_int_arrays = []
        for index_array, int_array in d:
            mz_array = self.timsdata.indexToMz(self.frame_id, index_array)
            mz_int_arrays.append(
                np.stack((mz_array, int_array), axis=-1).astype(np.float64)
            )
        return mz_int_arrays

    def centroid(
        self,
        mz_tolerance: float = 8,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        im_tolerance: float = 0.05,
        im_tolerance_type: Literal["relative", "absolute"] = "relative",
        min_peaks: int = 3,
        max_peaks: int | None = None,
        noise_filter=None,
        ion_mobility_type: Literal["ccs", "ook0", "voltage"] = "ook0",
    ) -> np.ndarray:
        """Centroid the spectrum for this DIA window using the specified parameters."""
        get_spectrum = partial(
            get_centroided_spectrum,
            self.timsdata,
            frame_id=self.frame_id,
            spectrum_index=None,
            ion_mobility_type=ion_mobility_type,
            mz_tolerance=mz_tolerance,
            mz_tolerance_type=mz_tolerance_type,
            im_tolerance=im_tolerance,
            im_tolerance_type=im_tolerance_type,
            min_peaks=min_peaks,
            max_peaks=max_peaks,
            noise_filter=noise_filter,
        )

        try:
            return get_spectrum(use_rust=True)
        except Exception:
            warnings.warn(
                f"Rust centroiding failed for frame {self.frame_id}. Falling back to Python implementation.",
                UserWarning,
                stacklevel=2,
            )
            return get_spectrum(use_rust=False)

    @property
    def ook0_begin(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.frame_id, [self.scan_num_begin])[0]

    @property
    def ook0_end(self) -> float:
        return self.timsdata.scanNumToOneOverK0(self.frame_id, [self.scan_num_end])[0]

    @property
    def ook0_range(self) -> tuple[float, float]:
        return (self.ook0_begin, self.ook0_end)

    @property
    def ccs_begin(self) -> float:
        return oneOverK0ToCCSforMz(self.ook0_begin, 1, self.isolation_mz)

    @property
    def ccs_end(self) -> float:
        return oneOverK0ToCCSforMz(self.ook0_end, 1, self.isolation_mz)

    @property
    def ccs_range(self) -> tuple[float, float]:
        return (self.ccs_begin, self.ccs_end)

    @property
    def voltage_begin(self) -> float:
        return self.timsdata.scanNumToVoltage(self.frame_id, [self.scan_num_begin])[0]

    @property
    def voltage_end(self) -> float:
        return self.timsdata.scanNumToVoltage(self.frame_id, [self.scan_num_end])[0]

    @property
    def voltage_range(self) -> tuple[float, float]:
        return (self.voltage_begin, self.voltage_end)


@dataclass
class DIAMs1Frame(Frame):
    """An MS1 frame from a DIA acquisition.

    Inherits all fields from `Frame`. The `dia_windows` field lists the DIA
    isolation windows that were active during this frame.
    """

    dia_windows: tuple[DiaWindow, ...]
    """All DIA windows associated with this MS1 frame."""


@dataclass
class _KeyDf:
    df: pd.DataFrame

    def __getitem__(self, key: str) -> str:
        if key not in self.df.index:
            raise KeyError(f"Calibration ID {key} not found.")
        return self.df.loc[key]


@dataclass
class Calibration(_KeyDf):
    """
    Example Calibration table keys:

    ```
    CalibrationDateTime              2018-08-21T16:50:31+02:00
    CalibrationUser                  Demo User
    CalibrationSoftware              Bruker otofControl
    CalibrationSoftwareVersion       5.1.81.714-13047
    MzCalibrationMode                3
    MzStandardDeviationPPM           0.130754
    ReferenceMassList                Tuning Mix ES-TOF (ESI)
    MobilityCalibrationDateTime      2018-08-21T16:49:17+02:00
    MobilityCalibrationUser          Demo User
    MobilityStandardDeviationPercent 0.000932
    ReferenceMobilityList            Tuning Mix ES-TOF (ESI)
    ```
    """

    @property
    def date(self) -> datetime.datetime:
        # Example: CalibrationDateTime 2018-08-21T16:50:31+02:00
        cdatetime = self["CalibrationDateTime"]
        return datetime.datetime.fromisoformat(cdatetime)

    @property
    def user(self) -> str:
        return self["CalibrationUser"]

    @property
    def software(self) -> str:
        return self["CalibrationSoftware"]

    @property
    def software_version(self) -> str:
        return self["CalibrationSoftwareVersion"]

    @property
    def mode(self) -> str:
        return self["CalibrationMode"]

    @property
    def std_ppm(self) -> float:
        return float(self["CalibrationStdPpm"])

    @property
    def reference_masses(self) -> str:
        return self["ReferenceMassList"]

    @property
    def mobility_calibration_date(self) -> datetime.datetime:
        mcdatetime = self["MobilityCalibrationDateTime"]
        return datetime.datetime.fromisoformat(mcdatetime)

    @property
    def mobility_calibration_user(self) -> str:
        return self["MobilityCalibrationUser"]

    @property
    def mobility_standard_deviation_percent(self) -> float:
        return float(self["MobilityStandardDeviationPercent"])

    @property
    def reference_mobility_list(self) -> str:
        return self["ReferenceMobilityList"]


@dataclass
class MetaData(_KeyDf):
    """
    Example GlobalMetaData table keys:

    ```
    SchemaType                  TDF
    SchemaVersionMajor          3
    SchemaVersionMinor          1
    AcquisitionSoftwareVendor   Bruker
    InstrumentVendor            Bruker
    TimsCompressionType         2
    ClosedProperly              1
    MaxNumPeaksPerScan          1412
    AnalysisId                  00000000-0000-0000-0000-000000000000
    MzAcqRangeLower             100.000000
    MzAcqRangeUpper             1700.000000
    OneOverK0AcqRangeLower      0.578703
    OneOverK0AcqRangeUpper      1.524471
    AcquisitionSoftware         Bruker otofControl
    AcquisitionSoftwareVersion  5.1.81.714-13047-vc110
    AcquisitionDateTime         2018-08-21T20:40:14.356+02:00
    InstrumentName              timsTOF Pro
    InstrumentSerialNumber      1844426.34
    OperatorName                Demo User
    SampleName                  200ngHeLaDIAPASEF_CE8V1st10VPASEF
    ```
    """

    @property
    def schema_type(self) -> str:
        """Schema type (typically 'TDF')."""
        return self["SchemaType"]

    @property
    def schema_version_major(self) -> int:
        """Major version of the TDF schema."""
        return int(self["SchemaVersionMajor"])

    @property
    def schema_version_minor(self) -> int:
        """Minor version of the TDF schema."""
        return int(self["SchemaVersionMinor"])

    @property
    def acquisition_software_vendor(self) -> str:
        """Vendor of acquisition software."""
        return self["AcquisitionSoftwareVendor"]

    @property
    def instrument_vendor(self) -> str:
        """Instrument vendor."""
        return self["InstrumentVendor"]

    @property
    def tims_compression_type(self) -> int:
        """TIMS data compression type."""
        return int(self["TimsCompressionType"])

    @property
    def closed_properly(self) -> bool:
        """Whether the acquisition was closed properly."""
        return bool(int(self["ClosedProperly"]))

    @property
    def max_num_peaks_per_scan(self) -> int:
        """Maximum number of peaks per scan."""
        return int(self["MaxNumPeaksPerScan"])

    @property
    def analysis_id(self) -> str:
        """Analysis UUID."""
        return self["AnalysisId"]

    @property
    def digitizer_num_samples(self) -> int:
        """Number of digitizer samples."""
        return int(self["DigitizerNumSamples"])

    @property
    def peak_list_index_scale_factor(self) -> int:
        """Peak list index scale factor."""
        return int(self["PeakListIndexScaleFactor"])

    @property
    def mz_acq_range_lower(self) -> float:
        """Lower m/z acquisition range."""
        return float(self["MzAcqRangeLower"])

    @property
    def mz_acq_range_upper(self) -> float:
        """Upper m/z acquisition range."""
        return float(self["MzAcqRangeUpper"])

    @property
    def mz_acq_range(self) -> tuple[float, float]:
        """M/z acquisition range as (lower, upper) tuple."""
        return (self.mz_acq_range_lower, self.mz_acq_range_upper)

    @property
    def one_over_k0_acq_range_lower(self) -> float:
        """Lower 1/K0 acquisition range."""
        return float(self["OneOverK0AcqRangeLower"])

    @property
    def one_over_k0_acq_range_upper(self) -> float:
        """Upper 1/K0 acquisition range."""
        return float(self["OneOverK0AcqRangeUpper"])

    @property
    def one_over_k0_acq_range(self) -> tuple[float, float]:
        """1/K0 acquisition range as (lower, upper) tuple."""
        return (self.one_over_k0_acq_range_lower, self.one_over_k0_acq_range_upper)

    # Acquisition software information
    @property
    def acquisition_software(self) -> str:
        """Acquisition software name."""
        return self["AcquisitionSoftware"]

    @property
    def acquisition_software_version(self) -> str:
        """Acquisition software version."""
        return self["AcquisitionSoftwareVersion"]

    @property
    def acquisition_firmware_version(self) -> str:
        """Acquisition firmware version."""
        return self["AcquisitionFirmwareVersion"]

    @property
    def acquisition_datetime(self) -> datetime.datetime:
        """Acquisition date and time."""
        adatetime = self["AcquisitionDateTime"]
        return datetime.datetime.fromisoformat(adatetime)

    # Instrument information
    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        return self["InstrumentName"]

    @property
    def instrument_family(self) -> int:
        """Instrument family code."""
        return int(self["InstrumentFamily"])

    @property
    def instrument_revision(self) -> int:
        """Instrument revision number."""
        return int(self["InstrumentRevision"])

    @property
    def instrument_source_type(self) -> int:
        """Instrument source type code."""
        return int(self["InstrumentSourceType"])

    @property
    def instrument_serial_number(self) -> str:
        """Instrument serial number."""
        return self["InstrumentSerialNumber"]

    # Sample and method information
    @property
    def operator_name(self) -> str:
        """Operator name."""
        return self["OperatorName"]

    @property
    def description(self) -> str:
        """Sample/acquisition description."""
        return self["Description"]

    @property
    def sample_name(self) -> str:
        """Sample name."""
        return self["SampleName"]

    @property
    def method_name(self) -> str:
        """Acquisition method name."""
        return self["MethodName"]
