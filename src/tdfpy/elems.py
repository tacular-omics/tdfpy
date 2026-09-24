"""Frame elements returned by the :class:`~tdfpy.DDA`, :class:`~tdfpy.DIA` and
:class:`~tdfpy.PRM` readers.

Every element is a frozen, slotted dataclass. Elements that can read spectra
keep a reference to the reader's :class:`~tdfpy.TimsData`; after the reader is
closed their spectral accessors raise :class:`~tdfpy.ReaderClosedError`.

Peak array shapes, the same on every element:

* ``raw_peaks()`` -> one ``(N, 3)`` float64 array of ``[m/z, intensity, ion_mobility]``.
* ``centroid()`` -> one ``(N, 3)`` float64 array of ``[m/z, intensity, ion_mobility]``.
* ``scan_peaks()`` -> a ``list`` with one ``(N_i, 2)`` ``[m/z, intensity]`` raw
  array per mobility scan.
* ``Precursor.merged_peaks()`` / ``PasefFrameMsmsInfo.merged_peaks()`` -> one
  ``(N, 2)`` ``[m/z, intensity]`` array, mobility-collapsed by a greedy merge.
  This is the most expensive accessor; call it once and keep the result.

Every spectral accessor is a method, so each call visibly reads and decodes
the frame; nothing is cached on the element.
"""

import datetime
import logging
import warnings
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Literal

import numpy as np
import numpy.typing as npt

from .calibration import ook0_to_ccs
from .centroiding import (
    get_centroided_spectrum,
    get_mobility_collapsed_spectrum,
    get_raw_peaks,
)
from .errors import ReaderClosedError, TdfpyError, TdfpyKeyError
from .noise import NoiseSpec
from .pipeline import Centroider, Smooth
from .regions import ChargeStateRegion
from .timsdata import TimsData

__all__ = [
    "Calibration",
    "DDAMs1Frame",
    "DIAMs1Frame",
    "DiaWindow",
    "DiaWindowGroup",
    "Frame",
    "MetaData",
    "MetaValue",
    "MsMsType",
    "PasefFrameMsmsInfo",
    "Polarity",
    "Precursor",
    "PRMMs1Frame",
    "PrmTarget",
    "PrmTransition",
]

logger = logging.getLogger(__name__)

IonMobilityType = Literal["ook0", "ccs", "voltage"]


class MsMsType(IntEnum):
    """The ``MsMsType`` column of the ``Frames`` table."""

    MS1 = 0
    DDA_MS2 = 8
    DIA_MS2 = 9
    PRM_MS2 = 10


class Polarity(StrEnum):
    """Ion polarity of a frame."""

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
            TdfpyError: If the string does not match any known polarity.
        """
        s = s.lower()
        if s in ("positive", "+"):
            return Polarity.POSITIVE
        if s in ("negative", "-"):
            return Polarity.NEGATIVE
        if s in ("unknown", "unkown", "?"):
            return Polarity.UNKNOWN
        if s in ("mixed", "mix"):
            return Polarity.MIXED
        raise TdfpyError(f"Unknown polarity string {s!r}. Expected one of (case-insensitive): 'positive'/'+', 'negative'/'-', 'unknown'/'?', 'mixed'/'mix'.")


# ---------------------------------------------------------------------------
# Mixins. They hold no state (``__slots__ = ()``) so the frozen, slotted
# dataclasses below can combine them freely.
# ---------------------------------------------------------------------------


class _TdfData:
    """Gives an element checked access to the reader's :class:`TimsData`."""

    __slots__ = ()
    _timsdata: TimsData

    @property
    def timsdata(self) -> TimsData:
        """The reader's open :class:`TimsData`.

        Raises:
            ReaderClosedError: If the reader that built this element was closed.
        """
        td = self._timsdata
        if td.handle is None:
            raise ReaderClosedError("The reader that built this element is closed. Read spectra inside the reader's `with` block.")
        return td


class _Spectrum(_TdfData):
    """``scan_peaks`` / ``raw_peaks`` / ``centroid`` over one frame's scan range."""

    __slots__ = ()
    frame_id: int

    def _scan_bounds(self) -> tuple[int, int] | None:
        """``[begin, end)`` mobility scans this element covers; ``None`` = whole frame."""
        raise NotImplementedError

    def scan_peaks(self) -> list[npt.NDArray[np.float64]]:
        """Raw peaks, one ``(N_i, 2)`` ``[m/z, intensity]`` float64 array per mobility scan.

        The list has one entry per scan in this element's scan range, in scan
        order; empty scans give a ``(0, 2)`` array.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        td = self.timsdata
        bounds = self._scan_bounds()
        num_scans = td.frame_metadata(self.frame_id).num_scans
        begin, end = bounds if bounds is not None else (0, num_scans)
        begin, end = max(0, begin), min(end, num_scans)
        if begin >= end:
            return []
        arrays = []
        for index_array, int_array in td.read_scans(self.frame_id, begin, end):
            mz_array = td.index_to_mz(self.frame_id, index_array)
            arrays.append(np.stack((mz_array, int_array), axis=-1).astype(np.float64))
        return arrays

    def raw_peaks(
        self,
        *,
        exclude: ChargeStateRegion | None = None,
        smooth: Smooth | None = None,
        noise: NoiseSpec = None,
        ion_mobility_type: IonMobilityType = "ook0",
    ) -> npt.NDArray[np.float64]:
        """Raw peaks as one ``(N, 3)`` ``[m/z, intensity, ion_mobility]`` array.

        Restricted to this element's scan range. See :func:`tdfpy.get_raw_peaks`
        for the keyword arguments.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        return get_raw_peaks(
            self.timsdata,
            self.frame_id,
            scan_range=self._scan_bounds(),
            exclude=exclude,
            smooth=smooth,
            noise=noise,
            ion_mobility_type=ion_mobility_type,
        )

    def centroid(
        self,
        *,
        exclude: ChargeStateRegion | None = None,
        smooth: Smooth | None = None,
        noise: NoiseSpec = None,
        ion_mobility_type: IonMobilityType = "ook0",
        centroid: Centroider | None = None,
    ) -> npt.NDArray[np.float64]:
        """Centroided peaks as one ``(N, 3)`` ``[m/z, intensity, ion_mobility]`` array.

        Restricted to this element's scan range. The raw-peak keywords are
        passed to :func:`tdfpy.get_raw_peaks`; centroider settings live on
        :class:`~tdfpy.Centroider` (default :class:`~tdfpy.MergePeaksCentroider`).

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        return get_centroided_spectrum(
            self.timsdata,
            self.frame_id,
            scan_range=self._scan_bounds(),
            exclude=exclude,
            smooth=smooth,
            noise=noise,
            ion_mobility_type=ion_mobility_type,
            centroid=centroid,
        )


class _IsolationWindow:
    """m/z range properties of an isolation window."""

    __slots__ = ()
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float

    @property
    def scan_num_range(self) -> tuple[int, int]:
        """``(scan_num_begin, scan_num_end)``, end exclusive."""
        return (self.scan_num_begin, self.scan_num_end)

    @property
    def mz_begin(self) -> float:
        """Lower isolation edge, ``isolation_mz - isolation_width / 2``."""
        return self.isolation_mz - self.isolation_width / 2

    @property
    def mz_end(self) -> float:
        """Upper isolation edge, ``isolation_mz + isolation_width / 2``."""
        return self.isolation_mz + self.isolation_width / 2

    @property
    def mz_range(self) -> tuple[float, float]:
        """``(mz_begin, mz_end)``."""
        return (self.mz_begin, self.mz_end)


class _MobilityWindow(_IsolationWindow, _Spectrum):
    """A scan range in one frame: mobility ranges plus spectrum access."""

    __slots__ = ()

    def _scan_bounds(self) -> tuple[int, int] | None:
        return (self.scan_num_begin, self.scan_num_end)

    def _edge_values(self, convert: Callable[[int, npt.ArrayLike], npt.NDArray[np.float64]]) -> tuple[float, float]:
        """``(low, high)`` of ``convert`` at the two scan edges. Scan order runs high to low 1/K0."""
        values = convert(self.frame_id, [self.scan_num_begin, self.scan_num_end])
        a, b = float(values[0]), float(values[1])
        return (a, b) if a <= b else (b, a)

    @property
    def ook0_range(self) -> tuple[float, float]:
        """``(ook0_begin, ook0_end)``: lowest and highest 1/K0 (V·s/cm²) of the scan range, low first."""
        return self._edge_values(self.timsdata.scan_num_to_ook0)

    @property
    def ook0_begin(self) -> float:
        """Lowest 1/K0 (V·s/cm²) of the scan range."""
        return self.ook0_range[0]

    @property
    def ook0_end(self) -> float:
        """Highest 1/K0 (V·s/cm²) of the scan range."""
        return self.ook0_range[1]

    @property
    def ccs_range(self) -> tuple[float, float]:
        """``(ccs_begin, ccs_end)``: CCS (Å²) range for charge 1 at ``isolation_mz``, low first."""
        lo, hi = self.ook0_range
        a, b = ook0_to_ccs(lo, 1, self.isolation_mz), ook0_to_ccs(hi, 1, self.isolation_mz)
        return (a, b) if a <= b else (b, a)

    @property
    def ccs_begin(self) -> float:
        """Lowest CCS (Å²) of the scan range, for charge 1 at ``isolation_mz``."""
        return self.ccs_range[0]

    @property
    def ccs_end(self) -> float:
        """Highest CCS (Å²) of the scan range, for charge 1 at ``isolation_mz``."""
        return self.ccs_range[1]

    @property
    def voltage_range(self) -> tuple[float, float]:
        """``(voltage_begin, voltage_end)``: TIMS voltage (V) range of the scan range, low first."""
        return self._edge_values(self.timsdata.scan_num_to_voltage)

    @property
    def voltage_begin(self) -> float:
        """Lowest TIMS voltage (V) of the scan range."""
        return self.voltage_range[0]

    @property
    def voltage_end(self) -> float:
        """Highest TIMS voltage (V) of the scan range."""
        return self.voltage_range[1]


def _timsdata_field() -> TimsData:
    return field(repr=False, compare=False)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# DDA
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PasefFrameMsmsInfo(_MobilityWindow):
    """A single PASEF MS/MS isolation window within an MS/MS frame.

    One row of the `PasefFrameMsMsInfo` table: a contiguous range of mobility
    scans acquired with one isolation window and collision energy, linked to a
    precursor.

    | Field | Type | Description |
    |---|---|---|
    | `frame_id` | `int` | MS/MS frame the window was acquired in (not the parent MS1 frame) |
    | `scan_num_begin` | `int` | First mobility scan (inclusive) |
    | `scan_num_end` | `int` | Mobility scan range end (exclusive), so the range is `[scan_num_begin, scan_num_end)` |
    | `isolation_mz` | `float` | Isolation window center m/z |
    | `isolation_width` | `float` | Isolation window width in Th |
    | `collision_energy` | `float` | Collision energy in eV |
    | `precursor_id` | `int \\| None` | Associated precursor ID (`None` when the row has no precursor) |
    | `rt` | `float` | Retention time in seconds of that MS/MS frame |
    | `polarity` | `Polarity` | Ion polarity |
    """

    frame_id: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float
    precursor_id: int | None
    rt: float
    polarity: Polarity
    _timsdata: TimsData = _timsdata_field()

    @property
    def unique_id(self) -> tuple[int, int | None]:
        """``(frame_id, precursor_id)``. Warns when ``precursor_id`` is ``None``."""
        if self.precursor_id is None:
            warnings.warn(
                f"precursor_id is None for frame {self.frame_id}. Unique ID will be (frame_id, None).",
                UserWarning,
                stacklevel=2,
            )
        return (self.frame_id, self.precursor_id)

    def merged_peaks(self) -> npt.NDArray[np.float64]:
        """Mobility-collapsed MS/MS spectrum, one ``(N, 2)`` ``[m/z, intensity]`` array.

        Sums the window's scans (dropping ion mobility), then merges peaks
        greedily in m/z (30 ppm). This decodes and merges every scan, so it is
        the most expensive accessor; call it once and keep the result. For
        per-scan raw peaks use :meth:`scan_peaks`; for an ``(N, 3)`` spectrum
        that keeps ion mobility use :meth:`centroid`. See
        :func:`~tdfpy.get_mobility_collapsed_spectrum`.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        return get_mobility_collapsed_spectrum(
            self.timsdata,
            [(self.frame_id, self.scan_num_begin, self.scan_num_end)],
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class Precursor(_TdfData):
    """A detected precursor ion from a DDA acquisition.

    Combines one row of the `Precursors` table with its PASEF MS/MS windows.

    | Field | Type | Description |
    |---|---|---|
    | `precursor_id` | `int` | Unique precursor ID |
    | `largest_peak_mz` | `float` | m/z of the most intense isotope peak |
    | `average_mz` | `float` | Intensity-weighted average m/z |
    | `monoisotopic_mz` | `float \\| None` | Monoisotopic m/z (if determined) |
    | `charge` | `int \\| None` | Charge state (if determined) |
    | `scan_number` | `float` | Fractional mobility scan coordinate |
    | `intensity` | `float` | Summed precursor intensity |
    | `parent_frame_id` | `int` | MS1 frame the precursor was detected in |
    | `pasef_frame_msms_infos` | `tuple[PasefFrameMsmsInfo, ...]` | Associated PASEF MS/MS windows |
    | `rt` | `float` | Retention time in seconds of the parent frame |
    """

    precursor_id: int
    largest_peak_mz: float
    average_mz: float
    monoisotopic_mz: float | None
    charge: int | None
    scan_number: float
    intensity: float
    parent_frame_id: int
    pasef_frame_msms_infos: tuple[PasefFrameMsmsInfo, ...]
    rt: float
    _timsdata: TimsData = _timsdata_field()

    @property
    def ook0(self) -> float:
        """1/K0 (V·s/cm²) at ``scan_number`` in the parent frame."""
        return float(self.timsdata.scan_num_to_ook0(self.parent_frame_id, [self.scan_number])[0])

    @property
    def ccs(self) -> float:
        """CCS (Å²). Uses charge 1 and ``largest_peak_mz`` when charge or monoisotopic m/z is unknown."""
        return ook0_to_ccs(self.ook0, self.charge or 1, self.monoisotopic_mz or self.largest_peak_mz)

    @property
    def voltage(self) -> float:
        """TIMS voltage (V) at ``scan_number`` in the parent frame."""
        return float(self.timsdata.scan_num_to_voltage(self.parent_frame_id, [self.scan_number])[0])

    def merged_peaks(self) -> npt.NDArray[np.float64]:
        """Mobility-collapsed MS/MS spectrum, one ``(N, 2)`` ``[m/z, intensity]`` array.

        Sums every PASEF window of this precursor (which may span several
        frames), drops ion mobility, then merges peaks greedily in m/z
        (30 ppm). This decodes every window, so it is the most expensive
        accessor; call it once and keep the result. See
        :func:`~tdfpy.get_mobility_collapsed_spectrum`.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        return get_mobility_collapsed_spectrum(
            self.timsdata,
            [(info.frame_id, info.scan_num_begin, info.scan_num_end) for info in self.pasef_frame_msms_infos],
        )

    def pasef_merged_peaks(self) -> list[npt.NDArray[np.float64]]:
        """:meth:`PasefFrameMsmsInfo.merged_peaks` of each PASEF window, one ``(N_i, 2)`` array per window.

        Raises:
            ReaderClosedError: If the reader was closed.
        """
        return [pasef_info.merged_peaks() for pasef_info in self.pasef_frame_msms_infos]

    def _single_value[T](self, values: set[T], attr: str) -> T | None:
        if len(values) == 1:
            return values.pop()
        reason = "No values" if not values else "Multiple values"
        warnings.warn(
            f"{reason} found for attribute '{attr}' in pasef_frame_msms_infos. Returning None.",
            UserWarning,
            stacklevel=3,
        )
        return None

    @property
    def scan_num_range(self) -> tuple[int, int] | None:
        """Shared scan range of the PASEF windows, or ``None`` (with a warning) if they differ."""
        return self._single_value({i.scan_num_range for i in self.pasef_frame_msms_infos}, "scan_num_range")

    @property
    def ook0_range(self) -> tuple[float, float] | None:
        """Shared 1/K0 range of the PASEF windows, or ``None`` (with a warning) if they differ."""
        return self._single_value({i.ook0_range for i in self.pasef_frame_msms_infos}, "ook0_range")

    @property
    def ccs_range(self) -> tuple[float, float] | None:
        """Shared CCS range of the PASEF windows, or ``None`` (with a warning) if they differ."""
        return self._single_value({i.ccs_range for i in self.pasef_frame_msms_infos}, "ccs_range")

    @property
    def voltage_range(self) -> tuple[float, float] | None:
        """Shared voltage range of the PASEF windows, or ``None`` (with a warning) if they differ."""
        return self._single_value({i.voltage_range for i in self.pasef_frame_msms_infos}, "voltage_range")

    @property
    def mz_range(self) -> tuple[float, float] | None:
        """Shared isolation m/z range of the PASEF windows, or ``None`` (with a warning) if they differ."""
        return self._single_value({i.mz_range for i in self.pasef_frame_msms_infos}, "mz_range")

    @property
    def collision_energy(self) -> float | None:
        """Shared collision energy (eV) of the PASEF windows, or ``None`` (with a warning) if they differ."""
        return self._single_value({i.collision_energy for i in self.pasef_frame_msms_infos}, "collision_energy")

    @property
    def polarity(self) -> Polarity:
        """Shared polarity of the PASEF windows; ``UNKNOWN`` if none, ``MIXED`` if they differ (both warn)."""
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


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class Frame(_Spectrum):
    """Base class for a single timsTOF MS1 frame.

    A frame is one complete TIMS-MS acquisition cycle. All fields below are
    present on `DDAMs1Frame`, `DIAMs1Frame` and `PRMMs1Frame`.
    """

    frame_id: int
    """Unique frame ID (1-based)."""
    rt: float
    """Retention time in seconds (the `Time` column)."""
    polarity: Polarity
    """Ion polarity of the acquisition."""
    scan_mode: int
    """Scan mode integer from the TDF schema."""
    msms_type: MsMsType
    """MS/MS type of the frame."""
    tims_id: int | None
    """Byte offset of this frame's data block in `analysis.tdf_bin` (the `TimsId` column)."""
    max_intensity: int
    """Maximum peak intensity across all scans in this frame."""
    summed_intensities: int
    """Sum of all peak intensities in this frame."""
    num_scans: int
    """Number of TIMS scans (mobility bins) in this frame."""
    num_peaks: int
    """Total number of peaks across all scans in this frame."""
    mz_calibration_id: int
    """ID of the row in the `MzCalibration` table."""
    t1: float
    """Temperature T1 (°C) used by the m/z calibration."""
    t2: float
    """Temperature T2 (°C) used by the m/z calibration."""
    tims_calibration_id: int
    """ID of the row in the `TimsCalibration` table."""
    property_group_id: int | None
    """ID of the row in the `PropertyGroups` table, if present."""
    accumulation_time: float
    """Ion accumulation time in milliseconds."""
    ramp_time: float
    """TIMS ramp time in milliseconds."""
    _timsdata: TimsData = _timsdata_field()

    def _scan_bounds(self) -> tuple[int, int] | None:
        return None


@dataclass(frozen=True, slots=True, kw_only=True)
class DDAMs1Frame(Frame):
    """An MS1 frame from a DDA acquisition.

    Inherits all fields from `Frame`. `precursors` lists every precursor
    detected in this frame.
    """

    precursors: tuple[Precursor, ...]
    """All precursors detected in this MS1 frame."""


# ---------------------------------------------------------------------------
# DIA
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class DiaWindowGroup(_IsolationWindow):
    """One window of a DIA window group, shared by every frame that uses the group.

    One row of the `DiaFrameMsMsWindows` table. `DiaWindow` adds the per-frame
    fields (`frame_id`, `rt`, `polarity`).

    | Field | Type | Description |
    |---|---|---|
    | `window_index` | `int` | 0-based row index in the `DiaFrameMsMsWindows` table (not per group) |
    | `window_group_id` | `int` | Window group ID (the `WindowGroup` column) |
    | `scan_num_begin` | `int` | First mobility scan (inclusive) |
    | `scan_num_end` | `int` | Mobility scan range end (exclusive), so the range is `[scan_num_begin, scan_num_end)` |
    | `isolation_mz` | `float` | Isolation window center m/z |
    | `isolation_width` | `float` | Isolation window width in Th |
    | `collision_energy` | `float` | Collision energy in eV |
    """

    window_index: int
    window_group_id: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float


@dataclass(frozen=True, slots=True, kw_only=True)
class DiaWindow(DiaWindowGroup, _MobilityWindow):
    """A DIA isolation window in one specific MS/MS frame.

    Extends `DiaWindowGroup` with the frame it was acquired in, and gives
    spectrum access and ion-mobility ranges for the window's scan range.
    """

    frame_id: int
    """MS/MS frame this window was acquired in."""
    rt: float
    """Retention time of that frame in seconds."""
    polarity: Polarity
    """Ion polarity."""
    _timsdata: TimsData = _timsdata_field()


@dataclass(frozen=True, slots=True, kw_only=True)
class DIAMs1Frame(Frame):
    """An MS1 frame from a DIA acquisition.

    Inherits all fields from `Frame`. DIA isolation windows belong to MS/MS
    frames; use the reader's `windows` lookup for them.
    """


# ---------------------------------------------------------------------------
# PRM
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class PrmTarget:
    """A predefined PRM target ion from the `PrmTargets` table.

    | Field | Type | Description |
    |---|---|---|
    | `target_id` | `int` | Unique target ID |
    | `external_id` | `str \\| None` | External identifier |
    | `rt` | `float` | Expected retention time in seconds |
    | `ook0` | `float` | Expected ion mobility 1/K0 (V·s/cm²) |
    | `monoisotopic_mz` | `float` | Target m/z |
    | `charge` | `int` | Charge state |
    | `description` | `str` | Target description |
    | `transitions` | `tuple[PrmTransition, ...]` | Transitions acquired for this target |

    ``transitions`` is excluded from equality and hashing (each transition
    points back at its target).
    """

    target_id: int
    external_id: str | None
    rt: float
    ook0: float
    monoisotopic_mz: float
    charge: int
    description: str
    transitions: tuple["PrmTransition", ...] = field(default=(), compare=False, repr=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class PrmTransition(_MobilityWindow):
    """A PRM isolation window in one specific MS/MS frame.

    One row of the `PrmFrameMsMsInfo` table: a contiguous range of mobility
    scans acquired with one isolation window and collision energy, linked to a
    PRM target.

    | Field | Type | Description |
    |---|---|---|
    | `frame_id` | `int` | Frame ID |
    | `scan_num_begin` | `int` | First mobility scan (inclusive) |
    | `scan_num_end` | `int` | Mobility scan range end (exclusive), so the range is `[scan_num_begin, scan_num_end)` |
    | `isolation_mz` | `float` | Isolation window center m/z |
    | `isolation_width` | `float` | Isolation window width in Th |
    | `collision_energy` | `float` | Collision energy in eV |
    | `target` | `PrmTarget` | Associated PRM target |
    | `rt` | `float` | Retention time in seconds |
    | `polarity` | `Polarity` | Ion polarity |
    """

    frame_id: int
    scan_num_begin: int
    scan_num_end: int
    isolation_mz: float
    isolation_width: float
    collision_energy: float
    target: PrmTarget
    rt: float
    polarity: Polarity
    _timsdata: TimsData = _timsdata_field()


@dataclass(frozen=True, slots=True, kw_only=True)
class PRMMs1Frame(Frame):
    """An MS1 frame from a PRM acquisition.

    Inherits all fields from `Frame`. PRM transitions belong to MS/MS frames;
    use the reader's `transitions` lookup for them.
    """


# ---------------------------------------------------------------------------
# Key/value tables
# ---------------------------------------------------------------------------

MetaValue = str | int | float | None
"""A value from the `GlobalMetadata` or `CalibrationInfo` table."""


@dataclass(frozen=True, slots=True, eq=False)
class _KeyTable(Mapping[str, MetaValue]):
    """Read-only key/value table. A missing key raises :class:`~tdfpy.TdfpyKeyError`."""

    table: Mapping[str, MetaValue]

    def __getitem__(self, key: str) -> MetaValue:
        try:
            return self.table[key]
        except KeyError:
            raise TdfpyKeyError(f"Key {key!r} not found in the {type(self).__name__} table. Available keys: {sorted(self.table)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.table)

    def __len__(self) -> int:
        return len(self.table)

    def _str(self, key: str) -> str:
        return str(self[key])

    def _int(self, key: str) -> int:
        return int(self._str(key))

    def _float(self, key: str) -> float:
        return float(self._str(key))

    def _datetime(self, key: str) -> datetime.datetime:
        return datetime.datetime.fromisoformat(self._str(key))


@dataclass(frozen=True, slots=True, eq=False)
class Calibration(_KeyTable):
    """The `CalibrationInfo` table as a read-only mapping of key to value.

    A missing key raises :class:`~tdfpy.TdfpyKeyError`, from indexing and from
    the typed properties below.

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
        """Mass calibration date (`CalibrationDateTime`)."""
        return self._datetime("CalibrationDateTime")

    @property
    def user(self) -> str:
        """User who ran the calibration."""
        return self._str("CalibrationUser")

    @property
    def software(self) -> str:
        """Calibration software name."""
        return self._str("CalibrationSoftware")

    @property
    def software_version(self) -> str:
        """Calibration software version."""
        return self._str("CalibrationSoftwareVersion")

    @property
    def mode(self) -> str:
        """m/z calibration mode (`MzCalibrationMode`), as text."""
        return self._str("MzCalibrationMode")

    @property
    def std_ppm(self) -> float:
        """m/z calibration standard deviation in ppm."""
        return self._float("MzStandardDeviationPPM")

    @property
    def reference_masses(self) -> str:
        """Reference mass list name."""
        return self._str("ReferenceMassList")

    @property
    def mobility_calibration_date(self) -> datetime.datetime:
        """Mobility calibration date."""
        return self._datetime("MobilityCalibrationDateTime")

    @property
    def mobility_calibration_user(self) -> str:
        """User who ran the mobility calibration."""
        return self._str("MobilityCalibrationUser")

    @property
    def mobility_standard_deviation_percent(self) -> float:
        """Mobility calibration standard deviation in percent."""
        return self._float("MobilityStandardDeviationPercent")

    @property
    def reference_mobility_list(self) -> str:
        """Reference mobility list name."""
        return self._str("ReferenceMobilityList")


@dataclass(frozen=True, slots=True, eq=False)
class MetaData(_KeyTable):
    """The `GlobalMetadata` table as a read-only mapping of key to value.

    A missing key raises :class:`~tdfpy.TdfpyKeyError`, from indexing and from
    the typed properties below.

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
        return self._str("SchemaType")

    @property
    def schema_version_major(self) -> int:
        """Major version of the TDF schema."""
        return self._int("SchemaVersionMajor")

    @property
    def schema_version_minor(self) -> int:
        """Minor version of the TDF schema."""
        return self._int("SchemaVersionMinor")

    @property
    def acquisition_software_vendor(self) -> str:
        """Vendor of acquisition software."""
        return self._str("AcquisitionSoftwareVendor")

    @property
    def instrument_vendor(self) -> str:
        """Instrument vendor."""
        return self._str("InstrumentVendor")

    @property
    def tims_compression_type(self) -> int:
        """TIMS data compression type."""
        return self._int("TimsCompressionType")

    @property
    def closed_properly(self) -> bool:
        """Whether the acquisition was closed properly."""
        return bool(self._int("ClosedProperly"))

    @property
    def max_num_peaks_per_scan(self) -> int:
        """Maximum number of peaks per scan."""
        return self._int("MaxNumPeaksPerScan")

    @property
    def analysis_id(self) -> str:
        """Analysis UUID."""
        return self._str("AnalysisId")

    @property
    def digitizer_num_samples(self) -> int:
        """Number of digitizer samples."""
        return self._int("DigitizerNumSamples")

    @property
    def peak_list_index_scale_factor(self) -> int:
        """Peak list index scale factor."""
        return self._int("PeakListIndexScaleFactor")

    @property
    def mz_acq_range_lower(self) -> float:
        """Lower m/z acquisition range."""
        return self._float("MzAcqRangeLower")

    @property
    def mz_acq_range_upper(self) -> float:
        """Upper m/z acquisition range."""
        return self._float("MzAcqRangeUpper")

    @property
    def mz_acq_range(self) -> tuple[float, float]:
        """M/z acquisition range as (lower, upper) tuple."""
        return (self.mz_acq_range_lower, self.mz_acq_range_upper)

    @property
    def ook0_acq_range_lower(self) -> float:
        """Lower 1/K0 acquisition range."""
        return self._float("OneOverK0AcqRangeLower")

    @property
    def ook0_acq_range_upper(self) -> float:
        """Upper 1/K0 acquisition range."""
        return self._float("OneOverK0AcqRangeUpper")

    @property
    def ook0_acq_range(self) -> tuple[float, float]:
        """1/K0 acquisition range as (lower, upper) tuple."""
        return (self.ook0_acq_range_lower, self.ook0_acq_range_upper)

    # Acquisition software information
    @property
    def acquisition_software(self) -> str:
        """Acquisition software name."""
        return self._str("AcquisitionSoftware")

    @property
    def acquisition_software_version(self) -> str:
        """Acquisition software version."""
        return self._str("AcquisitionSoftwareVersion")

    @property
    def acquisition_firmware_version(self) -> str:
        """Acquisition firmware version."""
        return self._str("AcquisitionFirmwareVersion")

    @property
    def acquisition_datetime(self) -> datetime.datetime:
        """Acquisition date and time."""
        return self._datetime("AcquisitionDateTime")

    # Instrument information
    @property
    def instrument_name(self) -> str:
        """Instrument name."""
        return self._str("InstrumentName")

    @property
    def instrument_family(self) -> int:
        """Instrument family code."""
        return self._int("InstrumentFamily")

    @property
    def instrument_revision(self) -> int:
        """Instrument revision number."""
        return self._int("InstrumentRevision")

    @property
    def instrument_source_type(self) -> int:
        """Instrument source type code."""
        return self._int("InstrumentSourceType")

    @property
    def instrument_serial_number(self) -> str:
        """Instrument serial number."""
        return self._str("InstrumentSerialNumber")

    # Sample and method information
    @property
    def operator_name(self) -> str:
        """Operator name."""
        return self._str("OperatorName")

    @property
    def description(self) -> str:
        """Sample/acquisition description."""
        return self._str("Description")

    @property
    def sample_name(self) -> str:
        """Sample name."""
        return self._str("SampleName")

    @property
    def method_name(self) -> str:
        """Acquisition method name."""
        return self._str("MethodName")
