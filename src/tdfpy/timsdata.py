"""Reader for Bruker timsTOF ``.d`` folders (``analysis.tdf`` + ``analysis.tdf_bin``).

Frame decoding and all coordinate conversions are pure Python/NumPy; see
:mod:`tdfpy.calibration` for the calibration models. Bruker's native library is
still required for the two peak-picker entry points
(:meth:`TimsData.readPasefMsMs` and
:meth:`TimsData.extractCentroidedSpectrumForFrame`).

Frame layout of ``analysis.tdf_bin``, per frame, at byte offset ``Frames.TimsId``::

    u32  byte_count     total packet size, including these 8 header bytes
    u32  scan_count     matches Frames.NumScans
    ...  payload        zstd-compressed, byte_count - 8 bytes

The decompressed payload is an ``(N, 4)`` matrix of u32 bytes stored
column-major, so all byte-0s precede all byte-1s and so on. Once transposed back
into u32 words the layout is::

    word[0]                     scan_count
    word[1 .. scan_count)       2 * peak count, for the first scan_count-1 scans
    word[scan_count ..]         interleaved (tof_delta, intensity) pairs

The last scan's peak count is implicit — it is whatever remains. TOF deltas are
1-based and accumulate within a scan, resetting at each scan boundary, so the TOF
index is a per-scan cumulative sum minus one.

Intensities are *not* returned as stored: Bruker normalises them to a 100 ms
accumulation window, and this reader reproduces that.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from ctypes import (
    CDLL,
    CFUNCTYPE,
    POINTER,
    c_char_p,
    c_double,
    c_float,
    c_int64,
    c_uint32,
    c_uint64,
    c_void_p,
    cdll,
    create_string_buffer,
)
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from .calibration import (
    MzCalibration,
    TimsCalibration,
    UnsupportedCalibrationError,
    ccs_to_one_over_k0,
    one_over_k0_to_ccs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PressureCompensationStrategy",
    "TimsData",
    "UnsupportedTdfError",
    "ccsToOneOverK0ToCCSforMz",
    "oneOverK0ToCCSforMz",
    "timsdata_connect",
]

#: The only ``GlobalMetadata['TimsCompressionType']`` this reader implements.
#: Type 1 is a legacy per-scan LZF format found on older acquisitions.
SUPPORTED_COMPRESSION_TYPE = 2

_EMPTY_U32 = np.zeros(0, dtype=np.uint32)


class UnsupportedTdfError(NotImplementedError):
    """Raised for a ``.d`` folder this reader has not been validated against."""


# ---------------------------------------------------------------------------
# zstd backend
# ---------------------------------------------------------------------------


def _resolve_zstd() -> Callable[[bytes], bytes]:
    """Pick a zstd implementation once, at import.

    Python 3.14 ships zstd in the standard library (PEP 784), but only when
    CPython was built against libzstd, so the import is still attempted rather
    than assumed. Otherwise ``zstandard`` or ``pyzstd`` will do. All three are
    used through their stateless one-shot entry points, which keeps concurrent
    frame reads safe.
    """
    if sys.version_info >= (3, 14):
        try:
            from compression.zstd import decompress

            return decompress
        except ImportError:  # pragma: no cover - build without libzstd
            pass
    try:
        from zstandard import decompress

        return decompress
    except ImportError:
        pass
    try:
        # Optional third fallback; not a declared dependency.
        from pyzstd import decompress  # ty: ignore[unresolved-import]

        return decompress
    except ImportError as exc:  # pragma: no cover - install-time failure
        raise ImportError(
            "tdfpy needs a zstd implementation to read analysis.tdf_bin. "
            "Install `zstandard` (or `pyzstd`), or use Python 3.14+ built with "
            "zstd in the standard library."
        ) from exc


_zstd_decompress = _resolve_zstd()


# ---------------------------------------------------------------------------
# Native library — retained only for Bruker's proprietary peak picker
# ---------------------------------------------------------------------------

_PLATFORM_LIB = {
    "win32": "timsdata.dll",
    "cygwin": "timsdata.dll",
    "linux": "libtimsdata.so",
}

MSMS_SPECTRUM_FUNCTOR = CFUNCTYPE(
    None, c_int64, c_uint32, POINTER(c_double), POINTER(c_float)
)


def _load_native() -> tuple[CDLL | None, Exception | None]:
    libname = next(
        (v for k, v in _PLATFORM_LIB.items() if sys.platform.startswith(k)), None
    )
    if libname is None:
        return None, OSError(
            f"Bruker's native library is not available for {sys.platform!r}."
        )
    path = os.path.join(os.path.dirname(__file__), libname)
    try:
        return cdll.LoadLibrary(path if os.path.exists(path) else libname), None
    except Exception as exc:  # noqa: BLE001 - re-raised on first native use
        logger.debug("could not load %s: %s", libname, exc)
        return None, exc


dll, _dll_load_error = _load_native()

if dll is not None:
    dll.tims_open_v2.argtypes = [c_char_p, c_uint32, c_uint32]
    dll.tims_open_v2.restype = c_uint64
    dll.tims_close.argtypes = [c_uint64]
    dll.tims_close.restype = None
    dll.tims_get_last_error_string.argtypes = [c_char_p, c_uint32]
    dll.tims_get_last_error_string.restype = c_uint32
    dll.tims_read_pasef_msms.argtypes = [
        c_uint64,
        POINTER(c_int64),
        c_uint32,
        MSMS_SPECTRUM_FUNCTOR,
    ]
    dll.tims_read_pasef_msms.restype = c_uint32
    dll.tims_extract_centroided_spectrum_for_frame_v2.argtypes = [
        c_uint64,
        c_int64,
        c_uint32,
        c_uint32,
        MSMS_SPECTRUM_FUNCTOR,
        c_void_p,
    ]
    dll.tims_extract_centroided_spectrum_for_frame_v2.restype = c_uint32


def _throw_last_native_error(dll_handle: CDLL) -> None:
    err_len = dll_handle.tims_get_last_error_string(None, 0)
    buf = create_string_buffer(err_len)
    dll_handle.tims_get_last_error_string(buf, err_len)
    msg = buf.value.decode("utf-8", errors="replace").strip()
    if not msg:
        msg = (
            "native call failed but returned no error string. Common causes: an "
            "invalid frame_id or scan range, or a handle that is already closed."
        )
    raise RuntimeError(f"timsdata native error: {msg}")


class PressureCompensationStrategy(Enum):
    """Bruker's per-frame mobility pressure-correction modes.

    Only :attr:`NoPressureCompensation` is implemented. The correction is
    believed to be driven by ``TimsCalibration.C8``/``C9``, which this reader's
    mobility model does not consume.
    """

    NoPressureCompensation = 0
    AnalysisGlobalPressureCompensation = 1
    PerFramePressureCompensation = 2


# ---------------------------------------------------------------------------
# Frame decoding
# ---------------------------------------------------------------------------


def _decode_frame(
    payload: bytes, scan_count: int
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]:
    """Decode one zstd frame payload.

    Returns ``(scan_starts, scan_counts, tof_indices, raw_intensities)``, where the
    per-scan slices are ``tof[start:start + count]``.
    """
    raw = np.frombuffer(_zstd_decompress(payload), dtype=np.uint8)
    if raw.size % 4:
        raise UnsupportedTdfError(
            f"decompressed frame is {raw.size} bytes, not a multiple of 4; "
            "the file may be corrupt or use an unexpected layout."
        )
    # Byte-plane de-interleaving: the payload stores an (N, 4) u32 byte matrix
    # column-major, so transposing the (4, N) view restores little-endian words.
    words = np.frombuffer(
        np.ascontiguousarray(raw.reshape(4, -1).T).tobytes(), dtype="<u4"
    )

    total_peaks = (words.size - scan_count) // 2
    counts = np.empty(scan_count, dtype=np.int64)
    counts[: scan_count - 1] = words[1:scan_count] >> 1  # stored as 2 * peak count
    counts[scan_count - 1] = total_peaks - int(counts[: scan_count - 1].sum())
    if counts[scan_count - 1] < 0:
        raise UnsupportedTdfError(
            "frame scan sizes exceed the decoded peak count; the file may be corrupt."
        )

    payload_words = words[scan_count:]
    tof_deltas = payload_words[0::2]
    intensities = payload_words[1::2]

    starts = np.zeros(scan_count, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    # TOF indices are a cumulative sum of deltas that resets at every scan
    # boundary. Done as one global cumsum minus the running total carried in at
    # each boundary, which avoids a Python loop over scans (5-9x faster).
    running = np.cumsum(tof_deltas, dtype=np.uint32)
    carry = np.zeros(scan_count, dtype=np.uint32)
    non_empty = counts > 0
    prev_index = starts[non_empty] - 1
    carry[non_empty] = np.where(
        prev_index >= 0, running[np.maximum(prev_index, 0)], np.uint32(0)
    )
    tof = running - np.repeat(carry, counts) - np.uint32(1)  # deltas are 1-based

    return starts, counts, tof, intensities


# ---------------------------------------------------------------------------
# TimsData
# ---------------------------------------------------------------------------


class TimsData:
    """Random-access reader for a Bruker ``.d`` folder.

    Metadata is loaded eagerly on open; spectral data is read from
    ``analysis.tdf_bin`` on demand.
    """

    def __init__(
        self,
        analysis_directory: str | os.PathLike[str],
        use_recalibrated_state: bool = False,
        pressure_compensation_strategy: PressureCompensationStrategy = PressureCompensationStrategy.NoPressureCompensation,
    ) -> None:
        analysis_directory = str(analysis_directory)

        if use_recalibrated_state:
            raise UnsupportedTdfError(
                "use_recalibrated_state=True is not supported; tdfpy reads the "
                "calibration recorded in analysis.tdf."
            )
        if (
            pressure_compensation_strategy
            is not PressureCompensationStrategy.NoPressureCompensation
        ):
            raise UnsupportedTdfError(
                f"{pressure_compensation_strategy.name} is not supported; only "
                "NoPressureCompensation is implemented."
            )

        if not os.path.isdir(analysis_directory):
            raise FileNotFoundError(
                f"Analysis directory not found: {analysis_directory!r}"
            )
        tdf_path = os.path.join(analysis_directory, "analysis.tdf")
        bin_path = os.path.join(analysis_directory, "analysis.tdf_bin")
        for path in (tdf_path, bin_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"{os.path.basename(path)} not found in {analysis_directory!r}"
                )

        self.analysis_directory = analysis_directory
        self.conn: sqlite3.Connection | None = sqlite3.connect(tdf_path)
        self.conn.row_factory = sqlite3.Row

        try:
            self._load_metadata()
        except Exception:
            self.conn.close()
            self.conn = None
            raise

        #: Open binary file object, or ``None`` once :meth:`close` has run.
        #: Callers use this only to test whether the reader is still open.
        self.handle: Any = open(bin_path, "rb")

        self._native_handle: int | None = None

    # -- setup ------------------------------------------------------------

    def _load_metadata(self) -> None:
        assert self.conn is not None
        meta = dict(self.conn.execute("SELECT Key, Value FROM GlobalMetadata"))

        compression = int(meta.get("TimsCompressionType", -1))
        if compression != SUPPORTED_COMPRESSION_TYPE:
            raise UnsupportedTdfError(
                f"TimsCompressionType {compression} is not supported (only "
                f"{SUPPORTED_COMPRESSION_TYPE} has been validated). Type 1 is a "
                "legacy LZF format; such files need Bruker's native library."
            )

        self._mz_calibrations = {
            int(row["Id"]): MzCalibration.from_row(row)
            for row in self.conn.execute("SELECT * FROM MzCalibration")
        }
        self._tims_calibrations = {
            int(row["Id"]): TimsCalibration.from_row(row)
            for row in self.conn.execute("SELECT * FROM TimsCalibration")
        }

        # (offset, num_scans, accumulation_time, T1, T2, mz_cal_id, tims_cal_id)
        self._frames: dict[int, tuple[int, int, float, float, float, int, int]] = {
            int(r["Id"]): (
                int(r["TimsId"]),
                int(r["NumScans"]),
                float(r["AccumulationTime"]),
                float(r["T1"]),
                float(r["T2"]),
                int(r["MzCalibration"]),
                int(r["TimsCalibration"]),
            )
            for r in self.conn.execute(
                "SELECT Id, TimsId, NumScans, AccumulationTime, T1, T2, "
                "MzCalibration, TimsCalibration FROM Frames"
            )
        }

    def _frame(self, frame_id: int) -> tuple[int, int, float, float, float, int, int]:
        try:
            return self._frames[int(frame_id)]
        except KeyError:
            if self._frames:
                lo, hi = min(self._frames), max(self._frames)
                valid = f"{lo}..{hi}"
            else:
                valid = "none (Frames table is empty)"
            raise ValueError(
                f"Frame {frame_id} not found in the Frames table "
                f"(valid frame IDs: {valid}). Frame IDs are 1-based."
            ) from None

    def _mz_cal(self, frame_id: int) -> tuple[MzCalibration, float, float]:
        _, _, _, t1, t2, mz_id, _ = self._frame(frame_id)
        return self._mz_calibrations[mz_id], t1, t2

    def _tims_cal(self, frame_id: int) -> TimsCalibration:
        return self._tims_calibrations[self._frame(frame_id)[6]]

    # -- lifecycle --------------------------------------------------------

    def __enter__(self) -> "TimsData":
        return self

    def __exit__(self, exit_type: Any, value: Any, traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            self.handle.close()
            self.handle = None
        if getattr(self, "_native_handle", None) is not None:
            dll.tims_close(self._native_handle)  # type: ignore[union-attr]
            self._native_handle = None
        if getattr(self, "conn", None) is not None:
            self.conn.close()  # type: ignore[union-attr]
            self.conn = None

    def _require_open(self) -> Any:
        if getattr(self, "handle", None) is None:
            raise RuntimeError("TimsData connection has been closed.")
        return self.handle

    # -- conversions ------------------------------------------------------

    def indexToMz(
        self, frame_id: int, indices: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Convert TOF sample indices to m/z for ``frame_id``."""
        cal, t1, t2 = self._mz_cal(frame_id)
        return cal.index_to_mz(indices, t1, t2)

    def mzToIndex(self, frame_id: int, mzs: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Convert m/z to (fractional) TOF sample indices for ``frame_id``."""
        cal, t1, t2 = self._mz_cal(frame_id)
        return cal.mz_to_index(mzs, t1, t2)

    def scanNumToOneOverK0(
        self, frame_id: int, scan_nums: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Convert scan numbers to inverse reduced mobility (1/K0)."""
        return self._tims_cal(frame_id).scan_to_one_over_k0(scan_nums)

    def oneOverK0ToScanNum(
        self, frame_id: int, mobilities: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Convert 1/K0 to (fractional) scan numbers."""
        return self._tims_cal(frame_id).one_over_k0_to_scan(mobilities)

    def scanNumToVoltage(
        self, frame_id: int, scan_nums: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Convert scan numbers to TIMS ramp voltage."""
        return self._tims_cal(frame_id).scan_to_voltage(scan_nums)

    def voltageToScanNum(
        self, frame_id: int, voltages: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Convert TIMS ramp voltage to (fractional) scan numbers."""
        return self._tims_cal(frame_id).voltage_to_scan(voltages)

    # -- spectral data ----------------------------------------------------

    def readScans(
        self, frame_id: int, scan_begin: int, scan_end: int
    ) -> list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]]:
        """Read scans ``[scan_begin, scan_end)`` of a frame.

        Returns one ``(tof_indices, intensities)`` pair per scan. Intensities are
        normalised to a 100 ms accumulation window, matching Bruker.
        """
        fh = self._require_open()
        offset, num_scans, accum_time, *_ = self._frame(frame_id)

        fh.seek(offset)
        header = fh.read(8)
        if len(header) < 8:
            raise UnsupportedTdfError(
                f"Frame {frame_id}: truncated header at offset {offset}."
            )
        byte_count, scan_count = np.frombuffer(header, dtype="<u4")
        payload = fh.read(int(byte_count) - 8)

        if not payload:
            return [(_EMPTY_U32, _EMPTY_U32) for _ in range(scan_begin, scan_end)]

        starts, counts, tof, raw_intensity = _decode_frame(payload, int(scan_count))

        # Bruker normalises raw digitiser sums to a 100 ms accumulation window.
        if accum_time > 0:
            intensity = np.floor(raw_intensity * (100.0 / accum_time) + 0.5).astype(
                np.uint32
            )
        else:
            logger.warning(
                "readScans: frame %d has AccumulationTime=0; "
                "returning un-normalised intensities.",
                frame_id,
            )
            intensity = raw_intensity.astype(np.uint32)

        result = []
        for i in range(scan_begin, scan_end):
            if i < 0 or i >= int(scan_count):
                result.append((_EMPTY_U32, _EMPTY_U32))
                continue
            start = int(starts[i])
            stop = start + int(counts[i])
            result.append((tof[start:stop], intensity[start:stop]))
        return result

    # -- native peak picker ----------------------------------------------

    def _native(self) -> int:
        """Lazily open Bruker's native handle, used only by the peak picker."""
        if dll is None:
            raise ImportError(
                f"Bruker's native library could not be loaded: {_dll_load_error}"
            ) from _dll_load_error
        if self._native_handle is None:
            handle = dll.tims_open_v2(self.analysis_directory.encode("utf-8"), 0, 0)
            if handle == 0:
                _throw_last_native_error(dll)
            self._native_handle = handle
        return self._native_handle

    def readPasefMsMs(self, precursor_list: list[int]) -> dict[int, tuple[Any, Any]]:
        """Bruker-centroided PASEF MS/MS spectra, keyed by precursor ID."""
        handle = self._native()
        precursors = np.array(precursor_list, dtype=np.int64)
        result: dict[int, tuple[Any, Any]] = {}

        @MSMS_SPECTRUM_FUNCTOR
        def callback(precursor_id: int, num_peaks: int, mzs: Any, areas: Any) -> None:
            result[precursor_id] = (mzs[0:num_peaks], areas[0:num_peaks])

        rc = dll.tims_read_pasef_msms(  # type: ignore[union-attr]
            handle,
            precursors.ctypes.data_as(POINTER(c_int64)),
            len(precursor_list),
            callback,
        )
        if rc == 0:
            _throw_last_native_error(dll)  # type: ignore[arg-type]
        return result

    def extractCentroidedSpectrumForFrame(
        self, frame_id: int, scan_begin: int, scan_end: int
    ) -> tuple[Any, Any] | None:
        """Bruker-centroided spectrum for a frame's scan range."""
        handle = self._native()
        result: tuple[Any, Any] | None = None

        @MSMS_SPECTRUM_FUNCTOR
        def callback(_id: int, num_peaks: int, mzs: Any, areas: Any) -> None:
            nonlocal result
            result = (mzs[0:num_peaks], areas[0:num_peaks])

        rc = dll.tims_extract_centroided_spectrum_for_frame_v2(  # type: ignore[union-attr]
            handle, frame_id, scan_begin, scan_end, callback, None
        )
        if rc == 0:
            _throw_last_native_error(dll)  # type: ignore[arg-type]
        return result


@contextmanager
def timsdata_connect(analysis_dir: str | os.PathLike[str]) -> Iterator[TimsData]:
    """Open a :class:`TimsData` and close it on exit."""
    td: TimsData | None = None
    try:
        td = TimsData(str(analysis_dir))
        yield td
    finally:
        if td:
            td.close()


def oneOverK0ToCCSforMz(ook0: float, charge: int, mz: float) -> float:
    """Convert 1/K0 to CCS for a given charge and m/z."""
    return one_over_k0_to_ccs(ook0, charge, mz)


def ccsToOneOverK0ToCCSforMz(ccs: float, charge: int, mz: float) -> float:
    """Convert CCS to 1/K0 for a given charge and m/z."""
    return ccs_to_one_over_k0(ccs, charge, mz)


# Re-exported so callers can catch a single error type for unsupported files.
UnsupportedTdfError.__module__ = __name__
_ = UnsupportedCalibrationError  # re-exported via tdfpy.calibration
