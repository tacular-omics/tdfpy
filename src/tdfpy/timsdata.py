"""Reader for Bruker timsTOF ``.d`` folders (``analysis.tdf`` + ``analysis.tdf_bin``).

Frame decoding and all coordinate conversions are pure Python/NumPy; see
:mod:`tdfpy.calibration` for the calibration models. Bruker's native library is
not used at all -- there is no ctypes, no shared object, and no platform
restriction.

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

The last scan's peak count is implicit — it is whatever remains. Because the
peaks are pairs, ``words.size - scan_count`` is necessarily even; that and the
other layout invariants are checked in :func:`_decode_frame` so a corrupt file
raises :class:`UnsupportedTdfError` instead of decoding into silent nonsense.

TOF deltas are 1-based and accumulate within a scan, resetting at each scan
boundary, so the TOF index is a per-scan cumulative sum minus one.

Intensities are *not* returned as stored: Bruker normalises them to a 100 ms
accumulation window, and this reader reproduces that.
"""

from __future__ import annotations

import logging
from pathlib import Path
import os
import sqlite3
import sys
import threading
import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

from .calibration import (
    MzCalibration,
    TimsCalibration,
    ccs_to_one_over_k0,
    one_over_k0_to_ccs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "PressureCompensationStrategy",
    "TimsData",
    "UnsupportedTdfError",
    "ccsToOneOverK0ToCCSforMz",
    "ccsToOneOverK0forMz",
    "oneOverK0ToCCSforMz",
    "timsdata_connect",
]

#: The only ``GlobalMetadata['TimsCompressionType']`` this reader implements.
#: Type 1 is a legacy per-scan LZF format found on older acquisitions.
SUPPORTED_COMPRESSION_TYPE = 2

_EMPTY_U32 = np.zeros(0, dtype=np.uint32)

#: ``os.pread`` is POSIX-only. Where it is missing (Windows) frame reads fall
#: back to a lock-guarded seek + read on the shared handle.
_PREAD: Callable[[int, int, int], bytes] | None = getattr(os, "pread", None)
_HAS_PREAD = _PREAD is not None


class UnsupportedTdfError(NotImplementedError):
    """Raised for a ``.d`` folder this reader has not been validated against."""


@dataclass(frozen=True)
class FrameMetadata:
    """Immutable frame metadata loaded when the reader opens. Time is in seconds."""

    frame_id: int
    time: float
    msms_type: int
    polarity: str
    num_scans: int
    num_peaks: int
    property_group: int | None
    mz_calibration: int
    tims_calibration: int
    t1: float
    t2: float


# ---------------------------------------------------------------------------
# zstd backend
# ---------------------------------------------------------------------------


def _resolve_zstd() -> Callable[[bytes], bytes]:
    """Pick a zstd implementation once, at import.

    Python 3.14 ships zstd in the standard library (PEP 784), but only when
    CPython was built against libzstd, so the import is still attempted rather
    than assumed. Otherwise ``zstandard`` or ``pyzstd`` will do. All three are
    used through their stateless one-shot entry points, so decompression itself
    carries no state between calls. That is only half of what concurrent frame
    reads need; the other half — not sharing a file position — is
    :meth:`TimsData._pread`'s job.
    """
    if sys.version_info >= (3, 14):
        try:
            from compression.zstd import decompress

            return decompress
        except ImportError:  # pragma: no cover - build without libzstd
            pass
    try:
        from zstandard import decompress  # ty: ignore[unresolved-import]

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
    payload: bytes, scan_count: int, frame_id: int
) -> tuple[
    npt.NDArray[np.int64],
    npt.NDArray[np.int64],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]:
    """Decode one zstd frame payload.

    Returns ``(scan_starts, scan_counts, tof_indices, raw_intensities)``, where the
    per-scan slices are ``tof[start:start + count]``.

    Every layout assumption is checked before any decoding arithmetic runs.
    Without those checks a corrupt payload escapes as an ``IndexError`` or a
    NumPy broadcast ``ValueError`` from deep inside the kernel — or, worse, is
    silently absorbed by the ``// 2`` and decodes into plausible nonsense.
    """
    try:
        decompressed = _zstd_decompress(payload)
    except MemoryError:
        raise
    except Exception as exc:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: invalid zstd payload ({exc})."
        ) from exc
    raw = np.frombuffer(decompressed, dtype=np.uint8)
    if raw.size % 4:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: decompressed payload is {raw.size} bytes, not a "
            "multiple of 4; the file may be corrupt or use an unexpected layout."
        )
    # Byte-plane de-interleaving: the payload stores an (N, 4) u32 byte matrix
    # column-major, so transposing the (4, N) view restores little-endian words.
    # .copy().view() reinterprets the one materialised buffer in place; going via
    # ascontiguousarray().tobytes() would copy it twice.
    words = raw.reshape(4, -1).T.copy().view("<u4").ravel()

    if scan_count == 0:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: Frames.NumScans is 0, but the frame carries a "
            f"payload decompressing to {words.size} u32 words. A frame with no "
            "scans must have an empty payload; the file may be corrupt."
        )
    if words.size < scan_count:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: decompressed payload holds {words.size} u32 "
            f"words, fewer than the {scan_count} scan-header words the layout "
            "requires; the file may be corrupt or truncated."
        )
    peak_words = words.size - scan_count
    if peak_words % 2:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: {peak_words} peak words follow the "
            f"{scan_count}-word scan header, but peaks are stored as "
            "(tof_delta, intensity) pairs and so must be even in number; the "
            "file may be corrupt."
        )
    # Verified on every frame of example_dda.d / example_dia.d / example_prm.d
    # (1710 frames): word[0] always equals the frame header's scan_count, which
    # in turn always equals Frames.NumScans.
    if int(words[0]) != scan_count:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: the payload's leading word declares "
            f"{int(words[0])} scans but the frame header and Frames.NumScans "
            f"say {scan_count}; the file may be corrupt."
        )

    total_peaks = peak_words // 2
    if np.any(words[1:scan_count] & 1):
        raise UnsupportedTdfError(
            f"Frame {frame_id}: scan peak-count words must be even."
        )
    counts = np.empty(scan_count, dtype=np.int64)
    counts[: scan_count - 1] = words[1:scan_count] >> 1  # stored as 2 * peak count
    counts[scan_count - 1] = total_peaks - int(counts[: scan_count - 1].sum())
    if counts[scan_count - 1] < 0:
        raise UnsupportedTdfError(
            f"Frame {frame_id}: scan sizes sum to more than the {total_peaks} "
            "peaks actually decoded; the file may be corrupt."
        )

    payload_words = words[scan_count:]
    tof_deltas = payload_words[0::2]
    intensities = payload_words[1::2]
    if np.any(tof_deltas == 0):
        raise UnsupportedTdfError(f"Frame {frame_id}: TOF deltas must be positive.")

    starts = np.zeros(scan_count, dtype=np.int64)
    np.cumsum(counts[:-1], out=starts[1:])

    # TOF indices are a cumulative sum of deltas that resets at every scan
    # boundary. Done as one global cumsum minus the running total carried in at
    # each boundary, which avoids a Python loop over scans (5-9x faster).
    running = np.cumsum(tof_deltas, dtype=np.uint64)
    carry = np.zeros(scan_count, dtype=np.uint64)
    non_empty = counts > 0
    prev_index = starts[non_empty] - 1
    carry[non_empty] = np.where(
        prev_index >= 0, running[np.maximum(prev_index, 0)], np.uint64(0)
    )
    tof = running - np.repeat(carry, counts) - np.uint64(1)
    if np.any(tof > np.iinfo(np.uint32).max):
        raise UnsupportedTdfError(f"Frame {frame_id}: TOF indices overflow uint32.")

    return starts, counts, tof.astype(np.uint32), intensities


# ---------------------------------------------------------------------------
# TimsData
# ---------------------------------------------------------------------------


class TimsData:
    """Random-access reader for a Bruker ``.d`` folder.

    Metadata is loaded eagerly on open; spectral data is read from
    ``analysis.tdf_bin`` on demand.

    Reading frames from several threads through one open reader is safe: frame
    bytes are fetched with :func:`os.pread`, which takes its offset as an
    argument and so shares no file position between threads, and decompression
    goes through stateless one-shot entry points. Where ``os.pread`` is
    unavailable (Windows) the seek + read pair is serialised by a lock instead.
    ``close()`` is *not* safe to race against an in-flight read, and the
    ``sqlite3`` connection on :attr:`conn` keeps sqlite3's own thread rules.
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
        self.conn: sqlite3.Connection | None = sqlite3.connect(
            Path(tdf_path).resolve().as_uri() + "?mode=ro", uri=True
        )
        self.conn.row_factory = sqlite3.Row

        try:
            self._load_metadata()
        except Exception:
            self.conn.close()
            self.conn = None
            raise

        #: Serialises seek + read on the shared handle for the no-``pread``
        #: fallback. Unused, but still created, when ``os.pread`` exists.
        self._read_lock = threading.Lock()
        #: Open binary file object, or ``None`` once :meth:`close` has run.
        #: Callers use this only to test whether the reader is still open.
        self.handle: Any = open(bin_path, "rb")
        self._fd: int = self.handle.fileno()

    # -- setup ------------------------------------------------------------

    def _load_metadata(self) -> None:
        assert self.conn is not None
        meta = dict(self.conn.execute("SELECT Key, Value FROM GlobalMetadata"))
        self._digitizer_num_samples = int(meta["DigitizerNumSamples"])
        self._peak_counts = dict(self.conn.execute("SELECT Id, NumPeaks FROM Frames"))
        self._frame_metadata = {
            int(r["Id"]): FrameMetadata(
                int(r["Id"]),
                float(r["Time"]),
                int(r["MsMsType"]),
                str(r["Polarity"]),
                int(r["NumScans"]),
                int(r["NumPeaks"]),
                r["PropertyGroup"],
                int(r["MzCalibration"]),
                int(r["TimsCalibration"]),
                float(r["T1"]),
                float(r["T2"]),
            )
            for r in self.conn.execute("SELECT * FROM Frames ORDER BY Id")
        }
        table_names = {
            r[0]
            for r in self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        # sqlite3.Row values are immutable and can be read by worker threads.
        self._metadata_tables = {
            name: tuple(self.conn.execute(f"SELECT * FROM {name}"))
            if name in table_names
            else ()
            for name in (
                "PropertyDefinitions",
                "GroupProperties",
                "DiaFrameMsMsWindows",
            )
        }
        self._gate_cache: dict[tuple, Any] = {}
        self._gate_lock = threading.RLock()

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

    @property
    def frame_ids(self) -> tuple[int, ...]:
        """Frame IDs in acquisition ID order. Requires an open reader."""
        self._require_open()
        return tuple(self._frame_metadata)

    def frame_metadata(self, frame_id: int) -> FrameMetadata:
        """Read eagerly loaded metadata without accessing SQLite."""
        self._require_open()
        self._frame(frame_id)
        return self._frame_metadata[frame_id]

    def metadata_table(self, name: str) -> tuple[sqlite3.Row, ...]:
        """Read an immutable snapshot of a gate metadata table.

        Supported names are PropertyDefinitions, GroupProperties, and
        DiaFrameMsMsWindows. An absent table produces an empty tuple.
        """
        self._require_open()
        return self._metadata_tables[name]

    def mz_calibration_key(self, frame_id: int) -> tuple[float, ...]:
        """Identify the effective m/z conversion, including temperature drift."""
        self._require_open()
        cal, t1, t2 = self._mz_cal(frame_id)
        return (
            cal.digitizer_timebase,
            cal.digitizer_delay,
            cal.c0,
            cal._c1_at(t1, t2),
            cal.c2,
        )

    def calibration_key(self, frame_id: int) -> tuple:
        """Identify the effective m/z and mobility conversions for caching."""
        return (self.mz_calibration_key(frame_id), self._tims_cal(frame_id))

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
        if getattr(self, "conn", None) is not None:
            self.conn.close()  # type: ignore[union-attr]
            self.conn = None

    def _require_open(self) -> Any:
        if getattr(self, "handle", None) is None:
            raise RuntimeError("TimsData connection has been closed.")
        return self.handle

    def _pread(self, count: int, offset: int) -> bytes:
        """Read ``count`` bytes of ``analysis.tdf_bin`` starting at ``offset``.

        Returns fewer than ``count`` bytes only at end of file, which the caller
        must treat as a truncated file. Positional reads keep concurrent frame
        reads from clobbering each other's file position.
        """
        self._require_open()
        if count <= 0:
            return b""
        if not _HAS_PREAD:  # the Windows path; tests force it on every platform
            with self._read_lock:
                handle = self._require_open()
                handle.seek(offset)
                return handle.read(count)

        pread = _PREAD
        assert pread is not None
        chunk = pread(self._fd, count, offset)
        if len(chunk) == count:
            return chunk
        # Short reads on a regular file mean EOF, but loop rather than assume it.
        chunks = [chunk]
        got = len(chunk)
        while got < count and chunk:
            chunk = pread(self._fd, count - got, offset + got)
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

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

    def _decode(
        self, frame_id: int
    ) -> (
        tuple[
            int,
            npt.NDArray[np.int64],
            npt.NDArray[np.int64],
            npt.NDArray[np.uint32],
            npt.NDArray[np.uint32],
        ]
        | None
    ):
        """Read and decode a whole frame, or ``None`` if it holds no data."""
        offset, num_scans, accum_time, *_ = self._frame(frame_id)

        header = self._pread(8, offset)
        if len(header) < 8:
            raise UnsupportedTdfError(
                f"Frame {frame_id}: truncated header at offset {offset} — "
                f"expected 8 bytes, got {len(header)}. analysis.tdf_bin is "
                "shorter than the Frames table says it should be."
            )
        byte_count, scan_count = (int(v) for v in np.frombuffer(header, dtype="<u4"))

        # byte_count includes the 8 header bytes, so anything below 8 is
        # nonsense. Left unchecked it became a negative read length: 7 means
        # read(-1), which swallows the entire rest of the file (tens of MB)
        # before zstd silently ignores the trailing garbage, and anything lower
        # raises an opaque "read length must be non-negative" ValueError from
        # the io layer. Neither tells the caller their file is corrupt.
        if byte_count < 8:
            raise UnsupportedTdfError(
                f"Frame {frame_id}: header at offset {offset} declares a "
                f"{byte_count}-byte packet, but the 8-byte header alone is "
                "larger than that; the file may be corrupt."
            )
        # Verified to hold on every frame of all three bundled fixtures.
        if scan_count != num_scans:
            raise UnsupportedTdfError(
                f"Frame {frame_id}: analysis.tdf_bin declares {scan_count} "
                f"scans but Frames.NumScans is {num_scans}; the metadata and "
                "the binary disagree, so one of them is corrupt."
            )

        want = byte_count - 8
        payload = self._pread(want, offset + 8)
        if len(payload) < want:
            raise UnsupportedTdfError(
                f"Frame {frame_id}: truncated payload at offset {offset + 8} — "
                f"expected {want} bytes, got {len(payload)}. analysis.tdf_bin "
                "is shorter than the Frames table says it should be."
            )
        if not payload:
            if self._peak_counts[frame_id] != 0:
                raise UnsupportedTdfError(
                    f"Frame {frame_id}: empty packet disagrees with Frames.NumPeaks "
                    f"({self._peak_counts[frame_id]})."
                )
            return None

        starts, counts, tof, raw_intensity = _decode_frame(
            payload, scan_count, frame_id
        )
        if tof.size != self._peak_counts[frame_id]:
            raise UnsupportedTdfError(
                f"Frame {frame_id}: decoded {tof.size} peaks but Frames.NumPeaks "
                f"is {self._peak_counts[frame_id]}."
            )
        if np.any(tof >= self._digitizer_num_samples):
            raise UnsupportedTdfError(
                f"Frame {frame_id}: TOF index exceeds DigitizerNumSamples."
            )

        # Bruker normalises raw digitiser sums to a 100 ms accumulation window.
        if accum_time > 0:
            intensity = np.floor(raw_intensity * (100.0 / accum_time) + 0.5).astype(
                np.uint32
            )
        else:
            logger.warning(
                "Frame %d has AccumulationTime=0; returning un-normalised intensities.",
                frame_id,
            )
            intensity = raw_intensity.astype(np.uint32)

        return scan_count, starts, counts, tof, intensity

    def read_frame_arrays(
        self, frame_id: int, scan_begin: int = 0, scan_end: int | None = None
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.uint32], npt.NDArray[np.uint32]]:
        """Read scans ``[scan_begin, scan_end)`` as three flat, parallel arrays.

        Returns ``(scan_indices, tof_indices, intensities)``, one entry per peak.
        This is the cheap path: peaks for a contiguous scan range are already
        contiguous in the decoded frame, so it slices rather than splitting the
        frame into per-scan arrays the way :meth:`readScans` must.

        Prefer this whenever you were going to concatenate ``readScans`` output
        back together.
        """
        decoded = self._decode(frame_id)
        if decoded is None:
            return (
                np.zeros(0, dtype=np.int64),
                _EMPTY_U32,
                _EMPTY_U32,
            )
        scan_count, starts, counts, tof, intensity = decoded

        if scan_end is None:
            scan_end = scan_count
        begin = max(0, min(int(scan_begin), scan_count))
        end = max(begin, min(int(scan_end), scan_count))
        if begin == end:
            return np.zeros(0, dtype=np.int64), _EMPTY_U32, _EMPTY_U32

        lo = int(starts[begin])
        hi = int(starts[end - 1] + counts[end - 1])
        scan_indices = np.repeat(
            np.arange(begin, end, dtype=np.int64), counts[begin:end]
        )
        return scan_indices, tof[lo:hi], intensity[lo:hi]

    def readScans(
        self, frame_id: int, scan_begin: int, scan_end: int
    ) -> list[tuple[npt.NDArray[np.uint32], npt.NDArray[np.uint32]]]:
        """Read scans ``[scan_begin, scan_end)`` of a frame.

        Returns one ``(tof_indices, intensities)`` pair per scan. Intensities are
        normalised to a 100 ms accumulation window, matching Bruker.

        See :meth:`read_frame_arrays` for a flat-array alternative that avoids
        materialising one array pair per scan.
        """
        decoded = self._decode(frame_id)
        if decoded is None:
            return [(_EMPTY_U32, _EMPTY_U32) for _ in range(scan_begin, scan_end)]
        scan_count, starts, counts, tof, intensity = decoded

        result = []
        for i in range(scan_begin, scan_end):
            if i < 0 or i >= scan_count:
                result.append((_EMPTY_U32, _EMPTY_U32))
                continue
            start = int(starts[i])
            stop = start + int(counts[i])
            result.append((tof[start:stop], intensity[start:stop]))
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


def ccsToOneOverK0forMz(ccs: float, charge: int, mz: float) -> float:
    """Convert CCS to 1/K0 for a given charge and m/z.

    The exact inverse of :func:`oneOverK0ToCCSforMz`.
    """
    return ccs_to_one_over_k0(ccs, charge, mz)


def ccsToOneOverK0ToCCSforMz(ccs: float, charge: int, mz: float) -> float:
    """Deprecated alias for :func:`ccsToOneOverK0forMz`.

    The old name was a copy-paste of the forward function's name and reads as
    "CCS to 1/K0 to CCS", which is not what it does. It stays exported so
    existing callers keep working.
    """
    warnings.warn(
        "ccsToOneOverK0ToCCSforMz is a misnamed alias and will be removed in a "
        "future release; use ccsToOneOverK0forMz instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return ccsToOneOverK0forMz(ccs, charge, mz)
