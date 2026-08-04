"""Capture peak lists from Bruker's proprietary peak picker.

Unlike the calibration golden file, these are *not* values tdfpy reproduces
exactly: Bruker's peak-picking algorithm is closed and appears to smooth before
picking, so tdfpy's mobility-collapse-and-merge produces a slightly different
peak list. The reference lists exist so that divergence stays measured and
bounded rather than drifting silently.

Run this only while ``libtimsdata.so`` / ``timsdata.dll`` is still present; the
committed JSON is the durable artifact.

    uv run python scripts/generate_peaks_golden.py
"""

from __future__ import annotations

import json
import sqlite3
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
)
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "tests" / "data" / "peaks_golden.json"
LIB = REPO / "src" / "tdfpy" / "libtimsdata.so"

FUNCTOR = CFUNCTYPE(None, c_int64, c_uint32, POINTER(c_double), POINTER(c_float))

DDA_PRECURSORS = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
N_DIA_WINDOWS = 12


class Native:
    def __init__(self, d: Path) -> None:
        self.dll = CDLL(str(LIB))
        self.dll.tims_open_v2.argtypes = [c_char_p, c_uint32, c_uint32]
        self.dll.tims_open_v2.restype = c_uint64
        self.dll.tims_close.argtypes = [c_uint64]
        self.dll.tims_read_pasef_msms.argtypes = [
            c_uint64,
            POINTER(c_int64),
            c_uint32,
            FUNCTOR,
        ]
        self.dll.tims_read_pasef_msms.restype = c_uint32
        self.dll.tims_extract_centroided_spectrum_for_frame_v2.argtypes = [
            c_uint64,
            c_int64,
            c_uint32,
            c_uint32,
            FUNCTOR,
            c_void_p,
        ]
        self.dll.tims_extract_centroided_spectrum_for_frame_v2.restype = c_uint32
        self.handle = self.dll.tims_open_v2(str(d).encode(), 0, 0)
        if self.handle == 0:
            raise RuntimeError(f"tims_open_v2 failed for {d}")

    def pasef_msms(self, precursor: int) -> tuple[list[float], list[float]] | None:
        out: dict[int, Any] = {}

        @FUNCTOR
        def cb(pid: int, n: int, mzs: Any, areas: Any) -> None:
            out[pid] = (list(mzs[0:n]), list(areas[0:n]))

        arr = np.array([precursor], dtype=np.int64)
        rc = self.dll.tims_read_pasef_msms(
            self.handle, arr.ctypes.data_as(POINTER(c_int64)), 1, cb
        )
        if rc == 0:
            raise RuntimeError("tims_read_pasef_msms failed")
        return out.get(precursor)

    def centroided_frame(
        self, frame: int, begin: int, end: int
    ) -> tuple[list[float], list[float]] | None:
        out: list[tuple[list[float], list[float]]] = []

        @FUNCTOR
        def cb(_pid: int, n: int, mzs: Any, areas: Any) -> None:
            out.append((list(mzs[0:n]), list(areas[0:n])))

        rc = self.dll.tims_extract_centroided_spectrum_for_frame_v2(
            self.handle, frame, begin, end, cb, None
        )
        if rc == 0:
            raise RuntimeError("tims_extract_centroided_spectrum_for_frame_v2 failed")
        return out[0] if out else None

    def close(self) -> None:
        self.dll.tims_close(self.handle)


def main() -> None:
    if not LIB.exists():
        raise SystemExit(f"native library not found: {LIB}")

    golden: dict[str, Any] = {
        "_comment": (
            "Peak lists from Bruker's proprietary peak picker. tdfpy does NOT "
            "reproduce these exactly; see tests/test_peaks_divergence.py for the "
            "bounds that are enforced. Regenerate with "
            "scripts/generate_peaks_golden.py (requires the native library)."
        ),
        "dda_precursors": [],
        "dia_windows": [],
    }

    # DDA: readPasefMsMs, keyed by precursor.
    dda = REPO / "tests" / "data" / "example_dda.d"
    conn = sqlite3.connect(dda / "analysis.tdf")
    native = Native(dda)
    for precursor in DDA_PRECURSORS:
        rows = conn.execute(
            "SELECT Frame, ScanNumBegin, ScanNumEnd FROM PasefFrameMsMsInfo "
            "WHERE Precursor = ? ORDER BY Frame",
            (precursor,),
        ).fetchall()
        peaks = native.pasef_msms(precursor)
        if not rows or peaks is None:
            continue
        golden["dda_precursors"].append(
            {
                "precursor": precursor,
                "scan_ranges": [list(map(int, r)) for r in rows],
                "mz": peaks[0],
                "intensity": peaks[1],
            }
        )
    native.close()
    conn.close()

    # DIA: extractCentroidedSpectrumForFrame, per isolation window.
    dia = REPO / "tests" / "data" / "example_dia.d"
    conn = sqlite3.connect(dia / "analysis.tdf")
    native = Native(dia)
    rows = conn.execute(
        "SELECT f.Id, w.ScanNumBegin, w.ScanNumEnd FROM Frames f "
        "JOIN DiaFrameMsMsInfo i ON i.Frame = f.Id "
        "JOIN DiaFrameMsMsWindows w ON w.WindowGroup = i.WindowGroup "
        "WHERE f.MsMsType = 9 LIMIT ?",
        (N_DIA_WINDOWS,),
    ).fetchall()
    for frame, begin, end in rows:
        peaks = native.centroided_frame(int(frame), int(begin), int(end))
        if peaks is None or len(peaks[0]) < 20:
            continue
        golden["dia_windows"].append(
            {
                "frame": int(frame),
                "scan_begin": int(begin),
                "scan_end": int(end),
                "mz": peaks[0],
                "intensity": peaks[1],
            }
        )
    native.close()
    conn.close()

    OUT.write_text(json.dumps(golden) + "\n")
    print(
        f"wrote {OUT}: {len(golden['dda_precursors'])} precursors, "
        f"{len(golden['dia_windows'])} DIA windows "
        f"({OUT.stat().st_size / 1024:.0f} KiB)"
    )


if __name__ == "__main__":
    main()
