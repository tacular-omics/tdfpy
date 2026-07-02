"""Precursor-space gates: drop MS1 signal the instrument never fragments.

Two acquisition-aware filters that mask out MS1 peaks lying in a region the
method can never schedule for fragmentation — i.e. signal that cannot become an
identification and so is noise for downstream search:

- :class:`SelectionPolygonGate` (**ddaPASEF**) — keeps only MS1 points inside the
  run's PASEF selection polygon (the "IMS PolygonFilter" stored in
  ``analysis.tdf``). A generalisation of :class:`tdfpy.regions.ChargeStateRegion`
  from a single line to the real acquisition polygon.
- :class:`DiaMs1WindowGate` (**diaPASEF**) — keeps only MS1 points inside the
  union of the isolation windows (``DiaFrameMsMsWindows``); everything outside is
  a precursor the method never isolates.

Both are ported from the ``dnoise`` Rust tool. They implement the
:class:`~tdfpy.noise.NoiseFilter` interface, so they compose with the statistical
filters (``noise=[SelectionPolygonGate(), MadThreshold(k=3)]``) and inherit the
:func:`tdfpy.pipeline.apply_noise` observability logging.

Both are **MS1-only**: apply them to MS1 frames. Each is calibration-free at test
time — the ``(m/z, 1/K0)`` region is converted once to per-scan integer TOF-index
intervals using the run calibration, then membership is a vectorised binary
search. When the run carries no polygon / no windows the gate is a no-op (keeps
everything) rather than dropping all points.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from weakref import WeakKeyDictionary

import numpy as np

from . import NoiseFilter

if TYPE_CHECKING:
    from ..timsdata import TimsData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure geometry — no TimsData, unit-testable with identity converters
# ---------------------------------------------------------------------------


def _scanline_spans(
    poly_mz: np.ndarray, poly_im: np.ndarray, y: float
) -> list[tuple[float, float]]:
    """m/z spans where the horizontal line ``1/K0 == y`` lies inside the polygon.

    Uses the even-odd ray-crossing rule over the closed edge ring (the last
    vertex connects back to the first). Returns sorted, pairwise ``(lo, hi)``
    m/z spans. The half-open crossing test counts each edge once and is robust
    at vertices.
    """
    n = len(poly_mz)
    xs: list[float] = []
    for i in range(n):
        j = (i + 1) % n
        yi, yj = poly_im[i], poly_im[j]
        if (yi > y) != (yj > y):
            t = (y - yi) / (yj - yi)
            xs.append(poly_mz[i] + t * (poly_mz[j] - poly_mz[i]))
    xs.sort()
    return [(xs[k], xs[k + 1]) for k in range(0, len(xs) - 1, 2)]


def _merge_mz_spans(
    spans: list[tuple[float, float]], mz_pad: float
) -> list[tuple[float, float]]:
    """Pad each span by ``mz_pad`` per side and merge overlapping spans (sorted)."""
    if not spans:
        return []
    spans = sorted(spans)
    merged: list[tuple[float, float]] = []
    cur_lo, cur_hi = spans[0][0] - mz_pad, spans[0][1] + mz_pad
    for lo, hi in spans[1:]:
        lo, hi = lo - mz_pad, hi + mz_pad
        if lo <= cur_hi:
            cur_hi = max(cur_hi, hi)
        else:
            merged.append((cur_lo, cur_hi))
            cur_lo, cur_hi = lo, hi
    merged.append((cur_lo, cur_hi))
    return merged


def _coalesce_intervals(
    intervals: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort inclusive integer intervals and coalesce touching/overlapping ones.

    Returns parallel ``(lo, hi)`` int64 arrays. A one-index gap is bridged so
    intervals that merely abut after rounding become one.
    """
    if not intervals:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    intervals = sorted(intervals)
    lo_out: list[int] = []
    hi_out: list[int] = []
    for lo, hi in intervals:
        if lo_out and lo <= hi_out[-1] + 1:
            hi_out[-1] = max(hi_out[-1], hi)
        else:
            lo_out.append(lo)
            hi_out.append(hi)
    return np.asarray(lo_out, dtype=np.int64), np.asarray(hi_out, dtype=np.int64)


@dataclass(frozen=True)
class PerScanTofIntervals:
    """Per-scan sorted TOF-index intervals defining a keep region.

    ``lo[s]`` / ``hi[s]`` are parallel int arrays of inclusive interval bounds
    for mobility scan ``s``; a point ``(scan, tof)`` is kept iff ``tof`` lands in
    one of scan ``scan``'s intervals. Empty rows keep nothing.
    """

    lo: tuple[np.ndarray, ...]
    hi: tuple[np.ndarray, ...]

    @property
    def num_scans(self) -> int:
        return len(self.lo)

    @property
    def is_empty(self) -> bool:
        """True when no scan covers any TOF interval (nothing would be kept)."""
        return all(a.size == 0 for a in self.lo)

    def contains(self, scan: int, tof: int) -> bool:
        """Scalar membership test (mainly for tests)."""
        if scan < 0 or scan >= self.num_scans:
            return False
        hi = self.hi[scan]
        i = int(np.searchsorted(hi, tof, side="left"))
        return i < hi.size and tof >= self.lo[scan][i]

    def keep_mask(
        self, scan_indices: np.ndarray, tof_indices: np.ndarray
    ) -> np.ndarray:
        """Vectorised per-point keep mask (input order)."""
        n = scan_indices.size
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out
        order = np.argsort(scan_indices, kind="stable")
        ss = scan_indices[order]
        # Group the sorted points by scan and test each group in one searchsorted.
        starts = np.concatenate(([0], np.flatnonzero(np.diff(ss)) + 1))
        ends = np.concatenate((starts[1:], [n]))
        for st, en in zip(starts, ends):
            s = int(ss[st])
            if s < 0 or s >= self.num_scans:
                continue
            hi = self.hi[s]
            if hi.size == 0:
                continue
            pts = order[st:en]
            tofs = tof_indices[pts]
            i = np.searchsorted(hi, tofs, side="left")  # first hi >= tof
            valid = i < hi.size
            iv = i[valid]
            inside = np.zeros(tofs.size, dtype=bool)
            inside[valid] = tofs[valid] >= self.lo[s][iv]
            out[pts] = inside
        return out


def build_polygon_intervals(
    poly_mz: np.ndarray,
    poly_im: np.ndarray,
    ook0_per_scan: np.ndarray,
    mz_to_tof: Callable[[np.ndarray], np.ndarray],
    *,
    mz_pad: float = 0.0,
    im_pad: float = 0.0,
) -> PerScanTofIntervals | None:
    """Convert polygon vertices to per-scan TOF intervals.

    ``ook0_per_scan[s]`` is the ``1/K0`` of scan ``s``; ``mz_to_tof`` maps an
    m/z array to (fractional) TOF indices and must be monotonic increasing.
    Returns ``None`` for a degenerate polygon (<3 vertices, mismatched lengths)
    or when the polygon covers no scan's mobility.
    """
    n = len(poly_mz)
    num_scans = len(ook0_per_scan)
    if n < 3 or len(poly_im) != n or num_scans == 0:
        return None

    owner: list[int] = []
    span_lo: list[float] = []
    span_hi: list[float] = []
    for s in range(num_scans):
        y0 = float(ook0_per_scan[s])
        ys = (y0,) if im_pad == 0.0 else (y0 - im_pad, y0, y0 + im_pad)
        spans: list[tuple[float, float]] = []
        for y in ys:
            spans.extend(_scanline_spans(poly_mz, poly_im, y))
        for lo, hi in _merge_mz_spans(spans, mz_pad):
            owner.append(s)
            span_lo.append(lo)
            span_hi.append(hi)

    if not owner:
        return None

    # Batch the monotone m/z -> TOF conversion; floor the low edge and ceil the
    # high edge so a point exactly on the boundary is kept. Clamp negative m/z.
    t_lo = np.floor(
        np.maximum(mz_to_tof(np.maximum(np.asarray(span_lo), 0.0)), 0.0)
    ).astype(np.int64)
    t_hi = np.ceil(
        np.maximum(mz_to_tof(np.maximum(np.asarray(span_hi), 0.0)), 0.0)
    ).astype(np.int64)

    rows: list[list[tuple[int, int]]] = [[] for _ in range(num_scans)]
    for k, s in enumerate(owner):
        if t_hi[k] >= t_lo[k]:
            rows[s].append((int(t_lo[k]), int(t_hi[k])))

    los, his = zip(*(_coalesce_intervals(r) for r in rows))
    gate = PerScanTofIntervals(lo=tuple(los), hi=tuple(his))
    return None if gate.is_empty else gate


def build_window_intervals(
    boxes: list[tuple[int, int, int, int]], num_scans: int
) -> PerScanTofIntervals | None:
    """Build per-scan TOF intervals from ``(scan_lo, scan_hi, tof_lo, tof_hi)`` boxes.

    All bounds are inclusive; scans are clamped into ``0..num_scans``. Returns
    ``None`` when there are no usable boxes (e.g. ddaPASEF).
    """
    if not boxes or num_scans == 0:
        return None
    rows: list[list[tuple[int, int]]] = [[] for _ in range(num_scans)]
    for scan_lo, scan_hi, tof_lo, tof_hi in boxes:
        if tof_hi < tof_lo:
            continue
        lo = min(scan_lo, num_scans - 1)
        hi = min(scan_hi, num_scans - 1)
        for s in range(lo, hi + 1):
            rows[s].append((tof_lo, tof_hi))
    los, his = zip(*(_coalesce_intervals(r) for r in rows))
    gate = PerScanTofIntervals(lo=tuple(los), hi=tuple(his))
    return None if gate.is_empty else gate


# ---------------------------------------------------------------------------
# TDF metadata readers
# ---------------------------------------------------------------------------


def _has_table(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def read_selection_polygon(td: "TimsData") -> tuple[np.ndarray, np.ndarray] | None:
    """Read the PASEF selection polygon (IMS PolygonFilter) as ``(mz, im)`` arrays.

    Bruker stores the two vertex arrays as little-endian ``f64`` BLOBs in
    ``GroupProperties``, keyed by the ``IMS_PolygonFilter_Mass`` /
    ``IMS_PolygonFilter_Mobility`` property definitions (resolved by permanent
    name — numeric ids are not stable). Returns ``None`` when the run carries no
    usable polygon (property absent, no value, <3 vertices, or length mismatch).
    """
    conn = td.conn
    if conn is None:
        return None
    if not _has_table(conn, "PropertyDefinitions") or not _has_table(
        conn, "GroupProperties"
    ):
        return None

    def prop_id(name: str) -> int | None:
        row = conn.execute(
            "SELECT Id FROM PropertyDefinitions WHERE PermanentName=?", (name,)
        ).fetchone()
        return int(row[0]) if row is not None else None

    mz_id = prop_id("IMS_PolygonFilter_Mass")
    im_id = prop_id("IMS_PolygonFilter_Mobility")
    if mz_id is None or im_id is None:
        return None

    def read_blob(prop: int) -> np.ndarray | None:
        row = conn.execute(
            "SELECT Value FROM GroupProperties WHERE Property=? LIMIT 1", (prop,)
        ).fetchone()
        if row is None or row[0] is None:
            return None
        return np.frombuffer(bytes(row[0]), dtype="<f8")

    mz = read_blob(mz_id)
    im = read_blob(im_id)
    if mz is None or im is None or mz.size < 3 or mz.size != im.size:
        return None
    return np.asarray(mz, dtype=np.float64), np.asarray(im, dtype=np.float64)


def read_dia_ms1_boxes(td: "TimsData") -> list[tuple[int, int, float, float]]:
    """Read distinct diaPASEF isolation windows as ``(scan_begin, scan_end, mz_lo, mz_hi)``.

    ``scan_end`` is exclusive (as stored). ``mz_lo``/``mz_hi`` derive from
    ``IsolationMz ± IsolationWidth/2``. Returns an empty list for ddaPASEF / when
    ``DiaFrameMsMsWindows`` is absent or carries no m/z info.
    """
    conn = td.conn
    if conn is None or not _has_table(conn, "DiaFrameMsMsWindows"):
        return []
    rows = conn.execute(
        "SELECT DISTINCT ScanNumBegin, ScanNumEnd, IsolationMz, IsolationWidth "
        "FROM DiaFrameMsMsWindows "
        "WHERE IsolationMz IS NOT NULL AND IsolationWidth IS NOT NULL"
    ).fetchall()
    boxes: list[tuple[int, int, float, float]] = []
    for scan_begin, scan_end, iso_mz, iso_width in rows:
        scan_begin, scan_end = int(scan_begin), int(scan_end)
        if scan_end <= scan_begin:
            continue
        half = float(iso_width) / 2.0
        boxes.append((scan_begin, scan_end, float(iso_mz) - half, float(iso_mz) + half))
    return boxes


# ---------------------------------------------------------------------------
# NoiseFilter gates (run-level gate cached per TimsData + params)
# ---------------------------------------------------------------------------

# Cache the built per-scan gate per TimsData so it isn't rebuilt for every frame.
# Keyed by (kind, num_scans, mz_pad, im_pad); the value ``None`` records "no gate
# for this run" (e.g. no polygon / no windows) so repeated calls stay cheap.
_GATE_CACHE: "WeakKeyDictionary[TimsData, dict[tuple, PerScanTofIntervals | None]]" = (
    WeakKeyDictionary()
)


def _cached(
    td: "TimsData", key: tuple, build: Callable[[], PerScanTofIntervals | None]
) -> PerScanTofIntervals | None:
    per_td = _GATE_CACHE.setdefault(td, {})
    if key not in per_td:
        per_td[key] = build()
    return per_td[key]


@dataclass(frozen=True)
class SelectionPolygonGate(NoiseFilter):
    """Keep only MS1 points inside the ddaPASEF selection polygon.

    Reads the run's IMS PolygonFilter and drops MS1 signal outside it — data the
    instrument never schedules as a precursor. No-op (keeps everything) when the
    run stores no polygon, or when the run is diaPASEF (there the same property
    stores window-placement quads, not one selection ring, so it is not used).

    Parameters (both in physical units, widening the kept region per side so an
    edge precursor keeps its isotopic envelope / mobility spread):
        mz_pad: m/z padding in Da (default 0).
        im_pad: 1/K0 padding (default 0).
    """

    mz_pad: float = 0.0
    im_pad: float = 0.0

    def keep_mask(
        self,
        scan_indices: np.ndarray,
        mz_indices: np.ndarray,
        intensities: np.ndarray,
        *,
        num_scans: int,
        td: "TimsData",
        frame_id: int,
    ) -> np.ndarray:
        key = ("polygon", num_scans, self.mz_pad, self.im_pad)
        gate = _cached(
            td, key, lambda: _build_polygon_gate(td, frame_id, num_scans, self)
        )
        if gate is None:
            logger.debug(
                "SelectionPolygonGate: no usable polygon for this run; keeping all "
                "%d points.",
                intensities.size,
            )
            return np.ones(intensities.size, dtype=bool)
        return gate.keep_mask(scan_indices, mz_indices)


@dataclass(frozen=True)
class DiaMs1WindowGate(NoiseFilter):
    """Keep only MS1 points inside the union of diaPASEF isolation windows.

    Drops MS1 signal in no isolation window — precursors the method never
    isolates. No-op (keeps everything) on ddaPASEF / when no windows are defined.

    Parameters (physical-unit padding per side, converted once via the run
    calibration, so an edge precursor keeps its isotopes / mobility spread):
        mz_pad: m/z padding in Da (default 5.0).
        im_pad: 1/K0 padding (default 0.05).
    """

    mz_pad: float = 5.0
    im_pad: float = 0.05

    def keep_mask(
        self,
        scan_indices: np.ndarray,
        mz_indices: np.ndarray,
        intensities: np.ndarray,
        *,
        num_scans: int,
        td: "TimsData",
        frame_id: int,
    ) -> np.ndarray:
        key = ("dia_ms1", num_scans, self.mz_pad, self.im_pad)
        gate = _cached(
            td, key, lambda: _build_dia_ms1_gate(td, frame_id, num_scans, self)
        )
        if gate is None:
            logger.debug(
                "DiaMs1WindowGate: no isolation windows for this run (ddaPASEF?); "
                "keeping all %d points.",
                intensities.size,
            )
            return np.ones(intensities.size, dtype=bool)
        return gate.keep_mask(scan_indices, mz_indices)


def _build_polygon_gate(
    td: "TimsData", frame_id: int, num_scans: int, params: SelectionPolygonGate
) -> PerScanTofIntervals | None:
    # diaPASEF stores multiple window quads under the same property, not one
    # selection ring — skip the polygon gate there (dia_ms1 handles diaPASEF MS1).
    if read_dia_ms1_boxes(td):
        return None
    poly = read_selection_polygon(td)
    if poly is None or num_scans == 0:
        return None
    poly_mz, poly_im = poly
    ook0_per_scan = np.asarray(
        td.scanNumToOneOverK0(frame_id, np.arange(num_scans))  # type: ignore[call-arg]
    )
    return build_polygon_intervals(
        poly_mz,
        poly_im,
        ook0_per_scan,
        lambda mz: np.asarray(td.mzToIndex(frame_id, mz)),
        mz_pad=params.mz_pad,
        im_pad=params.im_pad,
    )


def _build_dia_ms1_gate(
    td: "TimsData", frame_id: int, num_scans: int, params: DiaMs1WindowGate
) -> PerScanTofIntervals | None:
    boxes = read_dia_ms1_boxes(td)
    if not boxes or num_scans == 0:
        return None

    tof_boxes: list[tuple[int, int, int, int]] = []
    for scan_begin, scan_end, mz_lo, mz_hi in boxes:
        # m/z edges -> TOF indices (monotonic increasing), padded by mz_pad Da.
        t0, t1 = np.asarray(
            td.mzToIndex(frame_id, [mz_lo - params.mz_pad, mz_hi + params.mz_pad])
        )
        tof_lo = int(np.floor(max(min(t0, t1), 0.0)))
        tof_hi = int(np.ceil(max(max(t0, t1), 0.0)))

        # Scan range -> 1/K0 (monotonic decreasing), padded by im_pad, back to
        # scans. min/max keep it correct regardless of conversion direction.
        im0, im1 = np.asarray(td.scanNumToOneOverK0(frame_id, [scan_begin, scan_end]))
        s0, s1 = np.asarray(
            td.oneOverK0ToScanNum(
                frame_id, [max(im0, im1) + params.im_pad, min(im0, im1) - params.im_pad]
            )
        )
        scan_lo = int(np.floor(max(min(s0, s1), 0.0)))
        scan_hi = min(int(np.ceil(max(max(s0, s1), 0.0))), num_scans - 1)
        tof_boxes.append((scan_lo, scan_hi, tof_lo, tof_hi))

    return build_window_intervals(tof_boxes, num_scans)


__all__ = [
    "SelectionPolygonGate",
    "DiaMs1WindowGate",
    "PerScanTofIntervals",
    "build_polygon_intervals",
    "build_window_intervals",
    "read_selection_polygon",
    "read_dia_ms1_boxes",
]
