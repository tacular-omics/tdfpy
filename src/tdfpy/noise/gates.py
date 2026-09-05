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
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .._validation import nonnegative
from . import NoiseFilter

if TYPE_CHECKING:
    from ..timsdata import TimsData

logger = logging.getLogger(__name__)


def _is_ms1_frame(td: "TimsData", frame_id: int) -> bool:
    """Check immutable frame metadata without a per-call SQL query."""
    if td.handle is None:
        return False
    try:
        return td.frame_metadata(frame_id).msms_type == 0
    except ValueError:
        return False


def _ms1_only_noop(
    td: "TimsData", frame_id: int, intensities: np.ndarray, gate_name: str
) -> np.ndarray | None:
    """Keep-all mask (with a debug log) when ``gate_name`` must not gate this frame.

    Returns ``None`` when the frame is a confirmed MS1 frame and the gate should
    proceed. Shared by both gates so their MS1-only no-op contract stays in one
    place.
    """
    if _is_ms1_frame(td, frame_id):
        return None
    logger.debug(
        "%s applied to non-MS1 frame %d; it gates MS1 precursor selection only "
        "— keeping all %d points.",
        gate_name,
        frame_id,
        intensities.size,
    )
    return np.ones(intensities.size, dtype=bool)


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
        # read_spectrum emits scan_indices already sorted ascending; only pay for
        # a sort when a caller hands us out-of-order input. Reuse the same diff
        # for both the ordering check and the group-boundary scan.
        diffs = np.diff(scan_indices)
        if np.any(diffs < 0):
            order = np.argsort(scan_indices, kind="stable")
            ss = scan_indices[order]
            group_diffs = np.diff(ss)
        else:
            order = None  # identity mapping: group slices index directly
            ss = scan_indices
            group_diffs = diffs
        # Group the sorted points by scan and test each group in one searchsorted.
        starts = np.concatenate(([0], np.flatnonzero(group_diffs) + 1))
        ends = np.concatenate((starts[1:], [n]))
        for st, en in zip(starts, ends):
            s = int(ss[st])
            if s < 0 or s >= self.num_scans:
                continue
            hi = self.hi[s]
            if hi.size == 0:
                continue
            pts = slice(st, en) if order is None else order[st:en]
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

    All bounds are inclusive. Scan bounds are clamped into ``[0, num_scans)``;
    a box lying wholly outside that range (``scan_hi < 0`` or ``scan_lo >=
    num_scans``) is skipped rather than folded onto an edge scan. Returns
    ``None`` when there are no usable boxes (e.g. ddaPASEF).
    """
    if not boxes or num_scans == 0:
        return None
    rows: list[list[tuple[int, int]]] = [[] for _ in range(num_scans)]
    for scan_lo, scan_hi, tof_lo, tof_hi in boxes:
        if tof_hi < tof_lo:
            continue
        # Clamp into [0, num_scans); skip boxes lying wholly outside the range
        # (a negative hi or an lo past the last scan) rather than letting a
        # negative bound wrap around via Python indexing.
        lo = max(0, scan_lo)
        hi = min(scan_hi, num_scans - 1)
        if hi < lo:
            continue
        for s in range(lo, hi + 1):
            rows[s].append((tof_lo, tof_hi))
    los, his = zip(*(_coalesce_intervals(r) for r in rows))
    gate = PerScanTofIntervals(lo=tuple(los), hi=tuple(his))
    return None if gate.is_empty else gate


# ---------------------------------------------------------------------------
# TDF metadata readers
# ---------------------------------------------------------------------------


def read_selection_polygon(
    td: "TimsData", frame_id: int | None = None
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read a selection polygon from the opening-time metadata snapshot.

    When frame_id is supplied, use that frame's property group. Both vertex
    arrays must belong to the same property group.
    """
    if td.handle is None:
        return None
    definitions = {
        r["PermanentName"]: r["Id"] for r in td.metadata_table("PropertyDefinitions")
    }
    mz_id = definitions.get("IMS_PolygonFilter_Mass")
    im_id = definitions.get("IMS_PolygonFilter_Mobility")
    if mz_id is None or im_id is None:
        return None
    groups: dict[int | None, dict[int, Any]] = {}
    for row in td.metadata_table("GroupProperties"):
        groups.setdefault(row["PropertyGroup"], {})[row["Property"]] = row["Value"]
    if frame_id is None:
        candidates = list(groups.values())
    else:
        candidates = [groups.get(td.frame_metadata(frame_id).property_group, {})]
    for values in candidates:
        mz_blob, im_blob = values.get(mz_id), values.get(im_id)
        if mz_blob is None or im_blob is None:
            continue
        mz = np.frombuffer(mz_blob, dtype="<f8")
        im = np.frombuffer(im_blob, dtype="<f8")
        if mz.size >= 3 and mz.size == im.size:
            return mz, im
    return None


def read_dia_ms1_boxes(td: "TimsData") -> list[tuple[int, int, float, float]]:
    """Read distinct half-open DIA scan windows from the metadata snapshot."""
    if td.handle is None:
        return []
    boxes = []
    seen = set()
    for row in td.metadata_table("DiaFrameMsMsWindows"):
        begin, end = int(row["ScanNumBegin"]), int(row["ScanNumEnd"])
        mz, width = row["IsolationMz"], row["IsolationWidth"]
        if mz is None or width is None or end <= begin:
            continue
        box = (begin, end, float(mz) - float(width) / 2, float(mz) + float(width) / 2)
        if box not in seen:
            boxes.append(box)
            seen.add(box)
    return boxes


def _cached(
    td: "TimsData", key: tuple, build: Callable[[], PerScanTofIntervals | None]
) -> PerScanTofIntervals | None:
    # Keep caches on their reader and serialize first construction. Bound the
    # cache because temperature changes may produce many effective calibrations.
    with td._gate_lock:
        if key not in td._gate_cache:
            value = build()
            if len(td._gate_cache) >= 16:
                td._gate_cache.pop(next(iter(td._gate_cache)))
            td._gate_cache[key] = value
        return td._gate_cache[key]


@dataclass(frozen=True)
class SelectionPolygonGate(NoiseFilter):
    """Keep only MS1 points inside the ddaPASEF selection polygon.

    Reads the run's IMS PolygonFilter and drops MS1 signal outside it — data the
    instrument never schedules as a precursor. No-op (keeps everything) when the
    run stores no polygon, or when the run is diaPASEF (there the same property
    stores window-placement quads, not one selection ring, so it is not used).

    Parameters (both in physical units, widening the kept region per side so an
    edge precursor keeps its isotopic envelope / mobility spread rather than being
    clipped at a hard polygon boundary):
        mz_pad: m/z padding in Da (default 5.0).
        im_pad: 1/K0 padding (default 0.05).
    """

    mz_pad: float = 5.0
    im_pad: float = 0.05

    def __post_init__(self) -> None:
        nonnegative("mz_pad", self.mz_pad)
        nonnegative("im_pad", self.im_pad)

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
        noop = _ms1_only_noop(td, frame_id, intensities, "SelectionPolygonGate")
        if noop is not None:
            return noop
        key = (
            "polygon",
            num_scans,
            self.mz_pad,
            self.im_pad,
            td.calibration_key(frame_id),
            td.frame_metadata(frame_id).property_group,
        )
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

    def __post_init__(self) -> None:
        nonnegative("mz_pad", self.mz_pad)
        nonnegative("im_pad", self.im_pad)

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
        noop = _ms1_only_noop(td, frame_id, intensities, "DiaMs1WindowGate")
        if noop is not None:
            return noop
        key = (
            "dia_ms1",
            num_scans,
            self.mz_pad,
            self.im_pad,
            td.calibration_key(frame_id),
        )
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
    poly = read_selection_polygon(td, frame_id)
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
        # Preserve [begin, end) after padding. Snap numerical round-trip
        # residue at integer boundaries before taking the ceiling.
        bounds = np.array([min(s0, s1), max(s0, s1)])
        nearest = np.rint(bounds)
        bounds = np.where(np.abs(bounds - nearest) < 1e-9, nearest, bounds)
        scan_lo = max(0, int(np.ceil(bounds[0])))
        scan_hi = min(int(np.ceil(bounds[1])) - 1, num_scans - 1)
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
