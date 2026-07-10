"""Content-aware structural noise filters.

The vertical-noise filter identifies real ions by their characteristic
vertical streaks in ``(scan_number, TOF_index)`` space — see
``apps/ALGORITHM.md`` for the full algorithm write-up. It is the
canonical structural filter for timsTOF MS1 data.

The horizontal-halo filter (:class:`HorizontalHaloFilter`) removes the weak
m/z halo flanking each bright ion — comparing every peak to the maximum of its
surrounding box *excluding its own m/z column*, so it only ever drops
left/right neighbours and never the vertical streak above or below.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from . import NoiseFilter

try:
    from numba import njit as _njit  # ty: ignore[unresolved-import]
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

if TYPE_CHECKING:
    from ..timsdata import TimsData

logger = logging.getLogger(__name__)


@dataclass
class VerticalNoiseDiagnostics:
    """Diagnostics from a single or iterated pass of :class:`VerticalNoiseFilter`.

    Fields are populated from the *final* pass when ``num_iterations > 1``,
    except for :attr:`per_pass_kept` which traces all passes.
    """

    keep_point_mask: np.ndarray  # bool, len = num input points
    num_columns_evaluated: int  # one entry per unique mz_index seen
    num_columns_with_kept_runs: int
    num_kept_points: int
    feature_span_intensities: np.ndarray  # float64, total intensity per
    #   gap-closed run that cleared min_streak_scans (before the
    #   min_streak_intensity filter). Useful as a tuning histogram.
    per_pass_kept: list[int] = field(default_factory=list)
    #   Length num_iterations + 1; per_pass_kept[0] is the input count.


if _HAS_NUMBA:
    @_njit(cache=True)
    def _vertical_single_pass_njit(
        mz_sorted: np.ndarray,
        scan_sorted: np.ndarray,
        int_sorted: np.ndarray,
        first_idx: np.ndarray,
        num_scans: int,
        mz_idx_half_width: int,
        min_streak_scans: int,
        max_gap_scans: int,
        min_streak_intensity: float,
    ):
        """Numba single-pass kernel over m/z-sorted points.

        Mirrors :func:`_single_pass_filter_python` exactly. Uses a forward
        two-pointer window over the sorted points (centres increase
        monotonically, so the window bounds never move backward) and keeps an
        incremental per-scan intensity ``profile`` updated as points enter and
        leave the ``±mz_idx_half_width`` window. Returns ``(keep_sorted,
        num_columns_with_kept_runs)``.
        """
        n = mz_sorted.size
        u = first_idx.size
        keep = np.zeros(n, dtype=np.bool_)
        profile = np.zeros(num_scans, dtype=np.float64)
        run_lo = np.empty(num_scans, dtype=np.int64)
        run_hi = np.empty(num_scans, dtype=np.int64)
        left = 0
        right = 0
        n_cols_kept = 0

        for k in range(u):
            center = mz_sorted[first_idx[k]]
            lo_val = center - mz_idx_half_width
            hi_val = center + mz_idx_half_width
            while right < n and mz_sorted[right] <= hi_val:
                profile[scan_sorted[right]] += int_sorted[right]
                right += 1
            while left < n and mz_sorted[left] < lo_val:
                profile[scan_sorted[left]] -= int_sorted[left]
                left += 1

            # Walk scans, collecting gap-closed occupied runs that clear the
            # span + intensity thresholds.
            nkr = 0
            run_first = -1
            run_last = -1
            run_sum = 0.0
            for s in range(num_scans):
                p = profile[s]
                if p > 0.0:
                    if run_first == -1:
                        run_first = s
                        run_last = s
                        run_sum = p
                    elif (s - run_last - 1) > max_gap_scans:
                        if (run_last - run_first + 1) >= min_streak_scans and \
                                run_sum >= min_streak_intensity:
                            run_lo[nkr] = run_first
                            run_hi[nkr] = run_last
                            nkr += 1
                        run_first = s
                        run_last = s
                        run_sum = p
                    else:
                        run_last = s
                        run_sum += p
            if run_first != -1 and (run_last - run_first + 1) >= min_streak_scans and \
                    run_sum >= min_streak_intensity:
                run_lo[nkr] = run_first
                run_hi[nkr] = run_last
                nkr += 1
            if nkr > 0:
                n_cols_kept += 1

            # Keep this column's own points whose scan falls in a kept run.
            start = first_idx[k]
            end = first_idx[k + 1] if k + 1 < u else n
            for i in range(start, end):
                sc = scan_sorted[i]
                for r in range(nkr):
                    if run_lo[r] <= sc and sc <= run_hi[r]:
                        keep[i] = True
                        break
        return keep, n_cols_kept


def _single_pass_filter_python(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    num_scans: int,
    *,
    mz_idx_half_width: int,
    min_streak_scans: int,
    max_gap_scans: int,
    min_streak_intensity: float,
    collect_span_intensities: bool = False,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Pure-NumPy reference implementation of one pass (also the fallback).

    Returns ``(keep_mask, num_columns_evaluated, num_columns_with_kept_runs,
    feature_span_intensities)``. The last is populated only if
    ``collect_span_intensities`` is true (the per-frame tuning histogram).
    """
    n = scan_indices.size
    if n == 0:
        return (
            np.zeros(0, dtype=bool),
            0,
            0,
            np.zeros(0, dtype=np.float64),
        )

    order = np.argsort(mz_indices, kind="stable")
    mz_sorted = mz_indices[order]
    scan_sorted = scan_indices[order]
    int_sorted = intensities[order].astype(np.float64, copy=False)

    keep_sorted = np.zeros(n, dtype=bool)
    unique_mz, first_idx = np.unique(mz_sorted, return_index=True)
    pt_boundaries = np.concatenate([first_idx, [n]])
    gap_threshold = int(max_gap_scans) + 1

    num_columns_with_kept_runs = 0
    span_intensities: list[float] = []

    for k in range(unique_mz.size):
        center = int(unique_mz[k])
        left = int(np.searchsorted(mz_sorted, center - mz_idx_half_width, side="left"))
        right = int(np.searchsorted(mz_sorted, center + mz_idx_half_width, side="right"))

        window_scans = scan_sorted[left:right]
        window_int = int_sorted[left:right]
        profile = np.bincount(window_scans, weights=window_int, minlength=num_scans)

        occupied = profile > 0.0
        if not occupied.any():
            continue

        occ_scans = np.where(occupied)[0]
        diffs = np.diff(occ_scans)
        breaks = np.where(diffs > gap_threshold)[0] + 1
        run_starts = np.concatenate(([0], breaks))
        run_ends = np.concatenate((breaks, [occ_scans.size]))

        kept_scans = np.zeros(num_scans, dtype=bool)
        any_run_kept = False
        for run_start, run_end in zip(run_starts, run_ends):
            first_scan = int(occ_scans[run_start])
            last_scan = int(occ_scans[run_end - 1])
            span = last_scan - first_scan + 1
            if span < min_streak_scans:
                continue
            total_intensity = float(profile[first_scan : last_scan + 1].sum())
            if collect_span_intensities:
                span_intensities.append(total_intensity)
            if total_intensity < float(min_streak_intensity):
                continue
            kept_scans[first_scan : last_scan + 1] = True
            any_run_kept = True
        if any_run_kept:
            num_columns_with_kept_runs += 1

        pts_start = int(pt_boundaries[k])
        pts_end = int(pt_boundaries[k + 1])
        keep_sorted[pts_start:pts_end] = kept_scans[scan_sorted[pts_start:pts_end]]

    keep_mask = np.empty(n, dtype=bool)
    keep_mask[order] = keep_sorted
    return (
        keep_mask,
        int(unique_mz.size),
        num_columns_with_kept_runs,
        np.asarray(span_intensities, dtype=np.float64),
    )


def _single_pass_filter(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    num_scans: int,
    *,
    mz_idx_half_width: int,
    min_streak_scans: int,
    max_gap_scans: int,
    min_streak_intensity: float,
    collect_span_intensities: bool = False,
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """One pass of the vertical-noise filter.

    Dispatches to the Numba kernel when available (the common path); falls back
    to :func:`_single_pass_filter_python` when Numba is missing or when the
    per-run span-intensity histogram is requested (``collect_span_intensities``,
    used only by the tuning dashboard).
    """
    if not _HAS_NUMBA or collect_span_intensities:
        return _single_pass_filter_python(
            scan_indices, mz_indices, intensities, num_scans,
            mz_idx_half_width=mz_idx_half_width,
            min_streak_scans=min_streak_scans,
            max_gap_scans=max_gap_scans,
            min_streak_intensity=min_streak_intensity,
            collect_span_intensities=collect_span_intensities,
        )

    n = scan_indices.size
    if n == 0:
        return (np.zeros(0, dtype=bool), 0, 0, np.zeros(0, dtype=np.float64))

    order = np.argsort(mz_indices, kind="stable")
    mz_sorted = np.ascontiguousarray(mz_indices[order], dtype=np.int64)
    scan_sorted = np.ascontiguousarray(scan_indices[order], dtype=np.int64)
    int_sorted = np.ascontiguousarray(intensities[order], dtype=np.float64)
    _unique_mz, first_idx = np.unique(mz_sorted, return_index=True)

    keep_sorted, n_cols_kept = _vertical_single_pass_njit(
        mz_sorted, scan_sorted, int_sorted,
        np.ascontiguousarray(first_idx, dtype=np.int64), int(num_scans),
        int(mz_idx_half_width), int(min_streak_scans), int(max_gap_scans),
        float(min_streak_intensity),
    )
    keep_mask = np.empty(n, dtype=bool)
    keep_mask[order] = keep_sorted
    return (keep_mask, int(first_idx.size), int(n_cols_kept), np.zeros(0, dtype=np.float64))


@dataclass(frozen=True)
class VerticalNoiseFilter(NoiseFilter):
    """Keep points belonging to vertical streaks in (scan, TOF_index) space.

    A real ion produces an intensity streak along the ion-mobility axis at
    roughly the same TOF index across many consecutive scans. Noise tends
    to be isolated single hits or short streaks. This filter walks each
    TOF index, builds the IM intensity profile in a small m/z window,
    finds gap-closed runs of occupied scans, and keeps points whose scan
    falls inside a run that's long enough and intense enough.

    Iterated passes (``num_iterations > 1``) feed each pass the survivors
    of the previous one — points that only just survived because they sat
    next to barely-thick noise get dropped on a later pass once that noise
    is gone.

    See ``apps/ALGORITHM.md`` for the full write-up.
    """

    mz_idx_half_width: int = 3
    min_streak_scans: int = 5
    max_gap_scans: int = 1
    min_streak_intensity: float = 50.0
    num_iterations: int = 2

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
        return self.run(
            scan_indices, mz_indices, intensities, num_scans=num_scans,
            diagnostics=False,
        )

    @overload
    def run(
        self,
        scan_indices: np.ndarray,
        mz_indices: np.ndarray,
        intensities: np.ndarray,
        *,
        num_scans: int,
        diagnostics: Literal[False] = ...,
    ) -> np.ndarray: ...

    @overload
    def run(
        self,
        scan_indices: np.ndarray,
        mz_indices: np.ndarray,
        intensities: np.ndarray,
        *,
        num_scans: int,
        diagnostics: Literal[True],
    ) -> "VerticalNoiseDiagnostics": ...

    def run(
        self,
        scan_indices: np.ndarray,
        mz_indices: np.ndarray,
        intensities: np.ndarray,
        *,
        num_scans: int,
        diagnostics: bool = False,
    ) -> "np.ndarray | VerticalNoiseDiagnostics":
        """Run the filter on raw arrays.

        When ``diagnostics`` is False (default) returns the keep-mask only.
        When True returns a :class:`VerticalNoiseDiagnostics` carrying the mask
        and per-pass telemetry — used by the timsTOF viewer's IM-filter page.
        """
        n = scan_indices.size
        if n == 0:
            empty = np.zeros(0, dtype=bool)
            if not diagnostics:
                return empty
            return VerticalNoiseDiagnostics(
                keep_point_mask=empty,
                num_columns_evaluated=0,
                num_columns_with_kept_runs=0,
                num_kept_points=0,
                feature_span_intensities=np.zeros(0, dtype=np.float64),
                per_pass_kept=[0],
            )

        cumulative = np.ones(n, dtype=bool)
        per_pass_kept = [n]
        last_n_cols = 0
        last_n_cols_kept = 0
        last_span_intensities = np.zeros(0, dtype=np.float64)

        for _ in range(int(self.num_iterations)):
            active = np.nonzero(cumulative)[0]
            if active.size == 0:
                break
            mask, n_cols, n_cols_kept, spans = _single_pass_filter(
                scan_indices[cumulative],
                mz_indices[cumulative],
                intensities[cumulative],
                num_scans,
                mz_idx_half_width=self.mz_idx_half_width,
                min_streak_scans=self.min_streak_scans,
                max_gap_scans=self.max_gap_scans,
                min_streak_intensity=self.min_streak_intensity,
                collect_span_intensities=diagnostics,
            )
            kept = active[mask]
            cumulative = np.zeros(n, dtype=bool)
            cumulative[kept] = True
            per_pass_kept.append(int(cumulative.sum()))
            last_n_cols = n_cols
            last_n_cols_kept = n_cols_kept
            last_span_intensities = spans
            if not cumulative.any():
                break

        n_kept = int(cumulative.sum())
        logger.debug(
            "VerticalNoiseFilter: kept %d/%d points over %d pass(es) (per-pass: %s).",
            n_kept,
            n,
            len(per_pass_kept) - 1,
            per_pass_kept,
        )
        if n_kept == 0:
            logger.warning(
                "VerticalNoiseFilter: removed ALL %d points. The streak thresholds "
                "may be too strict (min_streak_scans=%d, min_streak_intensity=%.1f).",
                n,
                self.min_streak_scans,
                self.min_streak_intensity,
            )

        if not diagnostics:
            return cumulative
        return VerticalNoiseDiagnostics(
            keep_point_mask=cumulative,
            num_columns_evaluated=last_n_cols,
            num_columns_with_kept_runs=last_n_cols_kept,
            num_kept_points=int(cumulative.sum()),
            feature_span_intensities=last_span_intensities,
            per_pass_kept=per_pass_kept,
        )


# --------------------------------------------------------------------------
# Horizontal (m/z-axis) halo filter
# --------------------------------------------------------------------------


def _offcol_box_max_py(
    scan_sorted: np.ndarray,   # int64, sorted by (scan, mz_idx)
    mz_sorted: np.ndarray,     # int64
    int_sorted: np.ndarray,    # float64
    block_start: np.ndarray,   # int64, len num_scans (start index of each scan's block)
    block_len: np.ndarray,     # int64, len num_scans
    scan_half_width: int,
    mz_idx_half_width: int,
) -> np.ndarray:
    """Per-point max intensity over the box, excluding the point's own column.

    For each point, scans every source row in ``±scan_half_width`` and, within
    each row's m/z-sorted block, binary-searches the ``±mz_idx_half_width``
    window and tracks the largest intensity among peaks at a *different* TOF
    index. Windows are sparse (few peaks), so the inner scan is cheap.
    """
    n = scan_sorted.size
    num_scans = block_start.size
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        si = scan_sorted[i]
        mi = mz_sorted[i]
        lo_scan = si - scan_half_width
        hi_scan = si + scan_half_width
        if lo_scan < 0:
            lo_scan = 0
        if hi_scan > num_scans - 1:
            hi_scan = num_scans - 1
        lo_mz = mi - mz_idx_half_width
        hi_mz = mi + mz_idx_half_width
        best = 0.0
        for v in range(lo_scan, hi_scan + 1):
            bl = block_len[v]
            if bl == 0:
                continue
            bs = block_start[v]
            be = bs + bl
            # lower bound of lo_mz within [bs, be)
            left = bs
            right = be
            while left < right:
                mid = (left + right) // 2
                if mz_sorted[mid] < lo_mz:
                    left = mid + 1
                else:
                    right = mid
            j = left
            while j < be and mz_sorted[j] <= hi_mz:
                if mz_sorted[j] != mi and int_sorted[j] > best:
                    best = int_sorted[j]
                j += 1
        out[i] = best
    return out


if _HAS_NUMBA:
    _offcol_box_max = _njit(cache=True)(_offcol_box_max_py)
else:  # pragma: no cover
    _offcol_box_max = _offcol_box_max_py


@dataclass(frozen=True)
class HorizontalHaloFilter(NoiseFilter):
    """Remove the weak m/z halo flanking bright peaks — left/right only.

    High-intensity ions are flanked by a halo of weak peaks — likely from
    charge interactions within the fragment-ion cloud or detector effects —
    that are not resolvable to high-precision m/z values. A real ion forms a
    *vertical streak* along the ion-mobility axis (the same TOF index across
    many consecutive mobility scans), so this filter only removes peaks to the
    **left and right** (offset in TOF/m-z index) of a bright neighbour and
    **never above or below** (offset in ion mobility at the same index).

    For each peak it computes a local reference intensity — the **maximum
    intensity in the surrounding box** ``(±scan_half_width scans,
    ±mz_idx_half_width TOF indices)`` **excluding the peak's own m/z column**
    — and drops the peak if its intensity falls below ``peak_fraction`` of
    that reference. Excluding the peak's own column is what guarantees the
    vertical streak is never used against it: a bright peak directly above or
    below sits in the same column and so can never raise the threshold; only a
    genuine left/right neighbour can. A peak with no off-column neighbours in
    its box is always kept. Set ``scan_half_width=0`` for strictly per-row
    (same ion-mobility scan) behaviour.

    Operates entirely in integer ``(scan, TOF index)`` space — no unit
    conversion. The box max is JIT-compiled with Numba, with a pure-Python
    fallback.
    """

    peak_fraction: float = 0.15
    mz_idx_half_width: int = 100
    scan_half_width: int = 2

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
        n = intensities.size
        if n == 0:
            return np.ones(0, dtype=bool)
        # Sort by (scan, mz_idx) so each scan is a contiguous, m/z-sorted block.
        order = np.lexsort((mz_indices, scan_indices))
        scan_s = np.ascontiguousarray(scan_indices[order], dtype=np.int64)
        mz_s = np.ascontiguousarray(mz_indices[order], dtype=np.int64)
        int_s = np.ascontiguousarray(intensities[order], dtype=np.float64)

        block_start = np.zeros(num_scans, dtype=np.int64)
        block_len = np.zeros(num_scans, dtype=np.int64)
        uniq, starts, counts = np.unique(
            scan_s, return_index=True, return_counts=True
        )
        block_start[uniq] = starts
        block_len[uniq] = counts

        ref_sorted = _offcol_box_max(
            scan_s, mz_s, int_s, block_start, block_len,
            int(self.scan_half_width), int(self.mz_idx_half_width),
        )
        ref = np.empty(n, dtype=np.float64)
        ref[order] = ref_sorted
        # Keep unless strictly below the fraction of the left/right max.
        # Peaks with no off-column neighbours (ref == 0) are always kept.
        return intensities >= self.peak_fraction * ref
