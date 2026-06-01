"""Content-aware structural noise filters.

The vertical-noise filter identifies real ions by their characteristic
vertical streaks in ``(scan_number, TOF_index)`` space — see
``apps/ALGORITHM.md`` for the full algorithm write-up. It is the
canonical structural filter for timsTOF MS1 data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from . import NoiseFilter

if TYPE_CHECKING:
    from ..timsdata import TimsData


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
        and per-pass telemetry — used by the IM-feature-filter dashboard.
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
