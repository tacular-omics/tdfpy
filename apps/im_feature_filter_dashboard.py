"""Streamlit dashboard for testing the vertical-noise feature filter.

The filter operates in integer (scan_number, TOF-index) space — i.e. before
any m/z or 1/K0 conversion. Each input point is evaluated as the center of
its own bounding box: the mz dimension spans ±``mz_idx_half_width`` TOF
indices around the point, and the scan dimension spans the full IM axis.
The intuition: a real ion shows up as a vertical streak along the
ion-mobility (scan) axis at roughly the same m/z. TOF satellites and random
hits tend to be short vertical fragments or singletons.

Algorithm
---------
For each unique TOF index ``c`` present in the input (points sharing an
mz_index see identical windows, so we evaluate once per unique index):

1. Build a per-scan intensity profile by summing every input point whose
   ``mz_index ∈ [c - mz_idx_half_width, c + mz_idx_half_width]``.
2. Mark each scan as "occupied" if its profile value > 0.
3. Find maximal contiguous runs of occupied scans, merging runs separated
   by ≤ ``max_gap_scans`` empty scans (morphological closing along IM).
4. Mark runs whose total span (gap-inclusive) is ≥ ``min_streak_scans``
   as kept.
5. For every input point at mz_index == c: keep iff its scan number falls
   inside a kept run.

Each point therefore lives at the center of its own evaluation window —
no boundary-tiling, no peak splitting across bins.

Standalone testing tool — not part of the tdfpy package.

Run with::

    streamlit run apps/im_feature_filter_dashboard.py

Requires ``streamlit`` and ``plotly``::

    uv pip install streamlit plotly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

import tdfpy
from tdfpy.noise import IntensityThreshold, coerce_filters
from tdfpy.tdf import PandasTdf


def estimate_noise_level(intensities, method):
    """Local shim — bridges the old string/numeric API to the new
    :class:`tdfpy.noise.IntensityThreshold` subclasses so this standalone
    dashboard keeps working without a full rewrite."""
    (filt,) = coerce_filters(method)
    assert isinstance(filt, IntensityThreshold)
    return float(filt.compute_threshold(np.asarray(intensities)))

from tdfpy.noise import (  # type: ignore[import-not-found]
    VerticalNoiseDiagnostics as FeatureFilterResult,
    VerticalNoiseFilter,
)

# Keep the names referenced as locals so static analyzers see them used.
_ = FeatureFilterResult

NOISE_METHODS = ("mad", "percentile", "histogram", "baseline", "iterative_median")


# ---------------------------------------------------------------------------
# Watershed centroider — intensity-ordered region growing with a rectangular
# nearest-neighbor exclusion box. Walks points top-down: each point attaches
# to whichever group contains its nearest already-assigned point within the
# box, or becomes a new seed if no group is in reach.
# ---------------------------------------------------------------------------


@dataclass
class WatershedCentroidResult:
    centroids: np.ndarray  # (N, 3): [mz, intensity, ion_mobility]
    num_seeds_emitted: int  # final count after min_centroid_intensity filter
    num_seeds_promoted: int  # candidates that cleared min_seed_intensity
    num_followers: int  # points attached to an existing group
    num_orphans_dropped: int  # points with no in-box neighbor AND below min_seed_intensity


def smooth_intensities_box_average(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    *,
    smooth_scan_half_width: int,
    smooth_mz_idx_half_width: int,
) -> np.ndarray:
    """Replace each point's intensity with the mean intensity of every point
    within ``|Δscan| ≤ smooth_scan_half_width AND |Δmz_idx| ≤
    smooth_mz_idx_half_width`` (inclusive of self). Returns a new intensity
    array; positions are untouched.

    Uses the same bucket-grid lookup as the watershed centroider so per-point
    cost is O(k) where k is the number of points in the 3×3 cell
    neighborhood. The point itself is always in its own box, so the divisor
    is at least 1.
    """
    n = scan_indices.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    smooth_scan_half_width = max(1, int(smooth_scan_half_width))
    smooth_mz_idx_half_width = max(1, int(smooth_mz_idx_half_width))
    scan_arr = np.asarray(scan_indices, dtype=np.int64)
    mz_arr = np.asarray(mz_indices, dtype=np.int64)
    int_arr = np.asarray(intensities, dtype=np.float64)

    grid: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        cell = (
            int(scan_arr[i]) // smooth_scan_half_width,
            int(mz_arr[i]) // smooth_mz_idx_half_width,
        )
        grid.setdefault(cell, []).append(i)

    smoothed = np.empty(n, dtype=np.float64)
    for i in range(n):
        p_scan = int(scan_arr[i])
        p_mz = int(mz_arr[i])
        c_scan = p_scan // smooth_scan_half_width
        c_mz = p_mz // smooth_mz_idx_half_width
        total = 0.0
        count = 0
        for ds in (-1, 0, 1):
            for dm in (-1, 0, 1):
                bucket = grid.get((c_scan + ds, c_mz + dm))
                if bucket is None:
                    continue
                for q in bucket:
                    if abs(p_scan - int(scan_arr[q])) > smooth_scan_half_width:
                        continue
                    if abs(p_mz - int(mz_arr[q])) > smooth_mz_idx_half_width:
                        continue
                    total += float(int_arr[q])
                    count += 1
        smoothed[i] = total / count if count > 0 else float(int_arr[i])
    return smoothed


def watershed_centroid(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    mz_values: np.ndarray,
    ook0_values: np.ndarray,
    *,
    attach_scan_half_width: int,
    attach_mz_idx_half_width: int,
    min_seed_intensity: float = 0.0,
    min_centroid_intensity: float = 0.0,
) -> WatershedCentroidResult:
    """Intensity-ordered region growing with a rectangular NN-exclusion box.

    Walk all input points in descending intensity order. For each point,
    inspect the 3x3 spatial-grid neighborhood (cells sized
    ``(attach_scan_half_width, attach_mz_idx_half_width)``) for already-
    assigned points within ``|dscan| <= attach_scan_half_width AND |dmz_idx|
    <= attach_mz_idx_half_width``. The point's fate:

    * **At least one in-box neighbor exists** -> join the nearest neighbor's
      group (Manhattan tiebreak; further ties favor the group whose seed has
      higher intensity, giving a stable watershed boundary).
    * **No in-box neighbor AND intensity >= min_seed_intensity** -> promote
      to a new seed.
    * **No in-box neighbor AND intensity < min_seed_intensity** -> drop as
      an orphan (does not claim grid territory).

    Final centroids: intensity-weighted mean of every group member. Groups
    whose summed intensity falls below ``min_centroid_intensity`` are
    dropped.

    The grid neighborhood search guarantees correctness: with cells sized to
    the box dimensions, every point within the tolerance lives in one of the
    9 surrounding cells.
    """
    n = scan_indices.size
    if n == 0:
        return WatershedCentroidResult(
            centroids=np.empty((0, 3), dtype=np.float64),
            num_seeds_emitted=0,
            num_seeds_promoted=0,
            num_followers=0,
            num_orphans_dropped=0,
        )

    attach_scan_half_width = max(1, int(attach_scan_half_width))
    attach_mz_idx_half_width = max(1, int(attach_mz_idx_half_width))

    scan_arr = np.asarray(scan_indices, dtype=np.int64)
    mz_arr = np.asarray(mz_indices, dtype=np.int64)
    int_arr = np.asarray(intensities, dtype=np.float64)
    mz_val_arr = np.asarray(mz_values, dtype=np.float64)
    ook0_arr = np.asarray(ook0_values, dtype=np.float64)

    # group_id[i] = group index, or -1 if orphan (not assigned).
    group_id = np.full(n, -1, dtype=np.int64)
    # Per-group seed intensity, used as a tiebreak for equidistant neighbors.
    seed_intensities: list[float] = []

    # Spatial bucket grid -- sparse dict keyed by (scan_cell, mz_cell).
    # Cells are sized to the box dimensions so that the 3x3 cell neighborhood
    # is guaranteed to contain every point within the tolerance.
    grid: dict[tuple[int, int], list[int]] = {}

    intensity_order = np.argsort(int_arr, kind="stable")[::-1]
    num_orphans_dropped = 0

    for raw_i in intensity_order:
        i = int(raw_i)
        p_scan = int(scan_arr[i])
        p_mz = int(mz_arr[i])
        p_int = float(int_arr[i])
        c_scan = p_scan // attach_scan_half_width
        c_mz = p_mz // attach_mz_idx_half_width

        # Find the best in-box neighbor across the 9-cell neighborhood.
        best_group = -1
        best_dist = 0
        best_seed_int = -1.0
        for ds in (-1, 0, 1):
            for dm in (-1, 0, 1):
                cell = (c_scan + ds, c_mz + dm)
                bucket = grid.get(cell)
                if bucket is None:
                    continue
                for q in bucket:
                    d_scan = abs(p_scan - int(scan_arr[q]))
                    if d_scan > attach_scan_half_width:
                        continue
                    d_mz = abs(p_mz - int(mz_arr[q]))
                    if d_mz > attach_mz_idx_half_width:
                        continue
                    d = d_scan + d_mz  # Manhattan
                    q_group = int(group_id[q])
                    q_seed_int = seed_intensities[q_group]
                    if best_group < 0 or d < best_dist or (
                        d == best_dist and q_seed_int > best_seed_int
                    ):
                        best_group = q_group
                        best_dist = d
                        best_seed_int = q_seed_int

        if best_group >= 0:
            group_id[i] = best_group
            grid.setdefault((c_scan, c_mz), []).append(i)
        elif p_int >= min_seed_intensity:
            new_group = len(seed_intensities)
            group_id[i] = new_group
            seed_intensities.append(p_int)
            grid.setdefault((c_scan, c_mz), []).append(i)
        else:
            num_orphans_dropped += 1
            # Orphans do NOT enter the grid -- they don't claim territory.

    num_seeds_promoted = len(seed_intensities)
    if num_seeds_promoted == 0:
        return WatershedCentroidResult(
            centroids=np.empty((0, 3), dtype=np.float64),
            num_seeds_emitted=0,
            num_seeds_promoted=0,
            num_followers=0,
            num_orphans_dropped=num_orphans_dropped,
        )

    # Aggregate centroids per group via bincount over (group_id, intensity).
    assigned = group_id >= 0
    g = group_id[assigned]
    w = int_arr[assigned]
    total = np.bincount(g, weights=w, minlength=num_seeds_promoted)
    sum_mz = np.bincount(
        g, weights=mz_val_arr[assigned] * w, minlength=num_seeds_promoted
    )
    sum_ook0 = np.bincount(
        g, weights=ook0_arr[assigned] * w, minlength=num_seeds_promoted
    )
    safe_total = np.where(total > 0, total, 1.0)
    cent_mz = sum_mz / safe_total
    cent_ook0 = sum_ook0 / safe_total

    keep_groups = total >= float(min_centroid_intensity)
    centroids = np.column_stack(
        [cent_mz[keep_groups], total[keep_groups], cent_ook0[keep_groups]]
    )

    num_followers = int(assigned.sum()) - num_seeds_promoted
    return WatershedCentroidResult(
        centroids=centroids,
        num_seeds_emitted=int(keep_groups.sum()),
        num_seeds_promoted=num_seeds_promoted,
        num_followers=num_followers,
        num_orphans_dropped=num_orphans_dropped,
    )


# ---------------------------------------------------------------------------
# Streamlit app
# ---------------------------------------------------------------------------


st.set_page_config(page_title="tdfpy IM-feature filter", layout="wide")
st.title("tdfpy — vertical-IM feature noise filter (testing)")


@dataclass
class FrameRaw:
    scan_indices: np.ndarray
    mz_indices: np.ndarray
    intensities: np.ndarray
    mz_values: np.ndarray
    ook0_values: np.ndarray
    num_scans: int


@st.cache_data(show_spinner=False)
def _list_ms1_frames(analysis_dir: str) -> list[dict]:
    frames = PandasTdf(str(Path(analysis_dir) / "analysis.tdf")).frames
    ms1 = frames[frames["MsMsType"] == 0]
    return [
        {
            "frame_id": int(row["Id"]),
            "rt_min": float(row["Time"]) / 60.0,
            "num_peaks": int(row["NumPeaks"]),
            "num_scans": int(row["NumScans"]),
        }
        for _, row in ms1.iterrows()
    ]


@st.cache_data(show_spinner=True)
def _load_frame_raw(analysis_dir: str, frame_id: int) -> FrameRaw:
    """Load raw frame data in integer scan/TOF-index space + float conversions.

    Unlike ``tdfpy.centroiding.get_raw_peaks``, this preserves the integer
    ``scan_indices`` / ``mz_indices`` arrays — the filter needs them.
    """
    with tdfpy.timsdata_connect(analysis_dir) as td:
        cursor = td.conn.cursor()  # type: ignore[union-attr]
        cursor.execute("SELECT NumScans FROM Frames WHERE Id = ?", (frame_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Frame {frame_id} not found")
        (num_scans,) = row

        scans = td.readScans(frame_id, 0, num_scans)
        scan_lens = np.fromiter(
            (len(idx) for idx, _ in scans), dtype=np.int64, count=num_scans
        )
        if int(scan_lens.sum()) == 0:
            empty = np.empty(0, dtype=np.float64)
            return FrameRaw(
                scan_indices=np.empty(0, dtype=np.int64),
                mz_indices=np.empty(0, dtype=np.int64),
                intensities=empty,
                mz_values=empty,
                ook0_values=empty,
                num_scans=num_scans,
            )

        scan_indices = np.repeat(np.arange(num_scans, dtype=np.int64), scan_lens)
        mz_indices = np.concatenate([idx for idx, _ in scans]).astype(
            np.int64, copy=False
        )
        intensities = np.concatenate([i for _, i in scans]).astype(
            np.float64, copy=False
        )

        ook0_per_scan = np.asarray(
            td.scanNumToOneOverK0(frame_id, np.arange(num_scans))  # type: ignore[call-arg]
        )
        mz_values = np.asarray(td.indexToMz(frame_id, mz_indices))
        ook0_values = ook0_per_scan[scan_indices]

    return FrameRaw(
        scan_indices=scan_indices,
        mz_indices=mz_indices,
        intensities=intensities,
        mz_values=mz_values,
        ook0_values=ook0_values,
        num_scans=num_scans,
    )


@st.cache_data(show_spinner=True)
def _run_filter(
    analysis_dir: str,
    frame_id: int,
    mz_idx_half_width: int,
    min_streak_scans: int,
    max_gap_scans: int,
    min_streak_intensity: float,
    num_iterations: int,
) -> dict:
    """Streamlit-cached wrapper that loads the frame and runs the iterated
    filter. Same return contract as the old standalone ``_im_filter`` module.
    """
    raw = _load_frame_raw(analysis_dir, frame_id)
    diag = VerticalNoiseFilter(
        mz_idx_half_width=mz_idx_half_width,
        min_streak_scans=min_streak_scans,
        max_gap_scans=max_gap_scans,
        min_streak_intensity=min_streak_intensity,
        num_iterations=num_iterations,
    ).run(
        raw.scan_indices,
        raw.mz_indices,
        raw.intensities,
        num_scans=raw.num_scans,
        diagnostics=True,
    )
    return {
        "keep_point_mask": diag.keep_point_mask,
        "per_pass_kept": diag.per_pass_kept,
        "num_columns_evaluated": diag.num_columns_evaluated,
        "num_columns_with_kept_runs": diag.num_columns_with_kept_runs,
        "num_kept_points": diag.num_kept_points,
        "feature_span_intensities": diag.feature_span_intensities,
    }


@st.cache_data(show_spinner=True)
def _fetch_watershed_centroided(
    analysis_dir: str,
    frame_id: int,
    # filter params (cache invalidation when filter changes)
    mz_idx_half_width: int,
    min_streak_scans: int,
    max_gap_scans: int,
    min_streak_intensity: float,
    num_iterations: int,
    # smoothing params
    smooth_enabled: bool,
    smooth_scan_half_width: int,
    smooth_mz_idx_half_width: int,
    # centroider params
    attach_scan_half_width: int,
    attach_mz_idx_half_width: int,
    min_seed_intensity: float,
    min_centroid_intensity: float,
) -> dict:
    """Run the watershed centroider on the points surviving the IM filter.

    Cache key is the full param tuple so changing only centroider settings
    skips the data load + filter and re-runs just watershed_centroid.
    Returns the centroids plus diagnostic counts for the metric strip.
    """
    raw = _load_frame_raw(analysis_dir, frame_id)
    filt = _run_filter(
        analysis_dir,
        frame_id,
        mz_idx_half_width,
        min_streak_scans,
        max_gap_scans,
        min_streak_intensity,
        num_iterations,
    )
    keep = filt["keep_point_mask"]
    if not keep.any():
        return {
            "centroids": np.empty((0, 3), dtype=np.float64),
            "num_seeds_emitted": 0,
            "num_seeds_promoted": 0,
            "num_followers": 0,
            "num_orphans_dropped": 0,
        }
    intensities_for_centroid = raw.intensities[keep]
    if smooth_enabled:
        intensities_for_centroid = smooth_intensities_box_average(
            raw.scan_indices[keep],
            raw.mz_indices[keep],
            intensities_for_centroid,
            smooth_scan_half_width=smooth_scan_half_width,
            smooth_mz_idx_half_width=smooth_mz_idx_half_width,
        )
    result = watershed_centroid(
        scan_indices=raw.scan_indices[keep],
        mz_indices=raw.mz_indices[keep],
        intensities=intensities_for_centroid,
        mz_values=raw.mz_values[keep],
        ook0_values=raw.ook0_values[keep],
        attach_scan_half_width=attach_scan_half_width,
        attach_mz_idx_half_width=attach_mz_idx_half_width,
        min_seed_intensity=min_seed_intensity,
        min_centroid_intensity=min_centroid_intensity,
    )
    return {
        "centroids": result.centroids,
        "num_seeds_emitted": result.num_seeds_emitted,
        "num_seeds_promoted": result.num_seeds_promoted,
        "num_followers": result.num_followers,
        "num_orphans_dropped": result.num_orphans_dropped,
    }


# --- sidebar ---------------------------------------------------------------

with st.sidebar:
    st.header("Data source")
    analysis_dir = st.text_input(
        ".d folder path",
        value=st.session_state.get("analysis_dir", ""),
    )

path_ok = bool(analysis_dir) and Path(analysis_dir).exists() and Path(analysis_dir).is_dir()
if not analysis_dir:
    st.info("Enter a path to a `.d` folder in the sidebar to begin.")
    st.stop()
if not path_ok:
    st.error(f"Path does not exist or is not a directory: {analysis_dir}")
    st.stop()
if not (Path(analysis_dir) / "analysis.tdf").exists():
    st.error(f"`analysis.tdf` not found under {analysis_dir}")
    st.stop()
st.session_state["analysis_dir"] = analysis_dir

ms1_frames = _list_ms1_frames(analysis_dir)
if not ms1_frames:
    st.warning("No MS1 frames in this `.d` folder.")
    st.stop()

with st.sidebar:
    frame_labels = [
        f"Frame {f['frame_id']}  ·  RT {f['rt_min']:.2f} min  ·  {f['num_peaks']:,} peaks"
        for f in ms1_frames
    ]
    frame_choice = st.selectbox(
        "MS1 frame",
        options=range(len(ms1_frames)),
        format_func=lambda i: frame_labels[i],
    )
    frame_meta = ms1_frames[frame_choice]
    frame_id = frame_meta["frame_id"]

    st.header("Filter parameters")
    mz_idx_half_width = int(
        st.number_input(
            "mz_idx_half_width (TOF indices)",
            min_value=0,
            max_value=200,
            value=3,
            step=1,
            help="Half-width of the column window around each point's "
            "mz_index. ±N TOF indices on each side (window total = 2N+1).",
        )
    )
    min_streak_scans = int(
        st.number_input(
            "min_streak_scans",
            min_value=1,
            max_value=int(frame_meta["num_scans"]),
            value=5,
            step=1,
            help="Minimum total span (gap-inclusive) of a vertical streak, in scans.",
        )
    )
    max_gap_scans = int(
        st.number_input(
            "max_gap_scans",
            min_value=0,
            max_value=int(frame_meta["num_scans"]),
            value=1,
            step=1,
            help="Maximum number of consecutive empty scans tolerated inside a streak.",
        )
    )
    min_streak_intensity = float(
        st.number_input(
            "min_streak_intensity",
            min_value=0.0,
            value=50.0,
            step=10.0,
            help="Streak-level intensity floor: total summed intensity over "
            "the entire gap-closed run (column window × span). Applied after "
            "min_streak_scans. 0 disables. Use the diagnostic histogram "
            "below to pick a threshold.",
        )
    )
    num_iterations = int(
        st.number_input(
            "num_iterations",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Run the filter this many times, each pass operating on "
            "the survivors of the previous one. Useful on noisy data — "
            "points adjacent to barely-thick noise streaks get dropped on "
            "later passes when those streaks are gone. Diagnostics below "
            "show the per-pass survival counts.",
        )
    )

    st.header("Display")
    log_intensity = st.checkbox("Log-scale color (log10 intensity)", value=True)
    max_points = int(
        st.number_input(
            "Max points to plot per panel",
            min_value=10_000,
            max_value=2_000_000,
            value=400_000,
            step=50_000,
        )
    )

    st.header("Watershed centroiding")
    centroid_on = st.checkbox(
        "Run watershed centroiding on the filtered points",
        value=False,
        help=(
            "Intensity-ordered region growing with a rectangular NN box. "
            "Each point either joins the group of its nearest already-"
            "assigned neighbor (within the box) or becomes a new seed. "
            "Auto-handles peak splitting because there's no per-peak walk."
        ),
    )
    if centroid_on:
        st.markdown("**Pre-centroid intensity averaging**")
        smooth_enabled = st.checkbox(
            "Average each point's intensity over a box of neighbors",
            value=True,
            help=(
                "For each filtered point, replace its intensity with the "
                "mean intensity of all points within ±box scans and ±box "
                "TOF indices (inclusive of self). Smooths noisy per-pixel "
                "intensities before the centroider sees them, giving more "
                "stable seed ordering and centroid totals."
            ),
        )
        if smooth_enabled:
            smooth_scan_half_width = int(
                st.number_input(
                    "smooth_scan_half_width (scans)",
                    min_value=1,
                    max_value=int(frame_meta["num_scans"]),
                    value=5,
                    step=1,
                    help="±scan window for the averaging box. Independent "
                    "of the centroider's attach_scan_half_width.",
                )
            )
            smooth_mz_idx_half_width = int(
                st.number_input(
                    "smooth_mz_idx_half_width (TOF indices)",
                    min_value=1,
                    max_value=200,
                    value=3,
                    step=1,
                    help="±TOF-index window for the averaging box.",
                )
            )
        else:
            smooth_scan_half_width = 5
            smooth_mz_idx_half_width = 3

        st.markdown("**Watershed parameters**")
        attach_scan_half_width = int(
            st.number_input(
                "attach_scan_half_width (scans)",
                min_value=1,
                max_value=int(frame_meta["num_scans"]),
                value=10,
                step=1,
                help="±scan tolerance for NN attachment and seed exclusion. "
                "A new seed must be more than attach_scan_half_width away in "
                "scan AND more than attach_mz_idx_half_width away in mz from "
                "every assigned point.",
            )
        )
        attach_mz_idx_half_width = int(
            st.number_input(
                "attach_mz_idx_half_width (TOF indices)",
                min_value=1,
                max_value=200,
                value=3,
                step=1,
                help="±TOF-index tolerance for NN attachment and seed "
                "exclusion. Smaller than attach_scan_half_width because TOF "
                "peaks are sharper than IM peaks.",
            )
        )
        min_seed_intensity = float(
            st.number_input(
                "min_seed_intensity",
                min_value=0.0,
                value=0.0,
                step=10.0,
                help="Don't promote orphan points (no in-box neighbor) to "
                "seeds below this raw intensity. They are dropped instead. "
                "0 = every orphan becomes a seed.",
            )
        )
        min_centroid_intensity = float(
            st.number_input(
                "min_centroid_intensity",
                min_value=0.0,
                value=0.0,
                step=10.0,
                help="Drop final centroids whose summed group intensity is "
                "below this floor. 0 disables.",
            )
        )

        centroid_log_y = st.checkbox(
            "Log y-axis (intensity)",
            value=False,
        )

        centroid_noise_mode = st.selectbox(
            "Post-centroid noise filter",
            options=["off", "absolute", *NOISE_METHODS],
            index=0,
            help=(
                "Intensity floor applied to centroid totals. `mad` / "
                "`iterative_median` adapt to the per-frame distribution."
            ),
            key="centroid_noise_mode",
        )
        centroid_noise_filter: float | str | None
        if centroid_noise_mode == "off":
            centroid_noise_filter = None
        elif centroid_noise_mode == "absolute":
            centroid_noise_filter = float(
                st.number_input(
                    "Centroid absolute threshold",
                    min_value=0.0,
                    value=10.0,
                    step=1.0,
                    key="centroid_abs_threshold",
                )
            )
        else:
            centroid_noise_filter = centroid_noise_mode
    else:
        smooth_enabled = True
        smooth_scan_half_width = 5
        smooth_mz_idx_half_width = 3
        attach_scan_half_width = 10
        attach_mz_idx_half_width = 3
        min_seed_intensity = 0.0
        min_centroid_intensity = 0.0
        centroid_log_y = False
        centroid_noise_filter = None
        centroid_noise_mode = "off"


# --- data + filter ---------------------------------------------------------

raw = _load_frame_raw(analysis_dir, frame_id)
if raw.intensities.size == 0:
    st.warning("No peaks in this frame.")
    st.stop()

filter_out = _run_filter(
    analysis_dir,
    frame_id,
    mz_idx_half_width,
    min_streak_scans,
    max_gap_scans,
    min_streak_intensity,
    num_iterations,
)
keep_mask = filter_out["keep_point_mask"]

# Range sliders (always derived from the unfiltered data so we see the full extent).
mz_min, mz_max = float(raw.mz_values.min()), float(raw.mz_values.max())
ook0_min, ook0_max = float(raw.ook0_values.min()), float(raw.ook0_values.max())

with st.sidebar:
    st.header("Ranges")
    mz_range = st.slider(
        "m/z range",
        min_value=float(np.floor(mz_min)),
        max_value=float(np.ceil(mz_max)),
        value=(float(np.floor(mz_min)), float(np.ceil(mz_max))),
    )
    im_range = st.slider(
        "1/K0 range",
        min_value=ook0_min,
        max_value=ook0_max,
        value=(ook0_min, ook0_max),
    )


# --- metrics ---------------------------------------------------------------

num_raw_points = int(raw.intensities.size)
num_kept_points = int(filter_out["num_kept_points"])
reduction_pct = 100.0 * (1.0 - num_kept_points / max(num_raw_points, 1))

col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Frame", frame_id)
col_b.metric("RT (min)", f"{frame_meta['rt_min']:.2f}")
col_c.metric(
    "Raw → kept points",
    f"{num_kept_points:,}",
    delta=f"-{reduction_pct:.1f}% of {num_raw_points:,}",
    delta_color="off",
)
col_d.metric(
    "Columns (with run / total)",
    f"{filter_out['num_columns_with_kept_runs']:,}",
    delta=f"of {filter_out['num_columns_evaluated']:,}",
    delta_color="off",
)
col_e.metric(
    "Half-width / span / gap",
    f"±{mz_idx_half_width} / {min_streak_scans} / {max_gap_scans}",
)

per_pass = filter_out.get("per_pass_kept", [num_raw_points, num_kept_points])
if len(per_pass) > 2:
    # Render as "raw → p1 → p2 → ..." so the attrition per pass is visible.
    arrow_chain = " → ".join(f"{c:,}" for c in per_pass)
    st.caption(
        f"Iterative filter: {num_iterations} passes  ·  {arrow_chain}"
    )


# --- plotting --------------------------------------------------------------


def _scatter(
    mz: np.ndarray,
    ook0: np.ndarray,
    intensity: np.ndarray,
    title: str,
    *,
    point_size: int = 4,
) -> go.Figure:
    """Standard (m/z, 1/K0) scatter colored by intensity."""
    mask = (mz >= mz_range[0]) & (mz <= mz_range[1]) & (ook0 >= im_range[0]) & (ook0 <= im_range[1])
    mz = mz[mask]
    ook0 = ook0[mask]
    intensity = intensity[mask]
    if mz.size > max_points:
        keep = np.argpartition(intensity, -max_points)[-max_points:]
        mz, ook0, intensity = mz[keep], ook0[keep], intensity[keep]
    color = np.log10(intensity + 1.0) if log_intensity else intensity
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=mz,
            y=ook0,
            mode="markers",
            marker=dict(
                size=point_size,
                color=color,
                colorscale="Viridis",
                colorbar=dict(
                    title="log10(intensity+1)" if log_intensity else "intensity",
                ),
                showscale=True,
                opacity=0.7,
            ),
            customdata=np.column_stack([intensity]),
            hovertemplate=(
                "m/z: %{x:.4f}<br>"
                "1/K0: %{y:.4f}<br>"
                "intensity: %{customdata[0]:,.0f}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="m/z",
        yaxis_title="1/K0",
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        template="plotly_white",
    )
    return fig


col_left, col_right = st.columns(2)
with col_left:
    st.subheader(f"Raw — {num_raw_points:,} points")
    st.plotly_chart(
        _scatter(raw.mz_values, raw.ook0_values, raw.intensities, "Raw"),
        use_container_width=True,
    )

with col_right:
    sub = f"Filtered — {num_kept_points:,} points"
    if num_kept_points == 0:
        st.subheader(sub)
        st.info("No points survived the filter.")
    else:
        st.subheader(sub)
        st.plotly_chart(
            _scatter(
                raw.mz_values[keep_mask],
                raw.ook0_values[keep_mask],
                raw.intensities[keep_mask],
                "Filtered",
            ),
            use_container_width=True,
        )

# --- feature span intensity histogram (tuning diagnostic) -----------------

st.subheader("Feature span intensity distribution")
feature_ints = filter_out["feature_span_intensities"]
if feature_ints.size == 0:
    st.info(
        "No features cleared min_streak_scans yet — nothing to threshold "
        "on. Loosen the filter to see a distribution here."
    )
else:
    n_features = int(feature_ints.size)
    n_passing = int((feature_ints >= min_streak_intensity).sum())
    f_min = float(feature_ints.min())
    f_max = float(feature_ints.max())
    f_med = float(np.median(feature_ints))
    f_dyn = f_max / max(f_min, 1e-12)

    fcol1, fcol2, fcol3, fcol4 = st.columns(4)
    fcol1.metric("Features (≥ length)", f"{n_features:,}")
    fcol2.metric(
        "Pass intensity floor",
        f"{n_passing:,}",
        delta=f"of {n_features:,}",
        delta_color="off",
    )
    fcol3.metric("median / min / max", f"{f_med:,.0f} / {f_min:,.0f} / {f_max:,.0f}")
    fcol4.metric("threshold", f"{min_streak_intensity:g}")

    feat_log_x = st.checkbox(
        "Log-spaced bins (feature intensity)",
        value=f_dyn > 100.0,
        key="feat_int_log_x",
    )
    feat_log_y = st.checkbox(
        "Log count axis (feature intensity)",
        value=True,
        key="feat_int_log_y",
    )
    edges = (
        np.logspace(np.log10(max(f_min, 1.0)), np.log10(max(f_max, 1.0)), 81)
        if feat_log_x and f_max > 0
        else np.linspace(f_min, f_max, 81)
    )
    counts, _ = np.histogram(feature_ints, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    feat_hist = go.Figure()
    feat_hist.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker=dict(color="#10b981", line=dict(width=0)),
            hovertemplate=(
                "feature span intensity: %{x:,.0f}<br>"
                "count: %{y:,}<extra></extra>"
            ),
        )
    )
    if min_streak_intensity > 0:
        feat_hist.add_vline(
            x=float(min_streak_intensity),
            line=dict(color="#ef4444", dash="dash"),
            annotation_text=f"min_streak_intensity = {min_streak_intensity:g}",
            annotation_position="top right",
        )
    feat_hist.update_layout(
        xaxis_title="feature span intensity (sum over column × scan span)",
        yaxis_title="count",
        xaxis_type="log" if feat_log_x else "linear",
        yaxis_type="log" if feat_log_y else "linear",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_white",
        bargap=0,
    )
    st.plotly_chart(feat_hist, use_container_width=True)


# --- centroided spectrum --------------------------------------------------

if centroid_on:
    st.subheader("Watershed-centroided spectrum")
    centroid_out = _fetch_watershed_centroided(
        analysis_dir,
        frame_id,
        mz_idx_half_width,
        min_streak_scans,
        max_gap_scans,
        min_streak_intensity,
        num_iterations,
        smooth_enabled,
        smooth_scan_half_width,
        smooth_mz_idx_half_width,
        attach_scan_half_width,
        attach_mz_idx_half_width,
        min_seed_intensity,
        min_centroid_intensity,
    )
    centroided = centroid_out["centroids"]
    num_seeds_emitted = centroid_out["num_seeds_emitted"]
    num_followers = centroid_out["num_followers"]
    num_orphans_dropped = centroid_out["num_orphans_dropped"]

    pre_centroid_count = num_kept_points
    if centroided.size == 0:
        st.warning(
            "Watershed centroiding produced 0 centroids. Try lowering "
            "min_seed_intensity or min_centroid_intensity."
        )
    else:
        pre_noise_centroid_count = int(centroided.shape[0])
        centroid_threshold: float | None
        if centroid_noise_filter is not None:
            centroid_threshold = float(
                estimate_noise_level(
                    centroided[:, 1],
                    method=centroid_noise_filter,  # type: ignore[arg-type]
                )
            )
            centroided = centroided[centroided[:, 1] >= centroid_threshold]
        else:
            centroid_threshold = None

        c_mz = centroided[:, 0]
        c_int = centroided[:, 1]
        c_im = centroided[:, 2]

        # Apply the same display-range clipping as the raw plot.
        c_mask = (
            (c_mz >= mz_range[0])
            & (c_mz <= mz_range[1])
            & (c_im >= im_range[0])
            & (c_im <= im_range[1])
        )
        c_mz, c_int, c_im = c_mz[c_mask], c_int[c_mask], c_im[c_mask]

        if c_mz.size == 0:
            st.info("No centroids in the selected m/z / ion mobility window.")
        else:
            if c_mz.size > max_points:
                keep = np.argpartition(c_int, -max_points)[-max_points:]
                c_mz, c_int, c_im = c_mz[keep], c_int[keep], c_im[keep]
                st.info(
                    f"Downsampled centroids to top {max_points:,} "
                    f"of {c_mask.sum():,} matches."
                )

            reduction_pct = (
                100.0 * (1.0 - centroided.shape[0] / max(pre_centroid_count, 1))
            )
            if centroid_threshold is None:
                centroid_filter_label = "off"
                centroid_filter_delta = None
            else:
                dropped = pre_noise_centroid_count - centroided.shape[0]
                centroid_filter_label = (
                    f"{centroid_noise_mode} (thr={centroid_threshold:,.2f})"
                )
                centroid_filter_delta = (
                    f"-{dropped:,} of {pre_noise_centroid_count:,}"
                )

            col_c1, col_c2, col_c3, col_c4, col_c5, col_c6 = st.columns(6)
            col_c1.metric(
                "Centroids",
                f"{centroided.shape[0]:,}",
                delta=f"-{reduction_pct:.1f}% of {pre_centroid_count:,}",
                delta_color="off",
            )
            smooth_label = (
                f"±{smooth_scan_half_width} / ±{smooth_mz_idx_half_width}"
                if smooth_enabled
                else "off"
            )
            col_c2.metric(
                "attach (scan / mz) · smooth",
                f"±{attach_scan_half_width} / ±{attach_mz_idx_half_width}",
                delta=smooth_label,
                delta_color="off",
            )
            col_c3.metric(
                "followers / orphans",
                f"{num_followers:,} / {num_orphans_dropped:,}",
            )
            col_c4.metric("min_seed_intensity", f"{min_seed_intensity:g}")
            col_c5.metric("min_centroid_intensity", f"{min_centroid_intensity:g}")
            col_c6.metric(
                "centroid noise filter",
                centroid_filter_label,
                delta=centroid_filter_delta,
                delta_color="off",
            )

            # Stick spectrum colored by IM via per-bin Scattergl line traces +
            # a tip-marker trace carrying the continuous colorbar and hover.
            im_lo_v = float(im_range[0])
            im_hi_v = float(im_range[1])
            n_bins = 32
            if im_hi_v > im_lo_v:
                bin_edges = np.linspace(im_lo_v, im_hi_v, n_bins + 1)
                bin_idx = np.clip(
                    np.digitize(c_im, bin_edges) - 1, 0, n_bins - 1
                )
            else:
                bin_idx = np.zeros(c_im.shape, dtype=np.int64)
            sample_t = ((np.arange(n_bins) + 0.5) / n_bins).tolist()
            bin_colors = sample_colorscale("Viridis", sample_t)

            c_fig = go.Figure()
            for b in range(n_bins):
                stem_mask = bin_idx == b
                if not stem_mask.any():
                    continue
                bm = c_mz[stem_mask]
                bi = c_int[stem_mask]
                n = bm.size
                stem_x = np.empty(3 * n, dtype=np.float64)
                stem_y = np.empty(3 * n, dtype=np.float64)
                stem_x[0::3] = bm
                stem_x[1::3] = bm
                stem_x[2::3] = np.nan
                stem_y[0::3] = 0.0
                stem_y[1::3] = bi
                stem_y[2::3] = np.nan
                c_fig.add_trace(
                    go.Scattergl(
                        x=stem_x,
                        y=stem_y,
                        mode="lines",
                        line=dict(color=bin_colors[b], width=1.2),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
            c_fig.add_trace(
                go.Scattergl(
                    x=c_mz,
                    y=c_int,
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=c_im,
                        colorscale="Viridis",
                        cmin=im_lo_v,
                        cmax=im_hi_v,
                        colorbar=dict(title="1/K0"),
                        showscale=True,
                        line=dict(width=0),
                    ),
                    customdata=np.column_stack([c_im]),
                    hovertemplate=(
                        "m/z: %{x:.4f}<br>"
                        "intensity: %{y:,.0f}<br>"
                        "1/K0: %{customdata[0]:.4f}<extra></extra>"
                    ),
                    showlegend=False,
                )
            )
            c_fig.update_layout(
                xaxis_title="m/z",
                yaxis_title="intensity",
                xaxis_range=[mz_range[0], mz_range[1]],
                yaxis_type="log" if centroid_log_y else "linear",
                height=500,
                margin=dict(l=40, r=20, t=30, b=40),
                template="plotly_white",
            )
            st.plotly_chart(c_fig, use_container_width=True)


# --- intensity histogram (before vs after) --------------------------------

st.subheader("Intensity distribution — raw vs kept")
positive_raw = raw.intensities[raw.intensities > 0]
positive_kept = raw.intensities[keep_mask][raw.intensities[keep_mask] > 0]
if positive_raw.size == 0:
    st.info("No positive intensities to histogram.")
else:
    i_min = float(positive_raw.min())
    i_max = float(positive_raw.max())
    dynamic_range = i_max / max(i_min, 1e-12)
    log_x = st.checkbox(
        "Log-spaced bins (x)",
        value=dynamic_range > 100.0,
    )
    log_y = st.checkbox("Log count axis (y)", value=True)
    nbins = 80
    edges = (
        np.logspace(np.log10(i_min), np.log10(i_max), nbins + 1)
        if log_x
        else np.linspace(i_min, i_max, nbins + 1)
    )
    raw_counts, _ = np.histogram(positive_raw, bins=edges)
    kept_counts, _ = (
        np.histogram(positive_kept, bins=edges)
        if positive_kept.size > 0
        else (np.zeros_like(raw_counts), None)
    )
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    hist = go.Figure()
    hist.add_trace(
        go.Bar(
            x=centers,
            y=raw_counts,
            width=widths,
            name="raw",
            marker=dict(color="#cbd5e1", line=dict(width=0)),
        )
    )
    hist.add_trace(
        go.Bar(
            x=centers,
            y=kept_counts,
            width=widths,
            name="kept",
            marker=dict(color="#3b82f6", line=dict(width=0)),
        )
    )
    hist.update_layout(
        barmode="overlay",
        xaxis_title="intensity",
        yaxis_title="count",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        height=320,
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_white",
        bargap=0,
    )
    st.plotly_chart(hist, use_container_width=True)
