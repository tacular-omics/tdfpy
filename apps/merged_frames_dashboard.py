"""Streamlit dashboard for inspecting an MS1 frame with N frames on either
side summed in.

For each point ``(scan_index, mz_index)`` the dashboard sums intensities
across the ``2N + 1`` frames centered on the chosen frame. This lets you
see:

* whether peak positions drift in ``(scan, TOF_index)`` between adjacent
  frames (real signal should align tightly; misalignment shows up as
  twinned or smeared blobs in the merged view);
* what happens to noise — random noise should not coherently sum across
  frames the way real ions do, so the merged spectrum should have a
  higher SNR than any single frame.

Run with::

    streamlit run apps/merged_frames_dashboard.py

Requires ``streamlit`` and ``plotly``::

    uv pip install streamlit plotly
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

import tdfpy
from tdfpy import (
    ChargeStateRegion,
    MergePeaksCentroider,
    VerticalNoiseFilter,
    WatershedCentroider,
    get_acquisition_type,
)
from tdfpy.pipeline import Centroider, RawSpectrum, exclude_region, read_spectrum
from tdfpy.tdf import PandasTdf

st.set_page_config(page_title="tdfpy merged MS1 frames", layout="wide")
st.title("tdfpy — merged MS1 frames")
st.caption(
    "Sum intensities at identical (scan, TOF_index) over a ±N window of MS1 "
    "frames. Compare alignment, watch what coherent signal does vs. noise."
)


# ============================================================================
# Cached data accessors
# ============================================================================


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


@st.cache_data(show_spinner=False)
def _fetch_raw_spectrum_arrays(
    analysis_dir: str,
    frame_id: int,
    exclude: ChargeStateRegion | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read one frame's raw spectrum.

    Returns ``(scan_indices, mz_indices, intensities, num_scans)`` rather than
    a ``RawSpectrum`` because Streamlit's cache hashes by value and the
    dataclass-of-ndarrays is awkward there.
    """
    with tdfpy.timsdata_connect(analysis_dir) as td:
        spectrum = read_spectrum(td, frame_id)
        if exclude is not None:
            spectrum = exclude_region(spectrum, exclude, td=td, frame_id=frame_id)
        return (
            spectrum.scan_indices,
            spectrum.mz_indices,
            spectrum.intensities,
            spectrum.num_scans,
        )


@st.cache_data(show_spinner=False)
def _calibration_arrays(
    analysis_dir: str, frame_id: int, num_scans: int, max_mz_idx: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-scan 1/K0 and per-TOF-index m/z for the centre frame.

    Used only for the optional physical-units view.
    """
    with tdfpy.timsdata_connect(analysis_dir) as td:
        ook0 = np.asarray(td.scanNumToOneOverK0(frame_id, np.arange(num_scans)))
        mz_lookup = np.asarray(td.indexToMz(frame_id, np.arange(max_mz_idx + 1)))
    return ook0, mz_lookup


def _merge_frames(
    frames: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sum intensities at identical ``(scan_index, mz_index)`` across frames.

    Returns ``(scan_merged, mz_merged, intensity_summed, contributors)``
    where ``contributors`` is the number of frames contributing to each
    merged point.
    """
    if not frames:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    scans = np.concatenate([s for s, _, _ in frames]).astype(np.int64, copy=False)
    mzs = np.concatenate([m for _, m, _ in frames]).astype(np.int64, copy=False)
    ints = np.concatenate([i for _, _, i in frames]).astype(np.float64, copy=False)

    if scans.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )

    # Pack (scan, mz_idx) into a single int64 key. scan range << 32 bits;
    # mz_idx fits comfortably in the low 32 bits for any timsTOF run.
    if mzs.max() >= (1 << 32) or scans.max() >= (1 << 31):
        raise RuntimeError("scan or mz_index exceeds packed-key range")
    key = (scans << 32) | mzs

    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    int_sorted = ints[order]

    # Unique keys + groups (slices of identical keys)
    unique_keys, first_idx = np.unique(key_sorted, return_index=True)
    group_sums = np.add.reduceat(int_sorted, first_idx)

    # Contributors per merged point — same reduceat trick with ones
    ones = np.ones_like(int_sorted)
    contributors = np.add.reduceat(ones, first_idx).astype(np.int64)

    scan_merged = (unique_keys >> 32).astype(np.int64)
    mz_merged = (unique_keys & 0xFFFFFFFF).astype(np.int64)
    return scan_merged, mz_merged, group_sums, contributors


# ============================================================================
# Sidebar — path, frame picker, window size
# ============================================================================


with st.sidebar:
    st.header("Data source")
    analysis_dir = st.text_input(
        ".d folder path",
        value=st.session_state.get("analysis_dir", ""),
        help="Absolute path to a Bruker `.d` analysis folder on this machine.",
    )

path_ok = (
    bool(analysis_dir)
    and Path(analysis_dir).exists()
    and Path(analysis_dir).is_dir()
)
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

try:
    acquisition = get_acquisition_type(analysis_dir)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to read acquisition metadata: {exc}")
    st.stop()

ms1_frames = _list_ms1_frames(analysis_dir)
if not ms1_frames:
    st.warning("No MS1 frames found in this `.d` folder.")
    st.stop()


with st.sidebar:
    st.markdown(f"**Acquisition:** `{acquisition}`")
    st.markdown(f"**MS1 frames:** {len(ms1_frames)}")
    frame_labels = [
        f"Frame {f['frame_id']}  ·  RT {f['rt_min']:.2f} min  ·  {f['num_peaks']:,} peaks"
        for f in ms1_frames
    ]
    centre_choice = st.selectbox(
        "Centre MS1 frame",
        options=range(len(ms1_frames)),
        format_func=lambda i: frame_labels[i],
    )
    centre_meta = ms1_frames[centre_choice]
    centre_frame_id = centre_meta["frame_id"]

    n_neighbors = int(
        st.number_input(
            "N frames each side",
            min_value=0, max_value=50, value=2, step=1,
            help="Window is 2N+1 frames centered on the chosen frame.",
        )
    )

    st.header("Display")
    space = st.radio(
        "Coordinate space",
        options=["Integer (scan, TOF index)", "Physical (m/z, 1/K0)"],
        index=0,
        help=(
            "Integer space shows native (scan, TOF_index) — frame-to-frame "
            "shifts are visible here. Physical space converts via the centre "
            "frame's calibration."
        ),
    )

    view_mode = st.radio(
        "View",
        options=["Merged (summed)", "Overlay (per-frame colored)"],
        index=0,
        help=(
            "Merged: a single point per (scan, TOF) location with summed "
            "intensity. Overlay: every frame's points drawn separately, "
            "colored by frame offset — best for spotting drift."
        ),
    )

    max_points = int(
        st.number_input(
            "Max points to plot",
            min_value=10_000, max_value=5_000_000,
            value=100_000, step=10_000,
            help="If exceeded, top-N by intensity are kept.",
        )
    )

    log_intensity = st.checkbox("Log-scale color (log10 intensity)", value=True)

    st.header("Merged-site filters")
    st.caption(
        "Filter the merged sites by their summed intensity and by how many "
        "frames contributed. Applies to both views (overlay points are kept "
        "only at sites that pass)."
    )
    min_summed_intensity = float(
        st.number_input(
            "Min summed intensity",
            min_value=0.0, value=0.0, step=10.0, format="%.3f",
            help="Drop merged sites whose Σ intensity is below this value.",
        )
    )
    window_size = 2 * n_neighbors + 1
    min_contributors = int(
        st.slider(
            "Min contributing frames",
            min_value=1, max_value=window_size, value=1, step=1,
            help=(
                "Drop merged sites observed in fewer than this many frames. "
                "Bump to filter out single-frame noise; bump to "
                f"{window_size} to require coherence across the whole window."
            ),
        )
    )

    st.header("Post-merge vertical filter")
    vim_on = st.checkbox(
        "VerticalNoiseFilter (structural)",
        value=False,
        help=(
            "Content-aware: keeps points belonging to long vertical streaks "
            "in (scan, TOF-index) space. Applied AFTER merging — so streaks "
            "of summed intensity, not per-frame streaks."
        ),
    )
    vim_filter: VerticalNoiseFilter | None = None
    if vim_on:
        with st.expander("VerticalNoiseFilter knobs", expanded=True):
            vim_mz_idx_half_width = int(st.number_input(
                "mz_idx_half_width (TOF indices)",
                min_value=0, max_value=20, value=3, step=1,
            ))
            vim_min_streak_scans = int(st.number_input(
                "min_streak_scans",
                min_value=1, max_value=100, value=5, step=1,
            ))
            vim_max_gap_scans = int(st.number_input(
                "max_gap_scans",
                min_value=0, max_value=20, value=1, step=1,
            ))
            vim_min_streak_intensity = float(st.number_input(
                "min_streak_intensity",
                min_value=0.0, value=50.0, step=10.0, format="%.3f",
            ))
            vim_num_iterations = int(st.number_input(
                "num_iterations",
                min_value=1, max_value=10, value=2, step=1,
                help="Re-apply the filter to its own survivors.",
            ))
        vim_filter = VerticalNoiseFilter(
            mz_idx_half_width=vim_mz_idx_half_width,
            min_streak_scans=vim_min_streak_scans,
            max_gap_scans=vim_max_gap_scans,
            min_streak_intensity=vim_min_streak_intensity,
            num_iterations=vim_num_iterations,
        )

    st.header("Post-merge centroiding")
    centroid_on = st.checkbox(
        "Run centroider on merged spectrum",
        value=False,
        help=(
            "Centroid the merged + (optionally) vim-filtered spectrum. Uses "
            "the centre frame's calibration for m/z and 1/K0 conversion."
        ),
    )
    centroider: Centroider | None = None
    centroid_log_y = False
    if centroid_on:
        algo = st.radio(
            "Algorithm",
            options=["merge_peaks", "watershed"],
            horizontal=True,
            help=(
                "merge_peaks: greedy tolerance-based merge in float m/z. "
                "watershed: intensity-ordered region growing in integer "
                "(scan, TOF-index) space."
            ),
        )
        if algo == "merge_peaks":
            with st.expander("MergePeaksCentroider knobs", expanded=True):
                col_mz_tol, col_mz_unit = st.columns([2, 1])
                with col_mz_tol:
                    mp_mz_tol = float(st.number_input(
                        "m/z tolerance",
                        min_value=0.0, value=8.0, step=1.0, format="%.4f",
                    ))
                with col_mz_unit:
                    mp_mz_unit = st.selectbox(
                        "unit", options=["ppm", "da"], index=0, key="mp_mz_unit",
                    )
                col_im_tol, col_im_unit = st.columns([2, 1])
                with col_im_tol:
                    mp_im_tol = float(st.number_input(
                        "IM tolerance",
                        min_value=0.0, value=0.01, step=0.005, format="%.4f",
                    ))
                with col_im_unit:
                    mp_im_unit = st.selectbox(
                        "unit", options=["relative", "absolute"],
                        index=1, key="mp_im_unit",
                    )
                mp_min_peaks = int(st.number_input(
                    "min_peaks", min_value=0, max_value=50, value=1, step=1,
                    help="0 or 1 keeps all clusters.",
                ))
            centroider = MergePeaksCentroider(
                mz_tolerance=mp_mz_tol,
                mz_tolerance_type=mp_mz_unit,  # type: ignore[arg-type]
                im_tolerance=mp_im_tol,
                im_tolerance_type=mp_im_unit,  # type: ignore[arg-type]
                min_peaks=mp_min_peaks,
            )
        else:
            with st.expander("WatershedCentroider knobs", expanded=True):
                ws_attach_scan = int(st.number_input(
                    "attach_scan_half_width (scans)",
                    min_value=1, max_value=200, value=10, step=1,
                ))
                ws_attach_mz_idx = int(st.number_input(
                    "attach_mz_idx_half_width (TOF indices)",
                    min_value=1, max_value=200, value=3, step=1,
                ))
                ws_min_seed = float(st.number_input(
                    "min_seed_intensity",
                    min_value=0.0, value=0.0, step=10.0,
                ))
                ws_min_centroid = float(st.number_input(
                    "min_centroid_intensity",
                    min_value=0.0, value=0.0, step=10.0,
                ))
                st.caption("Pre-centroid box smoothing (0 = off)")
                col_bs1, col_bs2 = st.columns(2)
                with col_bs1:
                    ws_smooth_scan = int(st.number_input(
                        "smooth_scan_half_width",
                        min_value=0, max_value=50, value=5, step=1,
                    ))
                with col_bs2:
                    ws_smooth_mz = int(st.number_input(
                        "smooth_mz_idx_half_width",
                        min_value=0, max_value=50, value=3, step=1,
                    ))
                st.caption("Per-group leash from seed (0 = no limit)")
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    ws_leash_scan_raw = int(st.number_input(
                        "max_scan_from_seed",
                        min_value=0, max_value=1000, value=0, step=1,
                    ))
                with col_l2:
                    ws_leash_mz_raw = int(st.number_input(
                        "max_mz_idx_from_seed",
                        min_value=0, max_value=1000, value=10, step=1,
                    ))
                ws_leash_scan = ws_leash_scan_raw if ws_leash_scan_raw > 0 else None
                ws_leash_mz = ws_leash_mz_raw if ws_leash_mz_raw > 0 else None
            centroider = WatershedCentroider(
                attach_scan_half_width=ws_attach_scan,
                attach_mz_idx_half_width=ws_attach_mz_idx,
                min_seed_intensity=ws_min_seed,
                min_centroid_intensity=ws_min_centroid,
                smooth_scan_half_width=ws_smooth_scan,
                smooth_mz_idx_half_width=ws_smooth_mz,
                max_scan_from_seed=ws_leash_scan,
                max_mz_idx_from_seed=ws_leash_mz,
            )
        centroid_log_y = st.checkbox(
            "Log y-axis (centroid intensity)", value=False,
        )


# ============================================================================
# Region exclusion (applied per-frame, in integer-index space)
# ============================================================================


with st.sidebar:
    st.header("Region exclusion")
    exclude_on = st.checkbox(
        "Drop singly-charged region (m/z, 1/K0 line)",
        value=False,
        help="Applied per-frame before merging — in TOF-index space.",
    )
    exclude: ChargeStateRegion | None
    if exclude_on:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            mz_lo = float(st.number_input("m/z₁", value=350.0, step=10.0))
            ook0_lo = float(
                st.number_input("1/K0₁", value=0.7, step=0.05, format="%.3f")
            )
        with col_p2:
            mz_hi = float(st.number_input("m/z₂", value=1200.0, step=10.0))
            ook0_hi = float(
                st.number_input("1/K0₂", value=1.4, step=0.05, format="%.3f")
            )
        cap_at_upper = st.checkbox(
            "Cap at upper endpoint", value=True,
            help="Also drops anything above the line's higher 1/K0 endpoint.",
        )
        exclude = ChargeStateRegion(
            line=((mz_lo, ook0_lo), (mz_hi, ook0_hi)),
            cap_at_upper_endpoint=cap_at_upper,
        )
    else:
        exclude = None


# ============================================================================
# Resolve the window of frame IDs centered on the chosen frame
# ============================================================================


window_frame_ids: list[int] = []
window_offsets: list[int] = []
for offset in range(-n_neighbors, n_neighbors + 1):
    idx = centre_choice + offset
    if 0 <= idx < len(ms1_frames):
        window_frame_ids.append(ms1_frames[idx]["frame_id"])
        window_offsets.append(offset)


# ============================================================================
# Fetch per-frame raw spectra
# ============================================================================


per_frame: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
per_frame_stats: list[dict] = []
num_scans_centre = centre_meta["num_scans"]
mz_idx_max = 0

with st.spinner(f"Reading {len(window_frame_ids)} MS1 frames…"):
    for fid in window_frame_ids:
        scan, mz, intensities, num_scans = _fetch_raw_spectrum_arrays(
            analysis_dir, fid, exclude
        )
        per_frame.append((scan, mz, intensities))
        per_frame_stats.append(
            {
                "frame_id": fid,
                "n_points": int(scan.size),
                "intensity_sum": float(intensities.sum()) if scan.size else 0.0,
                "num_scans": num_scans,
            }
        )
        if scan.size:
            mz_idx_max = max(mz_idx_max, int(mz.max()))


# ============================================================================
# Merge
# ============================================================================


scan_m_all, mz_m_all, int_m_all, contrib_m_all = _merge_frames(per_frame)
total_in = sum(s["n_points"] for s in per_frame_stats)
n_merged_all = scan_m_all.size

# --- Stage 1: simple merged-site filter (intensity + contributors) ----------
site_mask = (int_m_all >= min_summed_intensity) & (contrib_m_all >= min_contributors)
scan_m = scan_m_all[site_mask]
mz_m = mz_m_all[site_mask]
int_m = int_m_all[site_mask]
contrib_m = contrib_m_all[site_mask]
n_after_simple_filter = scan_m.size

# --- Stage 2: optional VerticalNoiseFilter on the merged spectrum -----------
n_after_vim: int | None = None
if vim_filter is not None and scan_m.size:
    merged_for_vim = RawSpectrum(
        scan_indices=scan_m.astype(np.int64, copy=False),
        mz_indices=mz_m.astype(np.int64, copy=False),
        intensities=int_m.astype(np.float64, copy=False),
        num_scans=int(num_scans_centre),
    )
    with tdfpy.timsdata_connect(analysis_dir) as td:
        vim_mask = vim_filter.keep_mask(
            merged_for_vim.scan_indices,
            merged_for_vim.mz_indices,
            merged_for_vim.intensities,
            num_scans=merged_for_vim.num_scans,
            td=td,
            frame_id=centre_frame_id,
        )
    scan_m = scan_m[vim_mask]
    mz_m = mz_m[vim_mask]
    int_m = int_m[vim_mask]
    contrib_m = contrib_m[vim_mask]
    n_after_vim = scan_m.size

n_merged = scan_m.size
n_filtered_out = n_merged_all - n_merged
reduction = (1.0 - n_merged / max(total_in, 1)) * 100.0

# Set of (scan, mz_idx) sites surviving all post-merge filters — used to
# restrict the overlay view to the same sites.
passing_keys = (scan_m.astype(np.int64) << 32) | mz_m.astype(np.int64)
passing_keys_sorted = np.sort(passing_keys)

filter_active = (
    min_summed_intensity > 0 or min_contributors > 1 or vim_filter is not None
)


# ============================================================================
# Top-row metrics
# ============================================================================


col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Centre frame", centre_frame_id, delta=f"RT {centre_meta['rt_min']:.2f} min")
col_b.metric(
    "Window",
    f"{len(window_frame_ids)} frames",
    delta=f"±{n_neighbors} (offsets {window_offsets[0]} … {window_offsets[-1]})",
    delta_color="off",
)
col_c.metric(
    "Raw points (window)",
    f"{total_in:,}",
    delta=f"{total_in / max(len(window_frame_ids),1):,.0f} avg/frame",
    delta_color="off",
)
if filter_active:
    delta_bits = []
    if vim_filter is not None and n_after_vim is not None:
        delta_bits.append(f"vim: -{n_after_simple_filter - n_after_vim:,}")
    if min_summed_intensity > 0 or min_contributors > 1:
        delta_bits.append(f"simple: -{n_merged_all - n_after_simple_filter:,}")
    col_d.metric(
        "Sites (after filters)",
        f"{n_merged:,}",
        delta=" · ".join(delta_bits) if delta_bits else None,
        delta_color="off",
    )
else:
    col_d.metric(
        "Sites (after merge)",
        f"{n_merged:,}",
        delta=f"-{reduction:.1f}% vs raw",
        delta_color="off",
    )


# ============================================================================
# Convert to physical units if requested
# ============================================================================


use_physical = space.startswith("Physical")
if use_physical:
    if mz_idx_max == 0:
        st.warning("No peaks in the window after region exclusion.")
        st.stop()
    ook0_per_scan, mz_per_idx = _calibration_arrays(
        analysis_dir, centre_frame_id, num_scans_centre, mz_idx_max
    )


def _to_display(
    scan_arr: np.ndarray, mz_idx_arr: np.ndarray
) -> tuple[np.ndarray, np.ndarray, str, str]:
    if use_physical:
        x = mz_per_idx[mz_idx_arr]
        y = ook0_per_scan[scan_arr]
        return x, y, "m/z", "1/K0"
    return mz_idx_arr.astype(np.float64), scan_arr.astype(np.float64), "TOF index", "scan #"


# ============================================================================
# Main plot — branches on view_mode
# ============================================================================


if n_merged == 0:
    st.warning("No peaks survive the current region exclusion across the window.")
    st.stop()


if view_mode.startswith("Merged"):
    # --- Single sum view ---------------------------------------------------
    x_all, y_all, xlabel, ylabel = _to_display(scan_m, mz_m)

    if n_merged > max_points:
        keep = np.argpartition(int_m, -max_points)[-max_points:]
        x_plot, y_plot = x_all[keep], y_all[keep]
        int_plot, contrib_plot = int_m[keep], contrib_m[keep]
        st.info(
            f"Downsampled merged view to top {max_points:,} of {n_merged:,} "
            "unique sites (by summed intensity)."
        )
    else:
        x_plot, y_plot, int_plot, contrib_plot = x_all, y_all, int_m, contrib_m

    color = np.log10(int_plot + 1.0) if log_intensity else int_plot

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=x_plot, y=y_plot, mode="markers",
            marker=dict(
                size=4, color=color, colorscale="Viridis",
                colorbar=dict(
                    title=(
                        "log10(Σ intensity + 1)"
                        if log_intensity
                        else "Σ intensity"
                    )
                ),
                showscale=True, opacity=0.75,
            ),
            customdata=np.column_stack([int_plot, contrib_plot]),
            hovertemplate=(
                f"{xlabel}: " + "%{x:.4f}<br>"
                f"{ylabel}: " + "%{y:.4f}<br>"
                "Σ intensity: %{customdata[0]:,.0f}<br>"
                "contributors: %{customdata[1]} / "
                + str(len(window_frame_ids))
                + "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        xaxis_title=xlabel, yaxis_title=ylabel,
        height=700,
        margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Contributor histogram — how many frames a typical site is seen in
    st.subheader("Coverage — frames contributing per merged site")
    cov_counts = np.bincount(contrib_m, minlength=len(window_frame_ids) + 1)
    cov_fig = go.Figure(
        data=[
            go.Bar(
                x=np.arange(len(cov_counts)),
                y=cov_counts,
                marker=dict(color="#3b82f6"),
                hovertemplate=(
                    "%{x} contributing frames<br>%{y:,} sites<extra></extra>"
                ),
            )
        ]
    )
    cov_fig.update_layout(
        xaxis_title="contributing frames",
        yaxis_title="sites",
        yaxis_type="log",
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_white",
        bargap=0.1,
    )
    st.plotly_chart(cov_fig, use_container_width=True)
    st.caption(
        "A site at coverage = 1 was only seen in one frame (noise or "
        "drifting ion). Coverage near the window size = "
        f"{len(window_frame_ids)} indicates a stable, coherent ion."
    )

else:
    # --- Overlay view ------------------------------------------------------
    st.subheader("Overlay — each frame's raw points")

    # Cap total plotted points across frames by sorting by intensity globally.
    all_scan = np.concatenate([s for s, _, _ in per_frame])
    all_mz = np.concatenate([m for _, m, _ in per_frame])
    all_int = np.concatenate([i for _, _, i in per_frame])
    all_offsets = np.concatenate(
        [
            np.full(len(s), off, dtype=np.int64)
            for (s, _, _), off in zip(per_frame, window_offsets)
        ]
    )

    if filter_active and passing_keys_sorted.size and all_scan.size:
        # Restrict overlay to points sitting at a (scan, mz_idx) that passed
        # the merged-site filter. ``searchsorted`` membership test against
        # the sorted passing-keys array.
        point_keys = (all_scan.astype(np.int64) << 32) | all_mz.astype(np.int64)
        idx = np.searchsorted(passing_keys_sorted, point_keys)
        hits = (
            (idx < passing_keys_sorted.size)
            & (passing_keys_sorted[np.clip(idx, 0, passing_keys_sorted.size - 1)] == point_keys)
        )
        n_before = all_scan.size
        all_scan = all_scan[hits]
        all_mz = all_mz[hits]
        all_int = all_int[hits]
        all_offsets = all_offsets[hits]
        st.caption(
            f"Filter applied: kept {all_scan.size:,} of {n_before:,} raw "
            "points (those sitting at a passing merged site)."
        )

    if all_scan.size > max_points:
        keep = np.argpartition(all_int, -max_points)[-max_points:]
        all_scan = all_scan[keep]
        all_mz = all_mz[keep]
        all_int = all_int[keep]
        all_offsets = all_offsets[keep]
        st.info(
            f"Downsampled overlay to top {max_points:,} of "
            f"{int(total_in):,} total raw points (by intensity)."
        )

    x_all, y_all, xlabel, ylabel = _to_display(all_scan, all_mz)

    # One trace per offset — gives plotly a discrete legend
    fig = go.Figure()
    colorscale_t = (
        np.linspace(0.0, 1.0, max(2 * n_neighbors + 1, 2)).tolist()
        if n_neighbors >= 1
        else [0.5]
    )
    palette = sample_colorscale("RdBu", colorscale_t)
    for off, color in zip(range(-n_neighbors, n_neighbors + 1), palette):
        m = all_offsets == off
        if not m.any():
            continue
        fig.add_trace(
            go.Scattergl(
                x=x_all[m], y=y_all[m], mode="markers",
                marker=dict(size=3, color=color, opacity=0.55),
                name=f"offset {off:+d}",
                hovertemplate=(
                    f"{xlabel}: " + "%{x:.4f}<br>"
                    f"{ylabel}: " + "%{y:.4f}<br>"
                    "intensity: %{text}<br>"
                    f"offset: {off:+d}<extra></extra>"
                ),
                text=[f"{v:,.0f}" for v in all_int[m]],
            )
        )

    fig.update_layout(
        xaxis_title=xlabel, yaxis_title=ylabel,
        height=700, legend_title="frame offset",
        margin=dict(l=40, r=20, t=30, b=40),
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "If frames are well-aligned, points of different colors will "
        "overlap. Drift shows up as parallel offset clouds."
    )


# ============================================================================
# Centroided view (optional)
# ============================================================================


if centroid_on and centroider is not None and scan_m.size:
    st.subheader(f"Centroided merged spectrum — {type(centroider).__name__}")

    merged_for_centroid = RawSpectrum(
        scan_indices=scan_m.astype(np.int64, copy=False),
        mz_indices=mz_m.astype(np.int64, copy=False),
        intensities=int_m.astype(np.float64, copy=False),
        num_scans=int(num_scans_centre),
    )
    try:
        with tdfpy.timsdata_connect(analysis_dir) as td:
            centroids = centroider(
                merged_for_centroid, td, centre_frame_id,
                ion_mobility_type="ook0",
            )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Centroiding failed: {exc}")
        centroids = np.empty((0, 3), dtype=np.float64)

    if centroids.size == 0:
        st.warning(
            "Centroider produced 0 peaks. Try loosening tolerances or "
            "min_peaks / min_seed_intensity."
        )
    else:
        c_mz = centroids[:, 0]
        c_int = centroids[:, 1]
        c_im = centroids[:, 2]

        if c_mz.size > max_points:
            keep = np.argpartition(c_int, -max_points)[-max_points:]
            c_mz, c_int, c_im = c_mz[keep], c_int[keep], c_im[keep]
            st.info(
                f"Downsampled centroids to top {max_points:,} of "
                f"{centroids.shape[0]:,} (by intensity)."
            )

        reduction_pct = 100.0 * (1.0 - centroids.shape[0] / max(n_merged, 1))
        sig_retained = (
            100.0 * centroids[:, 1].sum() / max(int_m.sum(), 1.0)
        )
        col_c1, col_c2, col_c3 = st.columns(3)
        col_c1.metric(
            "Centroids",
            f"{centroids.shape[0]:,}",
            delta=f"-{reduction_pct:.1f}% of {n_merged:,}",
            delta_color="off",
        )
        col_c2.metric("Algorithm", type(centroider).__name__)
        col_c3.metric("Σ intensity retained", f"{sig_retained:.1f}%")

        cfg_lines = []
        for f in centroider.__dataclass_fields__:  # type: ignore[attr-defined]
            cfg_lines.append(f"{f}={getattr(centroider, f)!r}")
        st.caption("Centroider: " + ",  ".join(cfg_lines))

        # Stick spectrum colored by IM. Same pattern as raw_spectrum_dashboard.
        im_lo_v = float(c_im.min())
        im_hi_v = float(c_im.max())
        n_bins = 32
        if im_hi_v > im_lo_v:
            bin_edges = np.linspace(im_lo_v, im_hi_v, n_bins + 1)
            bin_idx = np.clip(np.digitize(c_im, bin_edges) - 1, 0, n_bins - 1)
        else:
            bin_idx = np.zeros(c_im.shape, dtype=np.int64)
        sample_t = ((np.arange(n_bins) + 0.5) / n_bins).tolist()
        bin_colors = sample_colorscale("Viridis", sample_t)

        c_fig = go.Figure()
        for b in range(n_bins):
            m = bin_idx == b
            if not m.any():
                continue
            bm = c_mz[m]
            bi = c_int[m]
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
                    x=stem_x, y=stem_y, mode="lines",
                    line=dict(color=bin_colors[b], width=1.2),
                    hoverinfo="skip", showlegend=False,
                )
            )
        c_fig.add_trace(
            go.Scattergl(
                x=c_mz, y=c_int, mode="markers",
                marker=dict(
                    size=4, color=c_im, colorscale="Viridis",
                    cmin=im_lo_v, cmax=im_hi_v,
                    colorbar=dict(title="1/K0"),
                    showscale=True, line=dict(width=0),
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
            yaxis_type="log" if centroid_log_y else "linear",
            height=500,
            margin=dict(l=40, r=20, t=30, b=40),
            template="plotly_white",
        )
        st.plotly_chart(c_fig, use_container_width=True)


# ============================================================================
# Intensity distributions — single centre frame vs merged window
# ============================================================================


st.subheader("Intensity distribution — centre frame vs merged window")

centre_intensities = per_frame[
    window_offsets.index(0) if 0 in window_offsets else 0
][2]
positive_centre = centre_intensities[centre_intensities > 0]
positive_merged = int_m[int_m > 0]

if positive_merged.size == 0 or positive_centre.size == 0:
    st.info("Not enough positive intensities to histogram.")
else:
    i_min = float(min(positive_centre.min(), positive_merged.min()))
    i_max = float(max(positive_centre.max(), positive_merged.max()))
    log_x = st.checkbox(
        "Log-spaced bins (x)", value=(i_max / max(i_min, 1e-12) > 100.0)
    )
    log_y = st.checkbox("Log count axis (y)", value=True)
    nbins = 80
    edges = (
        np.logspace(np.log10(i_min), np.log10(i_max), nbins + 1)
        if log_x
        else np.linspace(i_min, i_max, nbins + 1)
    )
    counts_c, _ = np.histogram(positive_centre, bins=edges)
    counts_m, _ = np.histogram(positive_merged, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    hist = go.Figure()
    hist.add_trace(
        go.Bar(
            x=centers, y=counts_c, width=widths,
            marker=dict(color="#94a3b8", line=dict(width=0)),
            name="centre frame only",
            hovertemplate="intensity: %{x:,.1f}<br>count: %{y:,}<extra></extra>",
        )
    )
    hist.add_trace(
        go.Bar(
            x=centers, y=counts_m, width=widths,
            marker=dict(color="#3b82f6", line=dict(width=0)),
            name=f"merged ({len(window_frame_ids)} frames)",
            hovertemplate="intensity: %{x:,.1f}<br>count: %{y:,}<extra></extra>",
        )
    )
    hist.update_layout(
        xaxis_title="intensity",
        yaxis_title="count (per intensity bin)",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        barmode="overlay", bargap=0,
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_white",
    )
    st.plotly_chart(hist, use_container_width=True)
    st.caption(
        "Coherent peaks add linearly across frames → the merged distribution "
        "should shift to higher intensities. Incoherent noise adds in "
        "quadrature → fewer extreme high-intensity counts than coherent "
        "summation would predict."
    )


# ============================================================================
# Per-frame summary table
# ============================================================================


st.subheader("Per-frame summary")

df = pd.DataFrame(
    {
        "offset": window_offsets,
        "frame_id": [s["frame_id"] for s in per_frame_stats],
        "n_points": [s["n_points"] for s in per_frame_stats],
        "intensity_sum": [s["intensity_sum"] for s in per_frame_stats],
    }
)
df["frac_of_total"] = df["intensity_sum"] / max(df["intensity_sum"].sum(), 1.0)
st.dataframe(
    df.style.format(
        {
            "n_points": "{:,}",
            "intensity_sum": "{:,.0f}",
            "frac_of_total": "{:.1%}",
        }
    ),
    use_container_width=True,
)
