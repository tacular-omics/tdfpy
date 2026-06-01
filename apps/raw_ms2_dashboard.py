"""Streamlit dashboard for viewing raw MS2 spectra in (scan, TOF-index) space.

Bypasses tdfpy's centroided MS2 API by reading directly from the bundled C
extension (libtimsdata.so) via ``TimsData.readScans`` — same call works for
MS1 and MS2 frames and returns raw profile-like (tof_idx, intensity) per IM
scan.

Run with::

    streamlit run apps/raw_ms2_dashboard.py

Requires ``streamlit`` and ``plotly``::

    uv pip install streamlit plotly
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import tdfpy
from tdfpy.noise import VerticalNoiseFilter
from tdfpy.tdf import PandasTdf


# Bruker MsMsType code → human label. Codes from the Bruker TDF schema docs.
MS_MS_TYPE_LABELS: dict[int, str] = {
    0: "MS1",
    2: "MRM",
    8: "PASEF DDA MS2",
    9: "PASEF DIA MS2",
    10: "PRM",
}


@dataclass
class FrameRaw:
    scan_indices: np.ndarray  # int64
    mz_indices: np.ndarray  # int64 (TOF index)
    intensities: np.ndarray  # float64
    num_scans: int
    # Optional float conversions, computed lazily by helpers below.


@st.cache_data(show_spinner=False)
def _list_ms2_frames(analysis_dir: str) -> list[dict]:
    """Return per-MS2 frame metadata for the frame picker."""
    frames = PandasTdf(str(Path(analysis_dir) / "analysis.tdf")).frames
    ms2 = frames[frames["MsMsType"] != 0]
    return [
        {
            "frame_id": int(row["Id"]),
            "rt_min": float(row["Time"]) / 60.0,
            "ms_ms_type": int(row["MsMsType"]),
            "num_peaks": int(row["NumPeaks"]),
            "num_scans": int(row["NumScans"]),
        }
        for _, row in ms2.iterrows()
    ]


@st.cache_data(show_spinner=True)
def _load_frame_raw(analysis_dir: str, frame_id: int) -> FrameRaw:
    """Read raw (scan, tof_idx, intensity) for a single frame.

    Goes straight to the C extension via ``TimsData.readScans`` — bypasses
    any centroiding the high-level API would normally apply for MS2.
    """
    with tdfpy.timsdata_connect(analysis_dir) as td:
        cursor = td.conn.cursor()  # type: ignore[union-attr]
        cursor.execute("SELECT NumScans FROM Frames WHERE Id = ?", (frame_id,))
        row = cursor.fetchone()
        if row is None:
            raise ValueError(f"Frame {frame_id} not found in database")
        (num_scans,) = row

        scans = td.readScans(frame_id, 0, num_scans)
        scan_lens = np.fromiter(
            (len(idx) for idx, _ in scans), dtype=np.int64, count=num_scans
        )
        if int(scan_lens.sum()) == 0:
            return FrameRaw(
                scan_indices=np.empty(0, dtype=np.int64),
                mz_indices=np.empty(0, dtype=np.int64),
                intensities=np.empty(0, dtype=np.float64),
                num_scans=num_scans,
            )

        scan_indices = np.repeat(np.arange(num_scans, dtype=np.int64), scan_lens)
        mz_indices = np.concatenate([idx for idx, _ in scans]).astype(
            np.int64, copy=False
        )
        intensities = np.concatenate([i for _, i in scans]).astype(
            np.float64, copy=False
        )

    return FrameRaw(
        scan_indices=scan_indices,
        mz_indices=mz_indices,
        intensities=intensities,
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
    """Apply the vertical-noise feature filter to this frame's raw points."""
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


@st.cache_data(show_spinner=False)
def _precursor_segments(analysis_dir: str, frame_id: int) -> list[dict]:
    """Return PASEF precursor isolation segments inside this frame, if any.

    Each entry describes one precursor's IM-scan range, isolation m/z window,
    and (when present) the precursor's monoisotopic m/z and charge. Used to
    draw overlay rectangles on top of the raw scatter so you can see which
    scan band corresponds to which precursor.
    """
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    try:
        pasef = pdf.pasef_frame_msms_info
    except Exception:  # noqa: BLE001
        return []
    pasef_for_frame = pasef[pasef["Frame"] == frame_id]
    if pasef_for_frame.empty:
        return []

    try:
        precursors = pdf.precursors
    except Exception:  # noqa: BLE001
        precursors = None

    segments: list[dict] = []
    for _, row in pasef_for_frame.iterrows():
        precursor_id = int(row["Precursor"])
        entry: dict = {
            "precursor_id": precursor_id,
            "scan_begin": int(row["ScanNumBegin"]),
            "scan_end": int(row["ScanNumEnd"]),
            "isolation_mz": float(row["IsolationMz"]),
            "isolation_width": float(row["IsolationWidth"]),
            "collision_energy": float(row["CollisionEnergy"]),
        }
        if precursors is not None:
            match = precursors[precursors["Id"] == precursor_id]
            if not match.empty:
                p = match.iloc[0]
                entry["precursor_mono_mz"] = (
                    float(p["MonoisotopicMz"])
                    if "MonoisotopicMz" in p and p["MonoisotopicMz"] is not None
                    else None
                )
                entry["precursor_charge"] = (
                    int(p["Charge"])
                    if "Charge" in p and p["Charge"] is not None
                    else None
                )
                entry["precursor_intensity"] = (
                    float(p["Intensity"]) if "Intensity" in p else None
                )
        segments.append(entry)
    return segments


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="tdfpy raw MS2 viewer", layout="wide")
st.title("tdfpy — raw MS2 spectrum viewer (scan × TOF-index)")

with st.sidebar:
    st.header("Data source")
    analysis_dir = st.text_input(
        ".d folder path",
        value=st.session_state.get("ms2_analysis_dir", ""),
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
st.session_state["ms2_analysis_dir"] = analysis_dir

ms2_frames = _list_ms2_frames(analysis_dir)
if not ms2_frames:
    st.warning("No MS2 frames found in this `.d` folder.")
    st.stop()


# Frame type filter — handy when DDA + DIA coexist or for navigating MRM data.
ms_types_present = sorted({f["ms_ms_type"] for f in ms2_frames})
with st.sidebar:
    st.markdown(f"**MS2 frames available:** {len(ms2_frames):,}")
    type_options = [
        f"{t} — {MS_MS_TYPE_LABELS.get(t, 'unknown')}" for t in ms_types_present
    ]
    type_choice_idx = st.selectbox(
        "MS2 type filter",
        options=range(len(type_options)),
        format_func=lambda i: type_options[i],
        index=0,
        help="Restrict the frame picker to one MsMsType code (8=PASEF DDA, 9=PASEF DIA, etc.).",
    )
    selected_type = ms_types_present[type_choice_idx]
    typed_frames = [f for f in ms2_frames if f["ms_ms_type"] == selected_type]

    frame_labels = [
        f"Frame {f['frame_id']}  ·  RT {f['rt_min']:.2f} min  ·  {f['num_peaks']:,} peaks"
        for f in typed_frames
    ]
    frame_choice = st.selectbox(
        "Frame",
        options=range(len(typed_frames)),
        format_func=lambda i: frame_labels[i],
    )
    frame_meta = typed_frames[frame_choice]
    frame_id = frame_meta["frame_id"]

    st.header("Display")
    log_intensity = st.checkbox(
        "Log-scale color (log10 intensity)",
        value=True,
    )
    invert_scan_axis = st.checkbox(
        "Scan 0 at top (image-style y-axis)",
        value=True,
        help="If on, scan numbers increase downward like an image. "
        "If off, scan numbers increase upward (Cartesian).",
    )
    show_precursor_overlay = st.checkbox(
        "Overlay PASEF precursor segments",
        value=True,
        help="Draws dashed horizontal bands showing each precursor's "
        "ScanNumBegin/End in PASEF DDA frames. Hidden for non-PASEF frames.",
    )
    point_size = int(
        st.slider(
            "Marker size",
            min_value=1,
            max_value=8,
            value=3,
        )
    )
    max_points = int(
        st.number_input(
            "Max points to plot",
            min_value=10_000,
            max_value=4_000_000,
            value=800_000,
            step=50_000,
            help="If the frame has more points than this, the top-N highest "
            "intensities are plotted.",
        )
    )

    st.header("Denoise (vertical-IM filter)")
    denoise_on = st.checkbox(
        "Apply vertical-IM feature filter",
        value=False,
        help=(
            "Same algorithm as the IM filter dashboard. Each unique TOF "
            "index is the center of a column window; points whose IM "
            "profile forms a long-enough contiguous run are kept. Useful "
            "for cleaning up MS2 fragment ion clouds before display."
        ),
    )
    if denoise_on:
        ms2_mz_idx_half_width = int(
            st.number_input(
                "mz_idx_half_width (TOF indices)",
                min_value=0,
                max_value=200,
                value=3,
                step=1,
                help="Column half-width: ±N TOF indices around each point.",
            )
        )
        ms2_min_streak_scans = int(
            st.number_input(
                "min_streak_scans",
                min_value=1,
                max_value=int(frame_meta["num_scans"]),
                value=5,
                step=1,
                help="Minimum total IM span (gap-inclusive) of a kept streak.",
            )
        )
        ms2_max_gap_scans = int(
            st.number_input(
                "max_gap_scans",
                min_value=0,
                max_value=int(frame_meta["num_scans"]),
                value=1,
                step=1,
                help="Max consecutive empty scans tolerated inside a streak.",
            )
        )
        ms2_min_streak_intensity = float(
            st.number_input(
                "min_streak_intensity",
                min_value=0.0,
                value=50.0,
                step=10.0,
                help="Total summed intensity required per kept streak.",
            )
        )
        ms2_num_iterations = int(
            st.number_input(
                "num_iterations",
                min_value=1,
                max_value=20,
                value=2,
                step=1,
                help="Re-apply the filter to its own survivors this many times.",
            )
        )
    else:
        ms2_mz_idx_half_width = 3
        ms2_min_streak_scans = 5
        ms2_max_gap_scans = 1
        ms2_min_streak_intensity = 50.0
        ms2_num_iterations = 2


# --- load data --------------------------------------------------------------

raw = _load_frame_raw(analysis_dir, frame_id)
if raw.intensities.size == 0:
    st.warning("This frame has zero raw points.")
    st.stop()

# Optional vertical-IM denoise. The filter result is cached so toggling
# parameters only re-runs the affected stage. When the toggle is off we
# fall through with a trivial all-True mask so the rest of the pipeline
# is unchanged.
if denoise_on:
    filter_out = _run_filter(
        analysis_dir,
        frame_id,
        ms2_mz_idx_half_width,
        ms2_min_streak_scans,
        ms2_max_gap_scans,
        ms2_min_streak_intensity,
        ms2_num_iterations,
    )
    denoise_mask = filter_out["keep_point_mask"]
    per_pass_kept = filter_out["per_pass_kept"]
else:
    denoise_mask = np.ones(raw.intensities.size, dtype=bool)
    per_pass_kept = [int(raw.intensities.size)]

precursor_segments = (
    _precursor_segments(analysis_dir, frame_id) if show_precursor_overlay else []
)

# --- ranges -----------------------------------------------------------------

mz_idx_min = int(raw.mz_indices.min())
mz_idx_max = int(raw.mz_indices.max())
scan_min = 0
scan_max = raw.num_scans - 1

with st.sidebar:
    st.header("Ranges")
    mz_idx_range = st.slider(
        "TOF index range",
        min_value=mz_idx_min,
        max_value=mz_idx_max,
        value=(mz_idx_min, mz_idx_max),
        step=max(1, (mz_idx_max - mz_idx_min) // 1000),
    )
    scan_range = st.slider(
        "Scan range",
        min_value=scan_min,
        max_value=scan_max,
        value=(scan_min, scan_max),
    )


# --- metrics ----------------------------------------------------------------

ms_type_label = MS_MS_TYPE_LABELS.get(frame_meta["ms_ms_type"], "unknown")
num_raw_points = int(raw.intensities.size)
num_kept_points = int(denoise_mask.sum())
col_a, col_b, col_c, col_d, col_e = st.columns(5)
col_a.metric("Frame", frame_id)
col_b.metric("RT (min)", f"{frame_meta['rt_min']:.2f}")
col_c.metric("MsMsType", f"{frame_meta['ms_ms_type']} — {ms_type_label}")
col_d.metric("Num scans", f"{frame_meta['num_scans']:,}")
if denoise_on:
    reduction_pct = 100.0 * (1.0 - num_kept_points / max(num_raw_points, 1))
    col_e.metric(
        "Raw → kept",
        f"{num_kept_points:,}",
        delta=f"-{reduction_pct:.1f}% of {num_raw_points:,}",
        delta_color="off",
    )
else:
    col_e.metric("Raw points", f"{num_raw_points:,}")

if denoise_on and len(per_pass_kept) > 2:
    arrow_chain = " → ".join(f"{c:,}" for c in per_pass_kept)
    st.caption(
        f"Iterative denoise: {ms2_num_iterations} passes  ·  {arrow_chain}"
    )

if precursor_segments:
    st.caption(
        f"PASEF precursors in this frame: {len(precursor_segments)}  ·  "
        + " · ".join(
            f"P{seg['precursor_id']} (scans {seg['scan_begin']}–{seg['scan_end']}, "
            f"iso {seg['isolation_mz']:.2f}±{seg['isolation_width']/2:.2f})"
            for seg in precursor_segments[:5]
        )
        + ("  …" if len(precursor_segments) > 5 else "")
    )


# --- filter + downsample for plotting --------------------------------------

scan_arr = raw.scan_indices[denoise_mask]
mz_idx_arr = raw.mz_indices[denoise_mask]
int_arr = raw.intensities[denoise_mask]

mask = (
    (mz_idx_arr >= mz_idx_range[0])
    & (mz_idx_arr <= mz_idx_range[1])
    & (scan_arr >= scan_range[0])
    & (scan_arr <= scan_range[1])
)
scan_arr = scan_arr[mask]
mz_idx_arr = mz_idx_arr[mask]
int_arr = int_arr[mask]

if scan_arr.size == 0:
    st.warning("No points fall inside the selected range.")
    st.stop()

if scan_arr.size > max_points:
    keep = np.argpartition(int_arr, -max_points)[-max_points:]
    scan_arr = scan_arr[keep]
    mz_idx_arr = mz_idx_arr[keep]
    int_arr = int_arr[keep]
    st.info(
        f"Downsampled to top {max_points:,} most intense of "
        f"{int(mask.sum()):,} points in range."
    )

color = np.log10(int_arr + 1.0) if log_intensity else int_arr


# --- plot -------------------------------------------------------------------

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=mz_idx_arr,
        y=scan_arr,
        mode="markers",
        marker=dict(
            size=point_size,
            color=color,
            colorscale="Viridis",
            colorbar=dict(
                title="log10(intensity+1)" if log_intensity else "intensity",
            ),
            showscale=True,
            opacity=0.75,
        ),
        customdata=np.column_stack([int_arr]),
        hovertemplate=(
            "TOF index: %{x}<br>"
            "scan: %{y}<br>"
            "intensity: %{customdata[0]:,.0f}<extra></extra>"
        ),
    )
)

# Optional PASEF precursor overlay — dashed band from x_min..x_max at each
# (ScanNumBegin, ScanNumEnd) range, plus a label at the right edge.
if precursor_segments:
    x_lo = float(mz_idx_range[0])
    x_hi = float(mz_idx_range[1])
    shapes = []
    annotations = []
    for seg in precursor_segments:
        if seg["scan_end"] < scan_range[0] or seg["scan_begin"] > scan_range[1]:
            continue
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x_lo,
                x1=x_hi,
                y0=seg["scan_begin"] - 0.5,
                y1=seg["scan_end"] + 0.5,
                line=dict(color="rgba(239,68,68,0.55)", dash="dash", width=1),
                fillcolor="rgba(239,68,68,0.0)",
                layer="above",
            )
        )
        label_parts = [f"P{seg['precursor_id']}"]
        if seg.get("precursor_mono_mz") is not None:
            charge = seg.get("precursor_charge")
            charge_str = f" {charge:+d}" if charge else ""
            label_parts.append(
                f"{seg['precursor_mono_mz']:.3f}{charge_str}"
            )
        else:
            label_parts.append(
                f"iso {seg['isolation_mz']:.2f}±{seg['isolation_width']/2:.2f}"
            )
        annotations.append(
            dict(
                x=x_hi,
                y=(seg["scan_begin"] + seg["scan_end"]) / 2,
                xref="x",
                yref="y",
                text=" ".join(label_parts),
                xanchor="right",
                yanchor="middle",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="rgba(239,68,68,0.55)",
                font=dict(size=10, color="#7f1d1d"),
            )
        )
    fig.update_layout(shapes=shapes, annotations=annotations)

fig.update_layout(
    xaxis_title="TOF index (m/z bucket)",
    yaxis_title="scan number (ion mobility)",
    height=720,
    margin=dict(l=40, r=20, t=30, b=40),
    template="plotly_white",
)
if invert_scan_axis:
    fig.update_yaxes(autorange="reversed")

st.plotly_chart(fig, use_container_width=True)


# --- precursor table (collapsible) -----------------------------------------

if precursor_segments:
    with st.expander(f"Precursor segments in this frame ({len(precursor_segments)})"):
        import pandas as pd

        df = pd.DataFrame(precursor_segments)
        ordered_cols = [
            c
            for c in (
                "precursor_id",
                "scan_begin",
                "scan_end",
                "isolation_mz",
                "isolation_width",
                "collision_energy",
                "precursor_mono_mz",
                "precursor_charge",
                "precursor_intensity",
            )
            if c in df.columns
        ]
        st.dataframe(df[ordered_cols], use_container_width=True)


# --- intensity distribution ------------------------------------------------

st.subheader("Intensity distribution (raw, in-range)")
positive = int_arr[int_arr > 0]
if positive.size == 0:
    st.info("No positive intensities to histogram.")
else:
    i_min = float(positive.min())
    i_max = float(positive.max())
    dynamic_range = i_max / max(i_min, 1e-12)
    auto_log = dynamic_range > 100.0
    log_x = st.checkbox("Log-spaced bins (x)", value=auto_log, key="ms2_hist_log_x")
    log_y = st.checkbox("Log count axis (y)", value=True, key="ms2_hist_log_y")
    edges = (
        np.logspace(np.log10(max(i_min, 1.0)), np.log10(max(i_max, 1.0)), 81)
        if log_x and i_max > 0
        else np.linspace(i_min, i_max, 81)
    )
    counts, _ = np.histogram(positive, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    hist = go.Figure()
    hist.add_trace(
        go.Bar(
            x=centers,
            y=counts,
            width=widths,
            marker=dict(color="#3b82f6", line=dict(width=0)),
            hovertemplate="intensity: %{x:,.1f}<br>count: %{y:,}<extra></extra>",
        )
    )
    hist.update_layout(
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
