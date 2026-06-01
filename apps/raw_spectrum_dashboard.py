"""Streamlit dashboard for browsing raw MS1 spectra from Bruker `.d` folders.

Run with:

    streamlit run apps/raw_spectrum_dashboard.py

Requires `streamlit` and `plotly` (not part of tdfpy's default deps):

    uv pip install streamlit plotly
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

import tdfpy
from tdfpy import (
    AbsoluteThreshold,
    BaselineThreshold,
    ChargeStateRegion,
    HistogramThreshold,
    IterativeMedianThreshold,
    MadThreshold,
    MergePeaksCentroider,
    NoiseFilter,
    PercentileThreshold,
    VerticalNoiseFilter,
    WatershedCentroider,
    get_acquisition_type,
    get_centroided_spectrum,
    get_raw_peaks,
)
from tdfpy.pipeline import Centroider
from tdfpy.tdf import PandasTdf


st.set_page_config(page_title="tdfpy raw MS1 viewer", layout="wide")
st.title("tdfpy — raw MS1 spectrum viewer")


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
def _count_raw_peaks(analysis_dir: str, frame_id: int) -> int:
    with tdfpy.timsdata_connect(analysis_dir) as td:
        return int(get_raw_peaks(td, frame_id).shape[0])


@st.cache_data(show_spinner=True, hash_funcs={NoiseFilter: lambda f: hash(f)})
def _fetch_raw_peaks(
    analysis_dir: str,
    frame_id: int,
    ion_mobility_type: str,
    noise_filters: tuple[NoiseFilter, ...],
    exclude: ChargeStateRegion | None,
) -> np.ndarray:
    with tdfpy.timsdata_connect(analysis_dir) as td:
        return get_raw_peaks(
            td,
            frame_id,
            ion_mobility_type=ion_mobility_type,  # type: ignore[arg-type]
            noise=list(noise_filters) if noise_filters else None,
            exclude=exclude,
        )


@st.cache_data(show_spinner=True, hash_funcs={NoiseFilter: lambda f: hash(f)})
def _fetch_centroided(
    analysis_dir: str,
    frame_id: int,
    ion_mobility_type: str,
    noise_filters: tuple[NoiseFilter, ...],
    exclude: ChargeStateRegion | None,
    centroider: Centroider,
) -> np.ndarray:
    with tdfpy.timsdata_connect(analysis_dir) as td:
        return get_centroided_spectrum(
            td,
            frame_id,
            ion_mobility_type=ion_mobility_type,  # type: ignore[arg-type]
            noise=list(noise_filters) if noise_filters else None,
            exclude=exclude,
            centroid=centroider,
        )


# ============================================================================
# Sidebar UI — assembles all filter configs, then we fetch data
# ============================================================================


with st.sidebar:
    st.header("Data source")
    analysis_dir = st.text_input(
        ".d folder path",
        value=st.session_state.get("analysis_dir", ""),
        help="Absolute path to a Bruker `.d` analysis folder on this machine.",
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

try:
    acquisition = get_acquisition_type(analysis_dir)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to read acquisition metadata: {exc}")
    st.stop()

ms1_frames = _list_ms1_frames(analysis_dir)
if not ms1_frames:
    st.warning("No MS1 frames found in this `.d` folder.")
    st.stop()


# -- Frame picker -----------------------------------------------------------

with st.sidebar:
    st.markdown(f"**Acquisition:** `{acquisition}`")
    st.markdown(f"**MS1 frames:** {len(ms1_frames)}")
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


# -- Display ----------------------------------------------------------------

with st.sidebar:
    st.header("Display")
    ion_mobility_type = st.selectbox(
        "Ion mobility axis",
        options=["ook0", "ccs", "voltage"],
        index=0,
    )
    log_intensity = st.checkbox("Log-scale color (log10 intensity)", value=True)
    max_points = int(
        st.number_input(
            "Max points to plot",
            min_value=10_000,
            max_value=2_000_000,
            value=500_000,
            step=50_000,
            help="If exceeded, top-N highest-intensity peaks are kept.",
        )
    )


# -- Region exclusion (ChargeStateRegion) -----------------------------------

with st.sidebar:
    st.header("Region exclusion")
    exclude_on = st.checkbox(
        "Drop singly-charged region (m/z, 1/K0 line)",
        value=False,
        help=(
            "Two-point line cap. Peaks above the line (typically the "
            "singly-charged region in timsTOF MS1) are removed before "
            "smoothing or m/z conversion — done in TOF-index space, no "
            "per-peak conversion."
        ),
    )
    exclude: ChargeStateRegion | None
    if exclude_on:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            mz_lo = float(st.number_input("m/z₁", value=350.0, step=10.0))
            ook0_lo = float(st.number_input("1/K0₁", value=0.7, step=0.05, format="%.3f"))
        with col_p2:
            mz_hi = float(st.number_input("m/z₂", value=1200.0, step=10.0))
            ook0_hi = float(st.number_input("1/K0₂", value=1.4, step=0.05, format="%.3f"))
        cap_at_upper = st.checkbox(
            "Cap at upper endpoint", value=True,
            help="When on, also drops anything above the line's higher 1/K0 endpoint.",
        )
        exclude = ChargeStateRegion(
            line=((mz_lo, ook0_lo), (mz_hi, ook0_hi)),
            cap_at_upper_endpoint=cap_at_upper,
        )
    else:
        exclude = None


# -- Noise filter pipeline --------------------------------------------------

with st.sidebar:
    st.header("Noise filters")

    # 1) Vertical-IM structural filter
    vim_on = st.checkbox(
        "Vertical-IM streak filter",
        value=False,
        help=(
            "Content-aware: keeps points belonging to long vertical streaks "
            "in (scan, TOF-index) space. Targets real ions; drops single "
            "hits and short streaks."
        ),
    )
    vim_filter: VerticalNoiseFilter | None = None
    if vim_on:
        with st.expander("VerticalNoiseFilter knobs", expanded=True):
            vim_mz_idx_half_width = int(st.number_input(
                "mz_idx_half_width (TOF indices)", min_value=0, max_value=20, value=3, step=1,
            ))
            vim_min_streak_scans = int(st.number_input(
                "min_streak_scans", min_value=1, max_value=100, value=5, step=1,
            ))
            vim_max_gap_scans = int(st.number_input(
                "max_gap_scans", min_value=0, max_value=20, value=1, step=1,
            ))
            vim_min_streak_intensity = float(st.number_input(
                "min_streak_intensity", min_value=0.0, value=50.0, step=10.0, format="%.3f",
            ))
            vim_num_iterations = int(st.number_input(
                "num_iterations", min_value=1, max_value=10, value=2, step=1,
                help="Re-apply the filter to its own survivors.",
            ))
        vim_filter = VerticalNoiseFilter(
            mz_idx_half_width=vim_mz_idx_half_width,
            min_streak_scans=vim_min_streak_scans,
            max_gap_scans=vim_max_gap_scans,
            min_streak_intensity=vim_min_streak_intensity,
            num_iterations=vim_num_iterations,
        )

    # 2) Intensity threshold (one of the IntensityThreshold subclasses, or off)
    NOISE_METHODS = (
        "off", "absolute", "mad", "percentile",
        "histogram", "baseline", "iterative_median",
    )
    threshold_method = st.selectbox(
        "Intensity threshold",
        options=NOISE_METHODS,
        index=0,
        help=(
            "Drop peaks below a computed intensity threshold. "
            "Adaptive methods (mad / iterative_median) derive the floor from "
            "the data; `absolute` is a fixed value."
        ),
    )
    threshold_filter: NoiseFilter | None = None
    if threshold_method == "absolute":
        thr_value = float(st.number_input(
            "Absolute threshold", min_value=0.0, value=1.0, step=1.0,
        ))
        threshold_filter = AbsoluteThreshold(value=thr_value)
    elif threshold_method == "mad":
        with st.expander("MadThreshold knobs"):
            mad_k = float(st.number_input(
                "k (× scale × MAD)", min_value=0.5, max_value=20.0,
                value=3.0, step=0.5, format="%.2f",
            ))
        threshold_filter = MadThreshold(k=mad_k)
    elif threshold_method == "percentile":
        with st.expander("PercentileThreshold knobs"):
            pct_q = float(st.slider(
                "q (percentile)", min_value=0.0, max_value=100.0, value=75.0, step=1.0,
            ))
        threshold_filter = PercentileThreshold(q=pct_q)
    elif threshold_method == "histogram":
        with st.expander("HistogramThreshold knobs"):
            hist_bins = int(st.number_input(
                "bins", min_value=10, max_value=1000, value=100, step=10,
            ))
            hist_k = float(st.number_input(
                "k (× std)", min_value=0.5, max_value=20.0, value=3.0, step=0.5,
            ))
        threshold_filter = HistogramThreshold(bins=hist_bins, k=hist_k)
    elif threshold_method == "baseline":
        with st.expander("BaselineThreshold knobs"):
            base_q = float(st.slider(
                "q (bottom percentile)", min_value=0.0, max_value=100.0,
                value=25.0, step=1.0,
            ))
            base_k = float(st.number_input(
                "k (× std)", min_value=0.5, max_value=20.0, value=3.0, step=0.5,
                key="base_k",
            ))
        threshold_filter = BaselineThreshold(q=base_q, k=base_k)
    elif threshold_method == "iterative_median":
        with st.expander("IterativeMedianThreshold knobs"):
            itm_passes = int(st.number_input(
                "passes", min_value=1, max_value=20, value=3, step=1,
            ))
            itm_inner = float(st.number_input(
                "inner_k", min_value=0.5, max_value=10.0, value=2.0, step=0.5,
            ))
            itm_final = float(st.number_input(
                "final_k", min_value=0.5, max_value=20.0, value=3.0, step=0.5,
            ))
        threshold_filter = IterativeMedianThreshold(
            passes=itm_passes, inner_k=itm_inner, final_k=itm_final,
        )

    # Compose: VerticalNoise first (content-aware), then intensity threshold.
    noise_filters: tuple[NoiseFilter, ...] = tuple(
        f for f in (vim_filter, threshold_filter) if f is not None
    )


# -- Centroiding ------------------------------------------------------------

with st.sidebar:
    st.header("Centroiding")
    centroid_on = st.checkbox(
        "Run centroiding",
        value=False,
        help="Runs the chosen centroider on the (filtered) raw peaks.",
    )
    centroider: Centroider | None = None
    centroid_log_y = False
    if centroid_on:
        algo = st.radio(
            "Algorithm",
            options=["merge_peaks", "watershed"],
            horizontal=True,
            help=(
                "merge_peaks: greedy tolerance-based merge in float m/z space. "
                "watershed: intensity-ordered region growing in integer "
                "(scan, TOF-index) space."
            ),
        )

        if algo == "merge_peaks":
            with st.expander("MergePeaksCentroider knobs", expanded=True):
                col_mz_tol, col_mz_unit = st.columns([2, 1])
                with col_mz_tol:
                    mp_mz_tol = float(st.number_input(
                        "m/z tolerance", min_value=0.0, value=8.0,
                        step=1.0, format="%.4f",
                    ))
                with col_mz_unit:
                    mp_mz_unit = st.selectbox(
                        "unit", options=["ppm", "da"], index=0, key="mp_mz_unit",
                    )
                col_im_tol, col_im_unit = st.columns([2, 1])
                with col_im_tol:
                    mp_im_tol = float(st.number_input(
                        "IM tolerance", min_value=0.0, value=0.01,
                        step=0.005, format="%.4f",
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
                mp_max_peaks_raw = int(st.number_input(
                    "max_peaks (0 = no limit)",
                    min_value=0, max_value=1_000_000, value=0, step=1000,
                ))
                mp_max_peaks = mp_max_peaks_raw if mp_max_peaks_raw > 0 else None
                mp_peak_noise = st.checkbox(
                    "Peak-satellite suppression",
                    value=False,
                    help=(
                        "After each centroid is formed, suppress raw points "
                        "within ±window Da of the anchor m/z (same IM window) "
                        "whose intensity falls below a linear ramp decaying "
                        "from anchor at d=0 to anchor × end_fraction at d=window."
                    ),
                )
                if mp_peak_noise:
                    mp_peak_window = float(st.number_input(
                        "peak_noise_window (Da)",
                        min_value=0.0001, max_value=10.0,
                        value=0.1, step=0.01, format="%.4f",
                    ))
                    mp_peak_end = float(st.number_input(
                        "peak_noise_end_fraction",
                        min_value=0.0, max_value=1.0,
                        value=0.1, step=0.05, format="%.3f",
                    ))
                else:
                    mp_peak_window = 0.1
                    mp_peak_end = 0.1
            centroider = MergePeaksCentroider(
                mz_tolerance=mp_mz_tol,
                mz_tolerance_type=mp_mz_unit,  # type: ignore[arg-type]
                im_tolerance=mp_im_tol,
                im_tolerance_type=mp_im_unit,  # type: ignore[arg-type]
                min_peaks=mp_min_peaks,
                max_peaks=mp_max_peaks,
                peak_noise_filter=mp_peak_noise,
                peak_noise_window=mp_peak_window,
                peak_noise_end_fraction=mp_peak_end,
            )

        else:  # watershed
            with st.expander("WatershedCentroider knobs", expanded=True):
                ws_attach_scan = int(st.number_input(
                    "attach_scan_half_width (scans)", min_value=1, max_value=200, value=10, step=1,
                ))
                ws_attach_mz_idx = int(st.number_input(
                    "attach_mz_idx_half_width (TOF indices)", min_value=1, max_value=200, value=3, step=1,
                ))
                ws_min_seed = float(st.number_input(
                    "min_seed_intensity", min_value=0.0, value=0.0, step=10.0,
                    help="Points below this can't promote to a new seed.",
                ))
                ws_min_centroid = float(st.number_input(
                    "min_centroid_intensity", min_value=0.0, value=0.0, step=10.0,
                    help="Final centroids below this summed intensity are dropped.",
                ))

                st.caption("Pre-centroid box smoothing (0 = off)")
                col_bs1, col_bs2 = st.columns(2)
                with col_bs1:
                    ws_smooth_scan = int(st.number_input(
                        "smooth_scan_half_width", min_value=0, max_value=50, value=5, step=1,
                    ))
                with col_bs2:
                    ws_smooth_mz = int(st.number_input(
                        "smooth_mz_idx_half_width", min_value=0, max_value=50, value=3, step=1,
                    ))

                st.caption("Per-group leash from seed (blank = no limit)")
                col_l1, col_l2 = st.columns(2)
                with col_l1:
                    ws_leash_scan_raw = int(st.number_input(
                        "max_scan_from_seed",
                        min_value=0, max_value=1000, value=0, step=1,
                        help="0 = no limit on the scan axis.",
                    ))
                with col_l2:
                    ws_leash_mz_raw = int(st.number_input(
                        "max_mz_idx_from_seed",
                        min_value=0, max_value=1000, value=10, step=1,
                        help="0 = no limit on the TOF-index axis.",
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
# Fetch and display
# ============================================================================

peaks = _fetch_raw_peaks(
    analysis_dir, frame_id,
    ion_mobility_type,
    noise_filters,
    exclude,
)
raw_count = _count_raw_peaks(analysis_dir, frame_id)

if peaks.size == 0:
    st.warning("No peaks survive the current filter chain. Loosen the noise filter or region exclusion.")
    st.stop()

mz = peaks[:, 0]
intensity = peaks[:, 1]
im = peaks[:, 2]

mz_min, mz_max = float(mz.min()), float(mz.max())
im_min, im_max = float(im.min()), float(im.max())

with st.sidebar:
    st.header("Ranges (display only)")
    mz_range = st.slider(
        "m/z range",
        min_value=float(np.floor(mz_min)),
        max_value=float(np.ceil(mz_max)),
        value=(float(np.floor(mz_min)), float(np.ceil(mz_max))),
    )
    im_range = st.slider(
        f"Ion mobility ({ion_mobility_type}) range",
        min_value=float(im_min),
        max_value=float(im_max),
        value=(float(im_min), float(im_max)),
    )

mask = (mz >= mz_range[0]) & (mz <= mz_range[1]) & (im >= im_range[0]) & (im <= im_range[1])
mz, intensity, im = mz[mask], intensity[mask], im[mask]

if mz.size == 0:
    st.warning("No peaks in the selected m/z and ion mobility window.")
    st.stop()

if mz.size > max_points:
    keep = np.argpartition(intensity, -max_points)[-max_points:]
    mz, intensity, im = mz[keep], intensity[keep], im[keep]
    st.info(
        f"Downsampled to top {max_points:,} most intense peaks of {mask.sum():,} matches."
    )

color = np.log10(intensity + 1.0) if log_intensity else intensity


# -- Metrics row ------------------------------------------------------------

post_filter_count = peaks.shape[0]
ratio = post_filter_count / max(raw_count, 1)

filter_label_parts = []
if exclude is not None:
    filter_label_parts.append("exclude")
if vim_filter is not None:
    filter_label_parts.append("vim")
if threshold_filter is not None:
    filter_label_parts.append(
        type(threshold_filter).__name__.replace("Threshold", "").lower()
    )
filter_label = " + ".join(filter_label_parts) if filter_label_parts else "—"

col_a, col_b, col_c, col_d = st.columns(4)
col_a.metric("Frame", frame_id)
col_b.metric("RT (min)", f"{frame_meta['rt_min']:.2f}")
col_c.metric(
    "Peaks (raw → post)",
    f"{post_filter_count:,}",
    delta=f"{ratio:.2f}× of {raw_count:,}",
    delta_color="off",
)
col_d.metric("Filter chain", filter_label)


# -- Raw scatter ------------------------------------------------------------

fig = go.Figure()
fig.add_trace(
    go.Scattergl(
        x=mz, y=im, mode="markers",
        marker=dict(
            size=4, color=color, colorscale="Viridis",
            colorbar=dict(
                title="log10(intensity + 1)" if log_intensity else "intensity",
            ),
            showscale=True, opacity=0.7,
        ),
        customdata=np.column_stack([intensity]),
        hovertemplate=(
            "m/z: %{x:.4f}<br>"
            f"{ion_mobility_type}: " + "%{y:.4f}<br>"
            "intensity: %{customdata[0]:,.0f}<extra></extra>"
        ),
    )
)
if exclude is not None and ion_mobility_type == "ook0":
    (mz_a, ook0_a), (mz_b, ook0_b) = exclude.line
    slope = (ook0_b - ook0_a) / (mz_b - mz_a)
    sample_mz = np.linspace(float(mz_range[0]), float(mz_range[1]), 200)
    line_ook0 = ook0_a + (sample_mz - mz_a) * slope
    if exclude.cap_at_upper_endpoint:
        line_ook0 = np.minimum(line_ook0, max(ook0_a, ook0_b))
    fig.add_trace(
        go.Scattergl(
            x=sample_mz, y=line_ook0, mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="exclude region", hoverinfo="skip", showlegend=False,
        )
    )

fig.update_layout(
    xaxis_title="m/z",
    yaxis_title=f"Ion mobility ({ion_mobility_type})",
    height=700,
    margin=dict(l=40, r=20, t=30, b=40),
    template="plotly_white",
)
st.plotly_chart(fig, use_container_width=True)


# -- Centroided spectrum ----------------------------------------------------

if centroid_on and centroider is not None:
    st.subheader(f"Centroided spectrum — {type(centroider).__name__}")
    try:
        centroided = _fetch_centroided(
            analysis_dir, frame_id,
            ion_mobility_type,
            noise_filters, exclude,
            centroider,
        )
    except Exception as exc:  # noqa: BLE001
        st.error(f"Centroiding failed: {exc}")
        centroided = np.empty((0, 3), dtype=np.float64)

    if centroided.size == 0:
        st.warning("Centroiding produced 0 peaks. Try loosening tolerances or min_peaks / min_seed_intensity.")
    else:
        pre_centroid_count = post_filter_count
        c_mz = centroided[:, 0]
        c_int = centroided[:, 1]
        c_im = centroided[:, 2]
        c_mask = (
            (c_mz >= mz_range[0]) & (c_mz <= mz_range[1])
            & (c_im >= im_range[0]) & (c_im <= im_range[1])
        )
        c_mz, c_int, c_im = c_mz[c_mask], c_int[c_mask], c_im[c_mask]

        if c_mz.size == 0:
            st.info("No centroids in the selected display window.")
        else:
            if c_mz.size > max_points:
                keep = np.argpartition(c_int, -max_points)[-max_points:]
                c_mz, c_int, c_im = c_mz[keep], c_int[keep], c_im[keep]
                st.info(
                    f"Downsampled centroids to top {max_points:,} of {c_mask.sum():,} matches."
                )

            reduction_pct = 100.0 * (1.0 - centroided.shape[0] / max(pre_centroid_count, 1))
            col_c1, col_c2, col_c3 = st.columns(3)
            col_c1.metric(
                "Centroids",
                f"{centroided.shape[0]:,}",
                delta=f"-{reduction_pct:.1f}% of {pre_centroid_count:,}",
                delta_color="off",
            )
            col_c2.metric("Algorithm", type(centroider).__name__)
            # Total signal retained vs. raw input
            sig_retained = 100.0 * centroided[:, 1].sum() / max(peaks[:, 1].sum(), 1)
            col_c3.metric("Intensity retained", f"{sig_retained:.1f}%")

            # Show the centroider's knobs as a compact caption
            cfg_lines = []
            for f in centroider.__dataclass_fields__:  # type: ignore[attr-defined]
                cfg_lines.append(f"{f}={getattr(centroider, f)!r}")
            st.caption("Centroider: " + ",  ".join(cfg_lines))

            # Stick spectrum colored by IM (one line trace per IM bin so the
            # colors are correct; tip markers carry the continuous colorbar).
            im_lo_v = float(im_range[0])
            im_hi_v = float(im_range[1])
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
                bm = c_mz[m]; bi = c_int[m]
                n = bm.size
                stem_x = np.empty(3 * n, dtype=np.float64)
                stem_y = np.empty(3 * n, dtype=np.float64)
                stem_x[0::3] = bm; stem_x[1::3] = bm; stem_x[2::3] = np.nan
                stem_y[0::3] = 0.0; stem_y[1::3] = bi; stem_y[2::3] = np.nan
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
                        colorbar=dict(title=f"IM ({ion_mobility_type})"),
                        showscale=True, line=dict(width=0),
                    ),
                    customdata=np.column_stack([c_im]),
                    hovertemplate=(
                        "m/z: %{x:.4f}<br>"
                        "intensity: %{y:,.0f}<br>"
                        f"{ion_mobility_type}: " + "%{customdata[0]:.4f}<extra></extra>"
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


# -- Intensity distribution histogram ---------------------------------------

st.subheader("Intensity distribution (post-filter raw peaks)")

positive = intensity[intensity > 0]
if positive.size == 0:
    st.info("No positive intensities to histogram.")
else:
    i_min = float(positive.min())
    i_max = float(positive.max())
    dynamic_range = i_max / max(i_min, 1e-12)
    auto_log = dynamic_range > 100.0
    log_x = st.checkbox(
        "Log-spaced bins (x)",
        value=auto_log,
        help=f"Dynamic range {dynamic_range:.1f}× — log-spaced bins recommended above 100×.",
    )
    log_y = st.checkbox("Log count axis (y)", value=True)
    nbins = 80

    edges = (
        np.logspace(np.log10(i_min), np.log10(i_max), nbins + 1)
        if log_x
        else np.linspace(i_min, i_max, nbins + 1)
    )
    counts, _ = np.histogram(positive, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    hist = go.Figure()
    hist.add_trace(
        go.Bar(
            x=centers, y=counts, width=widths,
            marker=dict(color="#3b82f6", line=dict(width=0)),
            hovertemplate="intensity: %{x:,.1f}<br>count: %{y:,}<extra></extra>",
        )
    )
    # Mark the absolute threshold if one is set explicitly — adaptive
    # methods compute their threshold inside the kernel, so we can't draw
    # the line precisely here.
    if isinstance(threshold_filter, AbsoluteThreshold) and threshold_filter.value > 0:
        hist.add_vline(
            x=float(threshold_filter.value),
            line=dict(color="#ef4444", dash="dash"),
            annotation_text=f"threshold = {threshold_filter.value:g}",
            annotation_position="top right",
        )

    hist.update_layout(
        xaxis_title="intensity", yaxis_title="count",
        xaxis_type="log" if log_x else "linear",
        yaxis_type="log" if log_y else "linear",
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
        template="plotly_white",
        bargap=0,
    )
    st.plotly_chart(hist, use_container_width=True)

    q01, q50, q99 = (float(v) for v in np.quantile(positive, [0.01, 0.5, 0.99]))
    col_h1, col_h2, col_h3, col_h4 = st.columns(4)
    col_h1.metric("min / max", f"{i_min:,.1f} / {i_max:,.1f}")
    col_h2.metric("median", f"{q50:,.1f}")
    col_h3.metric("1% / 99%", f"{q01:,.1f} / {q99:,.1f}")
    col_h4.metric("dyn range", f"{dynamic_range:,.0f}×")
