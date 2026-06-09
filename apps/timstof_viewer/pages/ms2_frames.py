"""MS2 frames page — raw → denoise → centroid, mirroring the MS1 page.

Same single-flow experience as MS1: pick a frame, configure the
region/noise/centroid pipeline in the sidebar, and see the raw (m/z × 1/K0)
scatter, the centroided stick spectrum, and the intensity histogram stacked
in the main area.

The isolation bands fragmented in this frame — PASEF **precursors** (DDA),
**DIA windows** (DIA), or PRM **transitions** (PRM) — are drawn as (m/z × 1/K0)
rectangles over the raw scatter, the MS2 analogue of the MS1 feature overlay.
The pipeline can optionally be scoped to a single band's mobility scans.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _shared import (
    MS_MS_TYPE_LABELS,
    build_pipeline_ui,
    fetch_centroided,
    fetch_raw_peaks,
    filter_chain_label,
    full_axis_ranges,
    list_ms2_frames,
    ms2_segment_rects,
    require_analysis_dir,
    scatter_mz_im,
    select_frame_row,
    stick_spectrum_im,
)

st.title("MS2 frames")

analysis_dir = require_analysis_dir()
acquisition = st.session_state.get("acquisition", "Unknown")
BAND_NOUN = {"DDA": "precursor", "DIA": "window", "PRM": "transition"}.get(acquisition, "band")
ms2_frames = list_ms2_frames(analysis_dir)
if not ms2_frames:
    st.warning("No MS2 frames found in this `.d` folder.")
    st.stop()


# -- Sidebar: type filter ---------------------------------------------------

ms_types = sorted({f["ms_ms_type"] for f in ms2_frames})
with st.sidebar:
    st.header("MS2 frame")
    type_labels = [f"{t} — {MS_MS_TYPE_LABELS.get(t, 'unknown')}" for t in ms_types]
    type_idx = st.selectbox(
        "MS2 type filter", options=range(len(type_labels)),
        format_func=lambda i: type_labels[i], index=0,
        help="Restrict the table to one MsMsType (8=PASEF DDA, 9=PASEF DIA, ...).")
    selected_type = ms_types[type_idx]
    typed = [f for f in ms2_frames if f["ms_ms_type"] == selected_type]


# -- Main: frame selection table --------------------------------------------

st.caption("Select an MS2 frame (click a row):")
choice = select_frame_row(typed, key=f"ms2_frame_table_{selected_type}")
frame_meta = typed[choice]
frame_id = frame_meta["frame_id"]

segments = ms2_segment_rects(analysis_dir, frame_id, acquisition)


# -- Sidebar: display + band overlay + scan scope + pipeline ----------------

with st.sidebar:
    st.header("Display")
    ion_mobility_type = st.selectbox("Ion mobility axis", ["ook0", "ccs", "voltage"], index=0)
    log_intensity = st.checkbox("Log-scale color (log10 intensity)", value=True)
    max_points = int(st.number_input(
        "Max points to plot", 10_000, 2_000_000, 500_000, 50_000,
        help="If exceeded, top-N highest-intensity peaks are kept."))

    st.header(f"{BAND_NOUN.capitalize()} overlay")
    show_bands = st.checkbox(f"Overlay {BAND_NOUN} bands", value=True)
    scope_opts = ["Whole frame"] + [
        f"{s['label']} (scans {s['scan_begin']}–{s['scan_end']})" for s in segments
    ]
    scope_idx = st.selectbox(
        "Scan scope", options=range(len(scope_opts)),
        format_func=lambda i: scope_opts[i], index=0,
        help=f"Restrict the pipeline to one {BAND_NOUN}'s isolation scans.")
    scan_scope: tuple[int, int] | None = None
    if scope_idx > 0:
        seg = segments[scope_idx - 1]
        scan_scope = (seg["scan_begin"], seg["scan_end"] + 1)

    exclude, smooth, halo, noise_filters, centroider, centroid_log_y = build_pipeline_ui("ms2")


# -- Fetch raw peaks --------------------------------------------------------

peaks = fetch_raw_peaks(
    analysis_dir, frame_id, ion_mobility_type, noise_filters, exclude, scan_scope,
    smooth=smooth, halo=halo)
if peaks.size == 0:
    st.warning("No peaks survive the current filter chain / scan scope.")
    st.stop()

mz, intensity, im = peaks[:, 0], peaks[:, 1], peaks[:, 2]
mz_min, mz_max = float(mz.min()), float(mz.max())
im_min, im_max = float(im.min()), float(im.max())

# Fixed, frame-independent slider bounds from the acquisition metadata. Keeping
# the bounds stable (and giving each slider a key) means the selected range
# persists when you change frames or filters, instead of resetting every rerun.
# Fall back to this frame's data extent only when metadata is unavailable.
(fmz_lo, fmz_hi), (fim_lo, fim_hi) = full_axis_ranges(analysis_dir, ion_mobility_type)
mz_lo = fmz_lo if fmz_lo is not None else float(np.floor(mz_min))
mz_hi = fmz_hi if fmz_hi is not None else float(np.ceil(mz_max))
im_lo = fim_lo if fim_lo is not None else im_min
im_hi = fim_hi if fim_hi is not None else im_max

with st.sidebar:
    st.header("Ranges (display only)")
    mz_range = st.slider("m/z range", mz_lo, mz_hi, (mz_lo, mz_hi), key="ms2_mz_range")
    im_range = st.slider(
        f"Ion mobility ({ion_mobility_type}) range", im_lo, im_hi, (im_lo, im_hi),
        key=f"ms2_im_range_{ion_mobility_type}")

mask = (mz >= mz_range[0]) & (mz <= mz_range[1]) & (im >= im_range[0]) & (im <= im_range[1])
mz, intensity, im = mz[mask], intensity[mask], im[mask]
if mz.size == 0:
    st.warning("No peaks in the selected m/z and ion mobility window.")
    st.stop()

if mz.size > max_points:
    keep = np.argpartition(intensity, -max_points)[-max_points:]
    mz, intensity, im = mz[keep], intensity[keep], im[keep]
    st.info(f"Downsampled to top {max_points:,} most intense peaks of {int(mask.sum()):,} matches.")


# -- Metrics ----------------------------------------------------------------

ms_type_label = MS_MS_TYPE_LABELS.get(frame_meta["ms_ms_type"], "unknown")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Frame", frame_id)
c2.metric("RT (min)", f"{frame_meta['rt_min']:.2f}")
c3.metric("MsMsType", f"{frame_meta['ms_ms_type']} — {ms_type_label}")
c4.metric(f"{BAND_NOUN.capitalize()}s in frame", f"{len(segments):,}")
st.caption(f"Scope: {scope_opts[scope_idx]}  ·  Filter chain: {filter_chain_label(exclude, noise_filters)}")


# -- Raw scatter + isolation-band overlay -----------------------------------

fig = scatter_mz_im(
    mz, intensity, im,
    ion_mobility_type=ion_mobility_type, log_intensity=log_intensity,
    mz_range=mz_range, im_range=im_range, exclude=exclude)

if show_bands and segments and ion_mobility_type == "ook0":
    # Fragments span the full m/z axis, so each band is a full-width 1/K0
    # band (its mobility-scan range). The isolation m/z is the *precursor*
    # m/z and lives only in the hover text, not the rectangle width.
    x0, x1 = float(mz_range[0]), float(mz_range[1])
    shapes = []
    hx, hy, htext = [], [], []
    for seg in segments:
        half = seg["isolation_width"] / 2
        y0 = min(seg["ook0_begin"], seg["ook0_end"])
        y1 = max(seg["ook0_begin"], seg["ook0_end"])
        shapes.append(dict(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="#ef4444", width=2.0),
            fillcolor="rgba(239,68,68,0.12)", layer="above"))
        hx.append(x1)
        hy.append((y0 + y1) / 2)
        htext.append(
            f"{seg['label']}<br>precursor iso {seg['isolation_mz']:.2f} ± {half:.2f}<br>"
            f"1/K0 {y0:.3f}–{y1:.3f}<br>scans {seg['scan_begin']}–{seg['scan_end']}<br>"
            f"CE {seg['collision_energy']:.1f}")
    fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers", marker=dict(size=5, color="#7f1d1d"),
        text=htext, hovertemplate="%{text}<extra></extra>",
        name=f"{BAND_NOUN} bands", showlegend=False))
elif show_bands and segments and ion_mobility_type != "ook0":
    st.caption(f"{BAND_NOUN.capitalize()} band overlay is drawn only on the 1/K0 axis — switch ion mobility to `ook0`.")

st.plotly_chart(fig, use_container_width=True)


# -- Band table -------------------------------------------------------------

if segments:
    with st.expander(f"{BAND_NOUN.capitalize()} segments in this frame ({len(segments)})"):
        import pandas as pd

        st.dataframe(pd.DataFrame(segments), use_container_width=True)


# -- Centroided spectrum ----------------------------------------------------

if centroider is not None:
    st.subheader(f"Centroided spectrum — {type(centroider).__name__}")
    try:
        centroided = fetch_centroided(
            analysis_dir, frame_id, ion_mobility_type,
            noise_filters, exclude, centroider, scan_scope, smooth=smooth, halo=halo)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Centroiding failed: {exc}")
        centroided = np.empty((0, 3), dtype=np.float64)
    if centroided.size == 0:
        st.warning("Centroiding produced 0 peaks. Try loosening tolerances or min_peaks / min_seed_intensity.")
    else:
        c_mz, c_int, c_im = centroided[:, 0], centroided[:, 1], centroided[:, 2]
        cm = (c_mz >= mz_range[0]) & (c_mz <= mz_range[1]) & (c_im >= im_range[0]) & (c_im <= im_range[1])
        c_mz, c_int, c_im = c_mz[cm], c_int[cm], c_im[cm]
        if c_mz.size == 0:
            st.info("No centroids in the selected display window.")
        else:
            reduction = 100.0 * (1.0 - centroided.shape[0] / max(peaks.shape[0], 1))
            d1, d2, d3 = st.columns(3)
            d1.metric("Centroids", f"{centroided.shape[0]:,}", delta=f"-{reduction:.1f}% of raw", delta_color="off")
            d2.metric("Algorithm", type(centroider).__name__)
            d3.metric("Intensity retained", f"{100.0 * centroided[:, 1].sum() / max(peaks[:, 1].sum(), 1):.1f}%")
            fig_c = stick_spectrum_im(
                c_mz, c_int, c_im,
                ion_mobility_type=ion_mobility_type, im_range=im_range,
                mz_range=mz_range, log_y=centroid_log_y)
            st.plotly_chart(fig_c, use_container_width=True)
