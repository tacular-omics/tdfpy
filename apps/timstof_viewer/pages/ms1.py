"""MS1 page — raw → denoise → centroid with a mode-aware feature overlay.

The MS1 plane overlay adapts to the acquisition mode:

* **DDA** — detected precursors for the selected frame (m/z, 1/K0), sized by
  intensity and styled by charge.
* **PRM** — the scheduled targets (m/z, 1/K0); the same set on every frame.
* **DIA** — optionally overlay the DIA isolation window scheme (m/z × 1/K0
  rectangles) so you can see how the windows tile the MS1 plane.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from _shared import (
    build_pipeline_ui,
    dia_windows_ook0,
    fetch_centroided,
    fetch_raw_peaks,
    filter_chain_label,
    full_axis_ranges,
    list_ms1_frames,
    precursors_for_ms1_frame,
    prm_targets_overlay,
    require_analysis_dir,
    scatter_mz_im,
    select_frame_row,
    stick_spectrum_im,
)

analysis_dir = require_analysis_dir()
acquisition = st.session_state.get("acquisition", "Unknown")

OVERLAY_NOUN = {"DDA": "precursor", "PRM": "target"}.get(acquisition)
st.title(
    f"MS1 spectra + {OVERLAY_NOUN} overlay" if OVERLAY_NOUN else "MS1 spectra"
)

ms1_frames = list_ms1_frames(analysis_dir)
if not ms1_frames:
    st.warning("No MS1 frames found in this `.d` folder.")
    st.stop()


# -- Main: frame selection table --------------------------------------------

st.caption("Select an MS1 frame (click a row):")
jump_frame = st.session_state.pop("ms1_jump_frame", None)
choice = select_frame_row(ms1_frames, key="ms1_frame_table", jump_frame=jump_frame)
frame_meta = ms1_frames[choice]
frame_id = frame_meta["frame_id"]


# -- Sidebar: display + overlay + pipeline ----------------------------------

with st.sidebar:
    st.header("Display")
    ion_mobility_type = st.selectbox("Ion mobility axis", ["ook0", "ccs", "voltage"], index=0)
    log_intensity = st.checkbox("Log-scale color (log10 intensity)", value=True)
    max_points = int(st.number_input(
        "Max points to plot", 10_000, 2_000_000, 500_000, 50_000,
        help="If exceeded, top-N highest-intensity peaks are kept."))

    show_overlay = False
    show_dia_windows = False
    prec_mz_field = "monoisotopic_mz"
    if OVERLAY_NOUN:
        st.header(f"{OVERLAY_NOUN.capitalize()} overlay")
        show_overlay = st.checkbox(f"Overlay {OVERLAY_NOUN}s", value=True)
        if acquisition == "DDA":
            prec_mz_field = st.selectbox(
                "Precursor m/z", ["monoisotopic_mz", "largest_peak_mz", "average_mz"],
                index=0, help="Which precursor m/z to place on the m/z axis.",
            )
    elif acquisition == "DIA":
        st.header("DIA window overlay")
        show_dia_windows = st.checkbox(
            "Overlay DIA windows", value=False,
            help="Draw the isolation window scheme (m/z × 1/K0) over the MS1 plane.")

    exclude, smooth, gaussian, noise_filters, centroider, centroid_log_y = build_pipeline_ui("ms1")


# -- Fetch raw peaks --------------------------------------------------------

peaks = fetch_raw_peaks(
    analysis_dir, frame_id, ion_mobility_type, noise_filters, exclude,
    smooth=smooth, gaussian=gaussian)
if peaks.size == 0:
    st.warning("No peaks survive the current filter chain. Loosen the filters or region exclusion.")
    st.stop()

mz, intensity, im = peaks[:, 0], peaks[:, 1], peaks[:, 2]
mz_min, mz_max = float(mz.min()), float(mz.max())
im_min, im_max = float(im.min()), float(im.max())

# Default the axes to the full instrument acquisition range so frames are
# comparable, regardless of where this frame's data happens to fall.
(fmz_lo, fmz_hi), (fim_lo, fim_hi) = full_axis_ranges(analysis_dir, ion_mobility_type)
mz_lo = min(fmz_lo, mz_min) if fmz_lo is not None else float(np.floor(mz_min))
mz_hi = max(fmz_hi, mz_max) if fmz_hi is not None else float(np.ceil(mz_max))
im_lo = min(fim_lo, im_min) if fim_lo is not None else im_min
im_hi = max(fim_hi, im_max) if fim_hi is not None else im_max

with st.sidebar:
    st.header("Ranges (display only)")
    mz_range = st.slider("m/z range", mz_lo, mz_hi, (mz_lo, mz_hi))
    im_range = st.slider(f"Ion mobility ({ion_mobility_type}) range", im_lo, im_hi, (im_lo, im_hi))

mask = (mz >= mz_range[0]) & (mz <= mz_range[1]) & (im >= im_range[0]) & (im <= im_range[1])
mz, intensity, im = mz[mask], intensity[mask], im[mask]
if mz.size == 0:
    st.warning("No peaks in the selected m/z and ion mobility window.")
    st.stop()

if mz.size > max_points:
    keep = np.argpartition(intensity, -max_points)[-max_points:]
    mz, intensity, im = mz[keep], intensity[keep], im[keep]
    st.info(f"Downsampled to top {max_points:,} most intense peaks of {int(mask.sum()):,} matches.")


# -- Feature overlay for this frame (precursors / targets) ------------------

features: list[dict] = []
if show_overlay and acquisition == "DDA":
    features = precursors_for_ms1_frame(analysis_dir, frame_id)
elif show_overlay and acquisition == "PRM":
    features = prm_targets_overlay(analysis_dir, frame_id)

# DIA isolation windows (m/z × 1/K0 rectangles), optional overlay.
dia_windows = (
    dia_windows_ook0(analysis_dir, frame_id)
    if show_dia_windows and acquisition == "DIA" else []
)


# -- Metrics ----------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Frame", frame_id)
c2.metric("RT (min)", f"{frame_meta['rt_min']:.2f}")
if acquisition == "DIA":
    c3.metric("DIA windows", f"{len(dia_windows):,}")
else:
    c3.metric(f"{OVERLAY_NOUN.capitalize()}s" if OVERLAY_NOUN else "Features", f"{len(features):,}")
c4.metric("Filter chain", filter_chain_label(exclude, noise_filters))


# -- Raw scatter + feature overlay ------------------------------------------

fig = scatter_mz_im(
    mz, intensity, im,
    ion_mobility_type=ion_mobility_type, log_intensity=log_intensity,
    mz_range=mz_range, im_range=im_range, exclude=exclude,
)

if features and ion_mobility_type == "ook0" and acquisition == "DDA":
    # Each precursor as its full isolation (m/z) × mobility-scan (1/K0) box,
    # border colored by charge, drawn above the data so it stays visible. A
    # charge-colored centre marker carries the hover detail.
    charges_arr = np.array([f["charge"] or 0 for f in features], dtype=float)
    cspan = max(float(charges_arr.max() - charges_arr.min()), 1e-9)
    colors = sample_colorscale("Plasma", ((charges_arr - charges_arr.min()) / cspan).tolist())
    shapes, cx, cy, ccolor, ctext = [], [], [], [], []
    for f, color in zip(features, colors):
        mz_val = f.get(prec_mz_field) or f["largest_peak_mz"]
        hover = (
            f"P{f['precursor_id']}<br>m/z {mz_val:.4f}<br>z {f['charge']}<br>"
            f"1/K0 {f['ook0']:.4f}<br>intensity {f['intensity']:,.0f}"
        )
        if f["isolation_mz"] is not None and f["ook0_begin"] is not None:
            half = f["isolation_width"] / 2
            y0 = min(f["ook0_begin"], f["ook0_end"])
            y1 = max(f["ook0_begin"], f["ook0_end"])
            fill = color.replace("rgb(", "rgba(").replace(")", ", 0.15)")
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=f["isolation_mz"] - half, x1=f["isolation_mz"] + half,
                y0=y0, y1=y1,
                line=dict(color=color, width=2), fillcolor=fill, layer="above"))
            hover += (
                f"<br>isolation {f['isolation_mz']:.2f} ± {half:.2f}"
                f"<br>scans {f['scan_begin']}–{f['scan_end']}")
        cx.append(mz_val)
        cy.append(f["ook0"])
        ccolor.append(f["charge"] or 0)
        ctext.append(hover)
    if shapes:
        fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="markers",
        marker=dict(
            size=6, color=ccolor, colorscale="Plasma", symbol="x",
            line=dict(width=1, color="#111827"), showscale=False),
        text=ctext, hovertemplate="%{text}<extra></extra>",
        name="precursors", showlegend=False))

elif features and ion_mobility_type == "ook0":  # PRM targets — scheduled points
    px = [f["monoisotopic_mz"] for f in features]
    py = [f["ook0"] for f in features]
    charges = [f["charge"] or 0 for f in features]
    texts = [
        f"T{f['target_id']}<br>m/z {f['monoisotopic_mz']:.4f}<br>z {f['charge']}<br>"
        f"1/K0 {f['ook0']:.4f}<br>sched. RT {f['rt_min']:.2f} min"
        + (f"<br>{f['description']}" if f.get("description") else "")
        for f in features
    ]
    fig.add_trace(go.Scatter(
        x=px, y=py, mode="markers",
        marker=dict(
            size=12, color=charges, colorscale="Plasma",
            symbol="diamond-open", line=dict(width=1.5, color="#b91c1c"),
            showscale=False, opacity=0.9),
        text=texts, hovertemplate="%{text}<extra></extra>", name="targets"))

elif features and ion_mobility_type != "ook0":
    st.caption(f"{OVERLAY_NOUN.capitalize()} overlay is drawn only on the 1/K0 axis — switch ion mobility to `ook0`.")

# DIA window rectangles, colored by collision energy.
if dia_windows and ion_mobility_type == "ook0":
    ce = np.array([w["collision_energy"] for w in dia_windows], dtype=float)
    span = max(float(ce.max() - ce.min()), 1e-9)
    colors = sample_colorscale("Plasma", ((ce - ce.min()) / span).tolist())
    shapes = []
    hx, hy, htext = [], [], []
    for w, color in zip(dia_windows, colors):
        # Solid, thick border + a semi-transparent fill (alpha baked into the
        # fillcolor so the shape's own opacity doesn't wash out the border).
        fill = color.replace("rgb(", "rgba(").replace(")", ", 0.3)")
        shapes.append(dict(
            type="rect", xref="x", yref="y",
            x0=w["mz_begin"], x1=w["mz_end"],
            y0=min(w["ook0_begin"], w["ook0_end"]), y1=max(w["ook0_begin"], w["ook0_end"]),
            line=dict(color=color, width=2.5), fillcolor=fill, layer="above"))
        hx.append(w["isolation_mz"])
        hy.append((w["ook0_begin"] + w["ook0_end"]) / 2)
        htext.append(
            f"WG{w['window_group']}<br>m/z {w['mz_begin']:.1f}–{w['mz_end']:.1f}<br>"
            f"1/K0 {min(w['ook0_begin'], w['ook0_end']):.3f}–{max(w['ook0_begin'], w['ook0_end']):.3f}<br>"
            f"CE {w['collision_energy']:.1f}")
    fig.update_layout(shapes=shapes)
    fig.add_trace(go.Scatter(
        x=hx, y=hy, mode="markers", marker=dict(size=3, color="#111827"),
        text=htext, hovertemplate="%{text}<extra></extra>",
        name="DIA windows", showlegend=False))
elif show_dia_windows and ion_mobility_type != "ook0":
    st.caption("DIA window overlay is drawn only on the 1/K0 axis — switch ion mobility to `ook0`.")

st.plotly_chart(fig, use_container_width=True)


# -- Feature table ----------------------------------------------------------

if features:
    with st.expander(f"{OVERLAY_NOUN.capitalize()}s for this frame ({len(features)})"):
        import pandas as pd

        st.dataframe(pd.DataFrame(features), use_container_width=True)


# -- Centroided spectrum ----------------------------------------------------

if centroider is not None:
    st.subheader(f"Centroided spectrum — {type(centroider).__name__}")
    try:
        centroided = fetch_centroided(
            analysis_dir, frame_id, ion_mobility_type, noise_filters, exclude, centroider,
            smooth=smooth, gaussian=gaussian)
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
            m1, m2, m3 = st.columns(3)
            m1.metric("Centroids", f"{centroided.shape[0]:,}", delta=f"-{reduction:.1f}% of raw", delta_color="off")
            m2.metric("Algorithm", type(centroider).__name__)
            m3.metric("Intensity retained", f"{100.0 * centroided[:, 1].sum() / max(peaks[:, 1].sum(), 1):.1f}%")
            fig_c = stick_spectrum_im(
                c_mz, c_int, c_im,
                ion_mobility_type=ion_mobility_type, im_range=im_range,
                mz_range=mz_range, log_y=centroid_log_y,
            )
            st.plotly_chart(fig_c, use_container_width=True)
