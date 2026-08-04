"""Precursors page — accumulated MS2 spectrum per precursor.

Unlike the PASEF MS2 page (which shows full frames with multiple subscans),
this page shows the *accumulated* spectrum for a single precursor: summing its
PASEF subscans and collapsing the mobility dimension gives one 1-D
(m/z, intensity) profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _shared import (
    accumulated_precursor_spectrum,
    build_pipeline_ui,
    fetch_precursor_centroided,
    fetch_precursor_raw_peaks,
    filter_chain_label,
    list_precursors,
    precursor_pasef_info,
    require_analysis_dir,
    scatter_mz_im,
    select_table_row,
    stick_spectrum_im,
    stick_spectrum_simple,
)

st.title("Precursor spectra (accumulated)")

analysis_dir = require_analysis_dir()
prec_df = list_precursors(analysis_dir)
if prec_df.empty:
    st.warning("No precursors found in this `.d` folder.")
    st.stop()


# -- Sidebar: table shaping + display ---------------------------------------

with st.sidebar:
    st.header("Precursor table")
    sort_by = st.selectbox(
        "Sort by", ["Id", "Intensity", "MonoisotopicMz", "Charge"], index=1)
    ascending = sort_by == "Id"
    view = prec_df.sort_values(sort_by, ascending=ascending, na_position="last")

    charge_vals = sorted(int(c) for c in prec_df["Charge"].dropna().unique())
    if charge_vals:
        wanted = st.multiselect("Charge filter", charge_vals, default=charge_vals)
        if wanted:
            view = view[view["Charge"].isin(wanted)]

    st.header("Display")
    log_y = st.checkbox("Log y-axis (intensity)", value=False)
    annotate_iso = st.checkbox("Mark isotope spacing", value=True,
                               help="Draw expected isotope m/z positions for the charge state.")
    n_iso = int(st.number_input("Isotopes to mark", 1, 10, 4, 1)) if annotate_iso else 0

    st.header("Raw subscan view")
    show_raw_view = st.checkbox(
        "Show raw (m/z × 1/K0) subscans + pipeline", value=False,
        help="Combine every PASEF subscan of this precursor into one (m/z × "
        "1/K0) cloud and run the noise filter / centroider on it — instead of "
        "Bruker's pre-accumulated 1-D spectrum.")
    raw_ion_mobility = "ook0"
    raw_log_intensity = True
    raw_pipeline = None
    if show_raw_view:
        raw_ion_mobility = st.selectbox(
            "Ion mobility axis", ["ook0", "ccs", "voltage"], index=0, key="prec_im_axis")
        raw_log_intensity = st.checkbox(
            "Log-scale color (log10 intensity)", value=True, key="prec_log_int")
        raw_pipeline = build_pipeline_ui("prec")

if view.empty:
    st.warning("No precursors match the charge filter.")
    st.stop()


# -- Main: precursor selection table (above the plot) -----------------------

view = view.reset_index(drop=True)
table_cols = [c for c in (
    "Id", "MonoisotopicMz", "LargestPeakMz", "Charge", "ScanNumber",
    "Intensity", "Parent",
) if c in view.columns]
col_cfg = {
    "Id": st.column_config.NumberColumn("Precursor"),
    "MonoisotopicMz": st.column_config.NumberColumn("Mono m/z", format="%.4f"),
    "LargestPeakMz": st.column_config.NumberColumn("Largest m/z", format="%.4f"),
    "Charge": st.column_config.NumberColumn("z"),
    "ScanNumber": st.column_config.NumberColumn("Scan"),
    "Intensity": st.column_config.NumberColumn("Intensity", format="%.0f"),
    "Parent": st.column_config.NumberColumn("MS1 frame"),
}

ids = view["Id"].astype(int).tolist()
default_pos = 0
jump = st.session_state.pop("prec_jump_id", None)
if jump is not None and jump in ids:
    default_pos = ids.index(jump)

st.caption("Select a precursor (click a row):")
choice = select_table_row(
    view[table_cols], key="precursor_table", default_pos=default_pos,
    column_config={c: col_cfg[c] for c in table_cols})
precursor_id = int(view.iloc[choice]["Id"])

info = precursor_pasef_info(analysis_dir, precursor_id)


# -- Header metrics ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precursor", f"P{precursor_id}")
mono = info.get("monoisotopic_mz")
c2.metric("Monoisotopic m/z", f"{mono:.4f}" if mono else "—")
c3.metric("Charge", str(info.get("charge")) if info.get("charge") else "—")
c4.metric("Parent MS1 frame", str(info.get("parent_frame", "—")))

extra = []
if info.get("rt_min") is not None:
    extra.append(f"RT {info['rt_min']:.2f} min")
if info.get("isolation_mz") is not None:
    extra.append(f"isolation {info['isolation_mz']:.3f} ± {info['isolation_width']/2:.3f}")
if info.get("collision_energy") is not None:
    extra.append(f"CE {info['collision_energy']:.1f}")
if extra:
    st.caption("  ·  ".join(extra))

# Deep-link to the parent MS1 frame on the MS1 page.
if info.get("parent_frame"):
    if st.button(f"↳ View parent MS1 frame {info['parent_frame']} with precursor overlay"):
        st.session_state["ms1_jump_frame"] = int(info["parent_frame"])
        st.switch_page("pages/ms1.py")


# -- Accumulated spectrum ---------------------------------------------------

spectrum = accumulated_precursor_spectrum(analysis_dir, precursor_id)
if spectrum.size == 0:
    st.warning("No accumulated spectrum returned for this precursor.")
    st.stop()

mz, intensity = spectrum[:, 0], spectrum[:, 1]

markers: list[dict] = []
if annotate_iso and mono and info.get("charge"):
    z = int(info["charge"])
    for i in range(n_iso):
        iso_mz = mono + i * (1.00335 / z)  # neutron mass / charge
        markers.append({"x": iso_mz, "label": f"+{i}" if i else "mono", "color": "#10b981"})

st.subheader(f"Accumulated fragment spectrum — {mz.size:,} peaks")
fig = stick_spectrum_simple(mz, intensity, log_y=log_y, markers=markers)
st.plotly_chart(fig, use_container_width=True)

s1, s2, s3 = st.columns(3)
s1.metric("Peaks", f"{mz.size:,}")
s2.metric("Base peak m/z", f"{mz[int(np.argmax(intensity))]:.4f}")
s3.metric("Summed intensity", f"{intensity.sum():,.0f}")


# -- PASEF segment metadata -------------------------------------------------

segs = info.get("segments") or []
if segs:
    with st.expander(f"PASEF MS/MS segments contributing to this precursor ({len(segs)})"):
        st.dataframe(pd.DataFrame(segs), use_container_width=True)
    st.caption(
        "These segments are the per-frame subscans that were accumulated into "
        "the spectrum above. The PASEF MS2 page shows them as full frames."
    )


# -- Raw subscan view (m/z × 1/K0) + pipeline -------------------------------
# All of this precursor's PASEF subscans combined into one cloud, with the
# noise filter / centroider applied — the un-accumulated counterpart of the
# 1-D spectrum above.

if show_raw_view and raw_pipeline is not None:
    exclude, smooth, halo, noise_filters, centroider, centroid_log_y = raw_pipeline

    st.markdown("---")
    st.subheader("Raw subscans (m/z × 1/K0)")
    st.caption(
        f"Combined across {len(segs)} PASEF segment(s)  ·  "
        f"Filter chain: {filter_chain_label(exclude, noise_filters)}"
    )

    peaks = fetch_precursor_raw_peaks(
        analysis_dir, precursor_id, raw_ion_mobility, noise_filters, exclude,
        smooth=smooth, halo=halo)
    if peaks.size == 0:
        st.warning("No raw peaks survive the current filter chain for this precursor.")
    else:
        r_mz, r_int, r_im = peaks[:, 0], peaks[:, 1], peaks[:, 2]
        mz_range = (float(r_mz.min()), float(r_mz.max()))
        im_range = (float(r_im.min()), float(r_im.max()))
        fig_raw = scatter_mz_im(
            r_mz, r_int, r_im,
            ion_mobility_type=raw_ion_mobility, log_intensity=raw_log_intensity,
            mz_range=mz_range, im_range=im_range, exclude=exclude)
        st.plotly_chart(fig_raw, use_container_width=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Raw peaks", f"{r_mz.size:,}")
        r2.metric("Base peak m/z", f"{r_mz[int(np.argmax(r_int))]:.4f}")
        r3.metric("Summed intensity", f"{r_int.sum():,.0f}")

        if centroider is not None:
            st.subheader(f"Centroided subscans — {type(centroider).__name__}")
            try:
                centroided = fetch_precursor_centroided(
                    analysis_dir, precursor_id, raw_ion_mobility,
                    noise_filters, exclude, centroider, smooth=smooth, halo=halo)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Centroiding failed: {exc}")
                centroided = np.empty((0, 3), dtype=np.float64)
            if centroided.size == 0:
                st.warning("Centroiding produced 0 peaks. Try loosening tolerances.")
            else:
                c_mz, c_int, c_im = centroided[:, 0], centroided[:, 1], centroided[:, 2]
                reduction = 100.0 * (1.0 - centroided.shape[0] / max(peaks.shape[0], 1))
                d1, d2, d3 = st.columns(3)
                d1.metric("Centroids", f"{centroided.shape[0]:,}",
                          delta=f"-{reduction:.1f}% of raw", delta_color="off")
                d2.metric("Algorithm", type(centroider).__name__)
                d3.metric("Intensity retained",
                          f"{100.0 * c_int.sum() / max(r_int.sum(), 1):.1f}%")
                fig_c = stick_spectrum_im(
                    c_mz, c_int, c_im,
                    ion_mobility_type=raw_ion_mobility, im_range=im_range,
                    mz_range=mz_range, log_y=centroid_log_y)
                st.plotly_chart(fig_c, use_container_width=True)
