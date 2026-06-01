"""Precursors page — accumulated MS2 spectrum per precursor.

Unlike the PASEF MS2 page (which shows full frames with multiple subscans),
this page shows the *accumulated* spectrum for a single precursor — Bruker's
``readPasefMsMs`` sums the precursor's PASEF subscans into one 1-D
(m/z, intensity) profile.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _shared import (
    accumulated_precursor_spectrum,
    list_precursors,
    precursor_pasef_info,
    require_analysis_dir,
    select_table_row,
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
