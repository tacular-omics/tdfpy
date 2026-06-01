"""Data source page — set the `.d` folder once and review its stats.

The path entered here is stored in session state and used by every other
page, so the folder only needs to be entered once per session.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from _shared import data_source_stats, validate_analysis_dir

st.title("Data source")

analysis_dir = st.text_input(
    ".d folder path",
    value=st.session_state.get("analysis_dir", ""),
    help="Absolute path to a Bruker `.d` analysis folder on this machine.",
)

err = validate_analysis_dir(analysis_dir)
if analysis_dir and err:
    st.session_state["analysis_dir"] = ""
    st.error(err)
    st.stop()

st.session_state["analysis_dir"] = analysis_dir
if not analysis_dir:
    st.info("Enter the path to a Bruker `.d` folder above to load it.")
    st.stop()

try:
    stats = data_source_stats(analysis_dir)
except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to read metadata: {exc}")
    st.stop()

badge = {"DDA": "🟦", "DIA": "🟩", "PRM": "🎯"}.get(stats["acquisition"], "❔")
st.success(f"Loaded **{stats['name']}**  ·  {badge} **{stats['acquisition']}**")
if stats["sample"]:
    st.caption(f"Sample: {stats['sample']}")


# -- Headline stats ---------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Acquisition", stats["acquisition"])
c2.metric("Frames", f"{stats['n_frames']:,}")
c3.metric("MS1 / MS2", f"{stats['n_ms1']:,} / {stats['n_ms2']:,}")
c4.metric(stats["feature_label"], f"{stats['n_features']:,}")

c5, c6 = st.columns(2)
c5.metric("analysis.tdf", f"{stats['tdf_mb']:,.2f} MB")
c6.metric("analysis.tdf_bin", f"{stats['bin_mb']:,.2f} MB")


# -- Instrument / run metadata ----------------------------------------------

if stats["metadata_highlights"]:
    st.subheader("Instrument & run metadata")
    md = pd.DataFrame(
        sorted(stats["metadata_highlights"].items()), columns=["Key", "Value"])
    st.dataframe(md, use_container_width=True, hide_index=True)

st.caption(f"Path: `{Path(analysis_dir)}`")
st.info("Use the sidebar to navigate to the MS1, MS2, and mode-specific pages.")
