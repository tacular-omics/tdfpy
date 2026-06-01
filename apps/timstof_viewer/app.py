"""Unified timsTOF visualization app for Bruker `.d` folders (DDA / DIA / PRM).

A multi-page Streamlit app that detects the acquisition mode and adapts its
views accordingly:

* **MS1** — raw → denoise → centroid in (m/z, 1/K0) space. The MS1 overlay
  adapts to the mode: detected **precursors** (DDA), scheduled **targets**
  (PRM), or none (DIA).
* **MS2 frames** — raw MS2 frames in (scan, TOF-index) space showing the
  isolation bands (PASEF precursors / DIA windows / PRM transitions) plus the
  same denoise → centroid pipeline.
* **Mode page** — Precursor spectra (DDA), DIA window scheme (DIA), or PRM
  targets (PRM).
* **TDF tables** — browse any table in `analysis.tdf` and run ad-hoc SQL.

Run with::

    streamlit run apps/timstof_viewer/app.py

Requires ``streamlit`` and ``plotly`` (not part of tdfpy's default deps)::

    uv pip install streamlit plotly
"""

from __future__ import annotations

import streamlit as st

from _shared import acquisition_type

st.set_page_config(page_title="tdfpy — timsTOF viewer", layout="wide")

# The `.d` path is set once on the Data source page and shared via session
# state; the acquisition mode it implies drives which pages are shown.
analysis_dir = st.session_state.get("analysis_dir") or ""
acquisition = acquisition_type(analysis_dir) if analysis_dir else "Unknown"
st.session_state["acquisition"] = acquisition

with st.sidebar:
    st.title("tdfpy · timsTOF")


# ---------------------------------------------------------------------------
# Adaptive navigation — the mode page depends on the acquisition type.
# ---------------------------------------------------------------------------

pages = [
    st.Page("pages/data_source.py", title="Data source", icon="📁", default=True),
    st.Page("pages/ms1.py", title="MS1", icon="🔬"),
    st.Page("pages/ms2_frames.py", title="MS2 frames", icon="🧩"),
]
if acquisition == "DDA":
    pages.append(st.Page("pages/precursors.py", title="Precursor spectra", icon="🎯"))
elif acquisition == "DIA":
    pages.append(st.Page("pages/dia_windows.py", title="DIA windows", icon="🪟"))
elif acquisition == "PRM":
    pages.append(st.Page("pages/prm_targets.py", title="PRM targets", icon="🎯"))
pages.append(st.Page("pages/tdf_tables.py", title="TDF tables (SQL)", icon="🗄️"))

st.navigation(pages).run()
