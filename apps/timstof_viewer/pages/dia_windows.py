"""DIA windows page — the isolation window scheme.

DIA has no detected precursors; instead a fixed set of isolation windows
(grouped into window groups) tiles the (m/z, ion-mobility) plane. This page
visualizes that scheme as m/z × scan-range rectangles and lists the full
window table.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

from _shared import dia_window_scheme, require_analysis_dir

st.title("DIA window scheme")

analysis_dir = require_analysis_dir()
if st.session_state.get("acquisition") != "DIA":
    st.info("This page is only meaningful for DIA acquisitions.")
    st.stop()

windows = dia_window_scheme(analysis_dir)
if windows.empty:
    st.warning("No DIA windows found in this `.d` folder.")
    st.stop()

groups = sorted(int(g) for g in windows["WindowGroup"].unique())

c1, c2, c3 = st.columns(3)
c1.metric("Windows", f"{len(windows):,}")
c2.metric("Window groups", f"{len(groups):,}")
c3.metric("m/z span", f"{windows['mz_begin'].min():.0f}–{windows['mz_end'].max():.0f}")


# -- Window scheme map (m/z × scan range) -----------------------------------

with st.sidebar:
    st.header("Window map")
    sel_groups = st.multiselect(
        "Window groups", groups, default=groups,
        help="Filter the map / table to specific window groups.")
    color_by_ce = st.checkbox("Color by collision energy", value=True)

view = windows[windows["WindowGroup"].isin(sel_groups)] if sel_groups else windows

st.subheader("Isolation windows — m/z × mobility-scan range")
st.caption(
    "Each rectangle is one isolation window: its m/z width (x) over its "
    "mobility-scan range (y). Overlapping colors are different window groups."
)

# Color rectangles either by window group (categorical) or CE (continuous).
if color_by_ce:
    ce = view["CollisionEnergy"].to_numpy(dtype=float)
    ce_lo, ce_hi = float(ce.min()), float(ce.max())
    span = max(ce_hi - ce_lo, 1e-9)
    colors = sample_colorscale("Plasma", ((ce - ce_lo) / span).tolist())
else:
    group_to_t = {g: (i + 0.5) / len(groups) for i, g in enumerate(groups)}
    colors = sample_colorscale(
        "Turbo", [group_to_t[int(g)] for g in view["WindowGroup"]])

fig = go.Figure()
shapes = []
cx, cy, texts = [], [], []
for color, (_, r) in zip(colors, view.iterrows()):
    shapes.append(dict(
        type="rect", xref="x", yref="y",
        x0=float(r["mz_begin"]), x1=float(r["mz_end"]),
        y0=float(r["ScanNumBegin"]), y1=float(r["ScanNumEnd"]),
        line=dict(color=color, width=1), fillcolor=color, opacity=0.35, layer="below"))
    cx.append(float(r["IsolationMz"]))
    cy.append((float(r["ScanNumBegin"]) + float(r["ScanNumEnd"])) / 2)
    texts.append(
        f"WG{int(r['WindowGroup'])}<br>m/z {r['mz_begin']:.1f}–{r['mz_end']:.1f}<br>"
        f"scans {int(r['ScanNumBegin'])}–{int(r['ScanNumEnd'])}<br>"
        f"CE {r['CollisionEnergy']:.1f}")
fig.add_trace(go.Scatter(
    x=cx, y=cy, mode="markers", marker=dict(size=4, color="#111827"),
    text=texts, hovertemplate="%{text}<extra></extra>", showlegend=False))
fig.update_layout(
    shapes=shapes, xaxis_title="m/z", yaxis_title="mobility scan number",
    height=640, margin=dict(l=40, r=20, t=30, b=40), template="plotly_white")
fig.update_yaxes(autorange="reversed")
st.plotly_chart(fig, use_container_width=True)


# -- Full window table ------------------------------------------------------

st.subheader("Window table")
ordered = view[[
    "WindowGroup", "ScanNumBegin", "ScanNumEnd",
    "IsolationMz", "IsolationWidth", "mz_begin", "mz_end", "CollisionEnergy",
]].sort_values(["WindowGroup", "ScanNumBegin"])
st.dataframe(ordered, use_container_width=True, height=420)
st.download_button(
    "Download CSV", data=ordered.to_csv(index=False).encode("utf-8"),
    file_name="dia_windows.csv", mime="text/csv")

st.caption(
    "To inspect the fragments inside a window, open the **MS2 frames** page "
    "and use the per-window scan scope in the denoise → centroid tab."
)
