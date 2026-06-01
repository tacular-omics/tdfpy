"""PRM targets page — scheduled targets and their fragment spectra.

PRM acquisitions fragment a predefined list of targets. This page lets you
pick a target, see its scheduling (m/z, charge, 1/K0, RT), browse the
transitions (per-frame isolation windows) that were acquired for it, and view
the raw → denoise → centroid spectrum scoped to a chosen transition.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from _shared import (
    build_pipeline_ui,
    fetch_centroided,
    fetch_raw_peaks,
    prm_target_transitions,
    prm_targets_table,
    require_analysis_dir,
    scatter_mz_im,
    stick_spectrum_im,
)

st.title("PRM targets")

analysis_dir = require_analysis_dir()
if st.session_state.get("acquisition") != "PRM":
    st.info("This page is only meaningful for PRM acquisitions.")
    st.stop()

targets = prm_targets_table(analysis_dir)
if targets.empty:
    st.warning("No PRM targets found in this `.d` folder.")
    st.stop()


# -- Sidebar: target picker + pipeline --------------------------------------

with st.sidebar:
    st.header("Target")

    def _label(i: int) -> str:
        r = targets.iloc[i]
        desc = str(r["Description"]) if not pd.isna(r["Description"]) else ""
        z = int(r["Charge"]) if not pd.isna(r["Charge"]) else "?"
        return f"T{int(r['Id'])} · m/z {float(r['MonoisotopicMz']):.3f} · z {z}" + (f" · {desc}" if desc else "")

    tidx = st.selectbox("Target", options=range(len(targets)), format_func=_label)
    target = targets.iloc[tidx]
    target_id = int(target["Id"])

    # Transitions for this target (the per-frame isolation windows).
    transitions = prm_target_transitions(analysis_dir, target_id)
    st.header("Transition")
    if transitions:
        t_opts = [
            f"Frame {t['frame_id']}  ·  RT {t['rt_min']:.2f} min  ·  scans {t['scan_begin']}–{t['scan_end']}"
            for t in transitions
        ]
        sel = st.selectbox("Transition", options=range(len(transitions)), format_func=lambda i: t_opts[i])
    else:
        sel = None
        st.caption("This target has no transitions.")

    st.header("Display")
    ion_mobility_type = st.selectbox("Ion mobility axis", ["ook0", "ccs", "voltage"], index=0)
    log_intensity = st.checkbox("Log-scale color", value=True)

    exclude, smooth, gaussian, noise_filters, centroider, centroid_log_y = build_pipeline_ui("prmtgt")


# -- Target metadata --------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("Target", f"T{target_id}")
c2.metric("Monoisotopic m/z", f"{float(target['MonoisotopicMz']):.4f}")
c3.metric("Charge", str(int(target["Charge"])) if not pd.isna(target["Charge"]) else "—")
c4.metric("Scheduled 1/K0", f"{float(target['OneOverK0']):.4f}")

meta = [f"Scheduled RT {float(target['Time']) / 60.0:.2f} min"]
if not pd.isna(target["ExternalId"]):
    meta.append(f"ExternalId {target['ExternalId']}")
if not pd.isna(target["Description"]) and str(target["Description"]):
    meta.append(str(target["Description"]))
st.caption("  ·  ".join(meta))


# -- Transitions ------------------------------------------------------------

if not transitions:
    st.warning("This target has no transitions (PrmFrameMsMsInfo rows).")
    st.stop()

st.subheader(f"Transitions ({len(transitions)})")
st.caption(
    "Each transition is one MS2 frame's isolation window for this target. "
    "Pick one in the sidebar to view its fragment spectrum."
)
with st.expander("Transition table", expanded=False):
    st.dataframe(pd.DataFrame(transitions), use_container_width=True)

tr = transitions[sel]
scan_scope = (tr["scan_begin"], tr["scan_end"] + 1)
frame_id = tr["frame_id"]


# -- Scoped raw + centroided spectrum ---------------------------------------

peaks = fetch_raw_peaks(
    analysis_dir, frame_id, ion_mobility_type, noise_filters, exclude, scan_scope,
    smooth=smooth, gaussian=gaussian)
if peaks.size == 0:
    st.warning("No peaks survive the filter chain in this transition's scan scope.")
    st.stop()

mz, intensity, im = peaks[:, 0], peaks[:, 1], peaks[:, 2]
mz_range = (float(np.floor(mz.min())), float(np.ceil(mz.max())))
im_range = (float(im.min()), float(im.max()))

st.subheader("Fragment ions (raw, scoped to transition)")
st.caption(f"Raw points: {peaks.shape[0]:,}  ·  frame {frame_id}, scans {tr['scan_begin']}–{tr['scan_end']}")
fig = scatter_mz_im(
    mz, intensity, im,
    ion_mobility_type=ion_mobility_type, log_intensity=log_intensity,
    mz_range=mz_range, exclude=exclude)
st.plotly_chart(fig, use_container_width=True)

if centroider is not None:
    st.subheader(f"Centroided fragments — {type(centroider).__name__}")
    try:
        centroided = fetch_centroided(
            analysis_dir, frame_id, ion_mobility_type,
            noise_filters, exclude, centroider, scan_scope, smooth=smooth, gaussian=gaussian)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Centroiding failed: {exc}")
        centroided = np.empty((0, 3), dtype=np.float64)
    if centroided.size == 0:
        st.warning("Centroiding produced 0 peaks.")
    else:
        c_mz, c_int, c_im = centroided[:, 0], centroided[:, 1], centroided[:, 2]
        d1, d2 = st.columns(2)
        d1.metric("Centroids", f"{centroided.shape[0]:,}")
        d2.metric("Intensity retained", f"{100.0 * c_int.sum() / max(peaks[:, 1].sum(), 1):.1f}%")
        fig_c = stick_spectrum_im(
            c_mz, c_int, c_im,
            ion_mobility_type=ion_mobility_type, im_range=im_range,
            mz_range=mz_range, log_y=centroid_log_y)
        st.plotly_chart(fig_c, use_container_width=True)
