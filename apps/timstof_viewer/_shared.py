"""Shared helpers for the timsTOF viewer app.

This module factors the data loaders, the noise/centroid pipeline sidebar
builder, and the plotting helpers out of the individual pages so every page
renders consistently and re-uses the same cached accessors. The peak-processing
ops (smoothing, noise filtering, centroiding) are thin adapters over the public
``tdfpy`` pipeline — the app builds UI-tuple specs and hands them to
:func:`tdfpy.smooth`, :class:`tdfpy.HorizontalHaloFilter`, etc.

Nothing here is part of the public ``tdfpy`` package — this is internal dev
tooling under ``apps/`` only.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import sample_colorscale

import tdfpy
from tdfpy import (
    AbsoluteThreshold,
    BaselineThreshold,
    ChargeStateRegion,
    HorizontalHaloFilter,
    HistogramThreshold,
    IntensityThreshold,
    IterativeMedianThreshold,
    MadThreshold,
    MergePeaksCentroider,
    NoiseFilter,
    PercentileThreshold,
    RawSpectrum,
    VerticalNoiseFilter,
    WatershedCentroider,
    apply_noise,
    coerce_filters,
    convert,
    exclude_region,
    get_acquisition_type,
    get_raw_peaks,
    read_spectrum,
    smooth,
    subset_scans,
)
from tdfpy.pipeline import Centroider
from tdfpy.tdf import PandasTdf

# A smoothing config: ``(scan_half_width, mz_idx_half_width, "sum" | "mean")``.
SmoothSpec = tuple[int, int, str]
# A horizontal m/z-halo config: ``(peak_fraction, mz_idx_half_width, scan_half_width)``.
HaloSpec = tuple[float, int, int]

# Bruker MsMsType code → human label. Codes from the Bruker TDF schema docs.
MS_MS_TYPE_LABELS: dict[int, str] = {
    0: "MS1",
    2: "MRM",
    8: "PASEF DDA MS2",
    9: "PASEF DIA MS2",
    10: "PRM",
}

ION_MOBILITY_TYPES = ["ook0", "ccs", "voltage"]


# ===========================================================================
# Data-source helpers
# ===========================================================================


def require_analysis_dir() -> str:
    """Return the validated ``.d`` path from session state, or stop the page.

    The path is chosen once on the **Data source** page and shared across
    every page through ``st.session_state``.
    """
    analysis_dir = st.session_state.get("analysis_dir")
    if not analysis_dir:
        st.info("No `.d` folder loaded — open the **Data source** page to set one.")
        st.stop()
    return analysis_dir  # type: ignore[return-value]


def validate_analysis_dir(analysis_dir: str) -> str | None:
    """Return an error message if ``analysis_dir`` is not a usable ``.d`` folder."""
    if not analysis_dir:
        return None
    path = Path(analysis_dir)
    if not path.exists() or not path.is_dir():
        return f"Path does not exist or is not a directory: {analysis_dir}"
    if not (path / "analysis.tdf").exists():
        return f"`analysis.tdf` not found under {analysis_dir}"
    return None


# ===========================================================================
# Cached data accessors
# ===========================================================================


@dataclass
class FrameRaw:
    """Raw integer-index points for a single frame (scan × TOF-index)."""

    scan_indices: np.ndarray  # int64
    mz_indices: np.ndarray  # int64 (TOF index)
    intensities: np.ndarray  # float64
    num_scans: int


@st.cache_data(show_spinner=False)
def list_ms1_frames(analysis_dir: str) -> list[dict]:
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
def list_ms2_frames(analysis_dir: str) -> list[dict]:
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


@st.cache_data(show_spinner=False)
def run_summary(analysis_dir: str) -> dict:
    """Top-level counts + sample name for the entrypoint header."""
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    frames = pdf.frames
    n_ms1 = int((frames["MsMsType"] == 0).sum())
    n_ms2 = int((frames["MsMsType"] != 0).sum())
    try:
        meta = pdf.global_metadata
        kv = dict(zip(meta["Key"], meta["Value"]))
        sample = kv.get("SampleName", kv.get("Description", ""))
    except Exception:  # noqa: BLE001
        sample = ""
    acq = acquisition_type(analysis_dir)
    # Mode-appropriate "feature" count: precursors / DIA windows / PRM targets.
    feature_label, n_features = "Features", 0
    try:
        if acq == "DDA":
            feature_label, n_features = "Precursors", int(len(pdf.precursors))
        elif acq == "DIA":
            feature_label, n_features = "DIA windows", int(len(pdf.dia_frame_msms_windows))
        elif acq == "PRM":
            feature_label, n_features = "PRM targets", int(len(pdf.prm_targets))
    except Exception:  # noqa: BLE001
        n_features = 0
    return {
        "sample": sample,
        "n_frames": int(len(frames)),
        "n_ms1": n_ms1,
        "n_ms2": n_ms2,
        "feature_label": feature_label,
        "n_features": n_features,
        "acquisition": acq,
    }


@st.cache_data(show_spinner=False)
def acquisition_type(analysis_dir: str) -> str:
    """Detected acquisition mode: ``"DDA"`` / ``"DIA"`` / ``"PRM"`` / ``"Unknown"``."""
    try:
        return str(get_acquisition_type(analysis_dir))
    except Exception:  # noqa: BLE001
        return "Unknown"


@st.cache_data(show_spinner=False)
def full_axis_ranges(
    analysis_dir: str, ion_mobility_type: str
) -> tuple[tuple[float | None, float | None], tuple[float | None, float | None]]:
    """Full instrument acquisition ranges ``((mz_lo, mz_hi), (im_lo, im_hi))``.

    Read from ``GlobalMetadata`` so the plot axes can be fixed to the same
    range for every frame (making frames comparable) rather than auto-ranging
    to wherever the data lands. ``None`` entries mean "not available — fall
    back to the data extent". The 1/K0 range is only known from metadata for
    the ``ook0`` axis; ``ccs`` / ``voltage`` return ``(None, None)``.
    """
    try:
        meta = PandasTdf(str(Path(analysis_dir) / "analysis.tdf")).global_metadata
        kv = dict(zip(meta["Key"], meta["Value"]))
    except Exception:  # noqa: BLE001
        return (None, None), (None, None)

    def _f(key: str) -> float | None:
        try:
            return float(kv[key])
        except (KeyError, TypeError, ValueError):
            return None

    mz = (_f("MzAcqRangeLower"), _f("MzAcqRangeUpper"))
    if ion_mobility_type == "ook0":
        im = (_f("OneOverK0AcqRangeLower"), _f("OneOverK0AcqRangeUpper"))
    else:
        im = (None, None)
    return mz, im


@st.cache_data(show_spinner=False)
def data_source_stats(analysis_dir: str) -> dict:
    """Detailed stats for the Data source page (extends :func:`run_summary`)."""
    base = run_summary(analysis_dir)
    path = Path(analysis_dir)
    tdf = path / "analysis.tdf"
    tdf_bin = path / "analysis.tdf_bin"
    base["tdf_mb"] = round(tdf.stat().st_size / 1e6, 2) if tdf.exists() else 0.0
    base["bin_mb"] = round(tdf_bin.stat().st_size / 1e6, 2) if tdf_bin.exists() else 0.0
    base["name"] = path.name
    base["path"] = str(path)

    # Surface a handful of instrument / run fields from GlobalMetadata.
    highlights: dict[str, str] = {}
    try:
        meta = PandasTdf(str(tdf)).global_metadata
        kv = dict(zip(meta["Key"], meta["Value"]))
        for key in (
            "InstrumentName", "InstrumentSerialNumber", "AcquisitionSoftware",
            "AcquisitionSoftwareVersion", "AcquisitionDateTime",
            "MzAcqRangeLower", "MzAcqRangeUpper", "OneOverK0AcqRangeLower",
            "OneOverK0AcqRangeUpper",
        ):
            if key in kv and kv[key] not in (None, ""):
                highlights[key] = str(kv[key])
    except Exception:  # noqa: BLE001
        pass
    base["metadata_highlights"] = highlights
    return base


@st.cache_data(show_spinner=False)
def count_raw_peaks(analysis_dir: str, frame_id: int) -> int:
    with tdfpy.timsdata_connect(analysis_dir) as td:
        return int(get_raw_peaks(td, frame_id).shape[0])


def _smooth_spectrum(spectrum: RawSpectrum, smooth_spec: SmoothSpec) -> RawSpectrum:
    """Box-smooth intensities via the package :func:`tdfpy.smooth` op.

    ``smooth_spec`` is the UI tuple ``(scan_hw, mz_hw, "sum" | "mean")``.
    """
    scan_hw, mz_hw, mode = smooth_spec
    return smooth(
        spectrum,
        scan_half_width=max(0, int(scan_hw)),
        mz_idx_half_width=max(0, int(mz_hw)),
        mode=mode,  # type: ignore[arg-type]
    )


def _halo_filter_spectrum(
    spectrum: RawSpectrum, td, frame_id: int, halo: HaloSpec
) -> RawSpectrum:
    """Drop left/right m/z-halo peaks via :class:`tdfpy.HorizontalHaloFilter`.

    ``halo`` is the UI tuple
    ``(peak_fraction, mz_idx_half_width, scan_half_width)``.
    """
    if spectrum.empty:
        return spectrum
    peak_fraction, mz_idx_half_width, scan_half_width = halo
    filt = HorizontalHaloFilter(
        peak_fraction=peak_fraction,
        mz_idx_half_width=int(mz_idx_half_width),
        scan_half_width=int(scan_half_width),
    )
    return apply_noise(spectrum, (filt,), td=td, frame_id=frame_id)


def _build_spectrum(
    td,
    frame_id: int,
    *,
    scan_range: tuple[int, int] | None,
    exclude: ChargeStateRegion | None,
    smooth: SmoothSpec | None,
    halo: HaloSpec | None,
    noise_filters: tuple[NoiseFilter, ...],
) -> RawSpectrum:
    """Compose the integer-space pipeline:

    read → subset → exclude → **smooth** → vertical-IM → **halo** →
    intensity threshold. The halo filter runs after the structural (vertical)
    filter and before the global intensity threshold.
    """
    spectrum = read_spectrum(td, frame_id)
    if scan_range is not None:
        spectrum = subset_scans(spectrum, scan_num_begin=scan_range[0], scan_num_end=scan_range[1])
    if exclude is not None:
        spectrum = exclude_region(spectrum, exclude, td=td, frame_id=frame_id)
    if smooth is not None:
        spectrum = _smooth_spectrum(spectrum, smooth)

    # Split noise filters: structural (vertical) ones run before the halo
    # filter, global intensity thresholds run after it.
    filters = coerce_filters(list(noise_filters) if noise_filters else None)
    pre = [f for f in filters if not isinstance(f, IntensityThreshold)]
    post = [f for f in filters if isinstance(f, IntensityThreshold)]
    if pre:
        spectrum = apply_noise(spectrum, pre, td=td, frame_id=frame_id)
    if halo is not None:
        spectrum = _halo_filter_spectrum(spectrum, td, frame_id, halo)
    if post:
        spectrum = apply_noise(spectrum, post, td=td, frame_id=frame_id)
    return spectrum


@st.cache_data(show_spinner=True, hash_funcs={NoiseFilter: lambda f: hash(f)})
def fetch_raw_peaks(
    analysis_dir: str,
    frame_id: int,
    ion_mobility_type: str,
    noise_filters: tuple[NoiseFilter, ...],
    exclude: ChargeStateRegion | None,
    scan_range: tuple[int, int] | None = None,
    smooth: SmoothSpec | None = None,
    halo: HaloSpec | None = None,
) -> np.ndarray:
    with tdfpy.timsdata_connect(analysis_dir) as td:
        spectrum = _build_spectrum(
            td, frame_id, scan_range=scan_range, exclude=exclude,
            smooth=smooth, halo=halo, noise_filters=noise_filters)
        return convert(spectrum, td, frame_id, ion_mobility_type=ion_mobility_type)  # type: ignore[arg-type]


@st.cache_data(show_spinner=True, hash_funcs={NoiseFilter: lambda f: hash(f)})
def fetch_centroided(
    analysis_dir: str,
    frame_id: int,
    ion_mobility_type: str,
    noise_filters: tuple[NoiseFilter, ...],
    exclude: ChargeStateRegion | None,
    centroider: Centroider,
    scan_range: tuple[int, int] | None = None,
    smooth: SmoothSpec | None = None,
    halo: HaloSpec | None = None,
) -> np.ndarray:
    with tdfpy.timsdata_connect(analysis_dir) as td:
        spectrum = _build_spectrum(
            td, frame_id, scan_range=scan_range, exclude=exclude,
            smooth=smooth, halo=halo, noise_filters=noise_filters)
        if spectrum.empty:
            return np.empty((0, 3), dtype=np.float64)
        return centroider(
            spectrum, td, frame_id, ion_mobility_type=ion_mobility_type)  # type: ignore[arg-type]


@st.cache_data(show_spinner=True)
def load_frame_raw(analysis_dir: str, frame_id: int) -> FrameRaw:
    """Read raw ``(scan, tof_idx, intensity)`` for a frame via ``readScans``.

    Goes straight to the C extension — works identically for MS1 and MS2
    frames and bypasses any centroiding the high-level API would apply.
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
    return FrameRaw(scan_indices, mz_indices, intensities, num_scans)


@st.cache_data(show_spinner=False)
def precursor_segments(analysis_dir: str, frame_id: int) -> list[dict]:
    """PASEF precursor isolation segments inside an MS2 frame (overlay bands)."""
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


@st.cache_data(show_spinner=False)
def ms2_segments(analysis_dir: str, frame_id: int, acquisition: str) -> list[dict]:
    """Unified isolation segments inside an MS2 frame, across acquisition modes.

    Returns a list of normalized dicts with keys ``scan_begin``, ``scan_end``,
    ``isolation_mz``, ``isolation_width``, ``collision_energy``, ``ref_id``
    (precursor / window-group / target id) and a human ``label``. Drives both
    the band overlay and the per-segment scan-scope selector on the MS2 page.
    """
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))

    def _norm(scan_b, scan_e, mz, width, ce, ref_id, label) -> dict:
        return {
            "scan_begin": int(scan_b),
            "scan_end": int(scan_e),
            "isolation_mz": float(mz),
            "isolation_width": float(width),
            "collision_energy": float(ce),
            "ref_id": int(ref_id),
            "label": label,
        }

    out: list[dict] = []
    try:
        if acquisition == "DDA":
            pasef = pdf.pasef_frame_msms_info
            rows = pasef[pasef["Frame"] == frame_id]
            for _, r in rows.iterrows():
                out.append(_norm(
                    r["ScanNumBegin"], r["ScanNumEnd"], r["IsolationMz"],
                    r["IsolationWidth"], r["CollisionEnergy"], r["Precursor"],
                    f"P{int(r['Precursor'])}",
                ))
        elif acquisition == "DIA":
            info = pdf.dia_frame_msms_info
            wins = pdf.dia_frame_msms_windows
            frame_groups = info[info["Frame"] == frame_id]["WindowGroup"].tolist()
            for wg in frame_groups:
                for _, r in wins[wins["WindowGroup"] == wg].iterrows():
                    out.append(_norm(
                        r["ScanNumBegin"], r["ScanNumEnd"], r["IsolationMz"],
                        r["IsolationWidth"], r["CollisionEnergy"], wg,
                        f"WG{int(wg)} · {float(r['IsolationMz']):.1f}",
                    ))
        elif acquisition == "PRM":
            info = pdf.prm_frame_msms_info
            rows = info[info["Frame"] == frame_id]
            for _, r in rows.iterrows():
                out.append(_norm(
                    r["ScanNumBegin"], r["ScanNumEnd"], r["IsolationMz"],
                    r["IsolationWidth"], r["CollisionEnergy"], r["Target"],
                    f"T{int(r['Target'])} · {float(r['IsolationMz']):.2f}",
                ))
    except Exception:  # noqa: BLE001
        return []
    return out


@st.cache_data(show_spinner=False)
def ms2_segment_rects(analysis_dir: str, frame_id: int, acquisition: str) -> list[dict]:
    """:func:`ms2_segments` augmented with 1/K0 bounds for the band overlay.

    Each band's mobility-scan range is mapped to 1/K0 using the frame's
    calibration so the isolation bands can be drawn as (m/z × 1/K0) rectangles
    on the same plane as the raw peaks — mirroring the feature overlay on the
    MS1 page.
    """
    segs = ms2_segments(analysis_dir, frame_id, acquisition)
    if not segs:
        return []
    bounds = sorted({s["scan_begin"] for s in segs} | {s["scan_end"] for s in segs})
    with tdfpy.timsdata_connect(analysis_dir) as td:
        ook0 = np.asarray(td.scanNumToOneOverK0(frame_id, np.asarray(bounds, dtype=np.int64)))
    k0 = dict(zip(bounds, ook0.tolist()))
    return [
        {**s, "ook0_begin": float(k0[s["scan_begin"]]), "ook0_end": float(k0[s["scan_end"]])}
        for s in segs
    ]


@st.cache_data(show_spinner=False)
def precursors_for_ms1_frame(analysis_dir: str, frame_id: int) -> list[dict]:
    """Precursors detected in an MS1 frame, with isolation + scan windows.

    Drives the MS1 precursor overlay. Reads the ``Precursors`` table for rows
    whose ``Parent`` is this MS1 frame, then joins each precursor's PASEF
    isolation window (``IsolationMz`` / ``IsolationWidth`` and the isolated
    mobility-scan range ``ScanNumBegin``/``ScanNumEnd``). All scan numbers —
    the precursor apex and the isolation window bounds — are converted to 1/K0
    via the parent frame's calibration so the overlay can draw the full
    (m/z × 1/K0) isolation box, not just the centre point.
    """
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    try:
        precursors = pdf.precursors
    except Exception:  # noqa: BLE001
        return []
    here = precursors[precursors["Parent"] == frame_id]
    if here.empty:
        return []

    # One representative PASEF isolation row per precursor (the window is the
    # same wherever the precursor was fragmented).
    iso_by_prec: dict[int, dict] = {}
    try:
        pasef = pdf.pasef_frame_msms_info
        for _, r in pasef.iterrows():
            pid = int(r["Precursor"])
            if pid not in iso_by_prec:
                iso_by_prec[pid] = {
                    "isolation_mz": float(r["IsolationMz"]),
                    "isolation_width": float(r["IsolationWidth"]),
                    "scan_begin": int(r["ScanNumBegin"]),
                    "scan_end": int(r["ScanNumEnd"]),
                }
    except Exception:  # noqa: BLE001
        iso_by_prec = {}

    # Resolve 1/K0 for every scan number we need in one call.
    need = set(here["ScanNumber"].astype(int).tolist())
    for iso in iso_by_prec.values():
        need.add(iso["scan_begin"])
        need.add(iso["scan_end"])
    scan_list = sorted(need)
    with tdfpy.timsdata_connect(analysis_dir) as td:
        k0_vals = np.asarray(td.scanNumToOneOverK0(frame_id, np.asarray(scan_list, dtype=np.int64)))
    k0 = dict(zip(scan_list, k0_vals.tolist()))

    out: list[dict] = []
    for _, row in here.iterrows():
        pid = int(row["Id"])
        mono = row["MonoisotopicMz"]
        charge = row["Charge"]
        iso = iso_by_prec.get(pid)
        entry = {
            "precursor_id": pid,
            "largest_peak_mz": float(row["LargestPeakMz"]),
            "average_mz": float(row["AverageMz"]),
            "monoisotopic_mz": (
                float(mono) if mono is not None and not pd.isna(mono) else None
            ),
            "charge": (
                int(charge) if charge is not None and not pd.isna(charge) else None
            ),
            "scan_number": int(row["ScanNumber"]),
            "ook0": float(k0[int(row["ScanNumber"])]),
            "intensity": float(row["Intensity"]),
            "isolation_mz": iso["isolation_mz"] if iso else None,
            "isolation_width": iso["isolation_width"] if iso else None,
            "scan_begin": iso["scan_begin"] if iso else None,
            "scan_end": iso["scan_end"] if iso else None,
            "ook0_begin": float(k0[iso["scan_begin"]]) if iso else None,
            "ook0_end": float(k0[iso["scan_end"]]) if iso else None,
        }
        out.append(entry)
    return out


@st.cache_data(show_spinner=False)
def prm_targets_overlay(analysis_dir: str, frame_id: int) -> list[dict]:
    """PRM targets as MS1-overlay points (m/z, 1/K0).

    PRM targets are *scheduled* (not detected per-frame), so the same target
    set is overlaid on every MS1 frame — its expected (m/z, 1/K0) position.
    """
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    try:
        targets = pdf.prm_targets
    except Exception:  # noqa: BLE001
        return []
    out: list[dict] = []
    for _, row in targets.iterrows():
        out.append({
            "target_id": int(row["Id"]),
            "monoisotopic_mz": float(row["MonoisotopicMz"]),
            "ook0": float(row["OneOverK0"]),
            "charge": int(row["Charge"]) if not pd.isna(row["Charge"]) else None,
            "rt_min": float(row["Time"]) / 60.0,
            "description": str(row["Description"]) if not pd.isna(row["Description"]) else "",
        })
    return out


@st.cache_data(show_spinner=False)
def dia_window_scheme(analysis_dir: str) -> pd.DataFrame:
    """The full DIA window scheme (one row per window) with derived m/z bounds."""
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    df = pdf.dia_frame_msms_windows.copy()
    df["mz_begin"] = df["IsolationMz"] - df["IsolationWidth"] / 2
    df["mz_end"] = df["IsolationMz"] + df["IsolationWidth"] / 2
    return df


@st.cache_data(show_spinner=False)
def dia_windows_ook0(analysis_dir: str, frame_id: int) -> list[dict]:
    """DIA windows as (m/z, 1/K0) rectangles for overlaying on an MS1 frame.

    The DIA isolation scheme is fixed across frames, so its scan ranges are
    mapped to 1/K0 using the calibration of the given MS1 ``frame_id`` (any
    frame's calibration is close enough for a visual overlay).
    """
    windows = dia_window_scheme(analysis_dir)
    if windows.empty:
        return []
    scans = np.unique(
        np.concatenate([
            windows["ScanNumBegin"].to_numpy(dtype=np.int64),
            windows["ScanNumEnd"].to_numpy(dtype=np.int64),
        ])
    )
    with tdfpy.timsdata_connect(analysis_dir) as td:
        ook0 = np.asarray(td.scanNumToOneOverK0(frame_id, scans))
    scan_to_k0 = dict(zip(scans.tolist(), ook0.tolist()))
    out: list[dict] = []
    for _, r in windows.iterrows():
        out.append({
            "window_group": int(r["WindowGroup"]),
            "mz_begin": float(r["mz_begin"]),
            "mz_end": float(r["mz_end"]),
            "isolation_mz": float(r["IsolationMz"]),
            "ook0_begin": float(scan_to_k0[int(r["ScanNumBegin"])]),
            "ook0_end": float(scan_to_k0[int(r["ScanNumEnd"])]),
            "collision_energy": float(r["CollisionEnergy"]),
        })
    return out


@st.cache_data(show_spinner=False)
def prm_targets_table(analysis_dir: str) -> pd.DataFrame:
    """The ``PrmTargets`` table."""
    return PandasTdf(str(Path(analysis_dir) / "analysis.tdf")).prm_targets.copy()


@st.cache_data(show_spinner=False)
def prm_target_transitions(analysis_dir: str, target_id: int) -> list[dict]:
    """All PRM transitions (frame × scan-range rows) for one target."""
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    info = pdf.prm_frame_msms_info
    rows = info[info["Target"] == target_id]
    frames = pdf.frames
    rt_by_frame = dict(zip(frames["Id"].astype(int), frames["Time"].astype(float)))
    out: list[dict] = []
    for _, r in rows.iterrows():
        fid = int(r["Frame"])
        out.append({
            "frame_id": fid,
            "rt_min": rt_by_frame.get(fid, float("nan")) / 60.0,
            "scan_begin": int(r["ScanNumBegin"]),
            "scan_end": int(r["ScanNumEnd"]),
            "isolation_mz": float(r["IsolationMz"]),
            "isolation_width": float(r["IsolationWidth"]),
            "collision_energy": float(r["CollisionEnergy"]),
        })
    return out


@st.cache_data(show_spinner=False)
def list_precursors(analysis_dir: str) -> pd.DataFrame:
    """The full ``Precursors`` table as a tidy DataFrame for the picker."""
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    df = pdf.precursors.copy()
    return df


@st.cache_data(show_spinner=True)
def accumulated_precursor_spectrum(analysis_dir: str, precursor_id: int) -> np.ndarray:
    """The accumulated (summed-over-subscans) MS2 spectrum for one precursor.

    Uses ``readPasefMsMs``, which Bruker returns already accumulated across
    the precursor's PASEF subscans — a 1-D ``(m/z, intensity)`` profile,
    distinct from the per-frame multi-subscan view on the PASEF page.
    """
    with tdfpy.timsdata_connect(analysis_dir) as td:
        prec_map = td.readPasefMsMs([precursor_id])
        if precursor_id not in prec_map:
            return np.empty((0, 2), dtype=np.float64)
        scan = prec_map[precursor_id]
        mz = np.asarray(scan[0], dtype=np.float64)
        intensity = np.asarray(scan[1], dtype=np.float64)
    return np.column_stack((mz, intensity))


@st.cache_data(show_spinner=False)
def precursor_pasef_info(analysis_dir: str, precursor_id: int) -> dict:
    """Isolation / CE / scan-range metadata for a single precursor."""
    pdf = PandasTdf(str(Path(analysis_dir) / "analysis.tdf"))
    prec = pdf.precursors
    prow = prec[prec["Id"] == precursor_id]
    info: dict = {}
    if not prow.empty:
        p = prow.iloc[0]
        info["parent_frame"] = int(p["Parent"])
        info["scan_number"] = int(p["ScanNumber"])
        info["monoisotopic_mz"] = (
            float(p["MonoisotopicMz"]) if not pd.isna(p["MonoisotopicMz"]) else None
        )
        info["charge"] = int(p["Charge"]) if not pd.isna(p["Charge"]) else None
        info["intensity"] = float(p["Intensity"])
        info["rt_min"] = None
        frames = pdf.frames
        frow = frames[frames["Id"] == info["parent_frame"]]
        if not frow.empty:
            info["rt_min"] = float(frow.iloc[0]["Time"]) / 60.0
    try:
        pasef = pdf.pasef_frame_msms_info
        segs = pasef[pasef["Precursor"] == precursor_id]
        info["segments"] = segs.to_dict("records")
        if not segs.empty:
            s = segs.iloc[0]
            info["isolation_mz"] = float(s["IsolationMz"])
            info["isolation_width"] = float(s["IsolationWidth"])
            info["collision_energy"] = float(s["CollisionEnergy"])
    except Exception:  # noqa: BLE001
        info["segments"] = []
    return info


@st.cache_data(show_spinner=False)
def table_names(analysis_dir: str) -> list[str]:
    return PandasTdf(str(Path(analysis_dir) / "analysis.tdf")).get_table_names()


@st.cache_data(show_spinner=True)
def load_table(analysis_dir: str, name: str) -> pd.DataFrame:
    from tdfpy.tdf import convert_table_to_df

    return convert_table_to_df(str(Path(analysis_dir) / "analysis.tdf"), name)


@st.cache_data(show_spinner=True)
def run_sql(analysis_dir: str, query: str) -> pd.DataFrame:
    import sqlite3

    with sqlite3.connect(str(Path(analysis_dir) / "analysis.tdf")) as conn:
        return pd.read_sql_query(query, conn)


# ===========================================================================
# Pipeline sidebar builder — region → vertical-noise → threshold → centroider
# ===========================================================================


def build_pipeline_ui(
    prefix: str,
    *,
    show_centroider: bool = True,
) -> tuple[
    ChargeStateRegion | None,
    "SmoothSpec | None",
    "HaloSpec | None",
    tuple[NoiseFilter, ...],
    Centroider | None,
    bool,
]:
    """Render the shared filter/centroid sidebar and return the configs.

    ``prefix`` namespaces every widget key so multiple pages can each own an
    independent instance of the controls. Returns
    ``(exclude, smooth, halo, noise_filters, centroider, centroid_log_y)``.
    The halo filter runs after the vertical-IM filter and before the
    intensity threshold (handled in :func:`_build_spectrum`).
    """
    k = lambda name: f"{prefix}_{name}"  # noqa: E731

    # -- Smoothing (box sum / mean) — runs first, before the vertical filter
    st.header("Smoothing")
    smooth_on = st.checkbox(
        "Box sum / average over index window",
        value=False,
        key=k("smooth_on"),
        help=(
            "For every peak, gather all peaks within ±scan and ±TOF-index of it "
            "and replace its intensity with the window sum or mean. Runs first, "
            "before the vertical-IM and threshold filters."
        ),
    )
    smooth: SmoothSpec | None = None
    if smooth_on:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            sm_scan_hw = int(st.number_input(
                "± scan window", min_value=0, max_value=100, value=1, step=1, key=k("sm_scan")))
        with col_s2:
            sm_mz_hw = int(st.number_input(
                "± TOF-index window", min_value=0, max_value=200, value=2, step=1, key=k("sm_mz")))
        sm_mode = st.radio(
            "Combine", options=["sum", "mean"], horizontal=True, key=k("sm_mode"),
            help="sum: add all window intensities. mean: average them.")
        smooth = (sm_scan_hw, sm_mz_hw, sm_mode)

    # -- Region exclusion ---------------------------------------------------
    st.header("Region exclusion")
    exclude_on = st.checkbox(
        "Drop singly-charged region (m/z, 1/K0 line)",
        value=False,
        key=k("exclude_on"),
        help=(
            "Two-point line cap. Peaks above the line (typically the "
            "singly-charged region in timsTOF MS1) are removed before "
            "smoothing or m/z conversion — done in TOF-index space."
        ),
    )
    exclude: ChargeStateRegion | None = None
    if exclude_on:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            mz_lo = float(st.number_input("m/z₁", value=350.0, step=10.0, key=k("mz_lo")))
            ook0_lo = float(
                st.number_input("1/K0₁", value=0.7, step=0.05, format="%.3f", key=k("k0_lo"))
            )
        with col_p2:
            mz_hi = float(st.number_input("m/z₂", value=1200.0, step=10.0, key=k("mz_hi")))
            ook0_hi = float(
                st.number_input("1/K0₂", value=1.4, step=0.05, format="%.3f", key=k("k0_hi"))
            )
        cap_at_upper = st.checkbox(
            "Cap at upper endpoint", value=True, key=k("cap"),
            help="When on, also drops anything above the line's higher 1/K0 endpoint.",
        )
        exclude = ChargeStateRegion(
            line=((mz_lo, ook0_lo), (mz_hi, ook0_hi)),
            cap_at_upper_endpoint=cap_at_upper,
        )

    # -- Noise filters ------------------------------------------------------
    st.header("Noise filters")
    vim_on = st.checkbox(
        "Vertical-IM streak filter",
        value=False,
        key=k("vim_on"),
        help=(
            "Content-aware: keeps points belonging to long vertical streaks "
            "in (scan, TOF-index) space. Targets real ions; drops single "
            "hits and short streaks."
        ),
    )
    vim_filter: VerticalNoiseFilter | None = None
    if vim_on:
        with st.expander("VerticalNoiseFilter knobs", expanded=True):
            vim_mz = int(st.number_input(
                "mz_idx_half_width (TOF indices)", 0, 20, 3, 1, key=k("vim_mz")))
            vim_min = int(st.number_input(
                "min_streak_scans", 1, 100, 5, 1, key=k("vim_min")))
            vim_gap = int(st.number_input(
                "max_gap_scans", 0, 20, 1, 1, key=k("vim_gap")))
            vim_int = float(st.number_input(
                "min_streak_intensity", 0.0, value=50.0, step=10.0,
                format="%.3f", key=k("vim_int")))
            vim_iters = int(st.number_input(
                "num_iterations", 1, 10, 2, 1, key=k("vim_iters"),
                help="Re-apply the filter to its own survivors."))
        vim_filter = VerticalNoiseFilter(
            mz_idx_half_width=vim_mz,
            min_streak_scans=vim_min,
            max_gap_scans=vim_gap,
            min_streak_intensity=vim_int,
            num_iterations=vim_iters,
        )

    # 1b) Horizontal m/z-halo filter — runs after the vertical filter and
    #     before the intensity threshold.
    halo_on = st.checkbox(
        "Horizontal m/z-halo filter",
        value=False,
        key=k("halo_on"),
        help=(
            "Removes the weak m/z halo flanking bright peaks — left/right only, "
            "never above/below. Each peak is compared to the MAX intensity in "
            "its surrounding box EXCLUDING its own m/z column; if it's below "
            "peak_fraction of that, it's dropped. Same-column (vertical streak) "
            "neighbours can never trigger removal."
        ),
    )
    halo: HaloSpec | None = None
    if halo_on:
        with st.expander("Halo filter knobs", expanded=True):
            g_frac = float(st.number_input(
                "peak_fraction (drop below this × left/right max)", 0.0, 1.0, 0.10, 0.01,
                format="%.3f", key=k("g_frac")))
            g_mz_hw = int(st.number_input(
                "mz_idx_half_width (±TOF indices, ~247/Da)", 0, 1000, 100, 10,
                key=k("g_mzhw")))
            g_scan_hw = int(st.number_input(
                "scan_half_width (±mobility scans, box height)", 0, 200, 30, 5,
                key=k("g_scanhw")))
        halo = (g_frac, g_mz_hw, g_scan_hw)

    methods = ("off", "absolute", "mad", "percentile", "histogram", "baseline", "iterative_median")
    method = st.selectbox(
        "Intensity threshold", options=methods, index=0, key=k("thr_method"),
        help=(
            "Drop peaks below a computed intensity threshold. Adaptive methods "
            "(mad / iterative_median) derive the floor from the data; "
            "`absolute` is a fixed value."
        ),
    )
    threshold_filter: NoiseFilter | None = None
    if method == "absolute":
        v = float(st.number_input("Absolute threshold", 0.0, value=1.0, step=1.0, key=k("abs_v")))
        threshold_filter = AbsoluteThreshold(value=v)
    elif method == "mad":
        with st.expander("MadThreshold knobs"):
            mad_k = float(st.number_input("k (× scale × MAD)", 0.5, 20.0, 3.0, 0.5, format="%.2f", key=k("mad_k")))
        threshold_filter = MadThreshold(k=mad_k)
    elif method == "percentile":
        with st.expander("PercentileThreshold knobs"):
            q = float(st.slider("q (percentile)", 0.0, 100.0, 75.0, 1.0, key=k("pct_q")))
        threshold_filter = PercentileThreshold(q=q)
    elif method == "histogram":
        with st.expander("HistogramThreshold knobs"):
            bins = int(st.number_input("bins", 10, 1000, 100, 10, key=k("hist_bins")))
            hk = float(st.number_input("k (× std)", 0.5, 20.0, 3.0, 0.5, key=k("hist_k")))
        threshold_filter = HistogramThreshold(bins=bins, k=hk)
    elif method == "baseline":
        with st.expander("BaselineThreshold knobs"):
            bq = float(st.slider("q (bottom percentile)", 0.0, 100.0, 25.0, 1.0, key=k("base_q")))
            bk = float(st.number_input("k (× std)", 0.5, 20.0, 3.0, 0.5, key=k("base_k")))
        threshold_filter = BaselineThreshold(q=bq, k=bk)
    elif method == "iterative_median":
        with st.expander("IterativeMedianThreshold knobs"):
            passes = int(st.number_input("passes", 1, 20, 3, 1, key=k("itm_p")))
            inner = float(st.number_input("inner_k", 0.5, 10.0, 2.0, 0.5, key=k("itm_i")))
            final = float(st.number_input("final_k", 0.5, 20.0, 3.0, 0.5, key=k("itm_f")))
        threshold_filter = IterativeMedianThreshold(passes=passes, inner_k=inner, final_k=final)

    noise_filters: tuple[NoiseFilter, ...] = tuple(
        f for f in (vim_filter, threshold_filter) if f is not None
    )

    # -- Centroiding --------------------------------------------------------
    centroider: Centroider | None = None
    centroid_log_y = False
    if show_centroider:
        st.header("Centroiding")
        centroid_on = st.checkbox(
            "Run centroiding", value=False, key=k("centroid_on"),
            help="Runs the chosen centroider on the (filtered) raw peaks.",
        )
        if centroid_on:
            algo = st.radio(
                "Algorithm", options=["merge_peaks", "watershed"],
                horizontal=True, key=k("algo"),
                help=(
                    "merge_peaks: greedy tolerance-based merge in float m/z space. "
                    "watershed: intensity-ordered region growing in integer "
                    "(scan, TOF-index) space."
                ),
            )
            if algo == "merge_peaks":
                with st.expander("MergePeaksCentroider knobs", expanded=True):
                    c1, c2 = st.columns([2, 1])
                    with c1:
                        mz_tol = float(st.number_input(
                            "m/z tolerance", 0.0, value=8.0, step=1.0, format="%.4f", key=k("mp_mz_tol")))
                    with c2:
                        mz_unit = st.selectbox("unit", ["ppm", "da"], index=0, key=k("mp_mz_unit"))
                    c3, c4 = st.columns([2, 1])
                    with c3:
                        im_tol = float(st.number_input(
                            "IM tolerance", 0.0, value=0.01, step=0.005, format="%.4f", key=k("mp_im_tol")))
                    with c4:
                        im_unit = st.selectbox("unit", ["relative", "absolute"], index=1, key=k("mp_im_unit"))
                    min_peaks = int(st.number_input(
                        "min_peaks", 0, 50, 1, 1, key=k("mp_min"),
                        help="0 or 1 keeps all clusters."))
                    max_raw = int(st.number_input(
                        "max_peaks (0 = no limit)", 0, 1_000_000, 0, 1000, key=k("mp_max")))
                    max_peaks = max_raw if max_raw > 0 else None
                    peak_noise = st.checkbox(
                        "Peak-satellite suppression", value=False, key=k("mp_pn"),
                        help=(
                            "After each centroid is formed, suppress raw points within "
                            "±window Da of the anchor m/z whose intensity falls below a "
                            "linear ramp decaying from the anchor."
                        ),
                    )
                    if peak_noise:
                        pn_win = float(st.number_input(
                            "peak_noise_window (Da)", 0.0001, 10.0, 0.1, 0.01, format="%.4f", key=k("mp_pnw")))
                        pn_end = float(st.number_input(
                            "peak_noise_end_fraction", 0.0, 1.0, 0.1, 0.05, format="%.3f", key=k("mp_pne")))
                    else:
                        pn_win, pn_end = 0.1, 0.1
                centroider = MergePeaksCentroider(
                    mz_tolerance=mz_tol,
                    mz_tolerance_type=mz_unit,  # type: ignore[arg-type]
                    im_tolerance=im_tol,
                    im_tolerance_type=im_unit,  # type: ignore[arg-type]
                    min_peaks=min_peaks,
                    max_peaks=max_peaks,
                    peak_noise_filter=peak_noise,
                    peak_noise_window=pn_win,
                    peak_noise_end_fraction=pn_end,
                )
            else:  # watershed
                with st.expander("WatershedCentroider knobs", expanded=True):
                    a_scan = int(st.number_input(
                        "attach_scan_half_width", 1, 200, 10, 1, key=k("ws_as")))
                    a_mz = int(st.number_input(
                        "attach_mz_idx_half_width", 1, 200, 3, 1, key=k("ws_amz")))
                    min_seed = float(st.number_input(
                        "min_seed_intensity", 0.0, value=0.0, step=10.0, key=k("ws_seed"),
                        help="Points below this can't promote to a new seed."))
                    min_cent = float(st.number_input(
                        "min_centroid_intensity", 0.0, value=0.0, step=10.0, key=k("ws_cent"),
                        help="Final centroids below this summed intensity are dropped."))
                    st.caption("Pre-centroid box smoothing (0 = off; use the upstream Smoothing step instead)")
                    s1, s2 = st.columns(2)
                    with s1:
                        sm_scan = int(st.number_input("smooth_scan_half_width", 0, 50, 0, 1, key=k("ws_sms")))
                    with s2:
                        sm_mz = int(st.number_input("smooth_mz_idx_half_width", 0, 50, 0, 1, key=k("ws_smmz")))
                    st.caption("Per-group leash from seed (0 = no limit)")
                    l1, l2 = st.columns(2)
                    with l1:
                        leash_scan_raw = int(st.number_input("max_scan_from_seed", 0, 1000, 0, 1, key=k("ws_lscan")))
                    with l2:
                        leash_mz_raw = int(st.number_input("max_mz_idx_from_seed", 0, 1000, 10, 1, key=k("ws_lmz")))
                centroider = WatershedCentroider(
                    attach_scan_half_width=a_scan,
                    attach_mz_idx_half_width=a_mz,
                    min_seed_intensity=min_seed,
                    min_centroid_intensity=min_cent,
                    smooth_scan_half_width=sm_scan,
                    smooth_mz_idx_half_width=sm_mz,
                    max_scan_from_seed=leash_scan_raw if leash_scan_raw > 0 else None,
                    max_mz_idx_from_seed=leash_mz_raw if leash_mz_raw > 0 else None,
                )
            centroid_log_y = st.checkbox("Log y-axis (centroid intensity)", value=False, key=k("clog"))

    return exclude, smooth, halo, noise_filters, centroider, centroid_log_y


def filter_chain_label(
    exclude: ChargeStateRegion | None,
    noise_filters: tuple[NoiseFilter, ...],
) -> str:
    """Compact human label for the active filter chain."""
    parts: list[str] = []
    if exclude is not None:
        parts.append("exclude")
    for f in noise_filters:
        name = type(f).__name__
        if name == "VerticalNoiseFilter":
            parts.append("vim")
        else:
            parts.append(name.replace("Threshold", "").lower())
    return " + ".join(parts) if parts else "—"


# ===========================================================================
# Frame selection
# ===========================================================================


def select_table_row(
    df: pd.DataFrame,
    *,
    key: str,
    height: int = 260,
    default_pos: int = 0,
    column_config: dict | None = None,
) -> int:
    """Render a selectable single-row table; return the chosen positional index.

    Replaces a long selectbox with a scrollable, sortable table whose row
    selection drives the rest of the page. ``default_pos`` is returned when the
    table has no active selection (e.g. on first render or a deep link).
    """
    event = st.dataframe(
        df, key=key, on_select="rerun", selection_mode="single-row",
        hide_index=True, use_container_width=True, height=height,
        column_config=column_config or {},
    )
    rows = list(getattr(event.selection, "rows", []) or []) if event else []
    return int(rows[0]) if rows else default_pos


def select_frame_row(
    frames: list[dict],
    *,
    key: str,
    jump_frame: int | None = None,
    height: int = 260,
) -> int:
    """Selectable frame table; return the chosen list index.

    ``jump_frame`` preselects the row for that frame id when the table has no
    active selection (used for cross-page deep links).
    """
    df = pd.DataFrame(frames)
    cols = [c for c in (
        "frame_id", "rt_min", "ms_ms_type", "num_peaks", "num_scans",
    ) if c in df.columns]
    col_cfg = {
        "frame_id": st.column_config.NumberColumn("Frame"),
        "rt_min": st.column_config.NumberColumn("RT (min)", format="%.2f"),
        "ms_ms_type": st.column_config.NumberColumn("MsMsType"),
        "num_peaks": st.column_config.NumberColumn("Peaks"),
        "num_scans": st.column_config.NumberColumn("Scans"),
    }
    default_pos = 0
    if jump_frame is not None:
        for i, f in enumerate(frames):
            if f["frame_id"] == jump_frame:
                default_pos = i
                break
    return select_table_row(
        df[cols], key=key, height=height, default_pos=default_pos,
        column_config={c: col_cfg[c] for c in cols},
    )


# ===========================================================================
# Plot helpers
# ===========================================================================


def downsample_by_intensity(
    intensity: np.ndarray, max_points: int
) -> np.ndarray | None:
    """Return indices of the top-``max_points`` intensities, or ``None``."""
    if intensity.size <= max_points:
        return None
    return np.argpartition(intensity, -max_points)[-max_points:]


def scatter_mz_im(
    mz: np.ndarray,
    intensity: np.ndarray,
    im: np.ndarray,
    *,
    ion_mobility_type: str,
    log_intensity: bool,
    mz_range: tuple[float, float] | None = None,
    im_range: tuple[float, float] | None = None,
    exclude: ChargeStateRegion | None = None,
    point_size: int = 4,
    height: int = 700,
) -> go.Figure:
    """Raw (m/z, ion-mobility) scatter colored by intensity.

    ``mz_range`` / ``im_range`` pin the axes to a fixed window (e.g. the full
    acquisition range) so the plot reads the same across frames instead of
    auto-ranging to the data extent.
    """
    color = np.log10(intensity + 1.0) if log_intensity else intensity
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=mz, y=im, mode="markers",
            marker=dict(
                size=point_size, color=color, colorscale="Viridis",
                colorbar=dict(title="log10(intensity + 1)" if log_intensity else "intensity"),
                showscale=True, opacity=0.7,
            ),
            customdata=np.column_stack([intensity]),
            hovertemplate=(
                "m/z: %{x:.4f}<br>"
                f"{ion_mobility_type}: " + "%{y:.4f}<br>"
                "intensity: %{customdata[0]:,.0f}<extra></extra>"
            ),
            name="raw peaks",
            showlegend=False,  # the colorbar already labels intensity
        )
    )
    if exclude is not None and ion_mobility_type == "ook0" and mz_range is not None:
        (mz_a, ook0_a), (mz_b, ook0_b) = exclude.line
        slope = (ook0_b - ook0_a) / (mz_b - mz_a)
        sample_mz = np.linspace(float(mz_range[0]), float(mz_range[1]), 200)
        line = ook0_a + (sample_mz - mz_a) * slope
        if exclude.cap_at_upper_endpoint:
            line = np.minimum(line, max(ook0_a, ook0_b))
        fig.add_trace(
            go.Scattergl(
                x=sample_mz, y=line, mode="lines",
                line=dict(color="#ef4444", dash="dash", width=2),
                name="exclude region", hoverinfo="skip", showlegend=False,
            )
        )
    fig.update_layout(
        xaxis_title="m/z",
        yaxis_title=f"Ion mobility ({ion_mobility_type})",
        height=height, margin=dict(l=40, r=20, t=40, b=40), template="plotly_white",
        # Horizontal legend above the plot so overlay labels (precursors /
        # targets / bands) don't collide with the intensity colorbar.
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    if mz_range is not None:
        fig.update_xaxes(range=[mz_range[0], mz_range[1]])
    if im_range is not None:
        fig.update_yaxes(range=[im_range[0], im_range[1]])
    return fig


def stick_spectrum_im(
    c_mz: np.ndarray,
    c_int: np.ndarray,
    c_im: np.ndarray,
    *,
    ion_mobility_type: str,
    im_range: tuple[float, float],
    mz_range: tuple[float, float] | None = None,
    log_y: bool = False,
    height: int = 500,
) -> go.Figure:
    """Centroided stick spectrum, colored by ion mobility."""
    im_lo, im_hi = float(im_range[0]), float(im_range[1])
    n_bins = 32
    if im_hi > im_lo:
        edges = np.linspace(im_lo, im_hi, n_bins + 1)
        bin_idx = np.clip(np.digitize(c_im, edges) - 1, 0, n_bins - 1)
    else:
        bin_idx = np.zeros(c_im.shape, dtype=np.int64)
    sample_t = ((np.arange(n_bins) + 0.5) / n_bins).tolist()
    bin_colors = sample_colorscale("Viridis", sample_t)

    fig = go.Figure()
    for b in range(n_bins):
        m = bin_idx == b
        if not m.any():
            continue
        bm, bi = c_mz[m], c_int[m]
        n = bm.size
        sx = np.empty(3 * n)
        sy = np.empty(3 * n)
        sx[0::3], sx[1::3], sx[2::3] = bm, bm, np.nan
        sy[0::3], sy[1::3], sy[2::3] = 0.0, bi, np.nan
        fig.add_trace(go.Scattergl(
            x=sx, y=sy, mode="lines",
            line=dict(color=bin_colors[b], width=1.2),
            hoverinfo="skip", showlegend=False,
        ))
    fig.add_trace(go.Scattergl(
        x=c_mz, y=c_int, mode="markers",
        marker=dict(
            size=4, color=c_im, colorscale="Viridis", cmin=im_lo, cmax=im_hi,
            colorbar=dict(title=f"IM ({ion_mobility_type})"), showscale=True, line=dict(width=0),
        ),
        customdata=np.column_stack([c_im]),
        hovertemplate=(
            "m/z: %{x:.4f}<br>intensity: %{y:,.0f}<br>"
            f"{ion_mobility_type}: " + "%{customdata[0]:.4f}<extra></extra>"
        ),
        showlegend=False,
    ))
    layout: dict = dict(
        xaxis_title="m/z", yaxis_title="intensity",
        yaxis_type="log" if log_y else "linear",
        height=height, margin=dict(l=40, r=20, t=30, b=40), template="plotly_white",
    )
    if mz_range is not None:
        layout["xaxis_range"] = [mz_range[0], mz_range[1]]
    fig.update_layout(**layout)
    return fig


def stick_spectrum_simple(
    mz: np.ndarray,
    intensity: np.ndarray,
    *,
    log_y: bool = False,
    height: int = 480,
    color: str = "#3b82f6",
    markers: list[dict] | None = None,
) -> go.Figure:
    """A plain (m/z, intensity) stick spectrum (e.g. accumulated precursor)."""
    n = mz.size
    sx = np.empty(3 * n)
    sy = np.empty(3 * n)
    sx[0::3], sx[1::3], sx[2::3] = mz, mz, np.nan
    sy[0::3], sy[1::3], sy[2::3] = 0.0, intensity, np.nan
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=sx, y=sy, mode="lines", line=dict(color=color, width=1.0),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scattergl(
        x=mz, y=intensity, mode="markers",
        marker=dict(size=3, color=color),
        hovertemplate="m/z: %{x:.4f}<br>intensity: %{y:,.0f}<extra></extra>",
        showlegend=False,
    ))
    for mk in markers or []:
        fig.add_vline(
            x=mk["x"], line=dict(color=mk.get("color", "#ef4444"), dash="dash", width=1),
            annotation_text=mk.get("label", ""), annotation_position="top",
        )
    fig.update_layout(
        xaxis_title="m/z", yaxis_title="intensity",
        yaxis_type="log" if log_y else "linear",
        height=height, margin=dict(l=40, r=20, t=30, b=40), template="plotly_white",
    )
    return fig
