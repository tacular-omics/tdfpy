"""Bound how far tdfpy's peak picking may drift from Bruker's.

``Precursor.peaks`` and ``PasefFrameMsmsInfo.peaks`` used to call Bruker's
proprietary peak picker. tdfpy now sums the mobility dimension away and
centroids by greedy m/z merging, which is close but deliberately not identical:
Bruker's algorithm is closed and appears to smooth before picking.

These tests pin that difference. They are not equality assertions — they assert
the agreement measured when the reimplementation landed, so a future change that
degrades peak picking fails here instead of quietly shifting everyone's spectra.

Reference lists live in ``tests/data/peaks_golden.json`` (see
``scripts/generate_peaks_golden.py``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tdfpy import DDA
from tdfpy.centroiding import get_mobility_collapsed_spectrum
from tdfpy.timsdata import timsdata_connect

DATA_DIR = Path("tests/data")
GOLDEN_PATH = DATA_DIR / "peaks_golden.json"
DDA_PATH = DATA_DIR / "example_dda.d"
DIA_PATH = DATA_DIR / "example_dia.d"

# Bounds measured per item against the native peak picker on the bundled
# fixtures. Observed ranges across the 10 precursors and 12 DIA windows:
#
#   count ratio  0.948 .. 1.093
#   captured     0.928 .. 1.000
#   strong ppm   0.000 .. 1.901
#   TIC ratio    0.985 .. 1.039
#
# Limits sit outside those with room to absorb tuning, but not a regression.
# Note these are per-item, not averages: one badly picked window must fail even
# if the mean still looks healthy.
MAX_COUNT_RATIO = 1.25
MIN_COUNT_RATIO = 0.85
MIN_INTENSITY_CAPTURED = 0.90
MAX_STRONG_PEAK_PPM = 3.0
TIC_RATIO_RANGE = (0.97, 1.06)

#: Peaks within this many ppm count as the same peak when matching.
MATCH_PPM = 10.0


def _golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip(f"golden file not found: {GOLDEN_PATH}")
    return json.loads(GOLDEN_PATH.read_text())


GOLDEN = _golden() if GOLDEN_PATH.exists() else {"dda_precursors": [], "dia_windows": []}


def _agreement(
    ref_mz: np.ndarray, ref_intensity: np.ndarray, got: np.ndarray
) -> dict[str, float]:
    """Compare a produced peak list against Bruker's for the same scan range."""
    assert got.ndim == 2 and got.shape[1] == 2, f"expected (N, 2), got {got.shape}"
    order = np.argsort(got[:, 0])
    got_mz, got_intensity = got[order, 0], got[order, 1]

    # Nearest produced peak to each reference peak.
    right = np.clip(np.searchsorted(got_mz, ref_mz), 1, len(got_mz) - 1)
    nearest = np.where(
        np.abs(got_mz[right] - ref_mz) < np.abs(got_mz[right - 1] - ref_mz),
        right,
        right - 1,
    )
    ppm = np.abs(got_mz[nearest] - ref_mz) / ref_mz * 1e6

    matched = ppm < MATCH_PPM
    strong = ref_intensity >= np.percentile(ref_intensity, 80)
    return {
        "count_ratio": len(got_mz) / len(ref_mz),
        "captured": float(ref_intensity[matched].sum() / ref_intensity.sum()),
        "strong_ppm": float(np.median(ppm[strong])),
        "tic_ratio": float(got_intensity.sum() / ref_intensity.sum()),
    }


def _check(stats: dict[str, float], label: str) -> None:
    assert MIN_COUNT_RATIO <= stats["count_ratio"] <= MAX_COUNT_RATIO, (
        f"{label}: peak count ratio {stats['count_ratio']:.3f} outside "
        f"[{MIN_COUNT_RATIO}, {MAX_COUNT_RATIO}]"
    )
    assert stats["captured"] >= MIN_INTENSITY_CAPTURED, (
        f"{label}: only {stats['captured']:.3%} of Bruker's intensity is within "
        f"{MATCH_PPM} ppm of one of our peaks"
    )
    assert stats["strong_ppm"] <= MAX_STRONG_PEAK_PPM, (
        f"{label}: strong peaks disagree by {stats['strong_ppm']:.2f} ppm"
    )
    lo, hi = TIC_RATIO_RANGE
    assert lo <= stats["tic_ratio"] <= hi, (
        f"{label}: total ion current ratio {stats['tic_ratio']:.4f} outside [{lo}, {hi}]"
    )


@pytest.mark.parametrize(
    "entry", GOLDEN["dda_precursors"], ids=lambda e: f"precursor{e['precursor']}"
)
def test_pasef_precursor_peaks_track_bruker(entry: dict) -> None:
    if not DDA_PATH.is_dir():
        pytest.skip("Test data not found")
    with timsdata_connect(str(DDA_PATH)) as td:
        got = get_mobility_collapsed_spectrum(
            td, [tuple(r) for r in entry["scan_ranges"]]
        )
    _check(
        _agreement(np.asarray(entry["mz"]), np.asarray(entry["intensity"]), got),
        f"precursor {entry['precursor']}",
    )


@pytest.mark.parametrize(
    "entry", GOLDEN["dia_windows"], ids=lambda e: f"frame{e['frame']}"
)
def test_dia_window_peaks_track_bruker(entry: dict) -> None:
    if not DIA_PATH.is_dir():
        pytest.skip("Test data not found")
    with timsdata_connect(str(DIA_PATH)) as td:
        got = get_mobility_collapsed_spectrum(
            td, [(entry["frame"], entry["scan_begin"], entry["scan_end"])]
        )
    _check(
        _agreement(np.asarray(entry["mz"]), np.asarray(entry["intensity"]), got),
        f"DIA frame {entry['frame']}",
    )


def test_precursor_peaks_property_uses_collapsed_spectrum() -> None:
    """The public property must produce the same peaks as the helper."""
    if not DDA_PATH.is_dir():
        pytest.skip("Test data not found")
    entry = GOLDEN["dda_precursors"][0]
    with DDA(str(DDA_PATH)) as dda:
        precursor = dda.precursors[entry["precursor"]]
        via_property = precursor.peaks
        via_helper = get_mobility_collapsed_spectrum(
            dda.timsdata, [tuple(r) for r in entry["scan_ranges"]]
        )
    np.testing.assert_allclose(via_property, via_helper)


def test_collapsed_spectrum_conserves_total_intensity() -> None:
    """Merging redistributes intensity between peaks but must not create or lose it."""
    if not DDA_PATH.is_dir():
        pytest.skip("Test data not found")
    entry = GOLDEN["dda_precursors"][0]
    ranges = [tuple(r) for r in entry["scan_ranges"]]
    with timsdata_connect(str(DDA_PATH)) as td:
        raw_total = sum(
            int(intensities.sum())
            for frame_id, begin, end in ranges
            for _, intensities in td.readScans(frame_id, begin, end)
        )
        peaks = get_mobility_collapsed_spectrum(td, ranges)
    assert peaks[:, 1].sum() == pytest.approx(raw_total)


def test_empty_scan_ranges_return_empty_spectrum() -> None:
    if not DDA_PATH.is_dir():
        pytest.skip("Test data not found")
    with timsdata_connect(str(DDA_PATH)) as td:
        assert get_mobility_collapsed_spectrum(td, []).shape == (0, 2)
