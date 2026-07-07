"""Tests for tdfpy.noise.gates — selection-polygon and diaPASEF MS1 window gates."""

from collections.abc import Generator
import pathlib

import numpy as np
import pytest

from tdfpy import DiaMs1WindowGate, SelectionPolygonGate, timsdata
from tdfpy.noise.gates import (
    PerScanTofIntervals,
    build_polygon_intervals,
    build_window_intervals,
    read_dia_ms1_boxes,
    read_selection_polygon,
)

DDA_PATH = "tests/data/example_dda.d"
DIA_PATH = "tests/data/example_dia.d"

# Identity converters: TOF index == m/z, scan == 1/K0 — lets tests reason in
# polygon coordinates directly (mirrors dnoise's polygon.rs unit tests).
_IDENTITY = staticmethod(lambda a: np.asarray(a, dtype=np.float64))


def _square() -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned square (10,10)-(90,90) in (m/z, 1/K0)."""
    return np.array([10.0, 90.0, 90.0, 10.0]), np.array([10.0, 10.0, 90.0, 90.0])


def _square_gate(mz_pad: float = 0.0, im_pad: float = 0.0) -> PerScanTofIntervals:
    mz, im = _square()
    g = build_polygon_intervals(
        mz, im, np.arange(100.0), lambda a: np.asarray(a), mz_pad=mz_pad, im_pad=im_pad
    )
    assert g is not None
    return g


# ---------------------------------------------------------------------------
# Polygon geometry (pure, no TimsData)
# ---------------------------------------------------------------------------


def test_degenerate_polygon_yields_none():
    assert (
        build_polygon_intervals(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.arange(100.0), _IDENTITY
        )
        is None
    )


def test_square_keeps_inside_drops_outside():
    g = _square_gate()
    assert g.contains(50, 50)  # center
    assert g.contains(10, 10)  # boundary corner kept
    assert g.contains(89, 89)  # just inside upper-right
    assert not g.contains(50, 9)  # left of the m/z span
    assert not g.contains(50, 91)  # right of the m/z span
    assert not g.contains(5, 50)  # below the mobility range
    assert not g.contains(95, 50)  # above it


def test_mz_pad_widens_span():
    g = _square_gate(mz_pad=5.0)
    assert g.contains(50, 7)  # within the left pad
    assert g.contains(50, 94)  # within the right pad
    assert not g.contains(50, 4)  # beyond the pad


def test_im_pad_widens_mobility_band():
    g = _square_gate(im_pad=3.0)
    assert g.contains(8, 50)  # below the nominal mobility edge, within pad
    assert g.contains(92, 50)  # within the upper pad
    assert not g.contains(5, 50)  # beyond the pad


def test_concave_polygon_gives_two_spans_on_a_scan():
    # A "U": full base with a notch (m/z 30..70) cut from the top down to 1/K0 30.
    mz = np.array([0.0, 100.0, 100.0, 70.0, 70.0, 30.0, 30.0, 0.0])
    im = np.array([0.0, 0.0, 100.0, 100.0, 30.0, 30.0, 100.0, 0.0])
    g = build_polygon_intervals(mz, im, np.arange(101.0), _IDENTITY)
    assert g is not None
    assert g.contains(50, 15)  # left arm
    assert g.contains(50, 85)  # right arm
    assert not g.contains(50, 50)  # in the notch
    assert g.contains(10, 50)  # below the notch floor: one solid span


def test_keep_mask_matches_contains():
    g = _square_gate()
    scan = np.array([50, 50, 5])
    tof = np.array([50, 95, 50])
    np.testing.assert_array_equal(
        g.keep_mask(scan, tof), np.array([True, False, False])
    )


def test_keep_mask_empty_input():
    assert _square_gate().keep_mask(np.empty(0, int), np.empty(0, int)).shape == (0,)


# ---------------------------------------------------------------------------
# Window intervals (pure)
# ---------------------------------------------------------------------------


def test_window_box_membership():
    g = build_window_intervals([(10, 20, 1000, 2000)], num_scans=100)
    assert g is not None
    assert g.contains(15, 1500)  # center
    assert g.contains(10, 1000) and g.contains(20, 2000)  # inclusive corners
    assert not g.contains(15, 999)
    assert not g.contains(9, 1500)
    assert not g.contains(21, 1500)


def test_window_empty_yields_none():
    assert build_window_intervals([], num_scans=100) is None


def test_window_overlapping_boxes_merge():
    g = build_window_intervals(
        [(10, 20, 1000, 1500), (12, 18, 1490, 2000)], num_scans=100
    )
    assert g is not None
    assert g.contains(15, 1495)  # seam between the two TOF ranges is bridged
    assert g.contains(15, 1000) and g.contains(15, 2000)
    assert not g.contains(15, 2500)


def test_window_keep_mask_matches_contains():
    g = build_window_intervals([(10, 20, 1000, 2000)], num_scans=100)
    assert g is not None
    scan = np.array([15, 15, 9])
    tof = np.array([1500, 3000, 1500])
    np.testing.assert_array_equal(
        g.keep_mask(scan, tof), np.array([True, False, False])
    )


def test_window_negative_scan_lo_clamps_without_wrapping():
    # A negative scan_lo must clamp to 0 (not negative-index into the high end).
    g = build_window_intervals([(-2, 3, 1000, 2000)], num_scans=200)
    assert g is not None
    assert g.contains(0, 1500) and g.contains(3, 1500)  # intended low scans
    assert not g.contains(198, 1500) and not g.contains(199, 1500)  # no wrap


def test_window_box_wholly_below_range_is_skipped():
    # scan_hi < 0 means the box lies entirely below the scan axis -> no rows set.
    assert build_window_intervals([(-5, -1, 1000, 2000)], num_scans=100) is None


def test_keep_mask_handles_unsorted_scan_indices():
    # The fast path assumes sorted input; out-of-order input must still be correct.
    g = build_window_intervals([(10, 20, 1000, 2000)], num_scans=100)
    assert g is not None
    scan = np.array([9, 15, 15])  # descending-then-equal, not sorted ascending
    tof = np.array([1500, 3000, 1500])
    np.testing.assert_array_equal(
        g.keep_mask(scan, tof), np.array([False, False, True])
    )


# ---------------------------------------------------------------------------
# TDF metadata readers (fixtures)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dia_td() -> Generator[timsdata.TimsData, None, None]:
    if not pathlib.Path(DIA_PATH).exists():
        pytest.skip("DIA test data not found")
    with timsdata.timsdata_connect(DIA_PATH) as td:
        yield td


@pytest.fixture(scope="module")
def dda_td() -> Generator[timsdata.TimsData, None, None]:
    if not pathlib.Path(DDA_PATH).exists():
        pytest.skip("DDA test data not found")
    with timsdata.timsdata_connect(DDA_PATH) as td:
        yield td


def test_read_dia_boxes_on_dia(dia_td):
    boxes = read_dia_ms1_boxes(dia_td)
    assert len(boxes) > 0
    for scan_begin, scan_end, mz_lo, mz_hi in boxes:
        assert scan_end > scan_begin
        assert mz_hi > mz_lo


def test_read_dia_boxes_empty_on_dda(dda_td):
    assert read_dia_ms1_boxes(dda_td) == []


def test_read_polygon_none_on_dda(dda_td):
    # This DDA fixture stores no IMS PolygonFilter; the gate must no-op, not crash.
    assert read_selection_polygon(dda_td) is None


# ---------------------------------------------------------------------------
# Gates end-to-end (as NoiseFilters)
# ---------------------------------------------------------------------------


def _first_ms1_frame(td) -> int:
    cur = td.conn.cursor()
    cur.execute("SELECT Id FROM Frames WHERE MsMsType=0 ORDER BY Id LIMIT 1")
    return int(cur.fetchone()[0])


def test_dia_ms1_gate_drops_out_of_window_points(dia_td):
    from tdfpy import read_spectrum

    fid = _first_ms1_frame(dia_td)
    spec = read_spectrum(dia_td, fid)
    gate = DiaMs1WindowGate()
    mask = gate.keep_mask(
        spec.scan_indices,
        spec.mz_indices,
        spec.intensities,
        num_scans=spec.num_scans,
        td=dia_td,
        frame_id=fid,
    )
    # The gate should actually remove something but not everything.
    assert 0 < int(mask.sum()) < mask.size
    # Every kept point must fall inside the built window gate; every dropped
    # point must fall outside it (mask is exactly the gate membership).
    from tdfpy.noise.gates import _build_dia_ms1_gate

    built = _build_dia_ms1_gate(dia_td, fid, spec.num_scans, gate)
    assert built is not None
    np.testing.assert_array_equal(
        mask, built.keep_mask(spec.scan_indices, spec.mz_indices)
    )


def test_dia_ms1_gate_is_noop_on_ms2_frame(dia_td):
    # The gate encodes MS1 precursor selection; on an MS2 frame it must no-op
    # (keep all) rather than testing fragment peaks against the precursor region.
    from tdfpy import read_spectrum

    cur = dia_td.conn.cursor()
    row = cur.execute(
        "SELECT Id FROM Frames WHERE MsMsType=9 ORDER BY Id LIMIT 1"
    ).fetchone()
    if row is None:
        pytest.skip("no MS2 frame in fixture")
    fid = int(row[0])
    spec = read_spectrum(dia_td, fid)
    mask = DiaMs1WindowGate().keep_mask(
        spec.scan_indices,
        spec.mz_indices,
        spec.intensities,
        num_scans=spec.num_scans,
        td=dia_td,
        frame_id=fid,
    )
    assert mask.all()


def test_dia_ms1_gate_padding_keeps_more(dia_td):
    from tdfpy import read_spectrum

    fid = _first_ms1_frame(dia_td)
    spec = read_spectrum(dia_td, fid)

    def kept(gate):
        return int(
            gate.keep_mask(
                spec.scan_indices,
                spec.mz_indices,
                spec.intensities,
                num_scans=spec.num_scans,
                td=dia_td,
                frame_id=fid,
            ).sum()
        )

    tight = kept(DiaMs1WindowGate(mz_pad=0.0, im_pad=0.0))
    padded = kept(DiaMs1WindowGate(mz_pad=20.0, im_pad=0.2))
    assert padded >= tight


def test_dia_ms1_gate_is_noop_on_dda(dda_td):
    from tdfpy import read_spectrum

    fid = _first_ms1_frame(dda_td)
    spec = read_spectrum(dda_td, fid)
    mask = DiaMs1WindowGate().keep_mask(
        spec.scan_indices,
        spec.mz_indices,
        spec.intensities,
        num_scans=spec.num_scans,
        td=dda_td,
        frame_id=fid,
    )
    assert mask.all()  # no windows → keep everything


def test_polygon_gate_filters_with_injected_polygon(dda_td, monkeypatch):
    # This DDA fixture stores no polygon, so inject a synthetic one spanning
    # m/z 500..800 across the full mobility range to exercise the real
    # conversion path (td.mzToIndex / scanNumToOneOverK0) end-to-end.
    from tdfpy import read_spectrum
    from tdfpy.noise import gates as gates_mod

    poly = (
        np.array([500.0, 800.0, 800.0, 500.0]),
        np.array([0.5, 0.5, 1.7, 1.7]),
    )
    monkeypatch.setattr(gates_mod, "read_selection_polygon", lambda td: poly)
    gates_mod._GATE_CACHE.pop(dda_td, None)  # avoid a cached "no polygon" result

    fid = _first_ms1_frame(dda_td)
    spec = read_spectrum(dda_td, fid)
    mask = SelectionPolygonGate().keep_mask(
        spec.scan_indices,
        spec.mz_indices,
        spec.intensities,
        num_scans=spec.num_scans,
        td=dda_td,
        frame_id=fid,
    )
    assert 0 < int(mask.sum()) < mask.size
    # Every kept point's m/z must lie within the polygon's m/z band (±1 TOF
    # index of rounding slack, converted back to m/z).
    kept_mz = np.asarray(dda_td.indexToMz(fid, spec.mz_indices[mask]))
    assert kept_mz.min() > 495.0 and kept_mz.max() < 805.0
    gates_mod._GATE_CACHE.pop(dda_td, None)  # don't leak the injected gate


def test_polygon_gate_is_noop_without_polygon(dda_td):
    from tdfpy import read_spectrum

    fid = _first_ms1_frame(dda_td)
    spec = read_spectrum(dda_td, fid)
    mask = SelectionPolygonGate().keep_mask(
        spec.scan_indices,
        spec.mz_indices,
        spec.intensities,
        num_scans=spec.num_scans,
        td=dda_td,
        frame_id=fid,
    )
    assert mask.all()
