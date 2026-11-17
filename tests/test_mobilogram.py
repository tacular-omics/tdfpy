"""Extracted ion mobilograms: summed intensity in an m/z window against mobility.

The invariants, in order of strength: the summed intensity equals an independent
per-peak loop over the same frames, the apex of a precursor's own mobilogram sits
within a few scans of the ``Precursor.ook0`` Bruker derived, and the axis values
are exact scan conversions rather than interpolations.
"""

import numpy as np
import pytest

import tdfpy
from tdfpy import (
    DDA,
    DIA,
    PRM,
    ReaderClosedError,
    TdfpyError,
    ccs_to_ook0,
    extract_mobilogram,
    extract_mobilograms,
    get_raw_peaks,
)

DDA_PATH = "tests/data/example_dda.d"
DIA_PATH = "tests/data/example_dia.d"
PRM_PATH = "tests/data/example_prm.d"

#: A strong, isolated precursor of the DDA fixture.
DDA_MZ = 1292.6314


def _reference_total(td: tdfpy.TimsData, rows: list[tuple[int, int, int]], mz: float, ppm: float) -> float:
    """Summed intensity of every raw peak in the m/z window, by a different route."""
    low, high = mz * (1 - ppm / 1e6), mz * (1 + ppm / 1e6)
    total = 0.0
    for frame_id, begin, end in rows:
        peaks = get_raw_peaks(td, frame_id, scan_range=(begin, end))
        if len(peaks):
            total += float(peaks[(peaks[:, 0] >= low) & (peaks[:, 0] <= high), 1].sum())
    return total


def test_sum_rule_matches_an_independent_peak_loop() -> None:
    with DDA(DDA_PATH) as dda:
        td = dda.timsdata
        frames = [frame.frame_id for frame in dda.ms1][:10]
        mobilogram = extract_mobilogram(td, frames, mz=DDA_MZ, mz_tolerance=20.0)
        num_scans = td.frame_metadata(frames[0]).num_scans
        expected = _reference_total(td, [(frame_id, 0, num_scans) for frame_id in frames], DDA_MZ, 20.0)
        assert len(mobilogram) > 10, f"only {len(mobilogram)} rows, too few to prove anything"
        assert expected > 0, "the reference loop found no signal, so the comparison is vacuous"
        assert float(mobilogram[:, 1].sum()) == expected
        assert np.all(np.diff(mobilogram[:, 0]) > 0), "the axis must be ascending and have no repeated value"


def test_apex_agrees_with_the_precursor_ook0() -> None:
    """Bruker's own 1/K0 for a precursor must land on its mobility peak."""
    with DDA(DDA_PATH) as dda:
        td = dda.timsdata
        step = abs(float(np.diff(np.asarray(td.scan_num_to_ook0(1, np.array([0, 1]))))[0]))
        checked = 0
        for precursor in dda.precursors:
            if precursor.intensity is None or precursor.intensity < 5000:
                continue
            mobilogram = extract_mobilogram(td, [precursor.parent_frame_id], mz=precursor.largest_peak_mz, mz_tolerance=30.0)
            if len(mobilogram) < 3:
                continue
            apex = float(mobilogram[np.argmax(mobilogram[:, 1]), 0])
            assert abs(apex - precursor.ook0) < 5 * step, (
                f"precursor {precursor.precursor_id}: apex {apex} is {abs(apex - precursor.ook0) / step:.1f} scans from ook0 {precursor.ook0}"
            )
            checked += 1
            if checked == 4:
                break
        assert checked == 4, f"only {checked} precursors were strong enough to check"


def test_axis_values_are_exact_scan_conversions() -> None:
    with DDA(DDA_PATH) as dda:
        td = dda.timsdata
        frame_id = next(iter(dda.ms1)).frame_id
        mobilogram = extract_mobilogram(td, [frame_id], mz=DDA_MZ, mz_tolerance=30.0)
        grid = np.asarray(td.scan_num_to_ook0(frame_id, np.arange(td.frame_metadata(frame_id).num_scans)))
        assert len(mobilogram)
        assert np.isin(mobilogram[:, 0], grid).all(), "an axis value is not a scan of this frame"


def test_multi_target_equals_separate_calls() -> None:
    targets = [DDA_MZ, 700.8437, 565.3207]
    with DDA(DDA_PATH) as dda:
        td = dda.timsdata
        frames = [frame.frame_id for frame in dda.ms1][:5]
        together = extract_mobilograms(td, frames, targets=targets, mz_tolerance=30.0)
        apart = [extract_mobilogram(td, frames, mz=mz, mz_tolerance=30.0) for mz in targets]
        assert sum(len(m) for m in together) > 0
        for one, other in zip(together, apart, strict=True):
            np.testing.assert_array_equal(one, other)


def test_units_convert_the_axis_and_nothing_else() -> None:
    with DDA(DDA_PATH) as dda:
        td = dda.timsdata
        frames = [frame.frame_id for frame in dda.ms1][:5]
        ook0 = extract_mobilogram(td, frames, mz=DDA_MZ, mz_tolerance=30.0)
        ccs = extract_mobilogram(td, frames, mz=DDA_MZ, mz_tolerance=30.0, mobility_type="ccs", charge=2)
        voltage = extract_mobilogram(td, frames, mz=DDA_MZ, mz_tolerance=30.0, mobility_type="voltage")
        assert len(ook0) > 5
        for other in (ccs, voltage):
            assert other.shape == ook0.shape
            np.testing.assert_array_equal(np.sort(other[:, 1]), np.sort(ook0[:, 1]))
        back_from_ccs = np.array([ccs_to_ook0(float(value), 2, DDA_MZ) for value in ccs[:, 0]])
        np.testing.assert_allclose(np.sort(back_from_ccs), ook0[:, 0], rtol=1e-12)
        back_from_voltage = td.scan_num_to_ook0(frames[0], td.voltage_to_scan_num(frames[0], voltage[:, 0]))
        np.testing.assert_allclose(np.sort(np.asarray(back_from_voltage)), ook0[:, 0], rtol=1e-12)


def test_no_signal_gives_an_empty_array() -> None:
    with DDA(DDA_PATH) as dda:
        empty = extract_mobilogram(dda.timsdata, [frame.frame_id for frame in dda.ms1][:5], mz=1699.9, mz_tolerance=1.0)
        assert empty.shape == (0, 2)
        assert empty.dtype == np.float64


def test_closed_reader_raises() -> None:
    with DDA(DDA_PATH) as dda:
        td = dda.timsdata
        frames = [frame.frame_id for frame in dda.ms1][:2]
    with pytest.raises(ReaderClosedError):
        extract_mobilogram(td, frames, mz=DDA_MZ)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"mz": DDA_MZ, "mobility_type": "ccs"}, "needs charge"),
        ({"mz": DDA_MZ, "mobility_type": "fraction"}, "mobility_type"),
        ({"mz": DDA_MZ, "mz_tolerance": -1.0}, "mz_tolerance"),
        ({"mz": DDA_MZ, "mz_tolerance_unit": "ppb"}, "mz_tolerance_unit"),
        ({"mz": DDA_MZ, "ook0_bin_width": 0.0}, "ook0_bin_width"),
    ],
)
def test_bad_arguments_raise(kwargs: dict, match: str) -> None:
    with DDA(DDA_PATH) as dda:
        with pytest.raises(TdfpyError, match=match):
            extract_mobilogram(dda.timsdata, [next(iter(dda.ms1)).frame_id], **kwargs)


def test_selection_must_be_frames_or_windows() -> None:
    with DDA(DDA_PATH) as dda:
        with pytest.raises(TdfpyError, match="selection is empty"):
            extract_mobilogram(dda.timsdata, [], mz=DDA_MZ)
        with pytest.raises(TdfpyError, match="frame IDs or DIA windows"):
            extract_mobilogram(dda.timsdata, ["1"], mz=DDA_MZ)  # type: ignore[list-item]
        with pytest.raises(TdfpyError, match="targets is empty"):
            extract_mobilograms(dda.timsdata, [next(iter(dda.ms1)).frame_id], targets=[])


def test_reader_mobilogram_matches_the_function() -> None:
    with DDA(DDA_PATH) as dda:
        frames = [frame.frame_id for frame in dda.ms1.query_range(rt_range=(2400.0, 2410.0))]
        assert frames, "the RT window selected no frames"
        expected = extract_mobilogram(dda.timsdata, frames, mz=DDA_MZ, mz_tolerance=30.0)
        from_reader = dda.mobilogram(mz=DDA_MZ, rt_range=(2400.0, 2410.0), mz_tolerance=30.0)
        np.testing.assert_array_equal(from_reader, expected)
        np.testing.assert_array_equal(dda.mobilogram(mz=DDA_MZ, frame_ids=frames, mz_tolerance=30.0), expected)
        for kwargs in ({}, {"rt_range": (2400.0, 2410.0), "frame_ids": frames}):
            with pytest.raises(TdfpyError, match="exactly one of them"):
                dda.mobilogram(mz=DDA_MZ, **kwargs)  # type: ignore[arg-type]


def test_dia_windows_restrict_the_axis_to_the_window() -> None:
    with DIA(DIA_PATH) as dia:
        td = dia.timsdata
        # One window group repeats across frames at more than one scan position in
        # diaPASEF, so keep the windows that share a scan range to get a sharp bound.
        group = [window for window in dia.windows if window.window_group_id == 1]
        scan_range = group[0].scan_num_range
        windows = [window for window in group if window.scan_num_range == scan_range][:10]
        mobilogram = extract_mobilogram(td, windows, mz=662.7009, mz_tolerance=30.0)
        low, high = windows[0].ook0_range
        assert len(mobilogram) > 1, "the DIA window selection produced too little to check"
        assert mobilogram[:, 0].min() >= low and mobilogram[:, 0].max() <= high
        scans = td.frame_metadata(windows[0].frame_id).num_scans
        frame_low, frame_high = (float(value) for value in td.scan_num_to_ook0(windows[0].frame_id, np.array([scans - 1, 0])))
        assert high - low < (frame_high - frame_low) / 2, "the window must cover far less mobility than its frame"
        rows = [(window.frame_id, window.scan_num_begin, window.scan_num_end) for window in windows]
        assert float(mobilogram[:, 1].sum()) == _reference_total(td, rows, 662.7009, 30.0)


def test_mixed_calibrations_need_a_bin_width() -> None:
    """One PRM run carries two TIMS calibrations, which share no scan grid."""
    with PRM(PRM_PATH) as prm:
        td = prm.timsdata
        by_calibration: dict[int, list[int]] = {}
        for frame in prm.ms1:
            by_calibration.setdefault(td.frame_metadata(frame.frame_id).tims_calibration_id, []).append(frame.frame_id)
        assert len(by_calibration) == 2, f"this fixture is meant to mix calibrations, found {sorted(by_calibration)}"
        first, second = (by_calibration[key][:20] for key in sorted(by_calibration))
        mixed = first + second
        mz = 662.962

        with pytest.raises(TdfpyError, match="mixes 2 TIMS calibrations"):
            extract_mobilogram(td, mixed, mz=mz, mz_tolerance=40.0)
        with pytest.raises(TdfpyError, match="needs one TIMS calibration"):
            extract_mobilogram(td, mixed, mz=mz, mz_tolerance=40.0, mobility_type="voltage", ook0_bin_width=0.01)

        width = 0.01
        binned = extract_mobilogram(td, mixed, mz=mz, mz_tolerance=40.0, ook0_bin_width=width)
        rows = [(frame_id, 0, td.frame_metadata(frame_id).num_scans) for frame_id in mixed]
        expected = _reference_total(td, rows, mz, 40.0)
        assert expected > 0, "no signal in the mixed selection, so this proves nothing"
        assert float(binned[:, 1].sum()) == expected
        remainders = binned[:, 0] / width - np.floor(binned[:, 0] / width)
        np.testing.assert_allclose(remainders, 0.5, atol=1e-9)

        single = extract_mobilogram(td, first, mz=mz, mz_tolerance=40.0)
        assert float(single[:, 1].sum()) == _reference_total(td, [(f, 0, td.frame_metadata(f).num_scans) for f in first], mz, 40.0)


def test_a_da_tolerance_wider_than_the_target_mz_still_matches() -> None:
    """``mz - tolerance`` below zero has no TOF index, and a NaN bound matches nothing.

    The calibration takes a square root of the m/z, so a negative low edge came
    back NaN and every comparison against it was false: the window silently
    matched nothing instead of starting at the bottom of the spectrum.
    """
    with DDA(DDA_PATH) as reader:
        td = reader.timsdata
        inside = extract_mobilogram(td, [1], mz=300.0, mz_tolerance=300.0, mz_tolerance_unit="da")
        assert len(inside), "the 300 Da window is already empty, so the wider ones prove nothing"
        for tolerance in (301.0, 500.0, 5000.0):
            wider = extract_mobilogram(td, [1], mz=300.0, mz_tolerance=tolerance, mz_tolerance_unit="da")
            assert len(wider), f"a {tolerance} Da window around m/z 300 matched nothing: the low edge went negative"
            assert len(wider) >= len(inside), f"{tolerance} Da gave {len(wider)} rows, fewer than the 300 Da window's {len(inside)}"
            assert float(wider[:, 1].sum()) >= float(inside[:, 1].sum())


@pytest.mark.parametrize(("mz", "label"), [(300.0, "a window with signal"), (9000.0, "a window with none")])
def test_charge_zero_with_ccs_raises_whether_or_not_there_is_signal(mz: float, label: str) -> None:
    """CCS is undefined at charge 0, and the refusal must not depend on finding peaks.

    The conversion ran only on a non-empty result, so an empty window returned
    ``(0, 2)`` and a populated one raised a bare ``ValueError`` from the
    calibration rather than this package's own error.
    """
    with DDA(DDA_PATH) as reader:
        td = reader.timsdata
        with pytest.raises(TdfpyError, match="charge=0"):
            extract_mobilogram(td, [1], mz=mz, mz_tolerance=300.0, mz_tolerance_unit="da", mobility_type="ccs", charge=0)


def test_ccs_ignores_the_sign_of_the_charge() -> None:
    """What the charge-0 refusal claims: only the magnitude reaches the conversion."""
    with DDA(DDA_PATH) as reader:
        td = reader.timsdata
        options = {"mz": 300.0, "mz_tolerance": 300.0, "mz_tolerance_unit": "da", "mobility_type": "ccs"}
        positive = extract_mobilogram(td, [1], charge=2, **options)  # type: ignore[arg-type]
        negative = extract_mobilogram(td, [1], charge=-2, **options)  # type: ignore[arg-type]
        assert len(positive), "no signal in the window, so this proves nothing"
        np.testing.assert_array_equal(positive, negative)
