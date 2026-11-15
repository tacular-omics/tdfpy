"""``mobility_type`` chooses the reported unit, never the peak list.

The default centroider used to convert before merging, so a relative
``im_tolerance`` meant a different window in TIMS voltage (affine in scan
number) or in CCS (scaled per peak) than in 1/K0, and the same frame came back
as a different spectrum. Clustering now always runs in 1/K0 and only the
reported column is converted.
"""

import numpy as np
import pytest

import tdfpy
from tdfpy import (
    DDA,
    DIA,
    PRM,
    MergePeaksCentroider,
    WatershedCentroider,
    ccs_to_ook0,
    ook0_to_ccs,
)

DDA_PATH = "tests/data/example_dda.d"
DIA_PATH = "tests/data/example_dia.d"
PRM_PATH = "tests/data/example_prm.d"
UNITS = ("ook0", "ccs", "voltage")


def _assert_same_peaks(by_unit: dict[str, np.ndarray], label: str) -> None:
    """Every unit must give the same rows, differing only in the third column."""
    base = by_unit["ook0"]
    assert len(base), f"{label}: the 1/K0 spectrum is empty, so this proves nothing"
    for unit, peaks in by_unit.items():
        assert peaks.shape == base.shape, f"{label}: {unit} has shape {peaks.shape}, 1/K0 has {base.shape}"
        np.testing.assert_array_equal(peaks[:, 0], base[:, 0], err_msg=f"{label}: {unit} m/z column differs")
        np.testing.assert_array_equal(peaks[:, 1], base[:, 1], err_msg=f"{label}: {unit} intensity column differs")
        assert np.isfinite(peaks[:, 2]).all(), f"{label}: {unit} mobility column is not finite"


def test_ms1_peak_list_is_the_same_in_every_mobility_unit() -> None:
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        _assert_same_peaks({unit: frame.centroid(mobility_type=unit) for unit in UNITS}, f"DDA frame {frame.frame_id}")


def test_ms2_peak_list_is_the_same_in_every_mobility_unit() -> None:
    """Both centroiders, on an MS/MS window and on a PRM transition.

    The PRM fixture's transitions hold too few points for the greedy merger's
    ``min_peaks`` to keep any centroid, so that pair is covered by the
    watershed centroider only.
    """
    for centroider in (MergePeaksCentroider(), WatershedCentroider()):
        name = type(centroider).__name__
        with DIA(DIA_PATH) as dia:
            window = next(iter(dia.windows))
            by_unit = {unit: window.centroid(mobility_type=unit, centroid=centroider) for unit in UNITS}
            _assert_same_peaks(by_unit, f"{name}, DIA window of frame {window.frame_id}")
    with PRM(PRM_PATH) as prm:
        transition = next(iter(prm.transitions))
        by_unit = {unit: transition.centroid(mobility_type=unit, centroid=WatershedCentroider()) for unit in UNITS}
        _assert_same_peaks(by_unit, f"WatershedCentroider, PRM transition of frame {transition.frame_id}")


@pytest.mark.slow
def test_every_ms1_frame_of_every_fixture_is_unit_independent() -> None:
    """The exhaustive form of the test above: every MS1 frame, every fixture."""
    checked = 0
    for reader, path in ((DDA, DDA_PATH), (DIA, DIA_PATH), (PRM, PRM_PATH)):
        with reader(path) as run:
            for frame in run.ms1:
                by_unit = {unit: frame.centroid(mobility_type=unit) for unit in UNITS}
                if not len(by_unit["ook0"]):
                    continue  # an empty frame cannot show a difference
                _assert_same_peaks(by_unit, f"{reader.__name__} frame {frame.frame_id}")
                checked += 1
    assert checked >= 90, f"only {checked} non-empty MS1 frames were compared"


@pytest.mark.slow
def test_watershed_ms1_frames_are_unit_independent() -> None:
    for reader, path in ((DDA, DDA_PATH), (DIA, DIA_PATH)):
        with reader(path) as run:
            frame = next(iter(run.ms1))
            by_unit = {unit: frame.centroid(mobility_type=unit, centroid=WatershedCentroider()) for unit in UNITS}
            _assert_same_peaks(by_unit, f"WatershedCentroider, {reader.__name__} frame {frame.frame_id}")


def test_reported_columns_convert_back_to_the_merged_ook0_column() -> None:
    """CCS and voltage are the merged 1/K0 in another unit, invertibly so."""
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        ook0 = frame.centroid()
        ccs = frame.centroid(mobility_type="ccs")
        voltage = frame.centroid(mobility_type="voltage")
        sample = slice(0, 200)
        back_from_ccs = np.array([ccs_to_ook0(float(value), 1, float(mz)) for value, mz in zip(ccs[sample, 2], ccs[sample, 0], strict=True)])
        np.testing.assert_allclose(back_from_ccs, ook0[sample, 2], rtol=1e-12)
        back_from_voltage = dda.timsdata.scan_num_to_ook0(frame.frame_id, dda.timsdata.voltage_to_scan_num(frame.frame_id, voltage[:, 2]))
        np.testing.assert_allclose(back_from_voltage, ook0[:, 2], rtol=1e-12)


def test_ccs_column_equals_the_scalar_helper_applied_to_the_ook0_column() -> None:
    """The array conversion and the public scalar function must not drift apart."""
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        for label, ook0, ccs in (
            ("raw peaks", frame.raw_peaks(), frame.raw_peaks(mobility_type="ccs")),
            ("centroids", frame.centroid(), frame.centroid(mobility_type="ccs")),
        ):
            sample = slice(0, 2000)
            expected = np.array([ook0_to_ccs(float(value), 1, float(mz)) for value, mz in zip(ook0[sample, 2], ook0[sample, 0], strict=True)])
            np.testing.assert_array_equal(ccs[sample, 2], expected, err_msg=f"{label}: CCS column is not ook0_to_ccs of the 1/K0 column")


def test_ccs_conversion_does_not_call_the_scalar_helper_per_peak(monkeypatch: pytest.MonkeyPatch) -> None:
    """A whole spectrum converts in one NumPy expression, not one call per peak."""
    calls = 0

    def counting(ook0: float, charge: int, mz: float) -> float:
        nonlocal calls
        calls += 1
        return ook0_to_ccs(ook0, charge, mz)

    # raising=False: the pipeline no longer imports the scalar helper at all.
    monkeypatch.setattr(tdfpy.pipeline, "ook0_to_ccs", counting, raising=False)
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        ccs = frame.raw_peaks(mobility_type="ccs")
    assert len(ccs) > 100_000, f"the frame gave only {len(ccs)} peaks, too few to show anything"
    assert (ccs[:, 2] > 0).all()
    assert calls == 0, f"the CCS column cost {calls} Python calls for {len(ccs)} peaks"


def test_ook0_to_ccs_keeps_its_scalar_contract() -> None:
    value = ook0_to_ccs(1.0, 2, 500.0)
    assert type(value) is float
    assert value == ook0_to_ccs(1.0, -2, 500.0)
    assert ccs_to_ook0(value, 2, 500.0) == pytest.approx(1.0, rel=1e-12)
    with pytest.raises(ValueError, match="non-zero"):
        ook0_to_ccs(1.0, 0, 500.0)
    with pytest.raises(ValueError, match="non-zero"):
        ccs_to_ook0(value, 0, 500.0)
