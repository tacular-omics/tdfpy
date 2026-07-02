"""Tests for tdfpy.pipeline: subset_scans, convert, and error paths."""

from collections.abc import Generator
import pathlib

import numpy as np
import pytest

from tdfpy import (
    RawSpectrum,
    convert,
    read_spectrum,
    subset_scans,
    timsdata,
)

TDF_PATH = "tests/data/example_dda.d"


def _small_spectrum() -> RawSpectrum:
    return RawSpectrum(
        scan_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
        mz_indices=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        intensities=np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
        num_scans=5,
    )


# ---------------------------------------------------------------------------
# subset_scans — pure Python
# ---------------------------------------------------------------------------


def test_subset_scans_half_open_range() -> None:
    spectrum = _small_spectrum()
    subset = subset_scans(spectrum, scan_num_begin=1, scan_num_end=3)
    # scans 1 and 2 kept (end is exclusive); num_scans preserved.
    np.testing.assert_array_equal(subset.scan_indices, [1, 2])
    assert subset.num_scans == 5


def test_subset_scans_empty_spectrum_passes_through() -> None:
    empty = RawSpectrum.empty_like(5)
    assert subset_scans(empty, scan_num_begin=0, scan_num_end=3).empty


def test_subset_scans_negative_begin_raises() -> None:
    with pytest.raises(ValueError, match="Invalid scan range"):
        subset_scans(_small_spectrum(), scan_num_begin=-1, scan_num_end=3)


def test_subset_scans_end_before_begin_raises() -> None:
    with pytest.raises(ValueError, match="Invalid scan range"):
        subset_scans(_small_spectrum(), scan_num_begin=4, scan_num_end=2)


# ---------------------------------------------------------------------------
# convert — empty input is pure Python; other branches need the DLL fixture
# ---------------------------------------------------------------------------


def test_convert_empty_returns_empty_3col() -> None:
    empty = RawSpectrum.empty_like(5)
    out = convert(empty, td=None, frame_id=1)  # type: ignore[arg-type]
    assert out.shape == (0, 3)


@pytest.fixture(scope="module")
def td() -> Generator[timsdata.TimsData, None, None]:
    if not pathlib.Path(TDF_PATH).exists():
        pytest.skip("Test data not found")
    with timsdata.timsdata_connect(TDF_PATH) as conn:
        yield conn


@pytest.fixture(scope="module")
def ms1_frame_id(td: timsdata.TimsData) -> int:
    cursor = td.conn.cursor()
    cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1")
    return int(cursor.fetchone()[0])


@pytest.fixture(scope="module")
def ms1_spectrum(td: timsdata.TimsData, ms1_frame_id: int) -> RawSpectrum:
    spectrum = read_spectrum(td, ms1_frame_id)
    if spectrum.empty:
        pytest.skip("First MS1 frame has no peaks")
    return spectrum


def test_convert_ook0_shape_and_positive(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    out = convert(ms1_spectrum, td, ms1_frame_id, ion_mobility_type="ook0")
    assert out.shape == (ms1_spectrum.num_peaks, 3)
    assert np.all(out[:, 0] > 0)  # m/z
    assert np.all(np.isfinite(out[:, 2]))  # 1/K0


def test_convert_ccs_differs_from_ook0(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    ook0 = convert(ms1_spectrum, td, ms1_frame_id, ion_mobility_type="ook0")
    ccs = convert(ms1_spectrum, td, ms1_frame_id, ion_mobility_type="ccs")
    assert ccs.shape == ook0.shape
    # CCS is a different physical quantity; the mobility column must change.
    assert not np.allclose(ccs[:, 2], ook0[:, 2])


def test_convert_voltage_matches_scannum_mapping(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    # Regression for the fixed bug: the voltage branch must map per-peak *scan
    # numbers* (not 1/K0 values) through scanNumToVoltage. Verify each peak's
    # voltage equals the ground-truth voltage for its scan number. Feeding 1/K0
    # values in (the old bug) produced entirely different numbers.
    out = convert(ms1_spectrum, td, ms1_frame_id, ion_mobility_type="voltage")
    assert out.shape == (ms1_spectrum.num_peaks, 3)
    expected = np.asarray(
        td.scanNumToVoltage(ms1_frame_id, ms1_spectrum.scan_indices)
    )
    np.testing.assert_allclose(out[:, 2], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# read_spectrum error paths
# ---------------------------------------------------------------------------


def test_read_spectrum_unknown_frame_lists_valid_range(td: timsdata.TimsData) -> None:
    with pytest.raises(ValueError, match=r"valid frame IDs"):
        read_spectrum(td, 10_000_000)
