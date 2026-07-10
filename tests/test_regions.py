"""Tests for tdfpy.regions.ChargeStateRegion and pipeline.exclude_region."""

from collections.abc import Generator
import pathlib

import numpy as np
import pytest

from tdfpy import (
    ChargeStateRegion,
    RawSpectrum,
    exclude_region,
    read_spectrum,
    timsdata,
)

TDF_PATH = "tests/data/example_dda.d"


# ---------------------------------------------------------------------------
# __post_init__ validation — pure Python, no fixture needed
# ---------------------------------------------------------------------------


def test_equal_mz_endpoints_raises() -> None:
    with pytest.raises(ValueError, match="differ in m/z"):
        ChargeStateRegion(line=((350.0, 0.7), (350.0, 1.4)))


def test_equal_ook0_endpoints_raises() -> None:
    with pytest.raises(ValueError, match="differ in 1/K0"):
        ChargeStateRegion(line=((350.0, 0.7), (1200.0, 0.7)))


def test_negative_slope_raises() -> None:
    # Higher m/z endpoint has the *lower* 1/K0 → negative slope, unsupported.
    with pytest.raises(ValueError, match="non-negative"):
        ChargeStateRegion(line=((350.0, 1.4), (1200.0, 0.7)))


def test_default_region_constructs() -> None:
    region = ChargeStateRegion()
    assert region.cap_at_upper_endpoint is True
    assert region.line[0][0] < region.line[1][0]


def test_exclude_region_empty_spectrum_short_circuits() -> None:
    # An empty spectrum returns immediately without touching TimsData, so we can
    # safely pass td=None here.
    empty = RawSpectrum.empty_like(10)
    result = exclude_region(empty, ChargeStateRegion(), td=None, frame_id=1)  # type: ignore[arg-type]
    assert result.empty


# ---------------------------------------------------------------------------
# index_cutoff_per_scan / exclude_region — need the real DDA fixture
# ---------------------------------------------------------------------------


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


def test_index_cutoff_shape(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    cutoff = ChargeStateRegion().index_cutoff_per_scan(
        td, ms1_frame_id, ms1_spectrum.num_scans
    )
    assert cutoff.shape == (ms1_spectrum.num_scans,)


def test_index_cutoff_monotonic_in_scan(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    # 1/K0 is monotonic in scan number, so the m/z cutoff — and hence the index
    # cutoff — must be monotonic (one consistent direction) across scans.
    cutoff = ChargeStateRegion().index_cutoff_per_scan(
        td, ms1_frame_id, ms1_spectrum.num_scans
    )
    finite = cutoff[np.isfinite(cutoff)]
    diffs = np.diff(finite)
    assert np.all(diffs >= -1e-6) or np.all(diffs <= 1e-6)


def test_cap_at_upper_endpoint_produces_inf(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    # A low cap (upper endpoint at 1/K0=0.9) forces high-mobility scans to +inf
    # when capping is enabled, and never when it is disabled.
    line = ((350.0, 0.7), (1200.0, 0.9))
    capped = ChargeStateRegion(line=line).index_cutoff_per_scan(
        td, ms1_frame_id, ms1_spectrum.num_scans
    )
    uncapped = ChargeStateRegion(
        line=line, cap_at_upper_endpoint=False
    ).index_cutoff_per_scan(td, ms1_frame_id, ms1_spectrum.num_scans)
    assert np.isinf(capped).any()
    assert not np.isinf(uncapped).any()


def test_exclude_region_returns_subset(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    filtered = exclude_region(
        ms1_spectrum, ChargeStateRegion(), td=td, frame_id=ms1_frame_id
    )
    assert filtered.mz_indices.size <= ms1_spectrum.mz_indices.size
    assert filtered.num_scans == ms1_spectrum.num_scans


def test_exclude_region_drops_peaks_inside_band(
    td: timsdata.TimsData, ms1_frame_id: int, ms1_spectrum: RawSpectrum
) -> None:
    # An aggressive region (steep, high cap) should drop strictly more than the
    # default and never add peaks.
    aggressive = ChargeStateRegion(line=((100.0, 0.6), (1700.0, 1.5)))
    filtered = exclude_region(
        ms1_spectrum, aggressive, td=td, frame_id=ms1_frame_id
    )
    assert filtered.mz_indices.size <= ms1_spectrum.mz_indices.size
