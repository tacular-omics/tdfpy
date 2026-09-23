"""Tests for plot_centroiding."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest

import tdfpy
from tdfpy import DDA, MergePeaksCentroider, WatershedCentroider, get_centroided_spectrum, plot_centroiding

pytest.importorskip("matplotlib")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DDA_PATH = Path(__file__).parent / "data" / "example_dda.d"


@pytest.fixture(scope="module")
def frame_id() -> int:
    with DDA(DDA_PATH) as dda:
        return next(iter(dda.ms1)).frame_id


@pytest.fixture
def td() -> Iterator[tdfpy.TimsData]:
    with tdfpy.timsdata_connect(DDA_PATH) as handle:
        yield handle
    plt.close("all")


def _centroid_count(fig) -> int:
    for ax in fig.axes:
        m = re.match(r"Centroided\s+\(n=([\d,]+)\)", ax.get_title())
        if m:
            return int(m.group(1).replace(",", ""))
    raise AssertionError("no centroid panel found")


def test_default_matches_default_centroider(td, frame_id):
    """With no arguments the plot shows what frame.centroid() returns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fig = plot_centroiding(td, frame_id)
    assert _centroid_count(fig) == len(get_centroided_spectrum(td, frame_id))


@pytest.mark.parametrize(
    "centroider",
    [MergePeaksCentroider(mz_tolerance=20.0, im_tolerance=0.05), WatershedCentroider()],
    ids=["merge", "watershed"],
)
def test_centroid_kwarg(td, frame_id, centroider):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fig = plot_centroiding(td, frame_id, centroid=centroider)
    assert _centroid_count(fig) == len(get_centroided_spectrum(td, frame_id, centroid=centroider))


def test_legacy_tolerance_kwargs_warn_and_still_apply(td, frame_id):
    with pytest.warns(DeprecationWarning, match="centroid=MergePeaksCentroider"):
        fig = plot_centroiding(td, frame_id, mz_tolerance=20.0, im_tolerance=0.05, min_peaks=2)
    # Unspecified legacy knobs keep their old plot_centroiding defaults.
    expected = MergePeaksCentroider(mz_tolerance=20.0, im_tolerance=0.05, min_peaks=2)
    assert _centroid_count(fig) == len(get_centroided_spectrum(td, frame_id, centroid=expected))


def test_legacy_positional_tolerance_still_works(td, frame_id):
    with pytest.warns(DeprecationWarning):
        fig = plot_centroiding(td, frame_id, "ook0", 20.0)
    expected = MergePeaksCentroider(mz_tolerance=20.0, im_tolerance=0.01)
    assert _centroid_count(fig) == len(get_centroided_spectrum(td, frame_id, centroid=expected))


def test_centroid_and_legacy_kwargs_conflict(td, frame_id):
    with pytest.raises(TypeError, match="centroid"):
        plot_centroiding(td, frame_id, mz_tolerance=20.0, centroid=MergePeaksCentroider())
