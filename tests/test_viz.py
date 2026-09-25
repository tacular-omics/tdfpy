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


@pytest.fixture(scope="module")
def small_frame_id() -> int:
    """A PASEF MS2 frame of a few thousand points: watershed on an MS1 frame took ~7-18 s."""
    with DDA(DDA_PATH) as dda:
        info = next(iter(dda.precursors)).pasef_frame_msms_infos[0]
        return info.frame_id


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


def test_default_matches_default_centroider(td, small_frame_id):
    """With no arguments the plot shows what frame.centroid() returns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fig = plot_centroiding(td, small_frame_id)
    n = len(get_centroided_spectrum(td, small_frame_id))
    assert n > 0
    assert _centroid_count(fig) == n


@pytest.mark.parametrize(
    "centroider",
    [MergePeaksCentroider(mz_tolerance=20.0, im_tolerance=0.05), WatershedCentroider()],
    ids=["merge", "watershed"],
)
def test_centroid_kwarg(td, small_frame_id, centroider):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fig = plot_centroiding(td, small_frame_id, centroid=centroider)
    n = len(get_centroided_spectrum(td, small_frame_id, centroid=centroider))
    assert n > 0
    assert _centroid_count(fig) == n


@pytest.mark.parametrize("kwarg", ["mz_tolerance", "mz_tolerance_unit", "im_tolerance", "im_tolerance_unit", "min_peaks", "max_peaks"])
def test_removed_tolerance_kwargs_raise(td, frame_id, kwarg):
    """5.0 removed the deprecated tolerance kwargs; pass ``centroid=`` instead."""
    with pytest.raises(TypeError, match=kwarg):
        plot_centroiding(td, frame_id, **{kwarg: 1})


def test_options_are_keyword_only(td, frame_id):
    with pytest.raises(TypeError, match="positional"):
        plot_centroiding(td, frame_id, "ook0")  # type: ignore[misc]
