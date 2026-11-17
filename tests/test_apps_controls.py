"""The viewer's sidebar must not be able to ask for a combination the library refuses.

``apps/`` is internal dev tooling, not part of the package, and it imports
Streamlit, which is not a dependency here. The control decision therefore lives
in ``apps/timstof_viewer/_controls.py``, which imports nothing, and this module
loads that file directly by path, as ``tests/test_apps_sql.py`` does for the
query box.

The case these tests exist for: ``MergePeaksCentroider`` clusters in 1/K0, so
``im_tolerance_unit="absolute"`` is a width in V·s/cm² and raises for
``mobility_type="ccs"`` or ``"voltage"``. The dashboard offered both units
whatever the axis, and defaulted to ``"absolute"``, so its own default controls
could reach that error. Every unit the control offers is therefore run against
the real centroider here, and the combination it withholds is asserted to raise,
so neither side can drift without this failing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from tdfpy import DDA, MergePeaksCentroider, TdfpyError, get_centroided_spectrum

_CONTROLS_PATH = Path(__file__).resolve().parents[1] / "apps" / "timstof_viewer" / "_controls.py"
DDA_PATH = "tests/data/example_dda.d"
AXES = ("ook0", "ccs", "voltage")


def _load_controls():
    """Import ``apps/timstof_viewer/_controls.py`` by path, with no Streamlit involved.

    A missing file is a failure, not a skip: a skip would silently retire the
    only coverage the sidebar's unit logic has.
    """
    assert _CONTROLS_PATH.is_file(), f"not found: {_CONTROLS_PATH}"
    spec = importlib.util.spec_from_file_location("_tdfpy_apps_controls", _CONTROLS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_both_units_stay_available_in_ook0() -> None:
    controls = _load_controls()
    units, hint = controls.im_tolerance_unit_options("ook0")
    assert units == ["relative", "absolute"]
    assert hint is None, "1/K0 can use either unit, so there is nothing to explain"
    assert units[-1] == "absolute", "the last option is the default, which was 'absolute' before this guard"


@pytest.mark.parametrize("axis", ("ccs", "voltage"))
def test_absolute_is_withheld_outside_ook0_with_a_hint_naming_the_unit(axis: str) -> None:
    controls = _load_controls()
    units, hint = controls.im_tolerance_unit_options(axis)
    assert units == ["relative"], f"{axis} still offers {units}"
    assert hint and "1/K0" in hint, f"{axis}: the hint must say which unit an absolute tolerance is in, got {hint!r}"


def test_every_unit_the_control_offers_is_one_the_centroider_accepts() -> None:
    """The point of the whole file: no reachable control combination raises."""
    controls = _load_controls()
    with DDA(DDA_PATH) as reader:
        td = reader.timsdata
        tried = 0
        for axis in AXES:
            units, _ = controls.im_tolerance_unit_options(axis)
            assert units, f"{axis} offers no unit at all"
            for unit in units:
                centroider = MergePeaksCentroider(im_tolerance=0.01, im_tolerance_unit=unit, min_peaks=1)
                peaks = get_centroided_spectrum(td, 1, mobility_type=axis, centroid=centroider)  # type: ignore[arg-type]
                assert len(peaks), f"{axis}/{unit} came back empty, so it proves nothing"
                tried += 1
        assert tried == 4, f"expected 4 offered combinations (2 for 1/K0, 1 each for CCS and voltage), ran {tried}"


@pytest.mark.parametrize("axis", ("ccs", "voltage"))
def test_the_withheld_combination_is_the_one_that_raises(axis: str) -> None:
    """Non-vacuity: the control withholds exactly what the library refuses."""
    controls = _load_controls()
    units, _ = controls.im_tolerance_unit_options(axis)
    assert "absolute" not in units
    centroider = MergePeaksCentroider(im_tolerance=0.01, im_tolerance_unit="absolute", min_peaks=1)
    with DDA(DDA_PATH) as reader:
        with pytest.raises(TdfpyError, match="width in 1/K0 units"):
            get_centroided_spectrum(reader.timsdata, 1, mobility_type=axis, centroid=centroider)  # type: ignore[arg-type]
