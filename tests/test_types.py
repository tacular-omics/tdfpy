"""Shared type aliases must match ``tacular.types`` (same names, values and order)."""

from typing import get_args

import tdfpy
from tdfpy import elems, types


def test_tolerance_unit_values():
    assert get_args(tdfpy.ToleranceUnit) == ("da", "ppm")


def test_polarity_values():
    assert get_args(tdfpy.Polarity) == ("positive", "negative")


def test_one_polarity_definition():
    assert tdfpy.Polarity is types.Polarity is elems.Polarity


def test_exported():
    assert {"Polarity", "ToleranceUnit"} <= set(tdfpy.__all__)
