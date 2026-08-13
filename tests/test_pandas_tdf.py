"""Tests for :class:`PandasTdf`, the pandas view over ``analysis.tdf``.

The fixture is built inside a pytest fixture rather than at class-body scope so
that absent test data *skips* these tests instead of breaking collection for the
whole suite.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tdfpy import PandasTdf
from tdfpy.constants import TableNames
from tdfpy.tdf import convert_table_to_df

DATA_DIR = Path("tests/data")
TDF_PATH = DATA_DIR / "example_dda.d" / "analysis.tdf"

#: Every DataFrame-returning property mapped to the SQLite table it actually
#: reads, so a newly added table cannot slip in untested and so the docstrings
#: can be checked against reality.
PROPERTY_TO_TABLE = {
    "calibration_info": TableNames.CALIBRATION_INFO.value,
    "dia_frame_msms_info": TableNames.DIA_FRAME_MSMS_INFO.value,
    "dia_frame_msms_window_groups": TableNames.DIA_FRAME_MSMS_WINDOW_GROUPS.value,
    "dia_frame_msms_windows": TableNames.DIA_FRAME_MSMS_WINDOWS.value,
    "error_log": TableNames.ERROR_LOG.value,
    "frame_msms_info": TableNames.FRAME_MSMS_WINDOW.value,
    "frame_properties": TableNames.FRAME_PROPERTIES.value,
    "frames": TableNames.FRAMES.value,
    "global_metadata": TableNames.GLOBAL_METADATA.value,
    "group_properties": TableNames.GROUP_PROPERTIES.value,
    "mz_calibration": TableNames.MZ_CALIBRATION.value,
    "pasef_frame_msms_info": TableNames.PASEF_FRAME_MSMS_INFO.value,
    "precursors": TableNames.PRECURSORS.value,
    "properties": TableNames.PROPERTIES.value,
    "property_definitions": TableNames.PROPERTY_DEFINITIONS.value,
    "property_groups": TableNames.PROPERTY_GROUPS.value,
    "segments": TableNames.SEGMENTS.value,
    "tims_calibration": TableNames.TIMS_CALIBRATION.value,
}


def _open(path: Path) -> PandasTdf:
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return PandasTdf(path)


@pytest.fixture
def pd_tdf() -> PandasTdf:
    """The DDA fixture, or a skip when the test data is absent."""
    return _open(TDF_PATH)


@pytest.mark.parametrize("prop", sorted(PROPERTY_TO_TABLE))
def test_table_property_returns_dataframe(pd_tdf: PandasTdf, prop: str) -> None:
    assert isinstance(getattr(pd_tdf, prop), pd.DataFrame)


@pytest.mark.parametrize(("prop", "table"), sorted(PROPERTY_TO_TABLE.items()))
def test_property_docstring_names_the_real_sqlite_table(prop: str, table: str) -> None:
    """The docstrings used to name the ``TableNames`` member, not the table."""
    doc = getattr(PandasTdf, prop).__doc__ or ""
    assert f"'{table}'" in doc, f"{prop} docstring does not name {table}"


# --------------------------------------------------------------------------
# Table discovery and the invalid-name guard
# --------------------------------------------------------------------------


def test_get_table_names_lists_the_real_tables(pd_tdf: PandasTdf) -> None:
    names = pd_tdf.get_table_names()
    assert isinstance(names, list)
    assert all(isinstance(n, str) for n in names)
    # Every .d has these, whatever the acquisition mode.
    assert {"Frames", "GlobalMetadata", "MzCalibration", "TimsCalibration"} <= set(
        names
    )


def test_get_table_names_is_repeatable(pd_tdf: PandasTdf) -> None:
    """Each call opens and closes its own connection; that must not leak state."""
    assert pd_tdf.get_table_names() == pd_tdf.get_table_names()


def test_unknown_table_name_is_rejected() -> None:
    """An arbitrary name must not reach the interpolated SELECT."""
    with pytest.raises(ValueError, match="Invalid table name"):
        convert_table_to_df(TDF_PATH, "Frames; DROP TABLE Frames")


def test_unknown_table_name_is_rejected_before_touching_the_file() -> None:
    """The name guard runs first, so a missing database is not the error."""
    with pytest.raises(ValueError, match="Invalid table name"):
        convert_table_to_df("does/not/exist.tdf", "NotATable")


def test_missing_database_is_reported() -> None:
    with pytest.raises(FileNotFoundError, match="TDF database not found"):
        PandasTdf(DATA_DIR / "no_such_file.tdf")


# --------------------------------------------------------------------------
# Acquisition-mode predicates
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fixture", "mode"),
    [
        ("example_dda.d", "dda"),
        ("example_dia.d", "dia"),
        ("example_prm.d", "prm"),
    ],
)
def test_acquisition_predicates_identify_exactly_one_mode(
    fixture: str, mode: str
) -> None:
    """Each fixture must match its own predicate and no other.

    The predicates key off different tables, so a file that satisfied two of
    them would send the high-level readers down the wrong path.
    """
    tdf = _open(DATA_DIR / fixture / "analysis.tdf")
    assert (tdf.is_dda, tdf.is_dia, tdf.is_prm) == (
        mode == "dda",
        mode == "dia",
        mode == "prm",
    )
    # None of the bundled fixtures is MALDI, which tdfpy does not support.
    assert tdf.is_maldi is False
