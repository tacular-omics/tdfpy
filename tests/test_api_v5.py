"""Contracts introduced by the 5.0 API cleanup.

Covers the error hierarchy, enums, frozen elements, lookup return types, the
renamed attributes and the names that 5.0 removed with no alias.
"""

import dataclasses
import pathlib
from collections.abc import Mapping

import numpy as np
import pytest

import tdfpy
from tdfpy import (
    DDA,
    DIA,
    PRM,
    AcquisitionType,
    MsMsType,
    PandasTdf,
    Polarity,
    ReaderClosedError,
    TdfpyError,
    TdfpyKeyError,
    UnsupportedCalibrationError,
    UnsupportedTdfError,
    get_acquisition_type,
    slice_d_folder,
)
from tdfpy.tdf import convert_table_to_df

DDA_PATH = "tests/data/example_dda.d"
DIA_PATH = "tests/data/example_dia.d"
PRM_PATH = "tests/data/example_prm.d"


# -- errors -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "builtin"),
    [
        (TdfpyError, ValueError),
        (TdfpyKeyError, KeyError),
        (ReaderClosedError, RuntimeError),
        (UnsupportedTdfError, NotImplementedError),
        (UnsupportedCalibrationError, NotImplementedError),
    ],
)
def test_error_hierarchy(cls, builtin):
    assert issubclass(cls, TdfpyError)
    assert issubclass(cls, ValueError)
    assert issubclass(cls, builtin)


def test_key_error_str_is_plain():
    assert str(TdfpyKeyError("no id 5")) == "no id 5"


def test_lookup_miss_raises_tdfpy_key_error():
    with DDA(DDA_PATH) as dda:
        with pytest.raises(TdfpyKeyError, match="99999"):
            dda.precursors[99999]
        with pytest.raises(TdfpyKeyError):
            dda.ms1[99999]


def test_closed_reader_raises_reader_closed_error():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
    with pytest.raises(ReaderClosedError):
        frame.centroid()
    with pytest.raises(ReaderClosedError):
        frame.scan_peaks()
    with pytest.raises(ReaderClosedError):
        list(dda.ms1)


def test_polarity_from_str_raises_tdfpy_error():
    assert Polarity.from_str("+") is Polarity.POSITIVE
    with pytest.raises(TdfpyError):
        Polarity.from_str("sideways")


# -- sqlite errors are wrapped at the boundary --------------------------------


@pytest.fixture
def not_sqlite(tmp_path: pathlib.Path) -> pathlib.Path:
    path = tmp_path / "analysis.tdf"
    path.write_bytes(b"this is not an sqlite database" * 100)
    return path


def test_convert_table_to_df_wraps_sqlite_error(not_sqlite):
    with pytest.raises(TdfpyError):
        convert_table_to_df(not_sqlite, "Frames")


def test_convert_table_to_df_unknown_table():
    with pytest.raises(TdfpyError):
        convert_table_to_df(pathlib.Path(DDA_PATH) / "analysis.tdf", "NoSuchTable")


def test_get_table_names_wraps_sqlite_error(not_sqlite):
    with pytest.raises(TdfpyError):
        PandasTdf(str(not_sqlite)).get_table_names()


def test_get_table_names_does_not_modify_file():
    path = pathlib.Path(DDA_PATH) / "analysis.tdf"
    before = path.stat().st_mtime_ns
    assert "Frames" in PandasTdf(str(path)).get_table_names()
    assert path.stat().st_mtime_ns == before


def test_slice_d_folder_wraps_sqlite_error(tmp_path, not_sqlite):
    (not_sqlite.parent / "analysis.tdf_bin").write_bytes(b"")
    with pytest.raises(TdfpyError):
        slice_d_folder(not_sqlite.parent, tmp_path / "out.d", 1, 2)


# -- enums ------------------------------------------------------------------


@pytest.mark.parametrize(("path", "expected"), [(DDA_PATH, "DDA"), (DIA_PATH, "DIA"), (PRM_PATH, "PRM")])
def test_acquisition_type_is_str_enum(path, expected):
    acq = get_acquisition_type(path)
    assert isinstance(acq, AcquisitionType)
    assert acq == expected
    assert acq is AcquisitionType(expected)


def test_acquisition_type_unknown_value():
    assert AcquisitionType.UNKNOWN == "Unknown"


def test_frame_msms_type_is_enum():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
    assert frame.msms_type is MsMsType.MS1
    assert isinstance(frame.polarity, Polarity)


# -- frozen, slotted, keyword-only elements ---------------------------------


def test_elements_are_frozen_and_hashable():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        precursor = next(iter(dda.precursors))
    for element in (frame, precursor):
        with pytest.raises(dataclasses.FrozenInstanceError):
            element.rt = 0.0  # type: ignore[misc]
        assert not hasattr(element, "__dict__")
        hash(element)


def test_elements_are_keyword_only():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
    values = [getattr(frame, f.name) for f in dataclasses.fields(frame)]
    with pytest.raises(TypeError):
        type(frame)(*values)


def test_prm_target_equality_ignores_transitions():
    with PRM(PRM_PATH) as prm:
        target = next(t for t in prm.targets if t.transitions)
    bare = dataclasses.replace(target, transitions=())
    assert bare == target
    assert hash(bare) == hash(target)


# -- renamed attributes -----------------------------------------------------


def test_renamed_frame_fields():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        precursor = next(iter(dda.precursors))
        ms1_ids = {f.frame_id for f in dda.ms1}
    for name in ("rt", "mz_calibration_id", "tims_calibration_id", "property_group_id"):
        assert hasattr(frame, name)
    for name in ("time", "mz_calibration", "tims_calibration", "property_group"):
        assert not hasattr(frame, name)
    assert precursor.parent_frame_id in ms1_ids
    assert not hasattr(precursor, "parent_frame")
    info = precursor.pasef_frame_msms_infos[0]
    assert info.precursor_id == precursor.precursor_id
    assert not hasattr(info, "precursor")


def test_renamed_prm_target_fields():
    with PRM(PRM_PATH) as prm:
        target = next(iter(prm.targets))
    assert isinstance(target.rt, float)
    assert isinstance(target.ook0, float)
    assert not hasattr(target, "time")
    assert not hasattr(target, "one_over_k0")


def test_metadata_ook0_acq_range():
    with DDA(DDA_PATH) as dda:
        md = dda.metadata
    lo, hi = md.ook0_acq_range
    assert (lo, hi) == (md.ook0_acq_range_lower, md.ook0_acq_range_upper)
    assert not hasattr(md, "one_over_k0_acq_range")


def test_metadata_is_a_mapping():
    with DDA(DDA_PATH) as dda:
        md = dda.metadata
        cal = dda.calibration
    for table in (md, cal):
        assert isinstance(table, Mapping)
        assert len(table) == len(list(table))
        key = next(iter(table))
        assert table[key] == table.table[key]
        with pytest.raises(TdfpyKeyError, match="nope"):
            table["nope"]
        assert table.get("nope") is None


# -- spectra ----------------------------------------------------------------


def test_scan_peaks_shape():
    with DIA(DIA_PATH) as dia:
        window = next(iter(dia.windows))
        frame = next(iter(dia.ms1))
        for element in (window, frame):
            scans = element.scan_peaks()
            assert isinstance(scans, list)
            assert scans
            assert all(s.ndim == 2 and s.shape[1] == 2 for s in scans)
            assert not hasattr(element, "peaks")


def test_prm_transition_scan_peaks_matches_scan_count():
    with PRM(PRM_PATH) as prm:
        tr = next(iter(prm.transitions))
        scans = tr.scan_peaks()
    assert len(scans) == tr.scan_num_end - tr.scan_num_begin


def test_pasef_msms_info_spectra():
    with DDA(DDA_PATH) as dda:
        info = next(iter(dda.precursors)).pasef_frame_msms_infos[0]
        raw = info.raw_peaks()
        cen = info.centroid()
        scans = info.scan_peaks()
        peaks = info.peaks
    assert raw.shape[1] == 3
    assert cen.shape[1] == 3
    assert len(scans) == info.scan_num_end - info.scan_num_begin
    assert peaks.ndim == 2 and peaks.shape[1] == 2


# -- lookups ----------------------------------------------------------------


def test_dia_window_lookup_returns_tuples():
    with DIA(DIA_PATH) as dia:
        group_id = dia.windows.ids()[0]
        group = dia.windows[group_id]
        assert isinstance(group, tuple)
        assert all(w.window_group_id == group_id for w in group)
        assert group_id in dia.windows
        assert -1 not in dia.windows
        assert dia.windows.get(-1) is None
        assert dia.windows.get(-1, ()) == ()
        assert isinstance(dia.window_groups, tuple)
        by_id = list(dia.windows.query(window_group=group_id))
        by_obj = list(dia.windows.query(window_group=group[0]))
        assert by_id == by_obj == list(group)


def test_prm_transition_lookup_returns_tuples():
    with PRM(PRM_PATH) as prm:
        target_id = prm.transitions.ids()[0]
        assert isinstance(prm.transitions[target_id], tuple)


def test_lookup_queries_are_keyword_only():
    with DDA(DDA_PATH) as dda:
        with pytest.raises(TypeError):
            dda.precursors.query(500.0)  # type: ignore[misc]
    with DIA(DIA_PATH) as dia:
        with pytest.raises(TypeError):
            dia.windows.query(1)  # type: ignore[misc]
        with pytest.raises(TypeError):
            dia.windows.query(window_group_index=1)  # type: ignore[call-arg]


def test_lookup_ids_are_sorted_and_complete():
    with DDA(DDA_PATH) as dda:
        ids = dda.precursors.ids()
        assert isinstance(ids, tuple)
        assert sorted(ids) == sorted(p.precursor_id for p in dda.precursors)


# -- removed names ----------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "name"),
    [
        ("tdfpy.centroiding", "batch_iterator"),
        ("tdfpy.centroiding", "calculate_nmass"),
        ("tdfpy.centroiding", "get_tdf_df"),
        ("tdfpy.centroiding", "Peak"),
        ("tdfpy.timsdata", "oneOverK0ToCCSforMz"),
        ("tdfpy.timsdata", "ccsToOneOverK0forMz"),
        ("tdfpy.calibration", "one_over_k0_to_ccs"),
        ("tdfpy.calibration", "ccs_to_one_over_k0"),
    ],
)
def test_removed_module_names(module, name):
    import importlib

    assert not hasattr(importlib.import_module(module), name)


def test_ccs_helpers_exported_from_top_level():
    ccs = tdfpy.ook0_to_ccs(1.0, 2, 800.0)
    assert tdfpy.ccs_to_ook0(ccs, 2, 800.0) == pytest.approx(1.0)
    assert np.isfinite(ccs)


def test_every_public_module_declares_all():
    import importlib

    for name in (
        "calibration",
        "centroiding",
        "constants",
        "elems",
        "errors",
        "lookup",
        "pipeline",
        "processing",
        "reader",
        "regions",
        "slicer",
        "tdf",
        "timsdata",
        "validation",
        "noise.intensity",
        "noise.structural",
        "noise.gates",
    ):
        mod = importlib.import_module(f"tdfpy.{name}")
        assert hasattr(mod, "__all__"), name
        for exported in mod.__all__:
            assert hasattr(mod, exported), f"{name}.{exported}"


# -- typed accessors --------------------------------------------------------


def _public_properties(obj: object) -> list[str]:
    return [name for name in dir(type(obj)) if not name.startswith("_") and isinstance(getattr(type(obj), name), property)]


@pytest.mark.parametrize("path", [DDA_PATH, DIA_PATH, PRM_PATH])
def test_every_metadata_property_is_typed_or_raises_key_error(path):
    reader = {DDA_PATH: DDA, DIA_PATH: DIA, PRM_PATH: PRM}[path]
    with reader(path) as r:
        tables = (r.metadata, r.calibration)
    for table in tables:
        for name in _public_properties(table):
            try:
                value = getattr(table, name)
            except TdfpyKeyError:
                continue
            assert value is not None, name


def test_precursor_aggregate_properties():
    with DDA(DDA_PATH) as dda:
        precursor = next(p for p in dda.precursors if len(p.pasef_frame_msms_infos) == 1)
        info = precursor.pasef_frame_msms_infos[0]
        assert precursor.scan_num_range == info.scan_num_range
        assert precursor.ook0_range == info.ook0_range
        assert precursor.ccs_range == info.ccs_range
        assert precursor.voltage_range == info.voltage_range == (info.voltage_begin, info.voltage_end)
        assert precursor.mz_range == info.mz_range
        assert precursor.collision_energy == info.collision_energy
        assert precursor.polarity is info.polarity
        assert np.isfinite(precursor.voltage)


def test_precursor_aggregate_warns_when_windows_disagree():
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        infos = precursor.pasef_frame_msms_infos
        shifted = dataclasses.replace(infos[0], collision_energy=infos[0].collision_energy + 1.0, polarity=Polarity.NEGATIVE)
        mixed = dataclasses.replace(precursor, pasef_frame_msms_infos=(infos[0], shifted))
        empty = dataclasses.replace(precursor, pasef_frame_msms_infos=())
        with pytest.warns(UserWarning, match="Multiple values"):
            assert mixed.collision_energy is None
        with pytest.warns(UserWarning, match="Multiple polarities"):
            assert mixed.polarity is Polarity.MIXED
        with pytest.warns(UserWarning, match="No values"):
            assert empty.mz_range is None
        with pytest.warns(UserWarning, match="No polarities"):
            assert empty.polarity is Polarity.UNKNOWN
