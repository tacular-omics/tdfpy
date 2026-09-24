"""Contracts introduced by the 5.0 API cleanup.

Covers the error hierarchy, enums, frozen elements, lookup return types, the
renamed attributes and the names that 5.0 removed with no alias.
"""

import dataclasses
import pathlib
from collections.abc import Mapping
from typing import get_args

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tdfpy
from tdfpy import (
    DDA,
    DIA,
    PRM,
    AcquisitionType,
    AcquisitionTypeError,
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
from tdfpy.constants import TableNames
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


def test_polarity_is_a_plain_string():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
        precursor = next(iter(dda.precursors))
    assert frame.polarity == "positive" and type(frame.polarity) is str
    assert precursor.pasef_frame_msms_infos[0].polarity == "positive"
    assert precursor.polarity == "positive"


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
    assert AcquisitionType.UNKNOWN == "unknown"


def test_frame_msms_type_is_enum():
    with DDA(DDA_PATH) as dda:
        frame = next(iter(dda.ms1))
    assert frame.msms_type is MsMsType.MS1
    assert frame.polarity in get_args(Polarity)


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
        peaks = info.merged_peaks()
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
        assert precursor.isolation_mz_range == info.isolation_mz_range
        assert precursor.collision_energy == info.collision_energy
        assert precursor.polarity is info.polarity
        assert np.isfinite(precursor.voltage)


def test_precursor_aggregate_warns_when_windows_disagree():
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        infos = precursor.pasef_frame_msms_infos
        shifted = dataclasses.replace(infos[0], collision_energy=infos[0].collision_energy + 1.0, polarity="negative")
        mixed = dataclasses.replace(precursor, pasef_frame_msms_infos=(infos[0], shifted))
        empty = dataclasses.replace(precursor, pasef_frame_msms_infos=())
        with pytest.warns(UserWarning, match="Multiple values"):
            assert mixed.collision_energy is None
        with pytest.warns(UserWarning, match="Multiple values found for attribute 'polarity'"):
            assert mixed.polarity is None
        with pytest.warns(UserWarning, match="No values"):
            assert empty.isolation_mz_range is None
        with pytest.warns(UserWarning, match="No values found for attribute 'polarity'"):
            assert empty.polarity is None


# -- review fixes: ranges are (low, high) ---------------------------------------


def _mobility_windows():
    with DDA(DDA_PATH) as dda:
        infos = [info for p in list(dda.precursors)[:5] for info in p.pasef_frame_msms_infos]
        precursors = list(dda.precursors)[:5]
        yield from ((w, w.ook0_range, w.ccs_range, w.voltage_range) for w in infos + precursors)
    with DIA(DIA_PATH) as dia:
        yield from ((w, w.ook0_range, w.ccs_range, w.voltage_range) for w in list(dia.windows)[:5])
    with PRM(PRM_PATH) as prm:
        yield from ((t, t.ook0_range, t.ccs_range, t.voltage_range) for t in list(prm.transitions)[:5])


def test_mobility_ranges_are_low_high():
    seen = 0
    for _, *ranges in _mobility_windows():
        for rng in ranges:
            assert rng is not None
            lo, hi = rng
            assert lo <= hi
        seen += 1
    assert seen > 10


def test_mobility_begin_end_match_range():
    with DIA(DIA_PATH) as dia:
        w = next(iter(dia.windows))
        assert (w.ook0_begin, w.ook0_end) == w.ook0_range
        assert (w.ccs_begin, w.ccs_end) == w.ccs_range
        assert (w.voltage_begin, w.voltage_end) == w.voltage_range
        # The low 1/K0 edge is the *high* scan number.
        assert w.ook0_begin == pytest.approx(float(dia.timsdata.scan_num_to_ook0(w.frame_id, [w.scan_num_end])[0]))


def test_ook0_range_feeds_query_range():
    """A target's own ook0 lies in a (low, high) range built around it."""
    with PRM(PRM_PATH) as prm:
        target = next(t for t in prm.targets if t.ook0 is not None)
        hits = list(prm.targets.query_range(ook0_range=(target.ook0 - 1e-6, target.ook0 + 1e-6)))
        assert target in hits


@pytest.fixture(scope="module")
def dda_info():
    with DDA(DDA_PATH) as dda:
        info = next(iter(dda.precursors)).pasef_frame_msms_infos[0]
        num_scans = dda.timsdata.frame_metadata(info.frame_id).num_scans
        yield info, num_scans


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=st.data())
def test_mobility_ranges_low_le_high_property(dda_info, data):
    info, num_scans = dda_info
    begin = data.draw(st.integers(0, num_scans - 1))
    end = data.draw(st.integers(begin + 1, num_scans))
    window = dataclasses.replace(info, scan_num_begin=begin, scan_num_end=end)
    for lo, hi in (window.ook0_range, window.ccs_range, window.voltage_range):
        assert lo <= hi


# -- review fixes: closed reader ------------------------------------------------


@pytest.mark.parametrize("name", ["timsdata", "metadata", "calibration", "pandas_tdf"])
def test_file_backed_properties_raise_after_close(name):
    with DDA(DDA_PATH) as dda:
        getattr(dda, name)
    with pytest.raises(ReaderClosedError):
        getattr(dda, name)


def test_precursor_mobility_raises_after_close():
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        info = precursor.pasef_frame_msms_infos[0]
    for call in (lambda: precursor.ook0, lambda: precursor.ook0_range, lambda: info.ook0_range, precursor.merged_peaks, info.merged_peaks):
        with pytest.raises(ReaderClosedError):
            call()


# -- review fixes: acquisition type is checked on open ---------------------------


@pytest.mark.parametrize(
    ("cls", "path", "found"),
    [(DDA, DIA_PATH, "DIA"), (DIA, DDA_PATH, "DDA"), (PRM, DDA_PATH, "DDA"), (DDA, PRM_PATH, "PRM")],
)
def test_wrong_reader_raises_acquisition_type_error(cls, path, found):
    with pytest.raises(AcquisitionTypeError, match=rf"not a {cls.__name__} acquisition \(found {found}\); use tdfpy.get_acquisition_type\(\)"):
        cls(path)
    assert issubclass(AcquisitionTypeError, TdfpyError)


# -- review fixes: unreadable analysis.tdf at reader level -----------------------


@pytest.mark.parametrize("content", [b"", b"this is not an sqlite database" * 100], ids=["empty", "corrupt"])
@pytest.mark.parametrize("cls", [DDA, DIA, PRM])
def test_reader_on_unreadable_tdf_raises_tdfpy_error(tmp_path, content, cls):
    d = tmp_path / "run.d"
    d.mkdir()
    (d / "analysis.tdf").write_bytes(content)
    (d / "analysis.tdf_bin").write_bytes(b"")
    with pytest.raises(TdfpyError):
        cls(str(d))
    with pytest.raises(TdfpyError):
        get_acquisition_type(str(d))


def test_convert_table_to_df_message_uses_plain_table_name(not_sqlite):
    with pytest.raises(TdfpyError) as info:
        convert_table_to_df(not_sqlite, TableNames.FRAMES)
    assert "'Frames'" in str(info.value)
    assert "TableNames" not in str(info.value)


# -- review fixes: removed dead fields, merged_peaks ------------------------------


def test_dead_ms1_fields_removed():
    field_names = {f.name for f in dataclasses.fields(tdfpy.DIAMs1Frame)} | {f.name for f in dataclasses.fields(tdfpy.PRMMs1Frame)}
    assert "dia_windows" not in field_names
    assert "prm_transitions" not in field_names


def test_merged_peaks_is_a_method():
    for cls in (tdfpy.Precursor, tdfpy.PasefFrameMsmsInfo):
        assert callable(cls.merged_peaks)
        assert not hasattr(cls, "peaks")
    assert not hasattr(tdfpy.Precursor, "pasef_peaks")
    assert not hasattr(tdfpy.Frame, "peaks")
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        merged = precursor.pasef_merged_peaks()
        assert len(merged) == len(precursor.pasef_frame_msms_infos)
        assert all(a.ndim == 2 and a.shape[1] == 2 for a in merged)


# -- review fixes: lookup query property ----------------------------------------


@pytest.fixture(scope="module")
def dda_precursors():
    with DDA(DDA_PATH) as dda:
        yield dda.precursors


def _precursor_mz(p):
    assert p.precursor_mz == (p.monoisotopic_mz if p.monoisotopic_mz is not None else p.largest_peak_mz)
    return p.precursor_mz


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    mz=st.one_of(st.none(), st.floats(100.0, 2000.0)),
    rt=st.one_of(st.none(), st.floats(0.0, 4000.0)),
    mz_tol=st.floats(0.0, 1e5),
    rt_tol=st.floats(0.0, 500.0),
    tol_type=st.sampled_from(["ppm", "da"]),
)
def test_precursor_query_matches_brute_force(dda_precursors, mz, rt, mz_tol, rt_tol, tol_type):
    got = list(dda_precursors.query(precursor_mz=mz, rt=rt, mz_tolerance=mz_tol, mz_tolerance_type=tol_type, rt_tolerance=rt_tol))
    width = None if mz is None else (mz * mz_tol / 1e6 if tol_type == "ppm" else mz_tol)
    expected = [
        p for p in dda_precursors if (mz is None or mz - width <= _precursor_mz(p) <= mz + width) and (rt is None or rt - rt_tol <= p.rt <= rt + rt_tol)
    ]
    assert got == expected
    precursor_mz_range = None if mz is None else (mz - width, mz + width)
    rt_range = None if rt is None else (rt - rt_tol, rt + rt_tol)
    assert list(dda_precursors.query_range(precursor_mz_range=precursor_mz_range, rt_range=rt_range)) == got


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(lo=st.floats(0.0, 5000.0), span=st.floats(0.0, 5000.0))
def test_precursor_query_range_is_inclusive_rt_filter(dda_precursors, lo, span):
    got = list(dda_precursors.query_range(rt_range=(lo, lo + span)))
    assert all(lo <= p.rt <= lo + span for p in got)
    assert len(got) == sum(lo <= p.rt <= lo + span for p in dda_precursors)


# -- window scan ranges must fit their frame ---------------------------------------


def test_window_scan_range_past_frame_raises():
    """Corrupt metadata raises on every spectral accessor instead of being clipped."""
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        info = precursor.pasef_frame_msms_infos[0]
        n = dda.timsdata.frame_metadata(info.frame_id).num_scans
        bad = dataclasses.replace(info, scan_num_end=n + 100)
        bad_precursor = dataclasses.replace(precursor, pasef_frame_msms_infos=(bad,))
        for call in (bad.scan_peaks, bad.raw_peaks, bad.centroid, bad.merged_peaks, bad_precursor.merged_peaks):
            with pytest.raises(TdfpyError, match="does not fit"):
                call()
        # An end equal to num_scans is legal: the end is exclusive.
        edge = dataclasses.replace(info, scan_num_begin=n - 1, scan_num_end=n)
        assert len(edge.scan_peaks()) == 1
        assert edge.raw_peaks().shape[1] == 3


def test_dia_window_scan_range_past_frame_raises():
    with DIA(DIA_PATH) as dia:
        window = next(iter(dia.windows))
        n = dia.timsdata.frame_metadata(window.frame_id).num_scans
        bad = dataclasses.replace(window, scan_num_begin=-1)
        with pytest.raises(TdfpyError, match="does not fit"):
            bad.centroid()
        bad = dataclasses.replace(window, scan_num_end=n + 100)
        with pytest.raises(TdfpyError, match="does not fit"):
            bad.raw_peaks()
