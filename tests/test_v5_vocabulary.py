"""5.0 reader vocabulary shared with mzmlpy, and the speedups that came with it.

Covers ``ms_level``/``msms_type``, ``precursor_mz``, the ``Polarity`` literal
(including the unknown-value warning), ``Ms1FrameLookup`` RT queries, the
sqlite-tuple reader init, scan-sliced frame decoding, the integer TOF sort ahead
of ``merge_peaks`` and :func:`tdfpy.iter_precursor_spectra`.
"""

import random
import shutil
import sqlite3
import warnings
from contextlib import closing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import tdfpy
from tdfpy import DDA, DIA, PRM, MsMsType, ReaderClosedError, TdfpyError, iter_precursor_spectra
from tdfpy import processing as processing_mod
from tdfpy.centroiding import _merge_peaks_numba, _tof_order, merge_peaks
from tdfpy.elems import DiaWindow, Frame, PasefFrameMsmsInfo, PrmTransition
from tdfpy.timsdata import timsdata_connect

DDA_PATH = Path("tests/data/example_dda.d")
DIA_PATH = Path("tests/data/example_dia.d")
PRM_PATH = Path("tests/data/example_prm.d")


def _need(path: Path) -> None:
    if not path.is_dir():
        pytest.skip("Test data not found")


def _copy(src: Path, tmp_path: Path) -> Path:
    _need(src)
    dest = tmp_path / src.name
    shutil.copytree(src, dest)
    return dest


# -- ms_level / msms_type ---------------------------------------------------


@pytest.mark.parametrize(
    ("msms_type", "level"),
    [(MsMsType.MS1, 1), (MsMsType.DDA_MS2, 2), (MsMsType.DIA_MS2, 2), (MsMsType.PRM_MS2, 2)],
)
def test_frame_ms_level_follows_msms_type(msms_type, level):
    assert Frame.ms_level.fget(SimpleNamespace(msms_type=msms_type)) == level


@pytest.mark.parametrize(
    ("cls", "msms_type"),
    [(PasefFrameMsmsInfo, MsMsType.DDA_MS2), (DiaWindow, MsMsType.DIA_MS2), (PrmTransition, MsMsType.PRM_MS2)],
)
def test_window_classes_declare_msms_type(cls, msms_type):
    assert cls.msms_type is msms_type


def test_readers_expose_ms_level():
    _need(DDA_PATH)
    _need(DIA_PATH)
    _need(PRM_PATH)
    with DDA(DDA_PATH) as dda, DIA(DIA_PATH) as dia, PRM(PRM_PATH) as prm:
        assert {f.ms_level for f in dda.ms1} == {1}
        assert {f.ms_level for f in dia.ms1} == {1}
        info = next(iter(dda.precursors)).pasef_frame_msms_infos[0]
        window = next(iter(dia.windows))
        transition = next(iter(prm.transitions))
        for item, msms_type in ((info, MsMsType.DDA_MS2), (window, MsMsType.DIA_MS2), (transition, MsMsType.PRM_MS2)):
            assert item.ms_level == 2
            assert item.msms_type is msms_type
            # The frame the window came from agrees with the class constant.
            assert int(item.timsdata.frame_metadata(item.frame_id).msms_type) == msms_type


# -- precursor_mz -------------------------------------------------------------


def test_precursor_mz_prefers_monoisotopic():
    _need(DDA_PATH)
    with DDA(DDA_PATH) as dda:
        precursors = list(dda.precursors)
        assert any(p.monoisotopic_mz is not None for p in precursors)
        for p in precursors:
            expected = p.monoisotopic_mz if p.monoisotopic_mz is not None else p.largest_peak_mz
            assert p.precursor_mz == expected


def test_precursor_mz_falls_back_to_largest_peak(tmp_path):
    d = _copy(DDA_PATH, tmp_path)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE Precursors SET MonoisotopicMz = NULL")
        conn.commit()
    with DDA(d) as dda:
        for p in dda.precursors:
            assert p.monoisotopic_mz is None
            assert p.precursor_mz == p.largest_peak_mz


def test_prm_target_precursor_mz():
    _need(PRM_PATH)
    with PRM(PRM_PATH) as prm:
        for target in prm.targets:
            assert isinstance(target.precursor_mz, float)
            assert not hasattr(target, "monoisotopic_mz")


# -- polarity -----------------------------------------------------------------


def test_unknown_polarity_is_none_with_one_warning(tmp_path):
    d = _copy(DDA_PATH, tmp_path)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        # The schema pins Polarity to '+'/'-'; a file that breaks that is the case under test.
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE Frames SET Polarity = '?'")
        conn.commit()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with DDA(d) as dda:
            assert {f.polarity for f in dda.ms1} == {None}
            precursor = next(iter(dda.precursors))
            assert precursor.pasef_frame_msms_infos[0].polarity is None
    polarity_warnings = [w for w in caught if "olarity" in str(w.message) and "'?'" in str(w.message)]
    assert len(polarity_warnings) == 1


def test_mixed_precursor_polarity_is_none(tmp_path):
    d = _copy(DDA_PATH, tmp_path)
    with DDA(DDA_PATH) as dda:
        precursor = next(p for p in dda.precursors if len({i.frame_id for i in p.pasef_frame_msms_infos}) > 1)
        pid, frame_id = precursor.precursor_id, precursor.pasef_frame_msms_infos[0].frame_id
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE Frames SET Polarity = '-' WHERE Id = ?", (frame_id,))
        conn.commit()
    with DDA(d) as dda:
        with pytest.warns(UserWarning, match="polarity"):
            assert dda.precursors[pid].polarity is None


# -- Ms1FrameLookup RT queries -------------------------------------------------


@pytest.fixture(scope="module")
def ms1_frames():
    _need(DDA_PATH)
    with DDA(DDA_PATH) as dda:
        yield dda.ms1, list(dda.ms1)


def test_ms1_query_range_none_keeps_all(ms1_frames):
    lookup, frames = ms1_frames
    assert list(lookup.query_range()) == frames
    assert list(lookup.query()) == frames


def test_ms1_query_rejects_negative_tolerance(ms1_frames):
    lookup, _ = ms1_frames
    with pytest.raises(TdfpyError):
        lookup.query(rt=10.0, rt_tolerance=-1.0)


def test_ms1_query_is_keyword_only(ms1_frames):
    lookup, _ = ms1_frames
    with pytest.raises(TypeError):
        lookup.query_range((0.0, 1.0))  # type: ignore[misc]
    with pytest.raises(TypeError):
        lookup.query(1.0)  # type: ignore[misc]


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(a=st.floats(-10, 200, allow_nan=False), b=st.floats(-10, 200, allow_nan=False), tol=st.floats(0, 50, allow_nan=False))
def test_ms1_rt_queries_match_brute_force(ms1_frames, a, b, tol):
    lookup, frames = ms1_frames
    lo, hi = min(a, b), max(a, b)
    assert list(lookup.query_range(rt_range=(lo, hi))) == [f for f in frames if lo <= f.rt <= hi]
    assert list(lookup.query(rt=a, rt_tolerance=tol)) == [f for f in frames if a - tol <= f.rt <= a + tol]


# -- sqlite reader init ---------------------------------------------------------


def test_reader_wraps_sqlite_errors(tmp_path):
    d = _copy(DDA_PATH, tmp_path)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("DROP TABLE PasefFrameMsMsInfo")
        conn.commit()
    with pytest.raises(TdfpyError, match="Failed to read TDF database"):
        DDA(d)


# -- scan-sliced decode -----------------------------------------------------------


@pytest.fixture(scope="module")
def dda_td():
    _need(DDA_PATH)
    with timsdata_connect(DDA_PATH) as td:
        yield td


@settings(max_examples=80, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(data=st.data())
def test_sliced_decode_matches_full_decode(dda_td, data):
    frame_id = data.draw(st.sampled_from(sorted(dda_td._peak_counts)))
    num_scans = dda_td.frame_metadata(frame_id).num_scans
    begin = data.draw(st.integers(0, num_scans))
    end = data.draw(st.integers(begin, num_scans))
    full = dda_td._decode(frame_id)
    sliced = dda_td._decode(frame_id, begin, end)
    if full is None:
        assert sliced is None
        return
    _, starts, counts, tof, intensity = full
    _, s_starts, s_counts, s_tof, s_intensity = sliced
    np.testing.assert_array_equal(counts, s_counts)
    for scan in range(begin, end):
        a, n = int(starts[scan]), int(counts[scan])
        b = int(s_starts[scan])
        np.testing.assert_array_equal(tof[a : a + n], s_tof[b : b + n])
        np.testing.assert_array_equal(intensity[a : a + n], s_intensity[b : b + n])
    assert s_tof.size == int(counts[begin:end].sum())
    scans, arr_tof, arr_int = dda_td.read_frame_arrays(frame_id, begin, end)
    np.testing.assert_array_equal(arr_tof, s_tof)
    np.testing.assert_array_equal(arr_int, s_intensity)
    assert scans.size == s_tof.size


# -- integer TOF sort -----------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    keys=st.lists(st.integers(0, 2_000), max_size=300) | st.lists(st.integers(0, 2**32 - 1), max_size=50)  # sparse span: comparison-sort fallback
)
def test_tof_order_is_a_stable_argsort(keys):
    arr = np.asarray(keys, dtype=np.int64)
    np.testing.assert_array_equal(_tof_order(arr), np.argsort(arr, kind="stable"))


@settings(max_examples=60, deadline=None)
@given(
    mz=st.lists(st.floats(100, 2000, allow_nan=False), min_size=1, max_size=80, unique=True),
    seed=st.integers(0, 2**16),
)
def test_merge_peaks_presorted_input_gives_same_result(mz, seed):
    """Skipping the argsort for ascending m/z must not change the centroids."""
    rng = np.random.default_rng(seed)
    mz_arr = np.asarray(mz)
    # Distinct intensities: equal ones make the greedy seed order a tie-break.
    intensity = rng.permutation(len(mz_arr)).astype(np.float64) + 1.0
    im = rng.uniform(0.6, 1.4, len(mz_arr))
    order = np.argsort(mz_arr)
    shuffled = _merge_peaks_numba(mz_arr, intensity, im, min_peaks=1)
    presorted = _merge_peaks_numba(mz_arr[order], intensity[order], im[order], min_peaks=1)
    np.testing.assert_array_equal(shuffled, presorted)
    np.testing.assert_allclose(presorted, merge_peaks(mz_arr, intensity, im, min_peaks=1, use_numba=False))


def test_merge_centroider_matches_sorted_merge(dda_td):
    """MergePeaksCentroider's pre-sort feeds merge_peaks ascending m/z."""
    from tdfpy.pipeline import MergePeaksCentroider, convert, read_spectrum

    frame_id = next(f for f, n in sorted(dda_td._peak_counts.items()) if n)
    spectrum = read_spectrum(dda_td, frame_id)
    got = MergePeaksCentroider()(spectrum, dda_td, frame_id)
    peaks = convert(spectrum, dda_td, frame_id)
    order = np.argsort(spectrum.mz_indices, kind="stable")
    np.testing.assert_array_equal(got, merge_peaks(peaks[order, 0], peaks[order, 1], peaks[order, 2]))


# -- iter_precursor_spectra -------------------------------------------------------


def test_iter_precursor_spectra_is_exported():
    assert "iter_precursor_spectra" in tdfpy.__all__
    assert tdfpy.iter_precursor_spectra is iter_precursor_spectra


def test_iter_precursor_spectra_matches_merged_peaks():
    _need(DDA_PATH)
    with DDA(DDA_PATH) as dda:
        precursors = list(dda.precursors)
        pairs = list(iter_precursor_spectra(precursors))
        assert [p for p, _ in pairs] == precursors
        for precursor, peaks in pairs:
            np.testing.assert_array_equal(peaks, precursor.merged_peaks())


def test_iter_precursor_spectra_decodes_each_frame_once(monkeypatch):
    _need(DDA_PATH)
    with DDA(DDA_PATH) as dda:
        calls: list[int] = []
        original = type(dda.timsdata)._decode

        def counting(self, frame_id, *args):
            calls.append(frame_id)
            return original(self, frame_id, *args)

        monkeypatch.setattr(type(dda.timsdata), "_decode", counting)
        list(iter_precursor_spectra(dda.precursors))
        assert len(calls) == len(set(calls))


def test_iter_precursor_spectra_scattered_order_with_tiny_cache(monkeypatch):
    _need(DDA_PATH)
    monkeypatch.setattr(processing_mod, "_PRECURSOR_FRAME_CACHE", 1)
    with DDA(DDA_PATH) as dda:
        precursors = list(dda.precursors)
        random.Random(0).shuffle(precursors)
        for precursor, peaks in iter_precursor_spectra(precursors):
            np.testing.assert_array_equal(peaks, precursor.merged_peaks())


def test_iter_precursor_spectra_empty_input():
    assert list(iter_precursor_spectra([])) == []


def test_iter_precursor_spectra_closed_reader():
    _need(DDA_PATH)
    dda = DDA(DDA_PATH)
    precursors = list(dda.precursors)
    dda.close()
    with pytest.raises(ReaderClosedError):
        next(iter_precursor_spectra(precursors))


# -- reader init edge cases -------------------------------------------------------


def _sql(d: Path, *statements: str) -> None:
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        for statement in statements:
            conn.execute(statement)
        conn.commit()


def test_reader_rejects_unknown_msms_type(tmp_path):
    d = _copy(DDA_PATH, tmp_path)
    _sql(d, "UPDATE Frames SET MsMsType = 5 WHERE Id = (SELECT MAX(Id) FROM Frames WHERE MsMsType = 8)")
    with pytest.raises(TdfpyError, match="Unrecognised MsMsType 5"):
        DDA(d)


def test_acquisition_type_unknown_without_ms2_frames(tmp_path):
    d = _copy(DDA_PATH, tmp_path)
    _sql(d, "UPDATE Frames SET MsMsType = 0")
    assert tdfpy.get_acquisition_type(d) == tdfpy.AcquisitionType.UNKNOWN


def test_acquisition_type_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        tdfpy.get_acquisition_type(tmp_path)


@pytest.mark.parametrize("missing", ["analysis.tdf", "analysis.tdf_bin"])
def test_reader_missing_files(tmp_path, missing):
    d = _copy(DDA_PATH, tmp_path)
    (d / missing).unlink()
    with pytest.raises(FileNotFoundError, match=missing):
        DDA(d)


def test_reader_raises_after_timsdata_closed():
    _need(DDA_PATH)
    with DDA(DDA_PATH) as dda:
        dda.timsdata.close()
        with pytest.raises(ReaderClosedError):
            dda._check_open()


def test_pasef_rows_without_precursor_are_reported(tmp_path, caplog):
    d = _copy(DDA_PATH, tmp_path)
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        info = precursor.pasef_frame_msms_infos[0]
        frame_id, scan_begin = info.frame_id, info.scan_num_begin
    _sql(d, f"UPDATE PasefFrameMsMsInfo SET Precursor = NULL WHERE Frame = {frame_id} AND ScanNumBegin = {scan_begin}")
    with caplog.at_level("WARNING"), DDA(d) as dda:
        assert all((i.frame_id, i.scan_num_begin) != (frame_id, scan_begin) for p in dda.precursors for i in p.pasef_frame_msms_infos)
    assert "NULL Precursor" in caplog.text


# -- fallbacks and empty ranges ----------------------------------------------------


def test_tof_order_falls_back_when_numba_fails(monkeypatch):
    from numba.core.errors import NumbaError

    from tdfpy import centroiding

    def broken(*_args):
        raise NumbaError("forced")

    monkeypatch.setattr(centroiding, "_counting_argsort_kernel", broken)
    keys = np.array([5, 1, 5, 0, 3], dtype=np.int64)
    np.testing.assert_array_equal(centroiding._tof_order(keys), np.argsort(keys, kind="stable"))


def test_iter_precursor_spectra_empty_window():
    import dataclasses

    _need(DDA_PATH)
    with DDA(DDA_PATH) as dda:
        precursor = next(iter(dda.precursors))
        info = precursor.pasef_frame_msms_infos[0]
        empty = dataclasses.replace(info, scan_num_end=info.scan_num_begin, _timsdata=info.timsdata)
        only_empty = dataclasses.replace(precursor, pasef_frame_msms_infos=(empty,), _timsdata=precursor.timsdata)
        [(_, peaks)] = list(iter_precursor_spectra([only_empty]))
        assert peaks.shape == (0, 2)
        np.testing.assert_array_equal(peaks, only_empty.merged_peaks())


def test_empty_lookup_miss_message():
    from tdfpy.lookup import _missing_id_error

    assert "none are loaded" in str(_missing_id_error("MS1 frame ID", 3, []))


# -- centroid tie-break ----------------------------------------------------------


@settings(max_examples=100, deadline=None)
@given(
    mz_pool=st.lists(st.floats(500, 500.2, allow_nan=False), min_size=1, max_size=20, unique=True),
    n=st.integers(1, 60),
    levels=st.integers(1, 3),
    seed=st.integers(0, 2**16),
)
def test_merge_peaks_equal_intensities_are_order_independent(mz_pool, n, levels, seed):
    """Equal intensities and repeated m/z values: input order must not matter.

    Points are drawn from a small m/z pool (so many share one m/z, like one TOF
    index seen in many scans) with few intensity and mobility levels, so the
    greedy seed order and the within-m/z order both decide the clusters. Two
    shuffles and the canonical (m/z, descending 1/K0, intensity) order must give
    the same centroids, in numba and in pure Python.
    """
    rng = np.random.default_rng(seed)
    mz_arr = rng.choice(np.asarray(mz_pool), n)
    intensity = rng.integers(1, levels + 1, n).astype(np.float64) * 100.0
    im = rng.choice([0.9, 1.0, 1.02], n)
    canonical = np.lexsort((intensity, -im, mz_arr))
    kwargs = {"mz_tolerance": 0.01, "mz_tolerance_unit": "da", "min_peaks": 1}
    for use_numba in (True, False):
        results = [
            merge_peaks(mz_arr[idx], intensity[idx], im[idx], use_numba=use_numba, **kwargs) for idx in (rng.permutation(n), rng.permutation(n), canonical)
        ]
        np.testing.assert_array_equal(results[0], results[2])
        np.testing.assert_array_equal(results[1], results[2])
        if use_numba:
            numba_result = results[2]
    np.testing.assert_allclose(numba_result, results[2], rtol=1e-12)


@pytest.mark.parametrize("use_numba", [True, False])
def test_merge_peaks_shuffled_frame_matches_tof_scan_order(dda_td, use_numba):
    """A real MS1 frame (many points per TOF index) gives the same centroids shuffled."""
    from tdfpy.pipeline import convert, read_spectrum

    frame_id = max(dda_td._peak_counts, key=dda_td._peak_counts.get)
    spectrum = read_spectrum(dda_td, frame_id)
    order = _tof_order(spectrum.mz_indices)
    peaks = convert(spectrum, dda_td, frame_id)[order]
    assert np.unique(peaks[:, 0]).size < len(peaks)  # repeated m/z values present
    expected = merge_peaks(peaks[:, 0], peaks[:, 1], peaks[:, 2], use_numba=use_numba)
    shuffled = peaks[np.random.default_rng(0).permutation(len(peaks))]
    got = merge_peaks(shuffled[:, 0], shuffled[:, 1], shuffled[:, 2], use_numba=use_numba)
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("use_numba", [True, False])
def test_merge_peaks_tie_seeds_lowest_mz_first(use_numba):
    """Three equal peaks 0.008 Da apart with a 0.01 Da tolerance: the lowest m/z seeds."""
    mz = np.array([500.016, 500.000, 500.008])
    intensity = np.full(3, 100.0)
    im = np.ones(3)
    got = merge_peaks(mz, intensity, im, mz_tolerance=0.01, mz_tolerance_unit="da", min_peaks=1, use_numba=use_numba)
    got = got[np.argsort(got[:, 0])]
    np.testing.assert_allclose(got[:, 0], [500.004, 500.016])
    np.testing.assert_allclose(got[:, 1], [200.0, 100.0])


# -- iter_precursor_spectra input check -------------------------------------------


def test_iter_precursor_spectra_rejects_dia_windows():
    _need(DIA_PATH)
    with DIA(DIA_PATH) as dia:
        windows = list(dia.windows)[:3]
        with pytest.raises(TdfpyError, match="iter_window_spectra"):
            list(iter_precursor_spectra(windows))  # type: ignore[arg-type]


# -- sliced reads check only the scans they decode --------------------------------


def test_sliced_read_does_not_check_tof_outside_the_slice(dda_td, monkeypatch):
    """A TOF index past DigitizerNumSamples in a scan outside the slice is not reported.

    The bounds check covers the decoded scans only (the price of decoding a slice);
    a full read of the same frame still raises.
    """
    for frame_id in sorted(dda_td._peak_counts):
        decoded = dda_td._decode(frame_id)
        if decoded is None:
            continue
        _, starts, counts, tof, _ = decoded
        per_scan = [(scan, int(tof[starts[scan] : starts[scan] + counts[scan]].max())) for scan in np.flatnonzero(counts)]
        low_scan, low_max = min(per_scan, key=lambda item: item[1])
        if low_max < int(tof.max()):
            break
    else:
        pytest.skip("no frame with a scan below the frame's max TOF")
    # Pretend the digitiser is shorter: only scans outside the chosen one now overflow.
    monkeypatch.setattr(dda_td, "_digitizer_num_samples", low_max + 1)
    scans, arr_tof, _ = dda_td.read_frame_arrays(frame_id, int(low_scan), int(low_scan) + 1)
    assert arr_tof.size == int(counts[low_scan])
    with pytest.raises(TdfpyError, match="DigitizerNumSamples"):
        dda_td.read_frame_arrays(frame_id)


def test_merge_peaks_unsigned_input_matches_float():
    """uint32 arrays are converted to float64 first, so the order check cannot wrap."""
    mz, intensity, im = np.array([300, 100, 200, 100], np.uint32), np.array([5, 9, 7, 9], np.uint32), np.array([2, 1, 3, 4], np.uint32)
    np.testing.assert_array_equal(
        merge_peaks(mz, intensity, im, min_peaks=1), merge_peaks(mz.astype(float), intensity.astype(float), im.astype(float), min_peaks=1)
    )
