"""Regressions for the repository maintenance review."""

import hashlib
import runpy
import sqlite3
import struct
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tdfpy import DDA, DiaMs1WindowGate, TimsData, UnsupportedTdfError, slice_d_folder
from tdfpy.centroiding import get_tdf_df
from tdfpy.noise.gates import _build_dia_ms1_gate, read_dia_ms1_boxes
from tdfpy.timsdata import _zstd_decompress


@pytest.mark.parametrize("mode", ["dda", "dia", "prm"])
@pytest.mark.parametrize("pread", [True, False])
def test_high_level_extraction_matches_concurrent_first_use(mode, pread, monkeypatch):
    from tdfpy import get_raw_peaks, SelectionPolygonGate
    from tdfpy import timsdata

    if pread and timsdata._PREAD is None:
        pytest.skip("pread is unavailable")
    monkeypatch.setattr(timsdata, "_HAS_PREAD", pread)
    path = f"tests/data/example_{mode}.d"
    filters = [SelectionPolygonGate(), DiaMs1WindowGate()]
    with TimsData(path) as serial:
        ids = [
            r[0]
            for r in serial.conn.execute(
                "SELECT Id FROM Frames WHERE MsMsType=0 LIMIT 3"
            )
        ]
        expected = {fid: get_raw_peaks(serial, fid, noise=filters) for fid in ids}
    with TimsData(path) as parallel:
        with ThreadPoolExecutor(4) as pool:
            results = list(
                pool.map(
                    lambda fid: get_raw_peaks(parallel, fid, noise=filters), ids * 2
                )
            )
    for fid, result in zip(ids * 2, results):
        np.testing.assert_array_equal(result, expected[fid])


def test_collapsed_spectrum_uses_each_frames_calibration(tmp_path):
    from tdfpy import get_mobility_collapsed_spectrum, merge_peaks
    from tdfpy.centroiding import _sum_by_tof_index

    d = slice_d_folder("tests/data/example_dda.d", tmp_path / "two.d", 1, 2)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE Frames SET T1=T1+100 WHERE Id=2")
        conn.commit()
    with TimsData(d) as td:
        ranges = [(1, 150, 175), (2, 150, 175)]
        chunks = []
        for fid, begin, end in ranges:
            _, tof, intensity = td.read_frame_arrays(fid, begin, end)
            bins, sums = _sum_by_tof_index(tof, intensity)
            chunks.append(np.column_stack([td.indexToMz(fid, bins), sums]))
        raw = np.concatenate(chunks)
        # Roll up equal physical coordinates before choosing greedy seeds.
        mz, inverse = np.unique(raw[:, 0], return_inverse=True)
        intensity = np.bincount(inverse, weights=raw[:, 1])
        expected = merge_peaks(
            mz,
            intensity,
            np.zeros(len(mz)),
            mz_tolerance=30,
            min_peaks=1,
            im_tolerance_type="absolute",
            im_tolerance=0,
        )[:, :2]
        actual = get_mobility_collapsed_spectrum(td, ranges)
        np.testing.assert_allclose(actual, expected, rtol=1e-12)


def test_calibration_generator_refuses_to_replace_references():
    golden = Path("tests/data/calibration_golden.json")
    original = hashlib.sha256(golden.read_bytes()).digest()
    module = runpy.run_path("scripts/generate_calibration_golden.py")

    # Redirect writes to an in-memory object before invoking the old generator.
    class NoWrites:
        def write_text(self, *args, **kwargs):
            pytest.fail("The generator attempted to replace independent references")

    module["main"].__globals__["OUT"] = NoWrites()
    with pytest.raises(SystemExit, match="independent"):
        module["main"]()
    assert hashlib.sha256(golden.read_bytes()).digest() == original


def test_precursor_coordinates_preserve_sqlite_precision():
    with DDA("tests/data/example_dda.d") as reader:
        for pid, scan in reader.timsdata.conn.execute(
            "SELECT Id, ScanNumber FROM Precursors"
        ):
            precursor = reader.precursors[pid]
            assert precursor.scan_number == scan
            expected = reader.timsdata.scanNumToOneOverK0(
                precursor.parent_frame, [scan]
            )[0]
            assert precursor.ook0 == pytest.approx(expected, abs=1e-12)


@pytest.fixture
def single_frame(tmp_path):
    return slice_d_folder("tests/data/example_dda.d", tmp_path / "one.d", 1, 1)


@pytest.mark.parametrize(
    "mutation",
    ["odd_count", "zero_delta", "overflow_delta", "missing_payload", "peak_count"],
)
def test_corrupt_frame_is_rejected(single_frame, mutation):
    d = single_frame
    binary = d / "analysis.tdf_bin"
    packet = binary.read_bytes()
    size, scans = struct.unpack("<II", packet[:8])
    if mutation == "missing_payload":
        binary.write_bytes(struct.pack("<II", 8, scans))
    elif mutation == "peak_count":
        with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
            conn.execute("UPDATE Frames SET NumPeaks = NumPeaks + 1")
            conn.commit()
    else:
        try:
            from compression.zstd import compress
        except ImportError:
            from zstandard import compress
        raw = np.frombuffer(_zstd_decompress(packet[8:size]), np.uint8)
        words = raw.reshape(4, -1).T.copy().view("<u4").ravel()
        if mutation == "odd_count":
            words[1] |= np.uint32(1)
        else:
            words[scans] = 0 if mutation == "zero_delta" else np.iinfo(np.uint32).max
        payload = compress(words.view(np.uint8).reshape(-1, 4).T.copy().tobytes())
        binary.write_bytes(struct.pack("<II", len(payload) + 8, scans) + payload)
    with TimsData(d) as td:
        with pytest.raises(UnsupportedTdfError, match="Frame 1"):
            td.read_frame_arrays(1)


def test_zero_padding_dia_gate_obeys_half_open_scans():
    with TimsData("tests/data/example_dia.d") as td:
        fid, ns = td.conn.execute(
            "SELECT Id, NumScans FROM Frames WHERE MsMsType=0 LIMIT 1"
        ).fetchone()
        boxes = read_dia_ms1_boxes(td)
        gate = _build_dia_ms1_gate(td, fid, ns, DiaMs1WindowGate(mz_pad=0, im_pad=0))
        for begin, end, lo, hi in boxes:
            tof = int(round(td.mzToIndex(fid, [(lo + hi) / 2])[0]))
            mz = td.indexToMz(fid, [tof])[0]
            for scan in (begin, end - 1, end):
                expected = any(
                    b <= scan < e and low <= mz <= high for b, e, low, high in boxes
                )
                assert gate.contains(scan, tof) == expected


def test_metadata_helper_opens_the_database_file():
    with TimsData("tests/data/example_dda.d") as td:
        result = get_tdf_df(td)
        assert len(result) == 2519
        assert "NeutralMass" in result


def test_gate_cache_respects_later_frame_calibration(tmp_path):
    from tdfpy import get_raw_peaks, read_spectrum

    d = slice_d_folder("tests/data/example_dia.d", tmp_path / "changed.d", 1, 40)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        ids = [
            r[0]
            for r in conn.execute("SELECT Id FROM Frames WHERE MsMsType=0 ORDER BY Id")
        ]
        columns = [r[1] for r in conn.execute("PRAGMA table_info(MzCalibration)")]
        row = list(conn.execute("SELECT * FROM MzCalibration LIMIT 1").fetchone())
        new_id = conn.execute("SELECT MAX(Id)+1 FROM MzCalibration").fetchone()[0]
        row[columns.index("Id")] = new_id
        row[columns.index("C1")] *= 1.1
        conn.execute(
            f"INSERT INTO MzCalibration VALUES ({','.join('?' for _ in row)})", row
        )
        conn.execute("UPDATE Frames SET MzCalibration=? WHERE Id=?", (new_id, ids[1]))
        conn.commit()
    cfg = DiaMs1WindowGate(mz_pad=0, im_pad=0)
    with TimsData(d) as td:
        get_raw_peaks(td, ids[0], noise=cfg)
        later = read_spectrum(td, ids[1])
        gate = _build_dia_ms1_gate(td, ids[1], later.num_scans, cfg)
        expected = gate.keep_mask(later.scan_indices, later.mz_indices)
        actual = cfg.keep_mask(
            later.scan_indices,
            later.mz_indices,
            later.intensities,
            num_scans=later.num_scans,
            td=td,
            frame_id=ids[1],
        )
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mz_tolerance_type": "PPM"},
        {"im_tolerance_type": "fraction"},
        {"mz_tolerance": float("nan")},
        {"im_tolerance": -1},
        {"max_peaks": 1.5},
        {"peak_noise_end_fraction": 2},
    ],
)
@pytest.mark.parametrize("use_numba", [False, True])
def test_merge_rejects_invalid_configuration_before_dispatch(kwargs, use_numba):
    from tdfpy import merge_peaks

    with pytest.raises(ValueError):
        merge_peaks(
            np.empty(0), np.empty(0), np.empty(0), use_numba=use_numba, **kwargs
        )


def test_invalid_filter_is_not_retried_as_a_numba_failure():
    from tdfpy import NoiseFilter

    class FailingFilter(NoiseFilter):
        calls = 0

        def keep_mask(self, *args, **kwargs):
            self.calls += 1
            raise ValueError("filter failure")

    f = FailingFilter()
    with DDA("tests/data/example_dda.d") as reader:
        with pytest.raises(ValueError, match="filter failure"):
            next(iter(reader.ms1)).centroid(noise=f)
    assert f.calls == 1


def test_bad_zstd_returns_a_frame_specific_validation_issue(single_frame):
    from tdfpy import validate_acquisition

    path = single_frame / "analysis.tdf_bin"
    packet = bytearray(path.read_bytes())
    packet[8:12] = b"bad!"
    path.write_bytes(packet)
    report = validate_acquisition(single_frame, full=True)
    assert not report.valid
    assert report.issues[0].frame_id == 1
    assert "zstd" in report.issues[0].message


def test_timsdata_metadata_connection_is_read_only(tmp_path):
    from tdfpy import slice_d_folder

    source = slice_d_folder(
        "tests/data/example_dia.d", tmp_path / "read only #1.d", 1, 1
    )
    with TimsData(source) as td:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            td.conn.execute("DELETE FROM Frames")
        assert td.frame_ids == (1,)
