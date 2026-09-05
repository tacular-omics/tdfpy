import shutil
import sqlite3
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tdfpy import timsdata
from tdfpy.calibration import MzCalibration, UnsupportedCalibrationError
from tdfpy.timsdata import (
    PressureCompensationStrategy,
    TimsData,
    UnsupportedTdfError,
)

TDF_PATH = r"tests/data/example_dda.d"


def _require_fixture() -> None:
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")


def test_timsdata_connect_opens_the_database() -> None:
    _require_fixture()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        assert td.conn is not None


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the DDA fixture so its metadata can be edited destructively."""
    src = Path(TDF_PATH)
    if not src.is_dir():
        pytest.skip("Test data not found")
    dest = tmp_path / "modified.d"
    shutil.copytree(src, dest)
    return dest


def _set_global_metadata(d: Path, key: str, value: str) -> None:
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE GlobalMetadata SET Value = ? WHERE Key = ?", (value, key))
        conn.commit()


# --------------------------------------------------------------------------
# Helpers for corrupting a *copy* of a real fixture. Synthesising a whole
# .tdf_bin would test a file this reader never sees; editing real bytes tests
# the guards against the layout they actually have to police.
# --------------------------------------------------------------------------


def _frame_header(d: Path, frame_id: int = 1) -> tuple[int, int]:
    """Return ``(TimsId offset, NumScans)`` for a frame."""
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        offset, num_scans = conn.execute(
            "SELECT TimsId, NumScans FROM Frames WHERE Id = ?", (frame_id,)
        ).fetchone()
    return int(offset), int(num_scans)


def _patch_bin_header(
    d: Path,
    offset: int,
    *,
    byte_count: int | None = None,
    scan_count: int | None = None,
) -> None:
    """Overwrite one or both u32 header fields of a frame in analysis.tdf_bin."""
    with open(d / "analysis.tdf_bin", "r+b") as fh:
        fh.seek(offset)
        stored_bytes, stored_scans = struct.unpack("<II", fh.read(8))
        fh.seek(offset)
        fh.write(
            struct.pack(
                "<II",
                stored_bytes if byte_count is None else byte_count,
                stored_scans if scan_count is None else scan_count,
            )
        )


def _set_num_scans(d: Path, frame_id: int, num_scans: int) -> None:
    """Keep Frames.NumScans in step with a patched binary header."""
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute(
            "UPDATE Frames SET NumScans = ? WHERE Id = ?", (num_scans, frame_id)
        )
        conn.commit()


def _decoded_word_count(d: Path, frame_id: int = 1) -> int:
    """How many u32 words frame ``frame_id`` decompresses to."""
    offset, _ = _frame_header(d, frame_id)
    with open(d / "analysis.tdf_bin", "rb") as fh:
        fh.seek(offset)
        byte_count, _ = struct.unpack("<II", fh.read(8))
        payload = fh.read(byte_count - 8)
    return len(timsdata._zstd_decompress(payload)) // 4


# --------------------------------------------------------------------------
# Guards. An unvalidated format must fail loudly: every one of these produces
# plausible-looking but silently wrong numbers if it is not rejected up front.
# --------------------------------------------------------------------------


def test_legacy_compression_type_is_rejected(tmp_path: Path) -> None:
    d = _copy_fixture(tmp_path)
    _set_global_metadata(d, "TimsCompressionType", "1")
    with pytest.raises(UnsupportedTdfError, match="TimsCompressionType 1"):
        TimsData(str(d))


def test_unknown_compression_type_is_rejected(tmp_path: Path) -> None:
    d = _copy_fixture(tmp_path)
    _set_global_metadata(d, "TimsCompressionType", "3")
    with pytest.raises(UnsupportedTdfError, match="TimsCompressionType 3"):
        TimsData(str(d))


def test_unknown_mz_calibration_model_is_rejected(tmp_path: Path) -> None:
    d = _copy_fixture(tmp_path)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE MzCalibration SET ModelType = 2")
        conn.commit()
    with pytest.raises(UnsupportedCalibrationError, match="MzCalibration.ModelType 2"):
        TimsData(str(d))


def test_unknown_tims_calibration_model_is_rejected(tmp_path: Path) -> None:
    d = _copy_fixture(tmp_path)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE TimsCalibration SET ModelType = 5")
        conn.commit()
    with pytest.raises(
        UnsupportedCalibrationError, match="TimsCalibration.ModelType 5"
    ):
        TimsData(str(d))


def test_recalibrated_state_is_rejected() -> None:
    _require_fixture()
    with pytest.raises(UnsupportedTdfError, match="use_recalibrated_state"):
        TimsData(TDF_PATH, use_recalibrated_state=True)


def test_pressure_compensation_is_rejected() -> None:
    _require_fixture()
    with pytest.raises(UnsupportedTdfError, match="PerFramePressureCompensation"):
        TimsData(
            TDF_PATH,
            pressure_compensation_strategy=PressureCompensationStrategy.PerFramePressureCompensation,
        )


def test_rejected_open_does_not_leak_the_sqlite_connection(tmp_path: Path) -> None:
    """A guard that fires during __init__ must not leave the database open."""
    d = _copy_fixture(tmp_path)
    _set_global_metadata(d, "TimsCompressionType", "1")
    with pytest.raises(UnsupportedTdfError):
        TimsData(str(d))
    # On Windows an open handle would block this; on POSIX it is simply hygiene.
    shutil.rmtree(d)


# --------------------------------------------------------------------------
# Frame reading
# --------------------------------------------------------------------------


def test_unknown_frame_id_names_the_valid_range() -> None:
    _require_fixture()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        with pytest.raises(ValueError, match=r"valid frame IDs"):
            td.readScans(10_000_000, 0, 10)


def test_read_scans_matches_frame_metadata() -> None:
    """Decoded peak counts must agree with Frames.NumPeaks."""
    _require_fixture()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        assert td.conn is not None
        rows = td.conn.execute(
            "SELECT Id, NumScans, NumPeaks FROM Frames ORDER BY Id LIMIT 25"
        ).fetchall()
        for row in rows:
            scans = td.readScans(row["Id"], 0, row["NumScans"])
            assert len(scans) == row["NumScans"]
            assert sum(len(idx) for idx, _ in scans) == row["NumPeaks"]
            for idx, inten in scans:
                assert idx.shape == inten.shape
                # TOF indices ascend strictly within a scan.
                assert np.all(np.diff(idx.astype(np.int64)) > 0)


def test_partial_scan_range_matches_full_read() -> None:
    _require_fixture()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        assert td.conn is not None
        fid, num_scans = td.conn.execute(
            "SELECT Id, NumScans FROM Frames ORDER BY Id LIMIT 1"
        ).fetchone()
        full = td.readScans(fid, 0, num_scans)
        part = td.readScans(fid, 10, 25)
        assert len(part) == 15
        for offset, (idx, inten) in enumerate(part):
            expected_idx, expected_int = full[10 + offset]
            np.testing.assert_array_equal(idx, expected_idx)
            np.testing.assert_array_equal(inten, expected_int)


@pytest.mark.parametrize(
    "scan_range", [(0, None), (0, 1), (5, 50), (300, 671), (668, 671)]
)
def test_read_frame_arrays_matches_read_scans(
    scan_range: tuple[int, int | None],
) -> None:
    """The flat path must be a faithful, cheaper view of ``readScans``.

    ``read_frame_arrays`` slices the decoded frame instead of splitting it per
    scan, so it is easy for the two to drift on scan-boundary arithmetic. They
    have to agree peak-for-peak, including which scan each peak belongs to.
    """
    _require_fixture()
    begin, end = scan_range
    with timsdata.timsdata_connect(TDF_PATH) as td:
        assert td.conn is not None
        for row in td.conn.execute(
            "SELECT Id, NumScans FROM Frames ORDER BY Id LIMIT 20"
        ).fetchall():
            fid, num_scans = row["Id"], row["NumScans"]
            stop = num_scans if end is None else end
            scans = td.readScans(fid, begin, stop)
            scan_indices, tof, intensity = td.read_frame_arrays(fid, begin, stop)

            np.testing.assert_array_equal(
                tof, np.concatenate([idx for idx, _ in scans])
            )
            np.testing.assert_array_equal(
                intensity, np.concatenate([val for _, val in scans])
            )
            np.testing.assert_array_equal(
                scan_indices,
                np.repeat(np.arange(begin, stop), [len(idx) for idx, _ in scans]),
            )


def test_read_frame_arrays_clamps_out_of_range_scans() -> None:
    _require_fixture()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        scan_indices, tof, intensity = td.read_frame_arrays(1, 10_000, 10_050)
        assert scan_indices.size == tof.size == intensity.size == 0


def test_use_after_close_raises() -> None:
    _require_fixture()
    td = TimsData(TDF_PATH)
    td.close()
    assert td.handle is None
    with pytest.raises(RuntimeError, match="closed"):
        td.readScans(1, 0, 10)


# --------------------------------------------------------------------------
# Corrupt .tdf_bin input. Each case is a byte-level edit to a *copy* of the
# real fixture; all of them must surface as UnsupportedTdfError naming the
# frame, never as an IndexError, a broadcast ValueError, or a raw zstd error.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("byte_count", [0, 4, 7])
def test_byte_count_below_header_size_is_rejected(
    tmp_path: Path, byte_count: int
) -> None:
    """A byte_count under 8 used to become a negative read length.

    7 is the nastiest: ``read(-1)`` swallowed the whole rest of the file — tens
    of megabytes — and zstd then quietly ignored the trailing garbage. Lower
    values raised an opaque io ``ValueError`` instead. Neither named the file as
    corrupt.
    """
    d = _copy_fixture(tmp_path)
    offset, _ = _frame_header(d)
    _patch_bin_header(d, offset, byte_count=byte_count)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(
            UnsupportedTdfError, match=rf"Frame 1: .*{byte_count}-byte packet"
        ):
            td.readScans(1, 0, 10)


def test_truncated_tdf_bin_is_rejected(tmp_path: Path) -> None:
    """A short read must name the frame and the expected/actual sizes."""
    d = _copy_fixture(tmp_path)
    offset, _ = _frame_header(d)
    bin_path = d / "analysis.tdf_bin"
    with open(bin_path, "r+b") as fh:
        fh.seek(offset)
        byte_count, _ = struct.unpack("<II", fh.read(8))
        fh.truncate(offset + 8 + byte_count // 2)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(
            UnsupportedTdfError, match=r"Frame 1: truncated payload .*expected \d+"
        ):
            td.readScans(1, 0, 10)


def test_truncated_frame_header_is_rejected(tmp_path: Path) -> None:
    d = _copy_fixture(tmp_path)
    offset, _ = _frame_header(d)
    with open(d / "analysis.tdf_bin", "r+b") as fh:
        fh.truncate(offset + 3)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(
            UnsupportedTdfError, match=r"Frame 1: truncated header .*got 3"
        ):
            td.readScans(1, 0, 10)


def test_zero_scan_count_with_a_payload_is_rejected(tmp_path: Path) -> None:
    """scan_count == 0 on a non-empty frame used to raise IndexError."""
    d = _copy_fixture(tmp_path)
    offset, _ = _frame_header(d)
    _patch_bin_header(d, offset, scan_count=0)
    _set_num_scans(d, 1, 0)  # keep the metadata consistent so this guard is reached
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(UnsupportedTdfError, match=r"Frame 1: Frames.NumScans is 0"):
            td.readScans(1, 0, 10)


def test_payload_shorter_than_the_scan_header_is_rejected(tmp_path: Path) -> None:
    """Fewer words than scans used to raise a NumPy broadcast ValueError."""
    d = _copy_fixture(tmp_path)
    offset, _ = _frame_header(d)
    too_many = _decoded_word_count(d) + 2
    _patch_bin_header(d, offset, scan_count=too_many)
    _set_num_scans(d, 1, too_many)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(UnsupportedTdfError, match=r"Frame 1: decompressed payload"):
            td.readScans(1, 0, 10)


def test_odd_peak_word_count_is_rejected(tmp_path: Path) -> None:
    """Peaks are (tof_delta, intensity) pairs, so an odd tail is corrupt.

    The ``// 2`` used to floor the stray word away silently. Measured on every
    frame of all three fixtures: ``(words.size - scan_count) % 2 == 0`` always.
    """
    d = _copy_fixture(tmp_path)
    offset, num_scans = _frame_header(d)
    _patch_bin_header(d, offset, scan_count=num_scans + 1)  # flips the parity
    _set_num_scans(d, 1, num_scans + 1)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(UnsupportedTdfError, match=r"Frame 1: \d+ peak words"):
            td.readScans(1, 0, 10)


def test_payload_scan_count_disagreeing_with_the_header_is_rejected(
    tmp_path: Path,
) -> None:
    """word[0] restates the scan count; the two must agree.

    Shifting by 2 keeps the peak-word parity valid so this guard, and not the
    parity one, is what fires.
    """
    d = _copy_fixture(tmp_path)
    offset, num_scans = _frame_header(d)
    _patch_bin_header(d, offset, scan_count=num_scans + 2)
    _set_num_scans(d, 1, num_scans + 2)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(
            UnsupportedTdfError, match=r"Frame 1: the payload's leading word"
        ):
            td.readScans(1, 0, 10)


def test_num_scans_disagreeing_between_tdf_and_tdf_bin_is_rejected(
    tmp_path: Path,
) -> None:
    """Frames.NumScans is read but used to be discarded.

    Measured: the binary header's scan_count equals Frames.NumScans on all 1710
    frames of the three fixtures, so a disagreement means one file is corrupt.
    """
    d = _copy_fixture(tmp_path)
    _, num_scans = _frame_header(d)
    _set_num_scans(d, 1, num_scans + 1)
    with timsdata.timsdata_connect(str(d)) as td:
        with pytest.raises(
            UnsupportedTdfError,
            match=rf"Frame 1: .*{num_scans} scans but Frames.NumScans is {num_scans + 1}",
        ):
            td.readScans(1, 0, 10)


def test_empty_frame_reads_as_no_peaks(tmp_path: Path) -> None:
    """byte_count == 8 is a legitimately empty frame, not a corrupt one."""
    d = _copy_fixture(tmp_path)
    offset, num_scans = _frame_header(d)
    _patch_bin_header(d, offset, byte_count=8)
    with closing(sqlite3.connect(d / "analysis.tdf")) as conn:
        conn.execute("UPDATE Frames SET NumPeaks=0 WHERE Id=1")
        conn.commit()
    with timsdata.timsdata_connect(str(d)) as td:
        scans = td.readScans(1, 0, num_scans)
        assert len(scans) == num_scans
        assert all(idx.size == 0 and inten.size == 0 for idx, inten in scans)

        scan_indices, tof, intensity = td.read_frame_arrays(1)
        assert scan_indices.size == tof.size == intensity.size == 0


# --------------------------------------------------------------------------
# Concurrency
# --------------------------------------------------------------------------


@pytest.fixture
def eager_preemption():
    """Switch threads as often as possible for the duration of a test.

    At the default 5 ms switch interval an unsynchronised ``seek`` + ``read``
    pair almost never gets preempted between the two calls, so the race this
    guards against would go unnoticed. At 1 ns it shows up reliably: reverting
    :meth:`TimsData._pread` to a bare seek + read fails the test below with
    hundreds of mismatched reads and decode errors.
    """
    previous = sys.getswitchinterval()
    sys.setswitchinterval(1e-9)
    try:
        yield
    finally:
        sys.setswitchinterval(previous)


def _assert_concurrent_reads_match_serial(td: TimsData, workers: int) -> None:
    assert td.conn is not None
    frame_ids = [
        int(r["Id"])
        for r in td.conn.execute("SELECT Id FROM Frames ORDER BY Id LIMIT 40")
    ]
    serial = {fid: td.read_frame_arrays(fid) for fid in frame_ids}

    def read(fid: int) -> tuple[int, tuple]:
        return fid, td.read_frame_arrays(fid)

    # Sweep the frames repeatedly so the workers keep overlapping in the file.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(read, frame_ids * 4))

    assert len(results) == len(frame_ids) * 4
    for fid, (scan_indices, tof, intensity) in results:
        expected = serial[fid]
        np.testing.assert_array_equal(scan_indices, expected[0])
        np.testing.assert_array_equal(tof, expected[1])
        np.testing.assert_array_equal(intensity, expected[2])


def test_concurrent_frame_reads_match_serial_reads(eager_preemption) -> None:
    """Threads reading through one open reader must not corrupt each other.

    Frame bytes are fetched positionally (``os.pread``) precisely so that no
    file position is shared. A plain seek + read pair interleaves here and hands
    one frame's header to another frame's payload.
    """
    _require_fixture()
    with timsdata.timsdata_connect(TDF_PATH) as td:
        _assert_concurrent_reads_match_serial(td, workers=2)


def test_concurrent_frame_reads_are_safe_without_pread(
    eager_preemption, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same, on the lock-guarded fallback used where os.pread is missing.

    ``os.pread`` is POSIX-only, so on Windows the fallback is the only path
    there is; forcing it here keeps it covered on every platform.
    """
    _require_fixture()
    monkeypatch.setattr(timsdata, "_HAS_PREAD", False)
    with timsdata.timsdata_connect(TDF_PATH) as td:
        _assert_concurrent_reads_match_serial(td, workers=4)


# --------------------------------------------------------------------------
# CCS conversions
# --------------------------------------------------------------------------


@pytest.mark.parametrize("charge", [1, 2, 3, 5])
def test_ccs_one_over_k0_roundtrip(charge: int) -> None:
    """``ccsToOneOverK0forMz`` must invert ``oneOverK0ToCCSforMz`` exactly."""
    for mz in (150.0, 500.0, 1000.0, 2500.0, 5000.0):
        for ook0 in (0.5, 0.8, 1.1, 1.5, 2.0):
            ccs = timsdata.oneOverK0ToCCSforMz(ook0, charge, mz)
            assert ccs > 0
            back = timsdata.ccsToOneOverK0forMz(ccs, charge, mz)
            assert back == pytest.approx(ook0, rel=1e-12), (
                f"charge={charge} mz={mz} 1/K0={ook0}"
            )


def test_ccs_roundtrip_from_the_ccs_side() -> None:
    """And the other direction, so neither is merely self-consistent."""
    for ccs in (100.0, 350.0, 800.0):
        ook0 = timsdata.ccsToOneOverK0forMz(ccs, 2, 700.0)
        assert timsdata.oneOverK0ToCCSforMz(ook0, 2, 700.0) == pytest.approx(
            ccs, rel=1e-12
        )


def test_misnamed_ccs_alias_still_works_but_warns() -> None:
    with pytest.warns(DeprecationWarning, match="ccsToOneOverK0forMz"):
        got = timsdata.ccsToOneOverK0ToCCSforMz(350.0, 2, 700.0)
    assert got == timsdata.ccsToOneOverK0forMz(350.0, 2, 700.0)


def test_both_ccs_names_are_exported() -> None:
    assert "ccsToOneOverK0forMz" in timsdata.__all__
    assert "ccsToOneOverK0ToCCSforMz" in timsdata.__all__


# --------------------------------------------------------------------------
# index -> m/z domain guard
# --------------------------------------------------------------------------


def _mz_calibration() -> MzCalibration:
    """A model shaped like the fixtures': DigitizerDelay far above C0."""
    return MzCalibration(
        model_type=1,
        digitizer_timebase=0.2,
        digitizer_delay=24864.0,
        t1=25.0,
        t2=100.0,
        dc1=0.0,
        dc2=0.0,
        c0=313.5776,
        c1=1.2e5,
        c2=3.3874e-4,
    )


def test_index_to_mz_rejects_indices_below_the_model_domain() -> None:
    """Below C0 the root goes negative and squaring it hides that completely."""
    cal = _mz_calibration()
    below = cal.min_tof_index - 1000.0
    assert below < 0
    with pytest.raises(ValueError, match="index_to_mz"):
        cal.index_to_mz([below], 25.0, 100.0)
    # One bad entry poisons the whole call, so the whole call must fail.
    with pytest.raises(ValueError, match="index_to_mz"):
        cal.index_to_mz([0.0, 1000.0, below], 25.0, 100.0)


def test_index_to_mz_accepts_the_domain_boundary() -> None:
    cal = _mz_calibration()
    mz = cal.index_to_mz([cal.min_tof_index], 25.0, 100.0)
    assert mz[0] == pytest.approx(0.0, abs=1e-9)


def test_index_to_mz_domain_covers_every_real_tof_index() -> None:
    """The guard must never fire on real data: min_tof_index is far below 0."""
    _require_fixture()
    for fixture in ("example_dda.d", "example_dia.d", "example_prm.d"):
        d = Path("tests/data") / fixture
        if not d.is_dir():
            continue
        with timsdata.timsdata_connect(str(d)) as td:
            assert td.conn is not None
            for row in td.conn.execute(
                "SELECT Id FROM Frames ORDER BY Id LIMIT 20"
            ).fetchall():
                fid = int(row["Id"])
                cal, _, _ = td._mz_cal(fid)
                assert cal.min_tof_index < 0
                _, tof, _ = td.read_frame_arrays(fid)
                if tof.size:
                    assert td.indexToMz(fid, tof.astype(np.float64)).min() > 0
