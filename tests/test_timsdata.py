import shutil
import sqlite3
import unittest
from pathlib import Path

import numpy as np
import pytest

from tdfpy import timsdata
from tdfpy.calibration import UnsupportedCalibrationError
from tdfpy.timsdata import (
    PressureCompensationStrategy,
    TimsData,
    UnsupportedTdfError,
)

TDF_PATH = r"tests/data/example_dda.d"


class TestTimsData(unittest.TestCase):
    def test_timsdata(self):
        with timsdata.timsdata_connect(TDF_PATH) as td:
            self.assertTrue(td.conn is not None)


def _copy_fixture(tmp_path: Path) -> Path:
    """Copy the DDA fixture so its metadata can be edited destructively."""
    src = Path(TDF_PATH)
    if not src.is_dir():
        pytest.skip("Test data not found")
    dest = tmp_path / "modified.d"
    shutil.copytree(src, dest)
    return dest


def _set_global_metadata(d: Path, key: str, value: str) -> None:
    with sqlite3.connect(d / "analysis.tdf") as conn:
        conn.execute("UPDATE GlobalMetadata SET Value = ? WHERE Key = ?", (value, key))


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
    with sqlite3.connect(d / "analysis.tdf") as conn:
        conn.execute("UPDATE MzCalibration SET ModelType = 2")
    with pytest.raises(UnsupportedCalibrationError, match="MzCalibration.ModelType 2"):
        TimsData(str(d))


def test_unknown_tims_calibration_model_is_rejected(tmp_path: Path) -> None:
    d = _copy_fixture(tmp_path)
    with sqlite3.connect(d / "analysis.tdf") as conn:
        conn.execute("UPDATE TimsCalibration SET ModelType = 5")
    with pytest.raises(
        UnsupportedCalibrationError, match="TimsCalibration.ModelType 5"
    ):
        TimsData(str(d))


def test_recalibrated_state_is_rejected() -> None:
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
    with pytest.raises(UnsupportedTdfError, match="use_recalibrated_state"):
        TimsData(TDF_PATH, use_recalibrated_state=True)


def test_pressure_compensation_is_rejected() -> None:
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
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
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
    with timsdata.timsdata_connect(TDF_PATH) as td:
        with pytest.raises(ValueError, match=r"valid frame IDs"):
            td.readScans(10_000_000, 0, 10)


def test_read_scans_matches_frame_metadata() -> None:
    """Decoded peak counts must agree with Frames.NumPeaks."""
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
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
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
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
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
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
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
    with timsdata.timsdata_connect(TDF_PATH) as td:
        scan_indices, tof, intensity = td.read_frame_arrays(1, 10_000, 10_050)
        assert scan_indices.size == tof.size == intensity.size == 0


def test_use_after_close_raises() -> None:
    if not Path(TDF_PATH).is_dir():
        pytest.skip("Test data not found")
    td = TimsData(TDF_PATH)
    td.close()
    assert td.handle is None
    with pytest.raises(RuntimeError, match="closed"):
        td.readScans(1, 0, 10)


if __name__ == "__main__":
    unittest.main()
