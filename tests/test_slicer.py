import os
import sqlite3
import struct
from pathlib import Path

import pytest

from tdfpy.slicer import slice_d_folder

TEST_DATA = Path("tests/data/200ngHeLaPASEF_1min.d")
SKIP_NO_DATA = pytest.mark.skipif(
    not TEST_DATA.exists(), reason="Test data not available"
)


@SKIP_NO_DATA
def test_slice_basic(tmp_path):
    """Slice to first 10 frames and verify structure."""
    dest = tmp_path / "sliced.d"
    result = slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    assert result == dest
    assert (dest / "analysis.tdf").exists()
    assert (dest / "analysis.tdf_bin").exists()

    with sqlite3.connect(dest / "analysis.tdf") as conn:
        frame_count = conn.execute("SELECT COUNT(*) FROM Frames").fetchone()[0]
        assert frame_count == 10

        ids = conn.execute("SELECT Id FROM Frames ORDER BY Id").fetchall()
        assert [r[0] for r in ids] == list(range(1, 11))


@SKIP_NO_DATA
def test_slice_binary_smaller(tmp_path):
    """Sliced binary file should be smaller than original."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    orig_size = os.path.getsize(TEST_DATA / "analysis.tdf_bin")
    sliced_size = os.path.getsize(dest / "analysis.tdf_bin")
    assert sliced_size < orig_size


@SKIP_NO_DATA
def test_slice_offsets_valid(tmp_path):
    """Each frame's TimsId offset should point to a valid blob."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    bin_size = os.path.getsize(dest / "analysis.tdf_bin")

    with sqlite3.connect(dest / "analysis.tdf") as conn:
        rows = conn.execute(
            "SELECT Id, TimsId FROM Frames ORDER BY Id"
        ).fetchall()

    with open(dest / "analysis.tdf_bin", "rb") as f:
        for frame_id, offset in rows:
            assert offset < bin_size, f"Frame {frame_id} offset out of range"
            f.seek(offset)
            header = f.read(4)
            (byte_count,) = struct.unpack("<I", header)
            assert byte_count > 8, f"Frame {frame_id} has invalid byte_count"
            assert offset + byte_count <= bin_size


@SKIP_NO_DATA
def test_slice_precursors_filtered(tmp_path):
    """Precursors outside frame range should be removed."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    with sqlite3.connect(dest / "analysis.tdf") as conn:
        orphaned = conn.execute(
            "SELECT COUNT(*) FROM Precursors WHERE Parent < 1 OR Parent > 10"
        ).fetchone()[0]
        assert orphaned == 0


@SKIP_NO_DATA
def test_slice_pasef_filtered(tmp_path):
    """PasefFrameMsMsInfo should only reference kept frames."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    with sqlite3.connect(dest / "analysis.tdf") as conn:
        orphaned = conn.execute(
            "SELECT COUNT(*) FROM PasefFrameMsMsInfo "
            "WHERE Frame < 1 OR Frame > 10"
        ).fetchone()[0]
        assert orphaned == 0


@SKIP_NO_DATA
def test_slice_middle_range(tmp_path):
    """Slicing a middle range should preserve frame IDs."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=100, frame_end=110)

    with sqlite3.connect(dest / "analysis.tdf") as conn:
        ids = conn.execute("SELECT Id FROM Frames ORDER BY Id").fetchall()
        assert [r[0] for r in ids] == list(range(100, 111))


@SKIP_NO_DATA
def test_slice_calibration_preserved(tmp_path):
    """Calibration and metadata tables should be untouched."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    with sqlite3.connect(TEST_DATA / "analysis.tdf") as orig_conn:
        orig_meta = orig_conn.execute(
            "SELECT * FROM GlobalMetadata"
        ).fetchall()

    with sqlite3.connect(dest / "analysis.tdf") as new_conn:
        new_meta = new_conn.execute(
            "SELECT * FROM GlobalMetadata"
        ).fetchall()

    assert orig_meta == new_meta


def test_slice_invalid_source(tmp_path):
    """Should raise FileNotFoundError for missing source."""
    with pytest.raises(FileNotFoundError):
        slice_d_folder(tmp_path / "nonexistent.d", tmp_path / "out.d", 1, 10)


def test_slice_invalid_range(tmp_path):
    """Should raise ValueError when frame_start > frame_end."""
    # Create minimal source dir to pass file checks.
    src = tmp_path / "src.d"
    src.mkdir()
    (src / "analysis.tdf").touch()
    (src / "analysis.tdf_bin").touch()

    with pytest.raises(ValueError, match="frame_start"):
        slice_d_folder(src, tmp_path / "out.d", 10, 5)


@SKIP_NO_DATA
def test_slice_dest_already_exists(tmp_path):
    """Should overwrite dest if it already exists."""
    dest = tmp_path / "sliced.d"
    dest.mkdir()
    (dest / "stale.txt").write_text("stale")

    result = slice_d_folder(TEST_DATA, dest, 1, 10)

    assert result == dest
    assert not (dest / "stale.txt").exists()
    assert (dest / "analysis.tdf").exists()
    assert (dest / "analysis.tdf_bin").exists()
