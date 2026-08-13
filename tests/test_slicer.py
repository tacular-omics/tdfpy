import os
import sqlite3
import struct
from pathlib import Path

import pytest

from tdfpy.slicer import slice_d_folder

TEST_DATA = Path("tests/data/example_dda.d")
SKIP_NO_DATA = pytest.mark.skipif(
    not TEST_DATA.exists(), reason="Test data not available"
)


def _minimal_source(path: Path) -> Path:
    """Create a stub .d folder that passes the source file checks."""
    path.mkdir(parents=True)
    (path / "analysis.tdf").touch()
    (path / "analysis.tdf_bin").touch()
    return path


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
    src = _minimal_source(tmp_path / "src.d")

    with pytest.raises(ValueError, match="frame_start"):
        slice_d_folder(src, tmp_path / "out.d", 10, 5)


def test_slice_dest_already_exists(tmp_path):
    """Should refuse to touch an existing dest rather than overwriting it."""
    src = _minimal_source(tmp_path / "src.d")

    dest = tmp_path / "sliced.d"
    dest.mkdir()
    (dest / "stale.txt").write_text("stale")

    with pytest.raises(FileExistsError, match="already exists"):
        slice_d_folder(src, dest, 1, 10)

    # The pre-existing contents must survive untouched.
    assert (dest / "stale.txt").read_text() == "stale"
    assert not (dest / "analysis.tdf").exists()


def test_slice_dest_is_existing_file(tmp_path):
    """An existing *file* at dest is also refused."""
    src = _minimal_source(tmp_path / "src.d")
    dest = tmp_path / "sliced.d"
    dest.write_text("not a folder")

    with pytest.raises(FileExistsError):
        slice_d_folder(src, dest, 1, 10)

    assert dest.read_text() == "not a folder"


def test_slice_dest_equals_source(tmp_path):
    """Slicing a .d folder onto itself must raise before any writes."""
    src = _minimal_source(tmp_path / "src.d")

    with pytest.raises(ValueError, match="source .d folder"):
        slice_d_folder(src, src, 1, 10)

    # Also catch the non-normalised spelling of the same path.
    with pytest.raises(ValueError, match="source .d folder"):
        slice_d_folder(src, src / "sub" / "..", 1, 10)

    assert (src / "analysis.tdf").exists()
    assert (src / "analysis.tdf_bin").exists()


def test_slice_dest_inside_source(tmp_path):
    """A destination nested inside the source folder must raise."""
    src = _minimal_source(tmp_path / "src.d")

    with pytest.raises(ValueError, match="inside the source"):
        slice_d_folder(src, src / "sliced.d", 1, 10)

    assert not (src / "sliced.d").exists()


@SKIP_NO_DATA
def test_slice_null_tims_id_removes_partial_dest(tmp_path):
    """A NULL TimsId must raise and leave no partial destination behind."""
    # Make a small valid source first, then corrupt one frame's TimsId.
    src = tmp_path / "small.d"
    slice_d_folder(TEST_DATA, src, frame_start=1, frame_end=5)

    with sqlite3.connect(src / "analysis.tdf") as conn:
        conn.execute("UPDATE Frames SET TimsId = NULL WHERE Id = 3")

    dest = tmp_path / "broken.d"
    with pytest.raises(ValueError, match="NULL TimsId"):
        slice_d_folder(src, dest, frame_start=1, frame_end=5)

    assert not dest.exists()

    # A range that excludes the broken frame still works.
    ok_dest = tmp_path / "ok.d"
    slice_d_folder(src, ok_dest, frame_start=4, frame_end=5)
    assert (ok_dest / "analysis.tdf_bin").exists()


@SKIP_NO_DATA
def test_slice_failure_removes_partial_dest(tmp_path):
    """A mid-write failure must not leave a half-written .d folder."""
    src = tmp_path / "small.d"
    slice_d_folder(TEST_DATA, src, frame_start=1, frame_end=5)

    # Truncate the binary so the blob for a kept frame can no longer be read.
    with open(src / "analysis.tdf_bin", "r+b") as f:
        f.truncate(16)

    dest = tmp_path / "broken.d"
    with pytest.raises(IOError):
        slice_d_folder(src, dest, frame_start=1, frame_end=5)

    assert not dest.exists()
