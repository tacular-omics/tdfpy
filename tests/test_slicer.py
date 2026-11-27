import os
import shutil
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from tdfpy import DDA
from tdfpy.errors import TdfpyError
from tdfpy.slicer import slice_d_folder

TEST_DATA = Path("tests/data/example_dda.d")
SKIP_NO_DATA = pytest.mark.skipif(not TEST_DATA.exists(), reason="Test data not available")


@SKIP_NO_DATA
def test_slice_removes_precursors_without_msms(tmp_path):
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=20)

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        assert (
            conn.execute("SELECT COUNT(*) FROM Precursors p WHERE NOT EXISTS (SELECT 1 FROM PasefFrameMsMsInfo i WHERE i.Precursor = p.Id)").fetchone()[0] == 0
        )
        assert conn.execute("SELECT COUNT(*) FROM Precursors").fetchone()[0] == 60
        assert conn.execute("SELECT COUNT(*) FROM PasefFrameMsMsInfo").fetchone()[0] == 109
        assert [row[0] for row in conn.execute("SELECT Id FROM Frames ORDER BY Id")] == list(range(1, 21))

    with DDA(TEST_DATA) as source, DDA(dest) as sliced:
        for precursor_id, shape in ((1, (471, 2)), (2, (511, 2)), (3, (1771, 2))):
            expected = source.precursors[precursor_id].merged_peaks()
            actual = sliced.precursors[precursor_id].merged_peaks()
            assert actual.shape == shape
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", ["dia", "prm"])
def test_slice_other_acquisition_modes(tmp_path, mode):
    source = Path(f"tests/data/example_{mode}.d")
    if not source.exists():
        pytest.skip("Test data not available")
    dest = tmp_path / f"{mode}.d"
    slice_d_folder(source, dest, frame_start=1, frame_end=20)
    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        assert [row[0] for row in conn.execute("SELECT Id FROM Frames ORDER BY Id")] == list(range(1, 21))


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

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
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

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        rows = conn.execute("SELECT Id, TimsId FROM Frames ORDER BY Id").fetchall()

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

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        orphaned = conn.execute("SELECT COUNT(*) FROM Precursors WHERE Parent < 1 OR Parent > 10").fetchone()[0]
        assert orphaned == 0


@SKIP_NO_DATA
def test_slice_pasef_filtered(tmp_path):
    """PasefFrameMsMsInfo should only reference kept frames."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        orphaned = conn.execute("SELECT COUNT(*) FROM PasefFrameMsMsInfo WHERE Frame < 1 OR Frame > 10").fetchone()[0]
        assert orphaned == 0


def _source_frames_kept(frame_start: int, frame_end: int) -> list[int]:
    """Frames of the DDA fixture in the range, minus PASEF MS/MS frames none of whose
    precursors was detected in an MS1 frame inside the range (read from the source)."""
    with closing(sqlite3.connect(TEST_DATA / "analysis.tdf")) as conn:
        return [
            r[0]
            for r in conn.execute(
                "SELECT f.Id FROM Frames f WHERE f.Id BETWEEN ?1 AND ?2 AND NOT ("
                " EXISTS (SELECT 1 FROM PasefFrameMsMsInfo p WHERE p.Frame = f.Id)"
                " AND NOT EXISTS (SELECT 1 FROM PasefFrameMsMsInfo p JOIN Precursors r ON r.Id = p.Precursor"
                "   WHERE p.Frame = f.Id AND r.Parent BETWEEN ?1 AND ?2)"
                ") ORDER BY f.Id",
                (frame_start, frame_end),
            )
        ]


@SKIP_NO_DATA
def test_slice_middle_range(tmp_path):
    """Slicing a middle range keeps the original frame IDs (no renumbering).

    PASEF MS/MS frames whose precursors all come from MS1 frames before the range are
    dropped (#101), so in 100..110 only the frames that can still be read remain.
    """
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=100, frame_end=110)

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        ids = [r[0] for r in conn.execute("SELECT Id FROM Frames ORDER BY Id")]
    assert ids == _source_frames_kept(100, 110)
    assert ids and set(ids) <= set(range(100, 111))


@SKIP_NO_DATA
def test_slice_calibration_preserved(tmp_path):
    """Calibration and metadata tables should be untouched."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=1, frame_end=10)

    with closing(sqlite3.connect(TEST_DATA / "analysis.tdf")) as orig_conn:
        orig_meta = orig_conn.execute("SELECT * FROM GlobalMetadata").fetchall()

    with closing(sqlite3.connect(dest / "analysis.tdf")) as new_conn:
        new_meta = new_conn.execute("SELECT * FROM GlobalMetadata").fetchall()

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

    with closing(sqlite3.connect(src / "analysis.tdf")) as conn:
        conn.execute("UPDATE Frames SET TimsId = NULL WHERE Id = 3")
        conn.commit()

    dest = tmp_path / "broken.d"
    with pytest.raises(ValueError, match="NULL TimsId"):
        slice_d_folder(src, dest, frame_start=1, frame_end=5)

    assert not dest.exists()

    # A range that excludes the broken frame still works. Frames 2..5 are PASEF MS/MS
    # frames of precursors from MS1 frame 1, so the range keeps frame 1 (#101).
    ok_dest = tmp_path / "ok.d"
    slice_d_folder(src, ok_dest, frame_start=1, frame_end=2)
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


PRM_DATA = Path("tests/data/example_prm.d")


@pytest.mark.skipif(not PRM_DATA.exists(), reason="Test data not available")
def test_slice_filters_frame_msms_info_and_ce_sweeping(tmp_path):
    """FrameMsMsInfo and CollisionEnergySweepingInfo rows for dropped frames must go.

    The bundled fixtures leave both tables empty, so rows are added here for every
    frame. Before the fix the slicer kept all of them, pointing at deleted frames.
    """
    src = tmp_path / "src.d"
    shutil.copytree(PRM_DATA, src)
    with closing(sqlite3.connect(src / "analysis.tdf")) as conn, conn:
        frame_ids = [r[0] for r in conn.execute("SELECT Id FROM Frames")]
        conn.executemany(
            "INSERT INTO FrameMsMsInfo (Frame, Parent, TriggerMass, IsolationWidth, PrecursorCharge, CollisionEnergy) VALUES (?, ?, 500.0, 2.0, 2, 30.0)",
            [(f, max(f - 1, 1)) for f in frame_ids],
        )
        conn.executemany(
            "INSERT INTO CollisionEnergySweepingInfo (Frame, CollisionId, CollisionEnergy, CollisionEnergyPercent) VALUES (?, ?, 30.0, 50.0)",
            [(f, c) for f in frame_ids for c in (1, 2)],
        )

    dest = tmp_path / "sliced.d"
    slice_d_folder(src, dest, frame_start=5, frame_end=9)

    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        kept = [r[0] for r in conn.execute("SELECT Frame FROM FrameMsMsInfo ORDER BY Frame")]
        sweeping = conn.execute("SELECT DISTINCT Frame FROM CollisionEnergySweepingInfo ORDER BY Frame").fetchall()
    assert kept == [5, 6, 7, 8, 9]
    assert [r[0] for r in sweeping] == [5, 6, 7, 8, 9]


@SKIP_NO_DATA
def test_slice_not_starting_at_frame_1_keeps_tables_in_step(tmp_path):
    """#101: Segments and PasefFrameMsMsInfo agree with the kept frames."""
    dest = tmp_path / "sliced.d"
    slice_d_folder(TEST_DATA, dest, frame_start=21, frame_end=40)

    with closing(sqlite3.connect(TEST_DATA / "analysis.tdf")) as src:
        source_rows = {
            frame: count for frame, count in src.execute("SELECT Frame, COUNT(*) FROM PasefFrameMsMsInfo WHERE Frame BETWEEN 21 AND 40 GROUP BY Frame")
        }
    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        ids = [r[0] for r in conn.execute("SELECT Id FROM Frames ORDER BY Id")]
        assert ids == _source_frames_kept(21, 40)
        assert ids[0] > 21  # frames 21..28 are MS/MS frames of precursors from before frame 21
        assert conn.execute("SELECT Id, FirstFrame, LastFrame, IsCalibrationSegment FROM Segments").fetchall() == [(1, ids[0], ids[-1], 0)]
        # Every kept MS/MS frame has its PasefFrameMsMsInfo rows, as in the source.
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM Frames f WHERE MsMsType = 8 AND NOT EXISTS (SELECT 1 FROM PasefFrameMsMsInfo p WHERE p.Frame = f.Id)"
            ).fetchone()[0]
            == 0
        )
        kept_rows = dict(conn.execute("SELECT Frame, COUNT(*) FROM PasefFrameMsMsInfo GROUP BY Frame").fetchall())
        assert set(kept_rows) == {f for f in ids if f in source_rows}
        # Every precursor's parent MS1 frame is in the slice.
        assert conn.execute("SELECT COUNT(*) FROM Precursors WHERE Parent NOT IN (SELECT Id FROM Frames)").fetchone()[0] == 0

    with DDA(TEST_DATA) as source, DDA(dest) as sliced:
        assert len(sliced.precursors) > 0
        # A precursor whose PASEF windows all lie in the slice reads back unchanged.
        whole = [p for p in sliced.precursors if {i.frame_id for i in source.precursors[p.precursor_id].pasef_frame_msms_infos} <= set(ids)]
        assert whole
        for precursor in whole[:3]:
            np.testing.assert_array_equal(precursor.merged_peaks(), source.precursors[precursor.precursor_id].merged_peaks())


@SKIP_NO_DATA
def test_slice_whose_frames_are_all_orphaned_msms_raises_and_cleans_up(tmp_path):
    dest = tmp_path / "sliced.d"
    with pytest.raises(TdfpyError, match="No frames are left"):
        slice_d_folder(TEST_DATA, dest, frame_start=21, frame_end=28)
    assert not dest.exists()


@pytest.mark.parametrize(
    ("mode", "frame_start", "frame_end", "expected"),
    [
        # DIA fixture: one segment 1..24888, clipped to the slice.
        ("dia", 101, 200, [(1, 101, 200, 0)]),
        # PRM fixture: six calibration segments; only the two overlapping 200..300 remain.
        ("prm", 200, 300, [(1, 200, 254, 1), (2, 255, 300, 1)]),
    ],
)
def test_slice_clips_segments(tmp_path, mode, frame_start, frame_end, expected):
    source = Path(f"tests/data/example_{mode}.d")
    if not source.exists():
        pytest.skip("Test data not available")
    dest = tmp_path / f"{mode}.d"
    slice_d_folder(source, dest, frame_start=frame_start, frame_end=frame_end)
    with closing(sqlite3.connect(dest / "analysis.tdf")) as conn:
        ids = [r[0] for r in conn.execute("SELECT Id FROM Frames ORDER BY Id")]
        assert ids == list(range(frame_start, frame_end + 1))
        rows = conn.execute("SELECT Id, FirstFrame, LastFrame, IsCalibrationSegment FROM Segments ORDER BY Id").fetchall()
    assert rows == expected
