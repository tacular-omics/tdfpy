"""
Utility for slicing a Bruker timsTOF .d folder to a subset of frames.

Produces a smaller, valid .d folder containing only the specified frame range,
with a rebuilt binary file and filtered SQLite metadata.
"""

import logging
import shutil
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

logger = logging.getLogger(__name__)

TDF_FILE = "analysis.tdf"
TDF_BIN_FILE = "analysis.tdf_bin"

# Tables that reference frame IDs and need filtering.
# Each entry is (table_name, column_name_referencing_frame_id).
_FRAME_DEPENDENT_TABLES = [
    ("FrameProperties", "Frame"),
    ("PasefFrameMsMsInfo", "Frame"),
    ("DiaFrameMsMsInfo", "Frame"),
    ("PrmFrameMsMsInfo", "Frame"),
    ("PrmFrameMeasurementMode", "Frame"),
]


def slice_d_folder(
    source_dir: str | Path,
    dest_dir: str | Path,
    frame_start: int,
    frame_end: int,
) -> Path:
    """Slice a .d folder to contain only frames in [frame_start, frame_end].

    Creates a new .d folder at ``dest_dir`` with a filtered SQLite database
    and a rebuilt binary file containing only the kept frames' data.

    Parameters
    ----------
    source_dir : str | Path
        Path to the source .d folder.
    dest_dir : str | Path
        Path for the output .d folder (must not already exist).
    frame_start : int
        First frame ID to keep (inclusive, 1-based).
    frame_end : int
        Last frame ID to keep (inclusive, 1-based).

    Returns
    -------
    Path
        The path to the created .d folder.

    Raises
    ------
    FileNotFoundError
        If the source .d folder is missing ``analysis.tdf`` or
        ``analysis.tdf_bin``.
    FileExistsError
        If ``dest_dir`` already exists. This function never overwrites or
        deletes pre-existing data.
    ValueError
        If ``frame_start > frame_end``, if ``dest_dir`` resolves to the source
        folder or to a location inside it, if no frames fall in the requested
        range, or if a kept frame has a NULL ``TimsId``.

    Notes
    -----
    If writing fails part-way through, the partially written destination
    folder is removed before the error propagates, so a failed slice never
    leaves a corrupt .d folder behind.
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    _validate_inputs(source_dir, dest_dir, frame_start, frame_end)

    logger.info(
        "Slicing .d folder %s → %s, frames [%d, %d] (inclusive).",
        source_dir,
        dest_dir,
        frame_start,
        frame_end,
    )

    dest_dir.mkdir(parents=True)
    try:
        frame_ids = _write_slice(source_dir, dest_dir, frame_start, frame_end)
    except BaseException:
        # Never leave a half-written .d folder behind — it would look valid to
        # a reader but contain truncated binary data.
        logger.warning(
            "slice_d_folder: writing %s failed; removing the partial destination.",
            dest_dir,
        )
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise

    logger.info(
        "slice_d_folder: wrote %s (%d frames, binary rebuilt).",
        dest_dir,
        len(frame_ids),
    )
    return dest_dir


def _write_slice(
    source_dir: Path, dest_dir: Path, frame_start: int, frame_end: int
) -> list[int]:
    """Populate an existing, empty ``dest_dir`` with the sliced .d contents.

    Returns the list of kept frame IDs.
    """
    # Step 1: Copy SQLite database and read original offsets before filtering.
    src_tdf = source_dir / TDF_FILE
    dst_tdf = dest_dir / TDF_FILE
    shutil.copy2(src_tdf, dst_tdf)

    with closing(sqlite3.connect(dst_tdf)) as conn, conn:
        # Read original offsets for frames we're keeping (before any DELETEs).
        rows = conn.execute(
            "SELECT Id, TimsId FROM Frames WHERE Id >= ? AND Id <= ? ORDER BY Id",
            (frame_start, frame_end),
        ).fetchall()

        if not rows:
            lo, hi = conn.execute("SELECT MIN(Id), MAX(Id) FROM Frames").fetchone()
            available = f"{lo}..{hi}" if lo is not None else "none (Frames table is empty)"
            raise ValueError(
                f"No frames in the requested range [{frame_start}, {frame_end}] "
                f"(inclusive). Source .d folder has frame IDs {available}; frame "
                "IDs are 1-based."
            )

        null_offset_ids = [r[0] for r in rows if r[1] is None]
        if null_offset_ids:
            preview = ", ".join(str(i) for i in null_offset_ids[:10])
            if len(null_offset_ids) > 10:
                preview += ", …"
            raise ValueError(
                f"{len(null_offset_ids)} frame(s) in the requested range "
                f"[{frame_start}, {frame_end}] have a NULL TimsId and therefore "
                f"no binary blob to copy (frame IDs: {preview}). Slice a range "
                "that excludes them, or repair the source .d folder."
            )

        frame_ids = [r[0] for r in rows]
        original_offsets = [r[1] for r in rows]

        logger.info(
            "slice_d_folder: keeping %d frames (IDs %d..%d).",
            len(frame_ids),
            frame_ids[0],
            frame_ids[-1],
        )

        # Step 2: Rebuild binary file with only kept frames.
        src_bin = source_dir / TDF_BIN_FILE
        dst_bin = dest_dir / TDF_BIN_FILE
        new_offsets = _rebuild_binary(src_bin, dst_bin, original_offsets)

        # Step 3: Filter SQLite tables.
        _filter_sqlite(conn, frame_start, frame_end)

        # Step 4: Update offsets in Frames table.
        conn.executemany(
            "UPDATE Frames SET TimsId = ? WHERE Id = ?",
            list(zip(new_offsets, frame_ids)),
        )

    # VACUUM must run outside a transaction.
    with closing(sqlite3.connect(dst_tdf)) as conn:
        conn.execute("VACUUM")

    return frame_ids


def _validate_inputs(
    source_dir: Path, dest_dir: Path, frame_start: int, frame_end: int
) -> None:
    if not (source_dir / TDF_FILE).exists():
        raise FileNotFoundError(f"{TDF_FILE} not found in {source_dir}")
    if not (source_dir / TDF_BIN_FILE).exists():
        raise FileNotFoundError(f"{TDF_BIN_FILE} not found in {source_dir}")
    if frame_start > frame_end:
        raise ValueError(
            f"frame_start ({frame_start}) must be <= frame_end ({frame_end})"
        )

    resolved_source = source_dir.resolve()
    resolved_dest = dest_dir.resolve()
    if resolved_dest == resolved_source:
        raise ValueError(
            f"dest_dir ({dest_dir}) resolves to the source .d folder "
            f"({source_dir}). Slicing in place is not supported; choose a "
            "different destination."
        )
    if resolved_source in resolved_dest.parents:
        raise ValueError(
            f"dest_dir ({dest_dir}) resolves to a location inside the source .d "
            f"folder ({source_dir}). Choose a destination outside the source."
        )
    if dest_dir.exists():
        raise FileExistsError(
            f"dest_dir ({dest_dir}) already exists. slice_d_folder never "
            "overwrites an existing path; remove it first or choose another "
            "destination."
        )


def _rebuild_binary(
    src_bin: Path, dst_bin: Path, original_offsets: list[int]
) -> list[int]:
    """Copy only the kept frames' blobs to a new binary file.

    Returns the list of new offsets corresponding to each kept frame.
    """
    new_offsets: list[int] = []

    with open(src_bin, "rb") as src, open(dst_bin, "wb") as dst:
        for offset in original_offsets:
            src.seek(offset)

            # Read byte_count (total blob size including this 4-byte field).
            header = src.read(4)
            if len(header) < 4:
                raise IOError(
                    f"Failed to read blob header at offset {offset}"
                )
            (byte_count,) = struct.unpack("<I", header)

            # Read remaining bytes (scan_count + compressed data).
            remaining = byte_count - 4
            if remaining < 0:
                raise IOError(
                    f"Invalid byte_count {byte_count} at offset {offset}"
                )
            blob_rest = src.read(remaining)
            if len(blob_rest) < remaining:
                raise IOError(
                    f"Truncated blob at offset {offset}: expected {remaining} "
                    f"bytes, got {len(blob_rest)}"
                )

            new_offsets.append(dst.tell())
            dst.write(header)
            dst.write(blob_rest)

    return new_offsets


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row[0] > 0


def _filter_sqlite(
    conn: sqlite3.Connection, frame_start: int, frame_end: int
) -> None:
    """Delete rows outside the kept frame range from all relevant tables."""
    # Delete from Frames.
    conn.execute(
        "DELETE FROM Frames WHERE Id < ? OR Id > ?",
        (frame_start, frame_end),
    )

    # Delete from tables that directly reference Frame IDs.
    for table, col in _FRAME_DEPENDENT_TABLES:
        if _table_exists(conn, table):
            conn.execute(
                f"DELETE FROM {table} WHERE {col} < ? OR {col} > ?",
                (frame_start, frame_end),
            )

    # Delete orphaned Precursors (Parent frame no longer exists).
    if _table_exists(conn, "Precursors"):
        conn.execute(
            "DELETE FROM Precursors WHERE Parent < ? OR Parent > ?",
            (frame_start, frame_end),
        )
        # Also clean up PasefFrameMsMsInfo rows referencing deleted precursors.
        if _table_exists(conn, "PasefFrameMsMsInfo"):
            conn.execute(
                "DELETE FROM PasefFrameMsMsInfo "
                "WHERE Precursor NOT IN (SELECT Id FROM Precursors)",
            )

    # Delete orphaned DIA windows.
    if _table_exists(conn, "DiaFrameMsMsWindows") and _table_exists(
        conn, "DiaFrameMsMsInfo"
    ):
        conn.execute(
            "DELETE FROM DiaFrameMsMsWindows "
            "WHERE WindowGroup NOT IN "
            "(SELECT DISTINCT WindowGroup FROM DiaFrameMsMsInfo)",
        )
