"""The viewer's ad-hoc SQL box must not be able to write to ``analysis.tdf``.

``apps/`` is internal dev tooling, not part of the package, and it imports
Streamlit, which is not a dependency here. The query box's two pure pieces
therefore live in ``apps/timstof_viewer/_sql.py``, which imports only
``sqlite3`` and ``pandas``, and this module loads that file directly by path.

The case these tests exist for: SQLite accepts a ``WITH`` clause in front of
``DELETE``, so ``WITH x AS (SELECT Id FROM Frames) DELETE FROM Frames WHERE Id
IN (SELECT Id FROM x)`` passes any ``startswith("with")`` check and empties the
table. The guarantee is the read-only connection, so every test here asserts on
the bytes of the database file, not only on the exception.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pandas as pd
import pytest

_SQL_MODULE_PATH = Path(__file__).resolve().parents[1] / "apps" / "timstof_viewer" / "_sql.py"


def _load_sql_module():
    """Import ``apps/timstof_viewer/_sql.py`` by path, with no Streamlit involved.

    A missing file is a failure, not a skip: a skip here would silently retire
    the only coverage the query box has.
    """
    assert _SQL_MODULE_PATH.is_file(), f"not found: {_SQL_MODULE_PATH}"
    spec = importlib.util.spec_from_file_location("_tdfpy_apps_sql", _SQL_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_sql = _load_sql_module()

DATA_DIR = Path(__file__).parent / "data"
SOURCE_D = DATA_DIR / "example_dda.d"

#: Statements that must never reach the database. The first two hide behind the
#: ``WITH`` prefix, which the old prefix-only check accepted.
WRITE_STATEMENTS = [
    "WITH doomed AS (SELECT Id FROM Frames) DELETE FROM Frames WHERE Id IN (SELECT Id FROM doomed)",
    "WITH t AS (SELECT 1) UPDATE Frames SET Time = 0",
    "DELETE FROM Frames",
    "UPDATE Frames SET Time = 0",
    "DROP TABLE Frames",
    "INSERT INTO Frames (Id) VALUES (999999)",
]


@pytest.fixture
def analysis_dir(tmp_path: Path) -> Path:
    """A throwaway copy of the DDA fixture, so a write would be visible."""
    dest = tmp_path / "copy.d"
    shutil.copytree(SOURCE_D, dest)
    return dest


def _digest(analysis_dir: Path) -> str:
    return hashlib.sha256((analysis_dir / "analysis.tdf").read_bytes()).hexdigest()


def _frame_count(analysis_dir: Path) -> int:
    with sqlite3.connect(analysis_dir / "analysis.tdf") as conn:
        return int(conn.execute("SELECT COUNT(*) FROM Frames").fetchone()[0])


@pytest.mark.parametrize("statement", WRITE_STATEMENTS)
def test_write_statements_leave_the_database_untouched(analysis_dir: Path, statement: str) -> None:
    before_digest, before_rows = _digest(analysis_dir), _frame_count(analysis_dir)
    assert before_rows > 0

    with pytest.raises((ValueError, sqlite3.Error, pd.errors.DatabaseError)):
        _sql.read_sql_query(analysis_dir, statement)

    assert _frame_count(analysis_dir) == before_rows
    assert _digest(analysis_dir) == before_digest


def test_with_delete_is_rejected_by_the_connection_not_the_prefix(analysis_dir: Path) -> None:
    """The ``WITH ... DELETE`` case passes the prefix check and must still fail."""
    statement = WRITE_STATEMENTS[0]
    assert _sql.check_select_only(statement) == statement, "the prefix check accepts this on purpose"

    with pytest.raises(pd.errors.DatabaseError, match="attempt to write a readonly database"):
        _sql.read_sql_query(analysis_dir, statement)


@pytest.mark.parametrize("statement", ["DELETE FROM Frames", "VACUUM", "PRAGMA writable_schema = 1"])
def test_non_select_prefixes_are_rejected_early(statement: str) -> None:
    with pytest.raises(ValueError, match="SELECT"):
        _sql.check_select_only(statement)


def test_empty_query_is_rejected() -> None:
    with pytest.raises(ValueError, match="Enter a query"):
        _sql.check_select_only("   \n  ")


def test_trailing_semicolon_and_whitespace_are_stripped() -> None:
    assert _sql.check_select_only("  SELECT 1 ;  ") == "SELECT 1"


def test_select_still_works(analysis_dir: Path) -> None:
    frame = _sql.read_sql_query(analysis_dir, "SELECT Id, Time FROM Frames ORDER BY Id LIMIT 5")
    assert list(frame.columns) == ["Id", "Time"]
    assert len(frame) == 5
    assert frame["Id"].tolist() == [1, 2, 3, 4, 5]


#: The query ``accumulated_precursor_spectrum`` runs, verbatim
#: (``apps/timstof_viewer/_shared.py``), and a precursor that has segments in the
#: DDA fixture.
SEGMENT_QUERY = "SELECT Frame, ScanNumBegin, ScanNumEnd FROM PasefFrameMsMsInfo WHERE Precursor = ? ORDER BY Frame"
SEGMENT_PRECURSOR_ID = 411


@pytest.mark.parametrize("statement", ["DELETE FROM Frames", "UPDATE Frames SET Time = 0", "DROP TABLE Frames"])
def test_connect_readonly_rejects_a_write(analysis_dir: Path, statement: str) -> None:
    """Writes fail on the handle itself, with no statement inspection involved.

    This is the guarantee the rest of the app leans on: every reader goes
    through :func:`connect_readonly`, so a stray write anywhere raises instead
    of editing the user's raw metadata.
    """
    before_digest, before_rows = _digest(analysis_dir), _frame_count(analysis_dir)
    assert before_rows > 0

    with closing(_sql.connect_readonly(analysis_dir)) as conn, pytest.raises(sqlite3.OperationalError, match="readonly database"):
        conn.execute(statement)

    assert _frame_count(analysis_dir) == before_rows
    assert _digest(analysis_dir) == before_digest


def test_connect_readonly_still_serves_the_precursor_segment_query(analysis_dir: Path) -> None:
    """The read the fix touched must keep returning the same rows."""
    with closing(_sql.connect_readonly(analysis_dir)) as conn:
        rows = conn.execute(SEGMENT_QUERY, (SEGMENT_PRECURSOR_ID,)).fetchall()

    assert len(rows) == 8, "precursor 411 has eight PASEF segments in the DDA fixture"
    assert all(len(row) == 3 for row in rows)
    assert rows == sorted(rows, key=lambda row: row[0]), "ORDER BY Frame"


def test_with_select_still_works(analysis_dir: Path) -> None:
    """A legitimate CTE must keep working: the fix is the connection, not a ban on ``WITH``."""
    frame = _sql.read_sql_query(
        analysis_dir,
        "WITH first_frames AS (SELECT Id FROM Frames ORDER BY Id LIMIT 3) SELECT COUNT(*) AS n FROM first_frames",
    )
    assert frame["n"].tolist() == [3]
