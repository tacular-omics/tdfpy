"""Tests for :func:`tdfpy._sqlite.connect_readonly`, the one way the package opens ``analysis.tdf``.

Each property here used to be a property of five separate copies of the same
connect expression, asserted in none of them.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

import tdfpy
from tdfpy._sqlite import connect_readonly


def _tiny_db(path: Path) -> Path:
    """A one-row SQLite file at ``path``, created with a writable connection."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute("CREATE TABLE GlobalMetadata (Key TEXT, Value TEXT)")
        conn.execute("INSERT INTO GlobalMetadata VALUES ('SchemaType', 'TDF')")
        conn.commit()
    return path


def test_reads_rows(tmp_path: Path) -> None:
    db = _tiny_db(tmp_path / "analysis.tdf")
    with closing(connect_readonly(db)) as conn:
        assert conn.execute("SELECT Value FROM GlobalMetadata").fetchone()[0] == "TDF"


def test_refuses_writes(tmp_path: Path) -> None:
    """``mode=ro`` is what keeps the package off the acquisition it is reading."""
    db = _tiny_db(tmp_path / "analysis.tdf")
    with closing(connect_readonly(db)) as conn:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM GlobalMetadata")


def test_accepts_a_str_and_a_relative_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The call sites pass ``str``, ``Path`` and paths relative to the cwd."""
    db = _tiny_db(tmp_path / "analysis.tdf")
    with closing(connect_readonly(str(db))) as conn:
        assert conn.execute("SELECT count(*) FROM GlobalMetadata").fetchone()[0] == 1
    monkeypatch.chdir(tmp_path)
    with closing(connect_readonly(Path("analysis.tdf"))) as conn:
        assert conn.execute("SELECT count(*) FROM GlobalMetadata").fetchone()[0] == 1


def test_does_not_create_a_missing_database(tmp_path: Path) -> None:
    """Without ``mode=ro`` SQLite would create an empty file and report no tables."""
    missing = tmp_path / "analysis.tdf"
    with pytest.raises(sqlite3.OperationalError):
        with closing(connect_readonly(missing)) as conn:
            conn.execute("SELECT 1 FROM sqlite_master")
    assert not missing.exists()


def test_escapes_characters_a_uri_reserves(tmp_path: Path) -> None:
    """A ``#`` in a folder name is a URI fragment unless it is escaped."""
    db = _tiny_db(tmp_path / "read only #1.d" / "analysis.tdf")
    with closing(connect_readonly(db)) as conn:
        assert conn.execute("SELECT Value FROM GlobalMetadata").fetchone()[0] == "TDF"


def test_every_metadata_connection_goes_through_the_helper() -> None:
    """One copy of the connect expression, so the URI and the mode cannot drift apart."""
    package = Path(tdfpy.__file__).parent
    modules = sorted(package.rglob("*.py"))
    assert len(modules) > 10, f"non-vacuity: only {len(modules)} modules found under {package}"
    callers = {p.relative_to(package).as_posix() for p in modules if "?mode=ro" in p.read_text(encoding="utf-8")}
    assert callers == {"_sqlite.py"}, f"the read-only connect expression is spelled out outside _sqlite.py: {sorted(callers - {'_sqlite.py'})}"
