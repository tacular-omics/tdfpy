"""
Shared SQLite access for ``analysis.tdf``.

One place for the read-only connection every reader of the metadata database
opens, so the URI, the mode and the resolution cannot drift between them.
"""

import sqlite3
from pathlib import Path

__all__ = ["connect_readonly"]


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """
    Open ``db_path`` read-only, the way every reader of ``analysis.tdf`` does.

    The database is opened through a ``file:`` URI with ``mode=ro``, so SQLite
    refuses to write to the acquisition and raises instead of creating a missing
    file. ``Path.as_uri`` needs an absolute path and percent-escapes the parts a
    URI reserves, such as ``#`` in a folder name, which plain string formatting
    would silently turn into a fragment.

    Args:
        db_path (str | Path): The path to the SQLite database.

    Returns:
        sqlite3.Connection: A new read-only connection. The caller owns it and
            should close it through :func:`contextlib.closing`, not the
            connection's own context manager, which only ends the transaction
            and would leave the handle open until garbage collection.

    Raises:
        sqlite3.Error: If SQLite cannot open the file. Callers translate this
            into :class:`~tdfpy.errors.TdfpyError` with their own message.
    """
    return sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
