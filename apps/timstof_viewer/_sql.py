"""Read-only SQL access for the viewer's ad-hoc query box.

Split out of :mod:`_shared` so it can be imported — and tested — without
Streamlit: this module needs nothing but ``sqlite3`` and ``pandas``.

``analysis.tdf`` is the user's raw acquisition metadata and there is no second
copy of it, so the query box opens the database **read-only** rather than
trusting the shape of the text it was handed. The prefix check in
:func:`check_select_only` is a friendly early error, not the boundary: SQLite
accepts a ``WITH`` clause in front of ``DELETE`` / ``UPDATE`` / ``INSERT``, so a
statement that passes any prefix test can still be a write.

Nothing here is part of the public ``tdfpy`` package — this is internal dev
tooling under ``apps/`` only.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pandas as pd

_ALLOWED_PREFIXES = ("select", "with")


def check_select_only(query: str) -> str:
    """Return ``query`` stripped of whitespace and a trailing ``;``.

    Args:
        query: Raw text from the query box.

    Returns:
        The stripped statement, ready to execute.

    Raises:
        ValueError: If the statement is empty or does not start with ``SELECT``
            or ``WITH``.
    """
    stripped = query.strip().rstrip(";").strip()
    if not stripped:
        raise ValueError("Enter a query.")
    if not stripped.lower().startswith(_ALLOWED_PREFIXES):
        raise ValueError("Only read-only `SELECT` / `WITH` queries are allowed.")
    return stripped


def connect_readonly(analysis_dir: str | Path) -> sqlite3.Connection:
    """Open ``analysis_dir/analysis.tdf`` with SQLite's ``mode=ro``.

    Every read of a user's run goes through here, so no code path in the app
    can write to the raw acquisition metadata: a write raises
    :class:`sqlite3.OperationalError` ("attempt to write a readonly database")
    and the file is untouched. The caller closes the handle — ``sqlite3``
    connections used as a context manager end a *transaction*, not the
    connection.

    Args:
        analysis_dir: Path to the ``.d`` folder.

    Returns:
        An open read-only connection.
    """
    tdf_path = Path(analysis_dir) / "analysis.tdf"
    uri = f"{tdf_path.resolve().as_uri()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def read_sql_query(analysis_dir: str | Path, query: str) -> pd.DataFrame:
    """Run a read-only query against ``analysis_dir/analysis.tdf``.

    The connection comes from :func:`connect_readonly`, so any statement that
    tries to write — including one hidden behind a ``WITH`` clause that
    :func:`check_select_only` cannot reject — fails with
    ``sqlite3.OperationalError`` and leaves the file untouched.

    Args:
        analysis_dir: Path to the ``.d`` folder.
        query: Statement to run; checked by :func:`check_select_only` first.

    Returns:
        The result rows as a :class:`pandas.DataFrame`.

    Raises:
        ValueError: If the statement does not pass :func:`check_select_only`.
        pandas.errors.DatabaseError: If SQLite rejects the statement, including
            every attempt to write to the read-only database (pandas wraps the
            underlying :class:`sqlite3.OperationalError`, which is why this is
            not an :class:`sqlite3.Error`).
    """
    statement = check_select_only(query)
    with closing(connect_readonly(analysis_dir)) as conn:
        return pd.read_sql_query(statement, conn)
