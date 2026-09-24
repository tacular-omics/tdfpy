"""Exception hierarchy for tdfpy.

Every error tdfpy raises for bad input or an unreadable ``.d`` folder is a
:class:`TdfpyError`, which subclasses :class:`ValueError`, so one ``except
tdfpy.TdfpyError`` catches them all. Subclasses also inherit the built-in type
the error used to be, so older ``except KeyError`` / ``except RuntimeError`` /
``except NotImplementedError`` handlers keep working.

Missing files still raise :class:`FileNotFoundError`, and a wrong-type argument
may raise :class:`TypeError`.
"""

__all__ = [
    "ReaderClosedError",
    "TdfpyError",
    "TdfpyKeyError",
    "UnsupportedCalibrationError",
    "UnsupportedTdfError",
]


class TdfpyError(ValueError):
    """Base class of every tdfpy error.

    Raised directly for invalid arguments (a negative tolerance, an unknown
    option name) and for SQLite errors while reading ``analysis.tdf``.
    """


class TdfpyKeyError(TdfpyError, KeyError):
    """An ID or key that is not in the acquisition.

    Raised by lookup indexing (``dda.precursors[99999]``), by the ``MetaData``
    and ``Calibration`` key accessors, and by :class:`~tdfpy.TimsData` for an
    unknown frame ID.
    """

    def __str__(self) -> str:
        # KeyError.__str__ wraps the message in quotes; show it plainly.
        return str(self.args[0]) if self.args else ""


class ReaderClosedError(TdfpyError, RuntimeError):
    """Spectral data was requested after the reader was closed.

    Frame elements keep a reference to the reader's :class:`~tdfpy.TimsData`;
    once the ``with`` block exits, their spectral accessors raise this instead of
    returning stale data.
    """


class UnsupportedTdfError(TdfpyError, NotImplementedError):
    """A ``.d`` folder this reader has not been validated against, or a corrupt one.

    Raised for legacy compression types, unsupported reader options and any
    ``analysis.tdf_bin`` whose layout checks fail.
    """


class UnsupportedCalibrationError(TdfpyError, NotImplementedError):
    """A calibration model type that has not been validated.

    Bruker ships several model types; only those seen on real data are
    implemented. An unknown model raises rather than returning plausible but
    wrong numbers.
    """
