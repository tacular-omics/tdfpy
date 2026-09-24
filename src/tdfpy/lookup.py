"""ID-indexed lookups returned by the reader properties (``dda.precursors``, ``dia.windows``, ...).

Every lookup is iterable, has ``len()``, is indexed by an integer ID and has
``get(id, default)``. A missing ID raises :class:`~tdfpy.TdfpyKeyError`.
Lookups that map one ID to several elements (``DiaWindowLookup``,
``PrmTransitionLookup``) return a ``tuple``. ``query`` / ``query_range``
arguments are keyword-only.

Argument names follow one rule: a ``(low, high)`` tuple is ``*_range``
(``rt_range``, ``ook0_range``, ``precursor_mz_range``) and goes to
``query_range``; a point value is ``rt`` / ``precursor_mz`` / ``ook0`` plus a
``*_tolerance`` and goes to ``query``. Ranges and tolerances are inclusive.
"""

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Literal, overload

from ._validation import choice, nonnegative
from .elems import DiaWindow, DiaWindowGroup, Frame, Precursor, PrmTarget, PrmTransition
from .errors import TdfpyKeyError

__all__ = [
    "DiaWindowLookup",
    "Ms1FrameLookup",
    "PrecursorLookup",
    "PrmTargetLookup",
    "PrmTransitionLookup",
]


def _missing_id_error(label: str, requested: int, available: Iterable[int]) -> TdfpyKeyError:
    """Build an actionable ``TdfpyKeyError`` for a lookup miss.

    Names the requested id and summarises what *is* available so callers (and
    LLM agents) can immediately see the valid range instead of guessing.
    """
    ids = sorted(available)
    if not ids:
        detail = "none are loaded"
    elif len(ids) == 1:
        detail = f"only {ids[0]} is loaded"
    else:
        detail = f"loaded range is {ids[0]}..{ids[-1]}, count={len(ids)}"
    return TdfpyKeyError(f"{label} {requested} not found ({detail}). Use .get(id, default) to avoid raising, or iterate this lookup to list what is available.")


def _tolerance_range(value: float | None, tolerance: float) -> tuple[float, float] | None:
    return None if value is None else (value - tolerance, value + tolerance)


def _mz_range(mz: float | None, tolerance: float, tolerance_type: Literal["ppm", "da"]) -> tuple[float, float] | None:
    choice("mz_tolerance_type", tolerance_type, ("ppm", "da"))
    nonnegative("mz_tolerance", tolerance)
    if mz is None:
        return None
    width = mz * tolerance / 1e6 if tolerance_type == "ppm" else tolerance
    return (mz - width, mz + width)


def _in(value: float, bounds: tuple[float, float] | None) -> bool:
    return bounds is None or bounds[0] <= value <= bounds[1]


class _IdLookup[V, R]:
    """Shared ID-indexing behaviour. ``R`` is what one ID maps to."""

    _label = "ID"

    def __init__(self, items: Iterable[V], index: Mapping[int, R]):
        self._items: tuple[V, ...] = tuple(items)
        self._index: dict[int, R] = dict(index)

    def __iter__(self) -> Iterator[V]:
        """Iterate over every element."""
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._index

    def __getitem__(self, key: int) -> R:
        """Look up by ID.

        Raises:
            TdfpyKeyError: If the ID is not present.
        """
        try:
            return self._index[key]
        except KeyError:
            raise _missing_id_error(self._label, key, self._index) from None

    @overload
    def get(self, key: int) -> R | None: ...
    @overload
    def get[D](self, key: int, default: D) -> R | D: ...
    def get(self, key: int, default: object = None) -> object:
        """Return the entry for ``key``, or ``default`` (``None``) if it is not present."""
        return self._index.get(key, default)

    def ids(self) -> tuple[int, ...]:
        """Every ID this lookup is indexed by, in insertion order."""
        return tuple(self._index)


def _group[V](items: Iterable[V], by: Callable[[V], int]) -> dict[int, tuple[V, ...]]:
    grouped: dict[int, list[V]] = {}
    for item in items:
        grouped.setdefault(by(item), []).append(item)
    return {k: tuple(v) for k, v in grouped.items()}


class Ms1FrameLookup[T: Frame](_IdLookup[T, T]):
    """MS1 frames of a reader, indexed by frame ID, with RT queries."""

    _label = "MS1 frame ID"

    def __init__(self, frames: Mapping[int, T]):
        super().__init__(frames.values(), frames)

    def query_range(self, *, rt_range: tuple[float, float] | None = None) -> Iterator[T]:
        """MS1 frames inside an RT range, in frame order.

        Args:
            rt_range: ``(min_rt, max_rt)`` in seconds, inclusive. ``None`` keeps every frame.

        Yields:
            Matching MS1 frames.
        """
        for frame in self._items:
            if _in(frame.rt, rt_range):
                yield frame

    def query(self, *, rt: float | None = None, rt_tolerance: float = 30.0) -> Iterator[T]:
        """MS1 frames within ``rt_tolerance`` of ``rt``, in frame order.

        Args:
            rt: Target retention time in seconds. ``None`` keeps every frame.
            rt_tolerance: RT tolerance in seconds (default 30).

        Yields:
            Matching MS1 frames.

        Raises:
            TdfpyError: If ``rt_tolerance`` is negative.
        """
        nonnegative("rt_tolerance", rt_tolerance)
        return self.query_range(rt_range=_tolerance_range(rt, rt_tolerance))


class DiaWindowLookup(_IdLookup[DiaWindow, tuple[DiaWindow, ...]]):
    """Per-frame DIA windows, indexed by window group ID.

    ``lookup[group_id]`` returns a tuple: every frame that used the group
    contributes one window per window definition.
    """

    _label = "DIA window group ID"

    def __init__(self, windows: Iterable[DiaWindow]):
        windows = tuple(windows)
        super().__init__(windows, _group(windows, lambda w: w.window_group_id))

    def query_range(
        self,
        *,
        window_group: int | DiaWindowGroup | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[DiaWindow]:
        """Windows in a window group and/or retention time range.

        Args:
            window_group: Window group ID, or a `DiaWindowGroup` (its
                `window_group_id` is used). ``None`` keeps every group.
            rt_range: ``(min_rt, max_rt)`` in seconds, inclusive. ``None`` skips RT filtering.

        Yields:
            Matching `DiaWindow` objects.
        """
        group_id = window_group.window_group_id if isinstance(window_group, DiaWindowGroup) else window_group
        for window in self._items:
            if group_id is not None and window.window_group_id != group_id:
                continue
            if _in(window.rt, rt_range):
                yield window

    def query(
        self,
        *,
        window_group: int | DiaWindowGroup | None = None,
        rt: float | None = None,
        rt_tolerance: float = 30.0,
    ) -> Iterator[DiaWindow]:
        """Windows in a window group and/or within ``rt_tolerance`` of ``rt``.

        Args:
            window_group: Window group ID or `DiaWindowGroup`. ``None`` keeps every group.
            rt: Target retention time in seconds. ``None`` skips RT filtering.
            rt_tolerance: RT tolerance in seconds (default 30).

        Yields:
            Matching `DiaWindow` objects.

        Raises:
            TdfpyError: If ``rt_tolerance`` is negative.
        """
        nonnegative("rt_tolerance", rt_tolerance)
        return self.query_range(window_group=window_group, rt_range=_tolerance_range(rt, rt_tolerance))


class PrecursorLookup(_IdLookup[Precursor, Precursor]):
    """DDA precursors, indexed by precursor ID, with m/z and RT queries."""

    _label = "Precursor ID"

    def __init__(self, precursors: Mapping[int, Precursor]):
        super().__init__(precursors.values(), precursors)

    def query_range(
        self,
        *,
        precursor_mz_range: tuple[float, float] | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[Precursor]:
        """Precursors inside precursor m/z and/or RT ranges (inclusive).

        Args:
            precursor_mz_range: ``(min_mz, max_mz)`` of :attr:`Precursor.precursor_mz`
                (monoisotopic when known, else the largest peak). ``None`` skips m/z filtering.
            rt_range: ``(min_rt, max_rt)`` in seconds. ``None`` skips RT filtering.

        Yields:
            Matching `Precursor` objects.
        """
        for precursor in self._items:
            if _in(precursor.precursor_mz, precursor_mz_range) and _in(precursor.rt, rt_range):
                yield precursor

    def query(
        self,
        *,
        precursor_mz: float | None = None,
        rt: float | None = None,
        mz_tolerance: float = 20.0,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        rt_tolerance: float = 30.0,
    ) -> Iterator[Precursor]:
        """Precursors within a tolerance of ``precursor_mz`` and/or ``rt``.

        Args:
            precursor_mz: Target precursor m/z. ``None`` skips m/z filtering.
            rt: Target retention time in seconds. ``None`` skips RT filtering.
            mz_tolerance: m/z tolerance (default 20).
            mz_tolerance_type: ``"ppm"`` (default) or ``"da"``.
            rt_tolerance: RT tolerance in seconds (default 30).

        Yields:
            Matching `Precursor` objects, matched on :attr:`Precursor.precursor_mz`.

        Raises:
            TdfpyError: If a tolerance is negative or ``mz_tolerance_type`` is unknown.
        """
        precursor_mz_range = _mz_range(precursor_mz, mz_tolerance, mz_tolerance_type)
        nonnegative("rt_tolerance", rt_tolerance)
        return self.query_range(precursor_mz_range=precursor_mz_range, rt_range=_tolerance_range(rt, rt_tolerance))


class PrmTargetLookup(_IdLookup[PrmTarget, PrmTarget]):
    """PRM targets, indexed by target ID, with m/z, RT and 1/K0 queries."""

    _label = "PRM target ID"

    def __init__(self, targets: Mapping[int, PrmTarget]):
        super().__init__(targets.values(), targets)

    def query_range(
        self,
        *,
        precursor_mz_range: tuple[float, float] | None = None,
        rt_range: tuple[float, float] | None = None,
        ook0_range: tuple[float, float] | None = None,
    ) -> Iterator[PrmTarget]:
        """Targets inside precursor m/z, RT and/or 1/K0 ranges (inclusive).

        Args:
            precursor_mz_range: ``(min_mz, max_mz)`` of :attr:`PrmTarget.precursor_mz`. ``None`` skips m/z filtering.
            rt_range: ``(min_rt, max_rt)`` in seconds. ``None`` skips RT filtering.
            ook0_range: ``(min_ook0, max_ook0)``. ``None`` skips 1/K0 filtering.

        Yields:
            Matching `PrmTarget` objects.
        """
        for target in self._items:
            if _in(target.precursor_mz, precursor_mz_range) and _in(target.rt, rt_range) and _in(target.ook0, ook0_range):
                yield target

    def query(
        self,
        *,
        precursor_mz: float | None = None,
        rt: float | None = None,
        ook0: float | None = None,
        mz_tolerance: float = 20.0,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        rt_tolerance: float = 30.0,
        ook0_tolerance: float = 0.05,
    ) -> Iterator[PrmTarget]:
        """Targets within a tolerance of ``precursor_mz``, ``rt`` and/or ``ook0``.

        Args:
            precursor_mz: Target precursor m/z. ``None`` skips m/z filtering.
            rt: Target retention time in seconds. ``None`` skips RT filtering.
            ook0: Target 1/K0. ``None`` skips 1/K0 filtering.
            mz_tolerance: m/z tolerance (default 20).
            mz_tolerance_type: ``"ppm"`` (default) or ``"da"``.
            rt_tolerance: RT tolerance in seconds (default 30).
            ook0_tolerance: Absolute 1/K0 tolerance (default 0.05).

        Yields:
            Matching `PrmTarget` objects.

        Raises:
            TdfpyError: If a tolerance is negative or ``mz_tolerance_type`` is unknown.
        """
        precursor_mz_range = _mz_range(precursor_mz, mz_tolerance, mz_tolerance_type)
        nonnegative("rt_tolerance", rt_tolerance)
        nonnegative("ook0_tolerance", ook0_tolerance)
        return self.query_range(
            precursor_mz_range=precursor_mz_range,
            rt_range=_tolerance_range(rt, rt_tolerance),
            ook0_range=_tolerance_range(ook0, ook0_tolerance),
        )


class PrmTransitionLookup(_IdLookup[PrmTransition, tuple[PrmTransition, ...]]):
    """PRM transitions, indexed by target ID.

    ``lookup[target_id]`` returns a tuple: a target is acquired in many frames.
    """

    _label = "PRM transition target ID"

    def __init__(self, transitions: Iterable[PrmTransition]):
        transitions = tuple(transitions)
        super().__init__(transitions, _group(transitions, lambda t: t.target.target_id))

    def query_range(
        self,
        *,
        target: int | PrmTarget | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[PrmTransition]:
        """Transitions of one target and/or inside an RT range.

        Args:
            target: Target ID or `PrmTarget`. ``None`` keeps every target.
            rt_range: ``(min_rt, max_rt)`` in seconds. ``None`` skips RT filtering.

        Yields:
            Matching `PrmTransition` objects.
        """
        target_id = target.target_id if isinstance(target, PrmTarget) else target
        for transition in self._items:
            if target_id is not None and transition.target.target_id != target_id:
                continue
            if _in(transition.rt, rt_range):
                yield transition

    def query(
        self,
        *,
        target: int | PrmTarget | None = None,
        rt: float | None = None,
        rt_tolerance: float = 30.0,
    ) -> Iterator[PrmTransition]:
        """Transitions of one target and/or within ``rt_tolerance`` of ``rt``.

        Args:
            target: Target ID or `PrmTarget`. ``None`` keeps every target.
            rt: Target retention time in seconds. ``None`` skips RT filtering.
            rt_tolerance: RT tolerance in seconds (default 30).

        Yields:
            Matching `PrmTransition` objects.

        Raises:
            TdfpyError: If ``rt_tolerance`` is negative.
        """
        nonnegative("rt_tolerance", rt_tolerance)
        return self.query_range(target=target, rt_range=_tolerance_range(rt, rt_tolerance))
