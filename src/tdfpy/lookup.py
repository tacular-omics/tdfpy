from collections.abc import Iterable, Iterator
from typing import Generic, Literal, TypeVar

from .elems import DiaWindow, DiaWindowGroup, Frame, Precursor, PrmTarget, PrmTransition

T = TypeVar("T", bound=Frame)


def _missing_id_error(label: str, requested: int, available: Iterable[int]) -> KeyError:
    """Build an actionable ``KeyError`` for a lookup miss.

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
    return KeyError(
        f"{label} {requested} not found ({detail}). Use .get(id, default) to avoid "
        "raising, or iterate this lookup to list what is available."
    )


class Ms1FrameLookup(Generic[T]):
    """
    A class to perform lookups on MS1 frames.
    Can be iterated over to yield all frames.
    Can be indexed by frame ID.
    """

    def __init__(self, frames: dict[int, T]):
        self._frames = frames

    def __iter__(self) -> Iterator[T]:
        """Iterate over all frames."""
        return iter(self._frames.values())

    def __getitem__(self, frame_id: int) -> T:
        """Get a frame by its ID."""
        if frame_id not in self._frames:
            raise _missing_id_error("MS1 frame ID", frame_id, self._frames)
        return self._frames[frame_id]

    def __len__(self) -> int:
        return len(self._frames)

    def get(self, frame_id: int, default=None):
        """Return the frame with the given ID, or `default` if not found."""
        return self._frames.get(frame_id, default)


class DiaWindowLookup:
    """
    A class to perform lookups on DIA windows.
    Can be iterated over to yield all windows.
    Can be indexed by window ID (which is equivalent to window_group).
    """

    def __init__(self, windows: list[DiaWindow]):
        self._windows = windows
        # Map window_group to list of windows with that group ID
        self._window_map: dict[int, list[DiaWindow]] = {}
        for w in windows:
            if w.window_group not in self._window_map:
                self._window_map[w.window_group] = []
            self._window_map[w.window_group].append(w)

    def __iter__(self) -> Iterator[DiaWindow]:
        """Iterate over all windows."""
        return iter(self._windows)

    def __getitem__(self, window_group_id: int) -> list[DiaWindow]:
        """Get windows by window_group ID. Returns a list as multiple frames can share a window group."""
        if window_group_id not in self._window_map:
            raise _missing_id_error(
                "DIA window group ID", window_group_id, self._window_map
            )
        return self._window_map[window_group_id]

    def __len__(self) -> int:
        return len(self._windows)

    def get(self, window_group_id: int, default=None):
        """Return windows for the given window group ID, or `default` if not found."""
        return self._window_map.get(window_group_id, default)

    def query_range(
        self,
        window_group_index: int | DiaWindowGroup | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[DiaWindow]:
        """
        Query windows by window group and/or retention time range.

        Args:
            window_group_index: Window group index or `DiaWindowGroup` to filter by.
                If None, all window groups are included.
            rt_range: Tuple of (min_rt, max_rt) in seconds. If None, RT filtering is skipped.

        Yields:
            DiaWindow objects matching the criteria.
        """
        for window in self._windows:
            if window_group_index is not None:
                if isinstance(window_group_index, DiaWindowGroup):
                    if window.window_group != window_group_index.window_group:
                        continue
                elif window.window_group != window_group_index:
                    continue
            if rt_range is not None:
                if not (rt_range[0] <= window.rt <= rt_range[1]):
                    continue
            yield window

    def query(
        self,
        window_group_index: int | DiaWindowGroup | None = None,
        rt: float | None = None,
        rt_tolerance: float = 30.0,
    ) -> Iterator[DiaWindow]:
        """
        Query windows by retention time.

        Args:
            rt: Target retention time (in seconds). If None, RT filtering is skipped.
            rt_tolerance: Tolerance for retention time matching (in seconds). Default is 30s.
        Yields:
            DiaWindow objects matching the criteria.
        """
        rt_range: tuple[float, float] | None = None
        if rt is not None:
            rt_range = (rt - rt_tolerance, rt + rt_tolerance)
        return self.query_range(
            window_group_index=window_group_index, rt_range=rt_range
        )


class PrecursorLookup:
    """
    A class to perform lookups on precursors.
    Can be iterated over to yield all precursors.
    Can be indexed by precursor ID.
    Provides methods to query by m/z and retention time.
    """

    def __init__(self, precursors: dict[int, Precursor]):
        self._precursors = precursors

    def __iter__(self) -> Iterator[Precursor]:
        """Iterate over all precursors."""
        return iter(self._precursors.values())

    def __getitem__(self, precursor_id: int) -> Precursor:
        """Get a precursor by its ID."""
        if precursor_id not in self._precursors:
            raise _missing_id_error("Precursor ID", precursor_id, self._precursors)
        return self._precursors[precursor_id]

    def __len__(self) -> int:
        return len(self._precursors)

    def get(self, precursor_id: int, default=None):
        """Return the precursor with the given ID, or `default` if not found."""
        return self._precursors.get(precursor_id, default)

    def query_range(
        self,
        mz_range: tuple[float, float] | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[Precursor]:
        """
        Query precursors by m/z and/or retention time ranges.

        Args:
            mz_range: Tuple of (min_mz, max_mz). If None, m/z filtering is skipped.
            rt_range: Tuple of (min_rt, max_rt) in seconds. If None, RT filtering is skipped.
        Yields:
            Precursor objects matching the criteria.
        """
        for precursor in self._precursors.values():
            if mz_range is not None:
                prec_mz = precursor.monoisotopic_mz
                if prec_mz is None:
                    prec_mz = precursor.largest_peak_mz
                if not (mz_range[0] <= prec_mz <= mz_range[1]):
                    continue

            if rt_range is not None:
                if not (rt_range[0] <= precursor.rt <= rt_range[1]):
                    continue

            yield precursor

    def query(
        self,
        mz: float | None = None,
        rt: float | None = None,
        mz_tolerance: float = 20.0,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        rt_tolerance: float = 30.0,
    ) -> Iterator[Precursor]:
        """
        Query precursors by m/z and/or retention time.

        Args:
            mz: Target m/z value. If None, m/z filtering is skipped.
            rt: Target retention time (in seconds). If None, RT filtering is skipped.
            mz_tolerance: Tolerance for m/z matching.
            mz_tolerance_type: Unit for m/z tolerance ("ppm" or "da"). Default is "ppm".
            rt_tolerance: Tolerance for retention time matching (in seconds). Default is 30s.

        Yields:
            Precursor objects matching the criteria.

        Note:
            Uses `monoisotopic_mz` if available, otherwise `largest_peak_mz`.
        """
        mz_range: tuple[float, float] | None = None
        if mz is not None:
            if mz_tolerance_type == "ppm":
                mz_range = (mz - mz * mz_tolerance / 1e6, mz + mz * mz_tolerance / 1e6)
            else:  # da
                mz_range = (mz - mz_tolerance, mz + mz_tolerance)

        rt_range: tuple[float, float] | None = None
        if rt is not None:
            rt_range = (rt - rt_tolerance, rt + rt_tolerance)

        return self.query_range(mz_range=mz_range, rt_range=rt_range)


class PrmTargetLookup:
    """Lookup for PRM targets by target ID, m/z, RT, and 1/K0."""

    def __init__(self, targets: dict[int, PrmTarget]):
        self._targets = targets

    def __iter__(self) -> Iterator[PrmTarget]:
        return iter(self._targets.values())

    def __getitem__(self, target_id: int) -> PrmTarget:
        if target_id not in self._targets:
            raise _missing_id_error("PRM target ID", target_id, self._targets)
        return self._targets[target_id]

    def __len__(self) -> int:
        return len(self._targets)

    def get(self, target_id: int, default=None):
        """Return the target with the given ID, or `default` if not found."""
        return self._targets.get(target_id, default)

    def query_range(
        self,
        mz_range: tuple[float, float] | None = None,
        rt_range: tuple[float, float] | None = None,
        ook0_range: tuple[float, float] | None = None,
    ) -> Iterator[PrmTarget]:
        """Query targets by m/z, RT, and/or 1/K0 ranges.

        Args:
            mz_range: Tuple of (min_mz, max_mz). If None, m/z filtering is skipped.
            rt_range: Tuple of (min_rt, max_rt) in seconds. If None, RT filtering is skipped.
            ook0_range: Tuple of (min_ook0, max_ook0). If None, 1/K0 filtering is skipped.

        Yields:
            PrmTarget objects matching the criteria.
        """
        for target in self._targets.values():
            if mz_range is not None:
                if not (mz_range[0] <= target.monoisotopic_mz <= mz_range[1]):
                    continue
            if rt_range is not None:
                if not (rt_range[0] <= target.time <= rt_range[1]):
                    continue
            if ook0_range is not None:
                if not (ook0_range[0] <= target.one_over_k0 <= ook0_range[1]):
                    continue
            yield target

    def query(
        self,
        mz: float | None = None,
        rt: float | None = None,
        ook0: float | None = None,
        mz_tolerance: float = 20.0,
        mz_tolerance_type: Literal["ppm", "da"] = "ppm",
        rt_tolerance: float = 30.0,
        ook0_tolerance: float = 0.05,
    ) -> Iterator[PrmTarget]:
        """Query targets by m/z, RT, and/or 1/K0 with tolerances.

        Args:
            mz: Target m/z value. If None, m/z filtering is skipped.
            rt: Target retention time (in seconds). If None, RT filtering is skipped.
            ook0: Target 1/K0 value. If None, 1/K0 filtering is skipped.
            mz_tolerance: Tolerance for m/z matching.
            mz_tolerance_type: Unit for m/z tolerance ("ppm" or "da"). Default is "ppm".
            rt_tolerance: Tolerance for retention time matching (in seconds). Default is 30s.
            ook0_tolerance: Absolute tolerance for 1/K0 matching. Default is 0.05.

        Yields:
            PrmTarget objects matching the criteria.
        """
        mz_range: tuple[float, float] | None = None
        if mz is not None:
            if mz_tolerance_type == "ppm":
                mz_range = (mz - mz * mz_tolerance / 1e6, mz + mz * mz_tolerance / 1e6)
            else:
                mz_range = (mz - mz_tolerance, mz + mz_tolerance)

        rt_range: tuple[float, float] | None = None
        if rt is not None:
            rt_range = (rt - rt_tolerance, rt + rt_tolerance)

        ook0_range: tuple[float, float] | None = None
        if ook0 is not None:
            ook0_range = (ook0 - ook0_tolerance, ook0 + ook0_tolerance)

        return self.query_range(mz_range=mz_range, rt_range=rt_range, ook0_range=ook0_range)


class PrmTransitionLookup:
    """Lookup for PRM transitions by target ID and RT."""

    def __init__(self, transitions: list[PrmTransition]):
        self._transitions = transitions
        self._target_map: dict[int, list[PrmTransition]] = {}
        for t in transitions:
            tid = t.target.target_id
            if tid not in self._target_map:
                self._target_map[tid] = []
            self._target_map[tid].append(t)

    def __iter__(self) -> Iterator[PrmTransition]:
        return iter(self._transitions)

    def __getitem__(self, target_id: int) -> list[PrmTransition]:
        """Get transitions by target ID. Returns a list as multiple frames target the same ion."""
        if target_id not in self._target_map:
            raise _missing_id_error(
                "PRM transition target ID", target_id, self._target_map
            )
        return self._target_map[target_id]

    def __len__(self) -> int:
        return len(self._transitions)

    def get(self, target_id: int, default=None):
        """Return transitions for the given target ID, or `default` if not found."""
        return self._target_map.get(target_id, default)

    def query_range(
        self,
        target: int | PrmTarget | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[PrmTransition]:
        """Query transitions by target and/or retention time range.

        Args:
            target: Target ID or PrmTarget to filter by. If None, all targets are included.
            rt_range: Tuple of (min_rt, max_rt) in seconds. If None, RT filtering is skipped.

        Yields:
            PrmTransition objects matching the criteria.
        """
        target_id: int | None = None
        if target is not None:
            target_id = target.target_id if isinstance(target, PrmTarget) else target

        for transition in self._transitions:
            if target_id is not None:
                if transition.target.target_id != target_id:
                    continue
            if rt_range is not None:
                if not (rt_range[0] <= transition.rt <= rt_range[1]):
                    continue
            yield transition

    def query(
        self,
        target: int | PrmTarget | None = None,
        rt: float | None = None,
        rt_tolerance: float = 30.0,
    ) -> Iterator[PrmTransition]:
        """Query transitions by target and/or retention time.

        Args:
            target: Target ID or PrmTarget to filter by. If None, all targets are included.
            rt: Target retention time (in seconds). If None, RT filtering is skipped.
            rt_tolerance: Tolerance for retention time matching (in seconds). Default is 30s.

        Yields:
            PrmTransition objects matching the criteria.
        """
        rt_range: tuple[float, float] | None = None
        if rt is not None:
            rt_range = (rt - rt_tolerance, rt + rt_tolerance)
        return self.query_range(target=target, rt_range=rt_range)
