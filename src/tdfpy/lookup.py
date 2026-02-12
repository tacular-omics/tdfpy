from typing import Generic, Iterator, Literal, TypeVar

from .elems import DiaWindow, DiaWindowGroup, Frame, Precursor

T = TypeVar("T", bound=Frame)


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
            raise KeyError(f"Frame ID {frame_id} not found.")
        return self._frames[frame_id]

    def __len__(self) -> int:
        return len(self._frames)

    def get(self, frame_id: int, default=None):
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
            raise KeyError(f"Window Group ID {window_group_id} not found.")
        return self._window_map[window_group_id]

    def __len__(self) -> int:
        return len(self._windows)

    def get(self, window_group_id: int, default=None):
        return self._window_map.get(window_group_id, default)

    def query_range(
        self,
        window_group_index: int | DiaWindowGroup | None = None,
        rt_range: tuple[float, float] | None = None,
    ) -> Iterator[DiaWindow]:
        """
        Query windows by retention time range.

        Args:
            rt_range: Tuple of (min_rt, max_rt) in seconds. If None, RT filtering is skipped.
        Yields:
            DiaWindow objects matching the criteria.
        """
        for window in self._windows:
            if window_group_index is not None:
                if isinstance(window_group_index, DiaWindowGroup):
                    if window.window_group != window_group_index.window_index:
                        continue
                elif window.window_index != window_group_index:
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
            raise KeyError(f"Precursor ID {precursor_id} not found.")
        return self._precursors[precursor_id]

    def __len__(self) -> int:
        return len(self._precursors)

    def get(self, precursor_id: int, default=None):
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
            mz_tolerance_type: Unit for m/z tolerance ("ppm" or "da"). Default is "da".
            rt_tolerance: Tolerance for retention time matching (in seconds). Default is 20s.

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
