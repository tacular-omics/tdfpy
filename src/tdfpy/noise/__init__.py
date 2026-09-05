"""Noise filtering for raw timsTOF peaks.

A noise filter inspects raw peaks in their native integer-index form
``(scan_indices, mz_indices, intensities)`` and returns a boolean keep
mask. Multiple filters can be chained in user-specified order — see
:func:`tdfpy.pipeline.apply_noise` for the runner.

Public surface:

- :class:`NoiseFilter` — abstract base class. Implement ``keep_mask``.
- :class:`IntensityThreshold` and its subclasses
  (:class:`AbsoluteThreshold`, :class:`MadThreshold`,
  :class:`PercentileThreshold`, :class:`HistogramThreshold`,
  :class:`BaselineThreshold`, :class:`IterativeMedianThreshold`).
- :class:`VerticalNoiseFilter` — content-aware vertical-streak filter.
- :class:`HorizontalHaloFilter` — left/right m/z halo remover.

The convenience entry points (``get_raw_peaks``, ``Frame.raw_peaks``,
etc.) also accept ``str`` / ``float`` shorthand which is coerced to a
sensible :class:`IntensityThreshold` subclass via :func:`coerce_filters`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..timsdata import TimsData


class NoiseFilter(ABC):
    """Base class for raw-peak noise filters.

    Subclasses are typically frozen dataclasses with their tunable knobs as
    fields. They implement a single method, :meth:`keep_mask`, which
    returns a boolean array of length ``len(intensities)`` indicating which
    points to keep.

    Filters operate on integer indices (TOF index + scan number) and raw
    intensity. Conversion to m/z and 1/K0 happens later in the pipeline so
    filters never need to do per-point unit conversions themselves.
    """

    @abstractmethod
    def keep_mask(
        self,
        scan_indices: np.ndarray,
        mz_indices: np.ndarray,
        intensities: np.ndarray,
        *,
        num_scans: int,
        td: "TimsData",
        frame_id: int,
    ) -> np.ndarray:
        """Return a boolean keep-mask of length ``len(intensities)``."""


from .intensity import (  # noqa: E402
    AbsoluteThreshold,
    BaselineThreshold,
    HistogramThreshold,
    IntensityThreshold,
    IterativeMedianThreshold,
    MadThreshold,
    PercentileThreshold,
)
from .structural import (  # noqa: E402
    HorizontalHaloFilter,
    VerticalNoiseDiagnostics,
    VerticalNoiseFilter,
)
from .gates import (  # noqa: E402
    DiaMs1WindowGate,
    SelectionPolygonGate,
)

_STRING_ALIASES: dict[str, type[IntensityThreshold]] = {
    "mad": MadThreshold,
    "percentile": PercentileThreshold,
    "histogram": HistogramThreshold,
    "baseline": BaselineThreshold,
    "iterative_median": IterativeMedianThreshold,
}


NoiseSpec = (
    NoiseFilter | str | float | int | list["NoiseSpec"] | tuple["NoiseSpec", ...] | None
)


def coerce_filters(spec: NoiseSpec) -> tuple[NoiseFilter, ...]:
    """Normalize a user-facing noise spec to a tuple of filter instances.

    Accepts:

    - ``None`` → empty tuple (no filtering)
    - a single :class:`NoiseFilter` instance → one-element tuple
    - a list/tuple of any of the above → flattened tuple
    - a string from ``"mad" | "percentile" | "histogram" | "baseline" |
      "iterative_median"`` → an :class:`IntensityThreshold` subclass with
      defaults
    - a numeric (``float`` / ``int``) → :class:`AbsoluteThreshold`

    Strings and numerics are how existing call sites stay terse; the tuple
    output is hashable for caching (e.g. Streamlit ``@cache_data``).
    """
    if spec is None:
        return ()
    if isinstance(spec, NoiseFilter):
        return (spec,)
    if isinstance(spec, str):
        try:
            cls = _STRING_ALIASES[spec]
        except KeyError as exc:
            raise ValueError(
                f"Unknown noise filter name {spec!r}. "
                f"Valid names: {sorted(_STRING_ALIASES)}"
            ) from exc
        return (cls(),)
    if isinstance(spec, (int, float)) and not isinstance(spec, bool):
        return (AbsoluteThreshold(value=float(spec)),)
    if isinstance(spec, (list, tuple)):
        out: list[NoiseFilter] = []
        for item in spec:
            out.extend(coerce_filters(item))
        return tuple(out)
    raise TypeError(
        f"Cannot coerce {type(spec).__name__} to a noise filter. "
        "Expected NoiseFilter, str, float, list/tuple, or None."
    )


__all__ = [
    "NoiseFilter",
    "NoiseSpec",
    "coerce_filters",
    "IntensityThreshold",
    "AbsoluteThreshold",
    "MadThreshold",
    "PercentileThreshold",
    "HistogramThreshold",
    "BaselineThreshold",
    "IterativeMedianThreshold",
    "VerticalNoiseFilter",
    "VerticalNoiseDiagnostics",
    "HorizontalHaloFilter",
    "SelectionPolygonGate",
    "DiaMs1WindowGate",
]
