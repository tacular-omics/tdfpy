"""Statistical intensity-threshold noise filters.

Each subclass of :class:`IntensityThreshold` exposes the knobs of its
estimator as dataclass fields so users can tune them with full type
support. The string shorthand ``"mad"`` / ``"percentile"`` / etc. (handled
in :mod:`tdfpy.noise`) maps to these classes with their default fields.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import numpy as np

from .._validation import integer, nonnegative
from . import NoiseFilter

if TYPE_CHECKING:
    from ..timsdata import TimsData

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntensityThreshold(NoiseFilter):
    """Drop points whose intensity is below a computed threshold.

    Subclasses implement :meth:`compute_threshold` to derive the threshold
    from the intensity distribution. The keep-mask is then the simple
    ``intensities >= threshold`` comparison.
    """

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name in ("bins", "passes", "min_remaining"):
                integer(field.name, value, minimum=1)
            elif field.name in ("value", "k", "scale", "inner_k", "final_k", "q"):
                nonnegative(field.name, value)
                if field.name == "q" and value > 100:
                    raise ValueError("q must not exceed 100.")

    @abstractmethod
    def compute_threshold(self, intensities: np.ndarray) -> float:
        """Return the intensity floor for this estimator."""

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
        if intensities.size == 0:
            return np.zeros(0, dtype=bool)
        threshold = self.compute_threshold(intensities)
        if not np.isfinite(threshold):
            # Degenerate input (e.g. all-equal intensities, or an empty
            # baseline slice) can make an estimator return NaN/inf. Rather
            # than silently drop every peak, keep them all and make it loud.
            logger.warning(
                "%s returned a non-finite threshold (%r) for frame %d "
                "(n=%d, min=%.4g, max=%.4g); keeping ALL points. This usually "
                "means degenerate input such as all-equal intensities.",
                type(self).__name__,
                threshold,
                frame_id,
                intensities.size,
                float(intensities.min()),
                float(intensities.max()),
            )
            return np.ones(intensities.size, dtype=bool)
        keep = intensities >= threshold
        n_kept = int(keep.sum())
        logger.debug(
            "%s[frame %d]: threshold=%.4g keeps %d/%d points",
            type(self).__name__,
            frame_id,
            threshold,
            n_kept,
            intensities.size,
        )
        if n_kept == 0:
            logger.warning(
                "%s[frame %d]: threshold %.4g removed ALL %d points "
                "(max intensity was %.4g). The threshold may be too high or the "
                "intensity distribution degenerate; downstream spectrum is empty.",
                type(self).__name__,
                frame_id,
                threshold,
                intensities.size,
                float(intensities.max()),
            )
        return keep


@dataclass(frozen=True)
class AbsoluteThreshold(IntensityThreshold):
    """Constant intensity floor, ignored estimator."""

    value: float = 0.0

    def compute_threshold(self, intensities: np.ndarray) -> float:
        return float(self.value)


@dataclass(frozen=True)
class MadThreshold(IntensityThreshold):
    """Median Absolute Deviation threshold: ``median + k · scale · MAD``.

    ``scale = 1.4826`` makes MAD a consistent estimator of the standard
    deviation for a Gaussian distribution.
    """

    k: float = 3.0
    scale: float = 1.4826

    def compute_threshold(self, intensities: np.ndarray) -> float:
        median = float(np.median(intensities))
        mad = float(np.median(np.abs(intensities - median)))
        return median + self.k * self.scale * mad


@dataclass(frozen=True)
class PercentileThreshold(IntensityThreshold):
    """Drop everything below the ``q``-th percentile of intensities."""

    q: float = 75.0

    def compute_threshold(self, intensities: np.ndarray) -> float:
        return float(np.percentile(intensities, self.q))


@dataclass(frozen=True)
class HistogramThreshold(IntensityThreshold):
    """Mode-of-histogram noise floor + ``k`` standard deviations.

    Bins the intensities into ``bins`` equal-width bins, takes the modal
    bin as the noise mode, estimates noise std from the FWHM around it,
    and returns ``mode + k · std``.
    """

    bins: int = 100
    k: float = 3.0

    def compute_threshold(self, intensities: np.ndarray) -> float:
        hist, edges = np.histogram(intensities, bins=self.bins)
        mode_idx = int(np.argmax(hist))
        mode = (edges[mode_idx] + edges[mode_idx + 1]) / 2
        half_max = hist[mode_idx] / 2
        left = mode_idx
        while left > 0 and hist[left] > half_max:
            left -= 1
        right = mode_idx
        while right < len(hist) - 1 and hist[right] > half_max:
            right += 1
        noise_std = (edges[right] - edges[left]) / 2.355  # FWHM → std
        return float(mode + self.k * noise_std)


@dataclass(frozen=True)
class BaselineThreshold(IntensityThreshold):
    """Bottom-quartile baseline: ``mean + k · std`` of the lowest ``q`` %."""

    q: float = 25.0
    k: float = 3.0

    def compute_threshold(self, intensities: np.ndarray) -> float:
        cutoff = float(np.percentile(intensities, self.q))
        baseline = intensities[intensities <= cutoff]
        return float(np.mean(baseline) + self.k * np.std(baseline))


@dataclass(frozen=True)
class IterativeMedianThreshold(IntensityThreshold):
    """Iteratively trim peaks above ``median + inner_k · scale · MAD``.

    Repeats up to ``passes`` times (or until fewer than ``min_remaining``
    points are left). The final threshold is ``median + final_k · std``
    of the surviving distribution.
    """

    passes: int = 3
    inner_k: float = 2.0
    final_k: float = 3.0
    scale: float = 1.4826
    min_remaining: int = 100

    def compute_threshold(self, intensities: np.ndarray) -> float:
        current = intensities.copy()
        for _ in range(self.passes):
            median = float(np.median(current))
            mad = float(np.median(np.abs(current - median)))
            cutoff = median + self.inner_k * self.scale * mad
            current = current[current <= cutoff]
            if len(current) < self.min_remaining:
                break
        return float(np.median(current) + self.final_k * np.std(current))
