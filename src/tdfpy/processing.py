"""Bounded frame reuse for isolation windows."""

from collections.abc import Iterable, Iterator
from itertools import groupby
from typing import Literal

import numpy as np

from .elems import DiaWindow, PrmTransition
from .noise import NoiseSpec
from .pipeline import (
    Centroider,
    MergePeaksCentroider,
    Smooth,
    _prepare_spectrum,
    read_spectrum,
)
from .regions import ChargeStateRegion


def iter_window_spectra(
    windows: Iterable[DiaWindow | PrmTransition],
    *,
    exclude: ChargeStateRegion | None = None,
    smooth: Smooth | None = None,
    noise: NoiseSpec = None,
    centroid: Centroider | None = None,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
) -> Iterator[tuple[DiaWindow | PrmTransition, np.ndarray]]:
    """Yield (window, peaks) pairs, decoding adjacent windows' frame once.

    Pass reader.windows or reader.transitions in their existing order. Peaks
    have shape (N, 3), matching window.centroid with the same options. Only the
    current frame is retained. Unsorted input preserves caller order and may
    decode a frame again when it reappears. Keep readers open while consuming
    the iterator. Results contain no diagnostic or provenance wrappers.
    """
    cfg = centroid if centroid is not None else MergePeaksCentroider()
    for (td, frame_id), group in groupby(
        windows, key=lambda w: (w.timsdata, w.frame_id)
    ):
        spectrum = read_spectrum(td, frame_id)
        for window in group:
            td._require_open()
            prepared = _prepare_spectrum(
                spectrum,
                td,
                frame_id,
                scan_range=(window.scan_num_begin, window.scan_num_end),
                exclude=exclude,
                smoothing=smooth,
                noise=noise,
                ion_mobility_type=ion_mobility_type,
            )
            peaks = (
                cfg(prepared, td, frame_id, ion_mobility_type=ion_mobility_type)
                if not prepared.empty
                else np.empty((0, 3), dtype=np.float64)
            )
            yield window, peaks
