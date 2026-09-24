"""Bounded frame reuse for isolation windows and PASEF precursors."""

from collections import OrderedDict
from collections.abc import Iterable, Iterator
from itertools import groupby
from typing import Literal

import numpy as np

from .centroiding import _collapsed_spectrum
from .elems import DiaWindow, Precursor, PrmTransition
from .errors import TdfpyError
from .noise import NoiseSpec
from .pipeline import (
    Centroider,
    MergePeaksCentroider,
    Smooth,
    _prepare_spectrum,
    read_spectrum,
)
from .regions import ChargeStateRegion
from .timsdata import TimsData

__all__ = ["iter_precursor_spectra", "iter_window_spectra"]

#: Decoded frames kept by :func:`iter_precursor_spectra`. A PASEF precursor
#: spans a handful of consecutive MS/MS frames, so this covers any precursor
#: order that is roughly by frame (the reader's order) with room to spare.
_PRECURSOR_FRAME_CACHE = 64


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
    for (td, frame_id), group in groupby(windows, key=lambda w: (w.timsdata, w.frame_id)):
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
            peaks = cfg(prepared, td, frame_id, ion_mobility_type=ion_mobility_type) if not prepared.empty else np.empty((0, 3), dtype=np.float64)
            yield window, peaks


def iter_precursor_spectra(precursors: Iterable[Precursor]) -> Iterator[tuple[Precursor, np.ndarray]]:
    """Yield ``(precursor, peaks)`` pairs, decoding each MS/MS frame once.

    ``peaks`` equals :meth:`Precursor.merged_peaks`: the ``(N, 2)`` ``[m/z,
    intensity]`` mobility-collapsed spectrum over all of the precursor's PASEF
    windows. A PASEF frame carries windows of many precursors (often ten or
    more), so looping over ``merged_peaks()`` decodes each frame that many
    times; this decodes it once and serves every window from it.

    Pass ``reader.precursors`` (or any subset) in its existing order. The most
    recent 64 decoded frames are kept, so any roughly frame-ordered input decodes
    each frame once; a very scattered order still gives correct results but may
    decode a frame again. Keep the reader open while consuming the iterator.

    Raises:
        ReaderClosedError: If the reader was closed.
        TdfpyError: If an item is not a :class:`Precursor`, or a window's scan
            range does not fit its frame (corrupt file).
    """
    cache: OrderedDict[tuple[TimsData, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None] = OrderedDict()

    for precursor in precursors:
        if not isinstance(precursor, Precursor):
            raise TdfpyError(
                f"iter_precursor_spectra takes DDA precursors, got {type(precursor).__name__}; use iter_window_spectra for DIA windows or PRM transitions."
            )
        td = precursor.timsdata

        def read(frame_id: int, begin: int, end: int, td=td) -> tuple[np.ndarray, np.ndarray]:
            td._require_open()
            key = (td, frame_id)
            if key in cache:
                cache.move_to_end(key)
                decoded = cache[key]
            else:
                full = td._decode(frame_id)
                decoded = None if full is None else full[1:]
                cache[key] = decoded
                if len(cache) > _PRECURSOR_FRAME_CACHE:
                    cache.popitem(last=False)
            if decoded is None or begin == end:
                return np.empty(0, dtype=np.uint32), np.empty(0, dtype=np.uint32)
            starts, counts, tof, intensity = decoded
            lo = int(starts[begin])
            hi = int(starts[end - 1] + counts[end - 1])
            return tof[lo:hi], intensity[lo:hi]

        ranges = [(info.frame_id, *info._window_scans()) for info in precursor.pasef_frame_msms_infos]
        yield precursor, _collapsed_spectrum(td, ranges, read)
