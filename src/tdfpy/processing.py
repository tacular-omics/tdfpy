"""Bounded frame reuse for isolation windows and PASEF precursors."""

from collections import OrderedDict
from collections.abc import Iterable, Iterator, Sequence
from itertools import groupby
from typing import Literal

import numpy as np

from ._validation import bounds, choice, nonnegative
from .calibration import _ccs_from_ook0
from .centroiding import _collapsed_spectrum
from .elems import DiaWindow, Frame, Precursor, PrmTransition
from .errors import TdfpyError
from .lookup import Ms1FrameLookup
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
from .types import ToleranceUnit

__all__ = ["extract_mobilogram", "extract_mobilograms", "iter_precursor_spectra", "iter_window_spectra"]

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
    mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
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
                mobility_type=mobility_type,
            )
            peaks = cfg(prepared, td, frame_id, mobility_type=mobility_type) if not prepared.empty else np.empty((0, 3), dtype=np.float64)
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


# ---------------------------------------------------------------------------
# Extracted ion mobilograms
# ---------------------------------------------------------------------------

#: One row per populated bin, ``[mobility, summed intensity]``.
_EMPTY_MOBILOGRAM = np.empty((0, 2), dtype=np.float64)


def _selection_rows(td: TimsData, selection: Iterable[int | DiaWindow | PrmTransition]) -> list[tuple[int, int, int]]:
    """``(frame_id, scan_begin, scan_end)`` per selected frame or window."""
    rows: list[tuple[int, int, int]] = []
    for item in selection:
        if isinstance(item, DiaWindow | PrmTransition):
            if item.timsdata is not td:
                raise TdfpyError("the selected windows belong to a different reader than the TimsData passed in.")
            rows.append((item.frame_id, *item._window_scans()))
        elif isinstance(item, bool) or not isinstance(item, int | np.integer):
            raise TdfpyError(
                f"selection takes frame IDs or DIA windows / PRM transitions, got {type(item).__name__}. "
                "Pass reader.ms1 frame IDs, reader.windows or reader.transitions."
            )
        else:
            rows.append((int(item), 0, td.frame_metadata(int(item)).num_scans))
    if not rows:
        raise TdfpyError("selection is empty: pass the frames or windows to extract from, never the whole run by default.")
    return rows


def _mz_index_window(td: TimsData, frame_id: int, mz: float, tolerance: float, unit: str) -> tuple[float, float]:
    width = mz * tolerance / 1e6 if unit == "ppm" else tolerance
    # A Da tolerance may be wider than the target m/z. A negative m/z has no TOF
    # index: the calibration takes a square root of it and returns NaN, and every
    # comparison against NaN is false, so the window would silently match nothing.
    edges = np.array([max(mz - width, 0.0), mz + width], dtype=np.float64)
    low, high = np.sort(np.asarray(td.mz_to_index(frame_id, edges)))
    return float(low), float(high)


def _mobility_axis(td: TimsData, frame_id: int, ook0: np.ndarray, mz: float, mobility_type: str, charge: int | None) -> np.ndarray:
    """The 1/K0 axis in the unit the caller asked for, converted once at the end."""
    if mobility_type == "ccs":
        return _ccs_from_ook0(ook0, 1 if charge is None else charge, np.full(ook0.shape, mz, dtype=np.float64))
    if mobility_type == "voltage":
        return np.asarray(td.scan_num_to_voltage(frame_id, td.ook0_to_scan_num(frame_id, ook0)), dtype=np.float64)
    return np.asarray(ook0, dtype=np.float64)


def _resolve_ms1_selection(
    ms1: "Ms1FrameLookup[Frame]",
    rt_range: tuple[float, float] | None,
    frame_ids: Sequence[int] | None,
) -> list[int]:
    """MS1 frame IDs from an RT window or an explicit list, never the whole run."""
    if (rt_range is None) == (frame_ids is None):
        raise TdfpyError(
            "pass rt_range=(low, high) or frame_ids=[...], exactly one of them: a mobilogram reads every frame it is\n"
            "given, so it never defaults to the whole run."
        )
    if frame_ids is not None:
        return [int(frame_id) for frame_id in frame_ids]
    bounds("rt_range", rt_range)
    return [frame.frame_id for frame in ms1.query_range(rt_range=rt_range)]


def extract_mobilograms(
    td: TimsData,
    selection: Iterable[int | DiaWindow | PrmTransition],
    *,
    targets: Sequence[float],
    mz_tolerance: float = 20.0,
    mz_tolerance_unit: ToleranceUnit = "ppm",
    mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    charge: int | None = None,
    ook0_bin_width: float | None = None,
) -> list[np.ndarray]:
    """One mobilogram per m/z target, decoding each selected frame once.

    The ion-mobility counterpart of a chromatogram: summed intensity inside an
    m/z window against ion mobility, over the frames or windows selected. Cost
    is the decode of that selection, so several targets in one call are far
    cheaper than one call each.

    Args:
        td: Open :class:`~tdfpy.timsdata.TimsData`.
        selection: MS1 frame IDs, or :class:`~tdfpy.DiaWindow` /
            :class:`~tdfpy.PrmTransition` elements, which also restrict the
            mobility axis to each window's ``[scan_num_begin, scan_num_end)``.
            There is no default: name the frames or windows to read.
        targets: Target m/z values, one mobilogram each, in the same order.
        mz_tolerance: Half-width of the m/z window.
        mz_tolerance_unit: ``"ppm"`` or ``"da"``.
        mobility_type: Unit of the reported axis. The quantity is always summed
            in 1/K0 and only the axis is converted, as the centroiders do.
        charge: Required for ``mobility_type="ccs"``, which is undefined without
            it. Ignored otherwise. ``0`` raises.
        ook0_bin_width: Width of the 1/K0 bins, in V*s/cm^2. Only needed when the
            selection mixes TIMS calibrations, where there is no common scan
            grid; otherwise the acquisition's own scan grid is used and the axis
            values are exact.

    Returns:
        One ``(M, 2)`` array per target, ``[mobility, summed intensity]``, in
        ascending mobility, one row per populated bin. A target with no signal
        gives ``(0, 2)``.

    Raises:
        ReaderClosedError: If the reader was closed.
        TdfpyError: If the selection is empty or holds something else, if
            ``targets`` is empty, if an argument is out of range, if
            ``mobility_type="ccs"`` comes without a charge or with ``charge=0``,
            or if the selection
            mixes TIMS calibrations and ``ook0_bin_width`` is missing (and for
            ``mobility_type="voltage"``, which has no common value across
            calibrations).
    """
    td._require_open()
    choice("mobility_type", mobility_type, ("ook0", "ccs", "voltage"))
    choice("mz_tolerance_unit", mz_tolerance_unit, ("da", "ppm"))
    nonnegative("mz_tolerance", mz_tolerance)
    mz_targets = [float(mz) for mz in targets]
    if not mz_targets:
        raise TdfpyError("targets is empty: pass at least one m/z to extract.")
    if mobility_type == "ccs" and charge is None:
        raise TdfpyError("mobility_type='ccs' needs charge=...: CCS is undefined without the ion's charge state.")
    if mobility_type == "ccs" and charge == 0:
        raise TdfpyError(
            "charge=0 cannot convert between 1/K0 and CCS: pass the ion's charge state, which is never 0. "
            "The sign is ignored, so charge=-2 and charge=2 give the same CCS."
        )
    if ook0_bin_width is not None and not ook0_bin_width > 0:
        raise TdfpyError(f"ook0_bin_width must be positive, got {ook0_bin_width}.")

    rows = _selection_rows(td, selection)
    grids: dict[tuple[int, int], list[int]] = {}
    for frame_id, _, _ in rows:
        meta = td.frame_metadata(frame_id)
        grids.setdefault((meta.tims_calibration_id, meta.num_scans), []).append(frame_id)
    one_grid = len(grids) == 1
    if not one_grid:
        detail = "; ".join(f"TimsCalibration {cal} with {scans} scans: frames {ids[0]}..{ids[-1]} ({len(ids)})" for (cal, scans), ids in sorted(grids.items()))
        if ook0_bin_width is None:
            raise TdfpyError(
                f"the selection mixes {len(grids)} TIMS calibrations, which share no scan grid, so pass ook0_bin_width=... to bin the 1/K0 axis. {detail}"
            )
        if mobility_type == "voltage":
            raise TdfpyError(f"mobility_type='voltage' needs one TIMS calibration: the same 1/K0 is a different voltage under each. {detail}")

    if one_grid and ook0_bin_width is None:
        (_, num_scans), frame_ids = next(iter(grids.items()))
        per_scan = np.zeros((len(mz_targets), num_scans), dtype=np.float64)
        for frame_id, begin, end in rows:
            spectrum = read_spectrum(td, frame_id)
            if spectrum.empty:
                continue
            in_window = (spectrum.scan_indices >= begin) & (spectrum.scan_indices < end)
            for index, mz in enumerate(mz_targets):
                low, high = _mz_index_window(td, frame_id, mz, mz_tolerance, mz_tolerance_unit)
                keep = in_window & (spectrum.mz_indices >= low) & (spectrum.mz_indices <= high)
                if keep.any():
                    np.add.at(per_scan[index], spectrum.scan_indices[keep], spectrum.intensities[keep])
        out = []
        for index, mz in enumerate(mz_targets):
            scans = np.nonzero(per_scan[index])[0]
            if not len(scans):
                out.append(_EMPTY_MOBILOGRAM.copy())
                continue
            ook0 = np.asarray(td.scan_num_to_ook0(frame_ids[0], scans), dtype=np.float64)
            axis = _mobility_axis(td, frame_ids[0], ook0, mz, mobility_type, charge)
            order = np.argsort(axis)
            out.append(np.column_stack([axis[order], per_scan[index][scans][order]]))
        return out

    assert ook0_bin_width is not None  # the mixed-grid branch above requires it
    binned: list[dict[int, float]] = [{} for _ in mz_targets]
    for frame_id, begin, end in rows:
        spectrum = read_spectrum(td, frame_id)
        if spectrum.empty:
            continue
        in_window = (spectrum.scan_indices >= begin) & (spectrum.scan_indices < end)
        num_scans = spectrum.num_scans
        for index, mz in enumerate(mz_targets):
            low, high = _mz_index_window(td, frame_id, mz, mz_tolerance, mz_tolerance_unit)
            keep = in_window & (spectrum.mz_indices >= low) & (spectrum.mz_indices <= high)
            if not keep.any():
                continue
            per_scan = np.zeros(num_scans, dtype=np.float64)
            np.add.at(per_scan, spectrum.scan_indices[keep], spectrum.intensities[keep])
            scans = np.nonzero(per_scan)[0]
            ook0 = np.asarray(td.scan_num_to_ook0(frame_id, scans), dtype=np.float64)
            for key, value in zip(np.floor(ook0 / ook0_bin_width).astype(np.int64), per_scan[scans], strict=True):
                binned[index][int(key)] = binned[index].get(int(key), 0.0) + float(value)
    out = []
    for index, mz in enumerate(mz_targets):
        if not binned[index]:
            out.append(_EMPTY_MOBILOGRAM.copy())
            continue
        keys = np.array(sorted(binned[index]), dtype=np.int64)
        centres = (keys + 0.5) * ook0_bin_width
        intensity = np.array([binned[index][int(key)] for key in keys], dtype=np.float64)
        axis = _mobility_axis(td, rows[0][0], centres, mz, mobility_type, charge)
        order = np.argsort(axis)
        out.append(np.column_stack([axis[order], intensity[order]]))
    return out


def extract_mobilogram(
    td: TimsData,
    selection: Iterable[int | DiaWindow | PrmTransition],
    *,
    mz: float,
    mz_tolerance: float = 20.0,
    mz_tolerance_unit: ToleranceUnit = "ppm",
    mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    charge: int | None = None,
    ook0_bin_width: float | None = None,
) -> np.ndarray:
    """One mobilogram: summed intensity in an m/z window against ion mobility.

    :func:`extract_mobilograms` with a single target, and the same arguments.
    Returns an ``(M, 2)`` array of ``[mobility, summed intensity]`` in ascending
    mobility, or ``(0, 2)`` when the window holds no signal.

    Raises:
        ReaderClosedError: If the reader was closed.
        TdfpyError: As :func:`extract_mobilograms`.
    """
    return extract_mobilograms(
        td,
        selection,
        targets=[mz],
        mz_tolerance=mz_tolerance,
        mz_tolerance_unit=mz_tolerance_unit,
        mobility_type=mobility_type,
        charge=charge,
        ook0_bin_width=ook0_bin_width,
    )[0]
