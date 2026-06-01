"""Composable pipeline ops for raw peak extraction and centroiding.

Each op takes a :class:`RawSpectrum` and returns a :class:`RawSpectrum`,
or in the case of :func:`convert` and :func:`centroid_peaks`, an
``np.ndarray`` of shape ``(N, 3)`` with columns ``[mz, intensity,
ion_mobility]``.

The convenience entry points :func:`tdfpy.get_raw_peaks` and
:func:`tdfpy.get_centroided_spectrum` orchestrate these ops in a fixed
order. Power users can call the ops directly for custom pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .noise import NoiseFilter
from .regions import ChargeStateRegion
from .timsdata import TimsData, oneOverK0ToCCSforMz

try:
    from numba import njit as _njit  # ty: ignore[unresolved-import]
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# --------------------------------------------------------------------------
# Data carrier
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RawSpectrum:
    """Raw peaks in integer-index (TOF / scan) space.

    The native form of Bruker raw data — TOF index and scan number are
    integers, intensity is a 32-bit-ish count. All pipeline ops operate
    on this representation; conversion to m/z and 1/K0 happens once at
    the end via :func:`convert`.
    """

    scan_indices: np.ndarray  # int64, len N
    mz_indices: np.ndarray  # int64, len N (TOF indices)
    intensities: np.ndarray  # float64, len N
    num_scans: int

    def __len__(self) -> int:
        return int(self.intensities.size)

    @property
    def num_peaks(self) -> int:
        return len(self)

    @property
    def empty(self) -> bool:
        return self.intensities.size == 0

    def filter(self, mask: np.ndarray) -> "RawSpectrum":
        """Return a new spectrum keeping only points where ``mask`` is True."""
        return RawSpectrum(
            scan_indices=self.scan_indices[mask],
            mz_indices=self.mz_indices[mask],
            intensities=self.intensities[mask],
            num_scans=self.num_scans,
        )

    @classmethod
    def empty_like(cls, num_scans: int) -> "RawSpectrum":
        return cls(
            scan_indices=np.empty(0, dtype=np.int64),
            mz_indices=np.empty(0, dtype=np.int64),
            intensities=np.empty(0, dtype=np.float64),
            num_scans=num_scans,
        )


# --------------------------------------------------------------------------
# Pipeline ops
# --------------------------------------------------------------------------


def read_spectrum(td: TimsData, frame_id: int) -> RawSpectrum:
    """Read a frame's raw peaks into integer-index form."""
    if td.conn is None:
        raise RuntimeError("TimsData connection is not open")

    cursor = td.conn.cursor()
    cursor.execute("SELECT NumScans FROM Frames WHERE Id = ?", (frame_id,))
    result = cursor.fetchone()
    if result is None:
        raise ValueError(f"Frame {frame_id} not found in database")
    (num_scans,) = result
    if num_scans == 0:
        return RawSpectrum.empty_like(0)

    scans = td.readScans(frame_id, 0, num_scans)
    scan_lengths = np.fromiter(
        (len(idx) for idx, _ in scans), dtype=np.int64, count=num_scans
    )
    total_peaks = int(scan_lengths.sum())
    if total_peaks == 0:
        return RawSpectrum.empty_like(num_scans)

    scan_indices = np.repeat(np.arange(num_scans, dtype=np.int64), scan_lengths)
    mz_indices = np.concatenate([idx for idx, _ in scans]).astype(np.int64, copy=False)
    intensities = np.concatenate(
        [intens for _, intens in scans]
    ).astype(np.float64, copy=False)
    return RawSpectrum(
        scan_indices=scan_indices,
        mz_indices=mz_indices,
        intensities=intensities,
        num_scans=num_scans,
    )


def exclude_region(
    spectrum: RawSpectrum,
    region: ChargeStateRegion,
    *,
    td: TimsData,
    frame_id: int,
) -> RawSpectrum:
    """Drop peaks lying inside the given region."""
    if spectrum.empty:
        return spectrum
    cutoff = region.index_cutoff_per_scan(td, frame_id, spectrum.num_scans)
    mask = spectrum.mz_indices >= cutoff[spectrum.scan_indices]
    return spectrum.filter(mask)


def subset_scans(
    spectrum: RawSpectrum,
    *,
    scan_num_begin: int,
    scan_num_end: int,
) -> RawSpectrum:
    """Restrict the spectrum to peaks in scans ``[scan_num_begin, scan_num_end)``.

    The bounds are half-open: ``begin`` inclusive, ``end`` exclusive — matching
    Bruker's ``readScans(frame_id, begin, end)`` semantics. The returned
    ``RawSpectrum`` keeps its ``num_scans`` field (i.e. the parent frame's
    full scan count) so downstream ops still address scans by their original
    index.

    Used by :class:`~tdfpy.DiaWindow` and :class:`~tdfpy.PrmTransition` to
    restrict centroiding / raw-peak extraction to the isolation window's
    scan range.
    """
    if spectrum.empty:
        return spectrum
    if scan_num_begin < 0 or scan_num_end < scan_num_begin:
        raise ValueError(
            f"invalid scan range [{scan_num_begin}, {scan_num_end})"
        )
    mask = (spectrum.scan_indices >= scan_num_begin) & (
        spectrum.scan_indices < scan_num_end
    )
    return spectrum.filter(mask)


def apply_noise(
    spectrum: RawSpectrum,
    filters: Iterable[NoiseFilter],
    *,
    td: TimsData,
    frame_id: int,
) -> RawSpectrum:
    """Apply each noise filter in order, threading the surviving peaks through."""
    for f in filters:
        if spectrum.empty:
            break
        mask = f.keep_mask(
            spectrum.scan_indices,
            spectrum.mz_indices,
            spectrum.intensities,
            num_scans=spectrum.num_scans,
            td=td,
            frame_id=frame_id,
        )
        spectrum = spectrum.filter(mask)
    return spectrum


def convert(
    spectrum: RawSpectrum,
    td: TimsData,
    frame_id: int,
    *,
    ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
) -> np.ndarray:
    """Convert integer indices to (m/z, intensity, ion_mobility).

    Returns a ``(N, 3)`` array. Empty input yields an empty array of the
    same shape so callers don't need to special-case.
    """
    if spectrum.empty:
        return np.empty((0, 3), dtype=np.float64)

    ook0_per_scan = np.asarray(
        td.scanNumToOneOverK0(frame_id, np.arange(spectrum.num_scans))  # type: ignore[call-arg]
    )
    ion_mobility_array = ook0_per_scan[spectrum.scan_indices]
    mz_array = td.indexToMz(frame_id, spectrum.mz_indices)

    if ion_mobility_type == "ccs":
        ion_mobility_array = np.array(
            [
                oneOverK0ToCCSforMz(ook0, 1, mz)
                for ook0, mz in zip(ion_mobility_array, mz_array)
            ],
            dtype=np.float64,
        )
    elif ion_mobility_type == "voltage":
        ion_mobility_array = td.scanNumToVoltage(frame_id, ion_mobility_array)

    return np.column_stack((mz_array, spectrum.intensities, ion_mobility_array))


# --------------------------------------------------------------------------
# Centroiding
# --------------------------------------------------------------------------


class Centroider(ABC):
    """Base class for centroiding algorithms.

    Subclasses are frozen dataclasses carrying their tunable knobs as fields
    and implement :meth:`__call__`, which takes the (filtered) raw spectrum
    and returns an ``(N, 3)`` array of ``[mz, intensity, ion_mobility]``
    centroids. Centroiders decide internally whether to operate in integer
    index space or after conversion to float m/z.
    """

    @abstractmethod
    def __call__(
        self,
        spectrum: RawSpectrum,
        td: TimsData,
        frame_id: int,
        *,
        ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class MergePeaksCentroider(Centroider):
    """Greedy m/z-tolerance centroider — wraps :func:`tdfpy.merge_peaks`.

    Operates on float m/z values. Real peaks are matched within an m/z
    tolerance (ppm or Da) and an ion mobility tolerance. Default algorithm
    used by :func:`tdfpy.get_centroided_spectrum`.
    """

    mz_tolerance: float = 8.0
    mz_tolerance_type: Literal["ppm", "da"] = "ppm"
    im_tolerance: float = 0.1
    im_tolerance_type: Literal["relative", "absolute"] = "relative"
    min_peaks: int = 3
    max_peaks: int | None = None
    peak_noise_filter: bool = False
    peak_noise_window: float = 0.1
    peak_noise_end_fraction: float = 0.1
    use_numba: bool = True

    def __call__(
        self,
        spectrum: RawSpectrum,
        td: TimsData,
        frame_id: int,
        *,
        ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    ) -> np.ndarray:
        from .centroiding import merge_peaks

        peaks = convert(spectrum, td, frame_id, ion_mobility_type=ion_mobility_type)
        if peaks.size == 0:
            return np.empty((0, 3), dtype=np.float64)
        return merge_peaks(
            peaks[:, 0],
            peaks[:, 1],
            peaks[:, 2],
            mz_tolerance=self.mz_tolerance,
            mz_tolerance_type=self.mz_tolerance_type,
            im_tolerance=self.im_tolerance,
            im_tolerance_type=self.im_tolerance_type,
            min_peaks=self.min_peaks,
            max_peaks=self.max_peaks,
            peak_noise_filter=self.peak_noise_filter,
            peak_noise_window=self.peak_noise_window,
            peak_noise_end_fraction=self.peak_noise_end_fraction,
            use_numba=self.use_numba,
        )


def _box_smooth_kernel(
    scan_arr: np.ndarray,
    mz_arr: np.ndarray,
    int_arr: np.ndarray,
    smooth_scan_half_width: int,
    smooth_mz_idx_half_width: int,
) -> np.ndarray:
    """Box-mean kernel — position-preserving intensity smoothing.

    Used internally by :class:`WatershedCentroider` to stabilise seed
    selection. Not exposed as a pipeline op; the watershed is the
    only consumer in practice.
    """
    n = scan_arr.size
    grid: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        grid.setdefault(
            (
                int(scan_arr[i]) // smooth_scan_half_width,
                int(mz_arr[i]) // smooth_mz_idx_half_width,
            ),
            [],
        ).append(i)

    smoothed = np.empty(n, dtype=np.float64)
    for i in range(n):
        p_scan = int(scan_arr[i])
        p_mz = int(mz_arr[i])
        c_scan = p_scan // smooth_scan_half_width
        c_mz = p_mz // smooth_mz_idx_half_width
        total = 0.0
        count = 0
        for ds in (-1, 0, 1):
            for dm in (-1, 0, 1):
                bucket = grid.get((c_scan + ds, c_mz + dm))
                if bucket is None:
                    continue
                for q in bucket:
                    if abs(p_scan - int(scan_arr[q])) > smooth_scan_half_width:
                        continue
                    if abs(p_mz - int(mz_arr[q])) > smooth_mz_idx_half_width:
                        continue
                    total += float(int_arr[q])
                    count += 1
        smoothed[i] = total / count if count > 0 else float(int_arr[i])
    return smoothed


def _box_smooth_intensities_arrays(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    *,
    smooth_scan_half_width: int,
    smooth_mz_idx_half_width: int,
) -> np.ndarray:
    """Array-input variant kept for the watershed centroider's internal use
    (it already has the arrays in hand and doesn't need to rewrap them)."""
    if scan_indices.size == 0:
        return np.zeros(0, dtype=np.float64)
    return _box_smooth_kernel(
        np.asarray(scan_indices, dtype=np.int64),
        np.asarray(mz_indices, dtype=np.int64),
        np.asarray(intensities, dtype=np.float64),
        max(1, int(smooth_scan_half_width)),
        max(1, int(smooth_mz_idx_half_width)),
    )


_CELL_STRIDE = np.int64(1_000_000_000)


if _HAS_NUMBA:
    @_njit(cache=True)
    def _watershed_njit_kernel(
        scan_arr, mz_arr, int_arr, mz_val_arr, im_arr,
        attach_scan_half_width, attach_mz_idx_half_width,
        min_seed_intensity, min_centroid_intensity,
        max_scan_from_seed, max_mz_idx_from_seed,
    ):
        """Numba-JIT watershed kernel.

        Avoids Python dicts by pre-sorting points into a contiguous
        cell-id-sorted array and using ``np.searchsorted`` to find the
        3×3-cell neighborhood at each query. An ``active`` mask marks
        which points have already joined or seeded a group.

        ``max_scan_from_seed`` and ``max_mz_idx_from_seed`` are leash bounds
        from each group's seed; ``-1`` means "no limit".
        """
        n = scan_arr.size
        if n == 0:
            return np.empty((0, 3), dtype=np.float64)

        cell_id = np.empty(n, dtype=np.int64)
        for i in range(n):
            cell_id[i] = (scan_arr[i] // attach_scan_half_width) * _CELL_STRIDE + (
                mz_arr[i] // attach_mz_idx_half_width
            )
        cell_order = np.argsort(cell_id)
        sorted_cell_id = cell_id[cell_order]

        group_id = np.full(n, -1, dtype=np.int64)
        active = np.zeros(n, dtype=np.bool_)
        seed_intensities = np.empty(n, dtype=np.float64)
        seed_scan = np.empty(n, dtype=np.int64)
        seed_mz = np.empty(n, dtype=np.int64)
        n_groups = np.int64(0)

        # Descending intensity order. argsort ascending then iterate in reverse.
        intensity_order = np.argsort(int_arr, kind="mergesort")

        for k in range(n - 1, -1, -1):
            i = intensity_order[k]
            p_scan = scan_arr[i]
            p_mz = mz_arr[i]
            p_int = int_arr[i]
            c_scan = p_scan // attach_scan_half_width
            c_mz = p_mz // attach_mz_idx_half_width

            best_group = np.int64(-1)
            best_dist = np.int64(0)
            best_seed_int = -1.0
            best_j = np.int64(-1)
            for ds in range(-1, 2):
                for dm in range(-1, 2):
                    target_cell = (c_scan + ds) * _CELL_STRIDE + (c_mz + dm)
                    left = np.searchsorted(sorted_cell_id, target_cell)
                    right = np.searchsorted(sorted_cell_id, target_cell + 1)
                    for jj in range(left, right):
                        j = cell_order[jj]
                        if not active[j]:
                            continue
                        d_scan = abs(p_scan - scan_arr[j])
                        if d_scan > attach_scan_half_width:
                            continue
                        d_mz = abs(p_mz - mz_arr[j])
                        if d_mz > attach_mz_idx_half_width:
                            continue
                        j_group = group_id[j]
                        # Leash: reject if too far from this group's seed.
                        if max_scan_from_seed >= 0 and abs(
                            p_scan - seed_scan[j_group]
                        ) > max_scan_from_seed:
                            continue
                        if max_mz_idx_from_seed >= 0 and abs(
                            p_mz - seed_mz[j_group]
                        ) > max_mz_idx_from_seed:
                            continue
                        d = d_scan + d_mz
                        j_seed_int = seed_intensities[j_group]
                        # Tiebreak: shorter Manhattan dist wins; on tie,
                        # higher seed intensity; on tie, smaller point
                        # index (makes the result independent of how the
                        # neighborhood is traversed).
                        if best_group < 0 or d < best_dist or (
                            d == best_dist and j_seed_int > best_seed_int
                        ) or (
                            d == best_dist
                            and j_seed_int == best_seed_int
                            and j < best_j
                        ):
                            best_group = j_group
                            best_dist = d
                            best_seed_int = j_seed_int
                            best_j = j

            if best_group >= 0:
                group_id[i] = best_group
                active[i] = True
            elif p_int >= min_seed_intensity:
                group_id[i] = n_groups
                seed_intensities[n_groups] = p_int
                seed_scan[n_groups] = p_scan
                seed_mz[n_groups] = p_mz
                n_groups += 1
                active[i] = True

        if n_groups == 0:
            return np.empty((0, 3), dtype=np.float64)

        total = np.zeros(n_groups, dtype=np.float64)
        sum_mz = np.zeros(n_groups, dtype=np.float64)
        sum_im = np.zeros(n_groups, dtype=np.float64)
        for i in range(n):
            if active[i]:
                g = group_id[i]
                w = int_arr[i]
                total[g] += w
                sum_mz[g] += mz_val_arr[i] * w
                sum_im[g] += im_arr[i] * w

        n_keep = np.int64(0)
        for g in range(n_groups):
            if total[g] >= min_centroid_intensity:
                n_keep += 1

        out = np.empty((n_keep, 3), dtype=np.float64)
        k_out = np.int64(0)
        for g in range(n_groups):
            if total[g] >= min_centroid_intensity:
                t = total[g]
                inv = 1.0 / t if t > 0.0 else 1.0
                out[k_out, 0] = sum_mz[g] * inv
                out[k_out, 1] = t
                out[k_out, 2] = sum_im[g] * inv
                k_out += 1
        return out


def _watershed_python_kernel(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    mz_values: np.ndarray,
    im_values: np.ndarray,
    *,
    attach_scan_half_width: int,
    attach_mz_idx_half_width: int,
    min_seed_intensity: float,
    min_centroid_intensity: float,
    max_scan_from_seed: int,
    max_mz_idx_from_seed: int,
) -> np.ndarray:
    """Pure-Python dict-of-lists watershed kernel — fallback for when
    Numba isn't installed. See :func:`_watershed_kernel` for the algorithm.
    """
    n = scan_indices.size
    if n == 0:
        return np.empty((0, 3), dtype=np.float64)

    attach_scan_half_width = max(1, int(attach_scan_half_width))
    attach_mz_idx_half_width = max(1, int(attach_mz_idx_half_width))

    scan_arr = np.asarray(scan_indices, dtype=np.int64)
    mz_arr = np.asarray(mz_indices, dtype=np.int64)
    int_arr = np.asarray(intensities, dtype=np.float64)
    mz_val_arr = np.asarray(mz_values, dtype=np.float64)
    im_arr = np.asarray(im_values, dtype=np.float64)

    group_id = np.full(n, -1, dtype=np.int64)
    seed_intensities: list[float] = []
    seed_scan: list[int] = []
    seed_mz: list[int] = []
    grid: dict[tuple[int, int], list[int]] = {}

    intensity_order = np.argsort(int_arr, kind="stable")[::-1]

    for raw_i in intensity_order:
        i = int(raw_i)
        p_scan = int(scan_arr[i])
        p_mz = int(mz_arr[i])
        p_int = float(int_arr[i])
        c_scan = p_scan // attach_scan_half_width
        c_mz = p_mz // attach_mz_idx_half_width

        best_group = -1
        best_dist = 0
        best_seed_int = -1.0
        best_j = -1
        for ds in (-1, 0, 1):
            for dm in (-1, 0, 1):
                bucket = grid.get((c_scan + ds, c_mz + dm))
                if bucket is None:
                    continue
                for q in bucket:
                    d_scan = abs(p_scan - int(scan_arr[q]))
                    if d_scan > attach_scan_half_width:
                        continue
                    d_mz = abs(p_mz - int(mz_arr[q]))
                    if d_mz > attach_mz_idx_half_width:
                        continue
                    q_group = int(group_id[q])
                    # Leash: reject if too far from this group's seed.
                    if max_scan_from_seed >= 0 and abs(
                        p_scan - seed_scan[q_group]
                    ) > max_scan_from_seed:
                        continue
                    if max_mz_idx_from_seed >= 0 and abs(
                        p_mz - seed_mz[q_group]
                    ) > max_mz_idx_from_seed:
                        continue
                    d = d_scan + d_mz
                    q_seed_int = seed_intensities[q_group]
                    # Tiebreak: shorter dist > higher seed intensity >
                    # smaller point index — matches the Numba kernel.
                    if best_group < 0 or d < best_dist or (
                        d == best_dist and q_seed_int > best_seed_int
                    ) or (
                        d == best_dist
                        and q_seed_int == best_seed_int
                        and q < best_j
                    ):
                        best_group = q_group
                        best_dist = d
                        best_seed_int = q_seed_int
                        best_j = q

        if best_group >= 0:
            group_id[i] = best_group
            grid.setdefault((c_scan, c_mz), []).append(i)
        elif p_int >= min_seed_intensity:
            new_group = len(seed_intensities)
            group_id[i] = new_group
            seed_intensities.append(p_int)
            seed_scan.append(p_scan)
            seed_mz.append(p_mz)
            grid.setdefault((c_scan, c_mz), []).append(i)
        # else: orphan — drop, don't enter grid

    num_groups = len(seed_intensities)
    if num_groups == 0:
        return np.empty((0, 3), dtype=np.float64)

    assigned = group_id >= 0
    g = group_id[assigned]
    w = int_arr[assigned]
    total = np.bincount(g, weights=w, minlength=num_groups)
    sum_mz = np.bincount(g, weights=mz_val_arr[assigned] * w, minlength=num_groups)
    sum_im = np.bincount(g, weights=im_arr[assigned] * w, minlength=num_groups)
    safe_total = np.where(total > 0, total, 1.0)
    cent_mz = sum_mz / safe_total
    cent_im = sum_im / safe_total

    keep = total >= float(min_centroid_intensity)
    return np.column_stack([cent_mz[keep], total[keep], cent_im[keep]])


def _watershed_kernel(
    scan_indices: np.ndarray,
    mz_indices: np.ndarray,
    intensities: np.ndarray,
    mz_values: np.ndarray,
    im_values: np.ndarray,
    *,
    attach_scan_half_width: int,
    attach_mz_idx_half_width: int,
    min_seed_intensity: float,
    min_centroid_intensity: float,
    max_scan_from_seed: int | None = None,
    max_mz_idx_from_seed: int | None = None,
    use_numba: bool = True,
) -> np.ndarray:
    """Intensity-ordered region growing in integer (scan, TOF-index) space.

    Walks points in descending intensity order. For each point:

    * If an already-assigned point lies within ``attach_scan_half_width`` /
      ``attach_mz_idx_half_width`` *and* the candidate's group seed is within
      ``max_scan_from_seed`` / ``max_mz_idx_from_seed`` of the point, join
      that group (nearest by Manhattan; ties broken by higher seed
      intensity, then smaller point index).
    * Else if ``intensity ≥ min_seed_intensity``, promote to a new seed.
    * Else drop as an orphan (does not claim grid territory).

    Final centroids are intensity-weighted means in float (m/z, IM) space.
    Groups whose summed intensity falls below ``min_centroid_intensity`` are
    dropped. See ``apps/ALGORITHM.md`` Stage 3 for the full write-up.

    ``max_*_from_seed`` is the "leash" — how far any group member can be
    from its seed. ``None`` disables the bound. The cell-neighbor box
    is the local attachment criterion; the leash is the group-wide one.

    Dispatches to a Numba-JIT'd kernel when ``use_numba=True`` (default)
    and Numba is available, falling back to the pure-Python implementation
    otherwise.
    """
    if scan_indices.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    attach_scan_half_width = max(1, int(attach_scan_half_width))
    attach_mz_idx_half_width = max(1, int(attach_mz_idx_half_width))

    scan_arr = np.asarray(scan_indices, dtype=np.int64)
    mz_arr = np.asarray(mz_indices, dtype=np.int64)
    int_arr = np.asarray(intensities, dtype=np.float64)
    mz_val_arr = np.asarray(mz_values, dtype=np.float64)
    im_arr = np.asarray(im_values, dtype=np.float64)

    # -1 sentinel = no limit
    max_s = -1 if max_scan_from_seed is None else int(max_scan_from_seed)
    max_m = -1 if max_mz_idx_from_seed is None else int(max_mz_idx_from_seed)

    if _HAS_NUMBA and use_numba:
        return _watershed_njit_kernel(
            scan_arr, mz_arr, int_arr, mz_val_arr, im_arr,
            attach_scan_half_width, attach_mz_idx_half_width,
            float(min_seed_intensity), float(min_centroid_intensity),
            max_s, max_m,
        )
    return _watershed_python_kernel(
        scan_arr, mz_arr, int_arr, mz_val_arr, im_arr,
        attach_scan_half_width=attach_scan_half_width,
        attach_mz_idx_half_width=attach_mz_idx_half_width,
        min_seed_intensity=min_seed_intensity,
        min_centroid_intensity=min_centroid_intensity,
        max_scan_from_seed=max_s,
        max_mz_idx_from_seed=max_m,
    )


@dataclass(frozen=True)
class WatershedCentroider(Centroider):
    """Intensity-ordered region-growing centroider in integer-index space.

    Operates on ``(scan_number, TOF_index)`` integers — avoiding the
    floating-point binning step that :class:`MergePeaksCentroider` does.
    Each point either joins the nearest already-assigned point's group
    (within a rectangular tolerance box) or promotes to a new seed.
    See ``apps/ALGORITHM.md`` Stage 3 for the full write-up.

    The optional ``smooth_*_half_width`` parameters apply a position-preserving
    box-mean filter to intensities *before* seed selection, which prevents
    noisy spikes from outranking the actual peak summit and stabilises
    seed ordering. Set both to ``0`` to skip.

    The optional ``max_*_from_seed`` parameters are per-group "leashes":
    a follower is rejected if its distance from the candidate group's
    *seed* (not its nearest member) exceeds the bound on either axis.
    This stops a group from wandering by chaining through followers.
    ``None`` disables the bound on that axis.
    """

    attach_scan_half_width: int = 10
    attach_mz_idx_half_width: int = 3
    min_seed_intensity: float = 0.0
    min_centroid_intensity: float = 0.0
    smooth_scan_half_width: int = 5
    smooth_mz_idx_half_width: int = 3
    max_scan_from_seed: int | None = None
    max_mz_idx_from_seed: int | None = 10
    use_numba: bool = True

    def __call__(
        self,
        spectrum: RawSpectrum,
        td: TimsData,
        frame_id: int,
        *,
        ion_mobility_type: Literal["ook0", "ccs", "voltage"] = "ook0",
    ) -> np.ndarray:
        if spectrum.empty:
            return np.empty((0, 3), dtype=np.float64)

        # Convert once for the final centroid coordinates only; the
        # algorithm itself runs on integer indices for stability.
        converted = convert(
            spectrum, td, frame_id, ion_mobility_type=ion_mobility_type
        )
        mz_values = converted[:, 0]
        im_values = converted[:, 2]

        intensities = spectrum.intensities
        if self.smooth_scan_half_width > 0 and self.smooth_mz_idx_half_width > 0:
            intensities = _box_smooth_intensities_arrays(
                spectrum.scan_indices,
                spectrum.mz_indices,
                intensities,
                smooth_scan_half_width=self.smooth_scan_half_width,
                smooth_mz_idx_half_width=self.smooth_mz_idx_half_width,
            )

        return _watershed_kernel(
            spectrum.scan_indices,
            spectrum.mz_indices,
            intensities,
            mz_values,
            im_values,
            attach_scan_half_width=self.attach_scan_half_width,
            attach_mz_idx_half_width=self.attach_mz_idx_half_width,
            min_seed_intensity=self.min_seed_intensity,
            min_centroid_intensity=self.min_centroid_intensity,
            max_scan_from_seed=self.max_scan_from_seed,
            max_mz_idx_from_seed=self.max_mz_idx_from_seed,
            use_numba=self.use_numba,
        )


def centroid_peaks(
    peaks: np.ndarray, centroider: MergePeaksCentroider
) -> np.ndarray:
    """Cluster ``(mz, intensity, ion_mobility)`` peaks into centroids.

    Convenience wrapper for users who already have a converted ``(N, 3)``
    array and want to skip back through a :class:`RawSpectrum`. Only
    supports :class:`MergePeaksCentroider` since :class:`WatershedCentroider`
    needs integer indices that aren't recoverable from float peaks.
    """
    from .centroiding import merge_peaks

    if peaks.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return merge_peaks(
        peaks[:, 0],
        peaks[:, 1],
        peaks[:, 2],
        mz_tolerance=centroider.mz_tolerance,
        mz_tolerance_type=centroider.mz_tolerance_type,
        im_tolerance=centroider.im_tolerance,
        im_tolerance_type=centroider.im_tolerance_type,
        min_peaks=centroider.min_peaks,
        max_peaks=centroider.max_peaks,
        peak_noise_filter=centroider.peak_noise_filter,
        peak_noise_window=centroider.peak_noise_window,
        peak_noise_end_fraction=centroider.peak_noise_end_fraction,
        use_numba=centroider.use_numba,
    )


__all__ = [
    "RawSpectrum",
    "Centroider",
    "MergePeaksCentroider",
    "WatershedCentroider",
    "read_spectrum",
    "subset_scans",
    "exclude_region",
    "apply_noise",
    "convert",
    "centroid_peaks",
]
