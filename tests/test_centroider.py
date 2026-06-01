"""Tests for the Centroider hierarchy.

The watershed kernel is tested against synthetic data; live-data integration
runs through tests/test_spectra.py via MergePeaksCentroider (the default).
"""

from __future__ import annotations

import numpy as np
import pytest

from tdfpy.pipeline import (
    Centroider,
    MergePeaksCentroider,
    WatershedCentroider,
    _box_smooth_intensities_arrays,
    _watershed_kernel,
)


# --------------------------------------------------------------------------
# ABC / dataclass plumbing
# --------------------------------------------------------------------------


class TestSubclassRelationships:
    def test_merge_peaks_is_centroider(self):
        assert issubclass(MergePeaksCentroider, Centroider)

    def test_watershed_is_centroider(self):
        assert issubclass(WatershedCentroider, Centroider)

    def test_centroider_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Centroider()  # type: ignore[abstract]

    def test_frozen_and_hashable(self):
        # Both must be hashable so they can serve as cache keys.
        hash((MergePeaksCentroider(), WatershedCentroider()))

    def test_equality(self):
        assert MergePeaksCentroider() == MergePeaksCentroider()
        assert (
            WatershedCentroider(attach_scan_half_width=10)
            == WatershedCentroider(attach_scan_half_width=10)
        )
        assert (
            WatershedCentroider(attach_scan_half_width=10)
            != WatershedCentroider(attach_scan_half_width=11)
        )


# --------------------------------------------------------------------------
# Watershed kernel on synthetic data
# --------------------------------------------------------------------------


def _two_blobs(
    seed_intensity_a: float = 1000.0,
    seed_intensity_b: float = 500.0,
    blob_radius: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build two well-separated rectangular blobs in (scan, TOF-index) space.

    Blob A is centered at (scan=50, mz_idx=1000); blob B at (scan=200, mz_idx=2000).
    Each blob is a ``(2*radius+1)²`` patch of points; the center has the highest
    intensity, others are at half. Returns the per-point arrays.
    """
    points = []
    for center_scan, center_mz, peak_int in [
        (50, 1000, seed_intensity_a),
        (200, 2000, seed_intensity_b),
    ]:
        for ds in range(-blob_radius, blob_radius + 1):
            for dm in range(-blob_radius, blob_radius + 1):
                intensity = peak_int if (ds == 0 and dm == 0) else peak_int * 0.5
                points.append((center_scan + ds, center_mz + dm, intensity))
    scan = np.array([p[0] for p in points], dtype=np.int64)
    mz_idx = np.array([p[1] for p in points], dtype=np.int64)
    intens = np.array([p[2] for p in points], dtype=np.float64)
    # m/z and IM are linear functions of index here; the actual values are
    # only used for the final centroid coordinates.
    mz_values = mz_idx.astype(np.float64) * 0.001 + 100.0
    im_values = scan.astype(np.float64) * 0.001 + 0.5
    return scan, mz_idx, intens, mz_values, im_values


class TestWatershedKernelSyntheticBlobs:
    def test_two_blobs_emit_two_centroids(self):
        scan, mz_idx, intens, mz_v, im_v = _two_blobs()
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=10,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        assert out.shape == (2, 3)

    def test_centroid_positions_match_seeds(self):
        scan, mz_idx, intens, mz_v, im_v = _two_blobs()
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=10,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        # Centroids sorted by intensity (descending) — A is brighter than B.
        out_sorted = out[np.argsort(out[:, 1])[::-1]]
        # Blob A: center at scan=50, mz_idx=1000 → mz=101.0, ook0=0.55
        # Blob B: center at scan=200, mz_idx=2000 → mz=102.0, ook0=0.7
        assert out_sorted[0, 0] == pytest.approx(101.0, abs=0.05)
        assert out_sorted[0, 2] == pytest.approx(0.55, abs=0.05)
        assert out_sorted[1, 0] == pytest.approx(102.0, abs=0.05)
        assert out_sorted[1, 2] == pytest.approx(0.7, abs=0.05)

    def test_total_intensity_conserved(self):
        scan, mz_idx, intens, mz_v, im_v = _two_blobs()
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=10,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        assert out[:, 1].sum() == pytest.approx(intens.sum())

    def test_min_seed_intensity_drops_weaker_blob(self):
        scan, mz_idx, intens, mz_v, im_v = _two_blobs(
            seed_intensity_a=1000.0, seed_intensity_b=100.0,
        )
        # Threshold between the two seeds.
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=10,
            min_seed_intensity=200.0, min_centroid_intensity=0.0,
        )
        # The blob B center can't seed (100 < 200), and its surrounding
        # points (intensity 50) also can't. Only blob A emerges.
        assert out.shape == (1, 3)

    def test_min_centroid_intensity_drops_small_groups(self):
        scan, mz_idx, intens, mz_v, im_v = _two_blobs(
            seed_intensity_a=1000.0, seed_intensity_b=100.0,
        )
        # Total intensity of blob A = 1000 + 24*500 = 13000. Blob B = 100 + 24*50 = 1300.
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=10,
            min_seed_intensity=0.0, min_centroid_intensity=5000.0,
        )
        assert out.shape == (1, 3)
        assert out[0, 1] == pytest.approx(13000.0)

    def test_high_seed_threshold_returns_empty(self):
        scan, mz_idx, intens, mz_v, im_v = _two_blobs()
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=10,
            min_seed_intensity=1e9, min_centroid_intensity=0.0,
        )
        assert out.shape == (0, 3)

    def test_empty_input(self):
        empty = np.empty(0, dtype=np.int64)
        empty_f = np.empty(0, dtype=np.float64)
        out = _watershed_kernel(
            empty, empty, empty_f, empty_f, empty_f,
            attach_scan_half_width=10, attach_mz_idx_half_width=3,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        assert out.shape == (0, 3)


class TestWatershedLeash:
    """``max_*_from_seed`` bounds chain-grown groups from wandering."""

    def _chain(self, n=30):
        # n collinear points along the TOF axis at scan=0; intensity
        # strictly decreasing so the leftmost point always seeds.
        scan = np.zeros(n, dtype=np.int64)
        mz_idx = np.arange(n, dtype=np.int64) * 3  # spaced 3 indices apart
        intens = np.linspace(1000.0, 100.0, n)
        mz_v = mz_idx * 0.001 + 100.0
        im_v = np.full(n, 0.5)
        return scan, mz_idx, intens, mz_v, im_v

    def test_no_leash_merges_chain(self):
        out = _watershed_kernel(
            *self._chain(),
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        assert out.shape[0] == 1

    def test_leash_splits_chain(self):
        out = _watershed_kernel(
            *self._chain(),
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_mz_idx_from_seed=10,
        )
        # Chain of 30 points × 3 indices apart = span 87. Leash=10 means
        # groups of ~4 points before a new seed has to take over.
        assert 1 < out.shape[0] < 30

    def test_tighter_leash_more_groups(self):
        loose = _watershed_kernel(
            *self._chain(),
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_mz_idx_from_seed=10,
        )
        tight = _watershed_kernel(
            *self._chain(),
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_mz_idx_from_seed=5,
        )
        assert tight.shape[0] > loose.shape[0]

    def test_total_intensity_conserved_under_leash(self):
        args = self._chain()
        out = _watershed_kernel(
            *args,
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_mz_idx_from_seed=10,
        )
        assert out[:, 1].sum() == pytest.approx(args[2].sum())

    def test_scan_axis_leash_independent(self):
        # Same chain laid along the scan axis.
        scan = np.arange(30, dtype=np.int64) * 3
        mz_idx = np.zeros(30, dtype=np.int64)
        intens = np.linspace(1000.0, 100.0, 30)
        mz_v = np.full(30, 100.0)
        im_v = scan * 0.001 + 0.5
        loose = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=5, attach_mz_idx_half_width=2,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        leashed = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=5, attach_mz_idx_half_width=2,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_scan_from_seed=10,
        )
        assert loose.shape[0] == 1
        assert leashed.shape[0] > 1

    def test_python_numba_agree_with_leash(self):
        py = _watershed_kernel(
            *self._chain(),
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_mz_idx_from_seed=10, use_numba=False,
        )
        nb = _watershed_kernel(
            *self._chain(),
            attach_scan_half_width=2, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            max_mz_idx_from_seed=10, use_numba=True,
        )
        canon = lambda a: np.array(sorted(map(tuple, np.round(a, 6))))  # noqa: E731
        assert np.array_equal(canon(py), canon(nb))


class TestWatershedSeparation:
    def test_blobs_outside_box_split(self):
        # Two blobs 100 scans apart, attach_scan_half_width=10 → must split.
        scan, mz_idx, intens, mz_v, im_v = _two_blobs()
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=10, attach_mz_idx_half_width=3,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        assert out.shape[0] == 2

    def test_blobs_inside_box_merge(self):
        # Same blobs but with a huge box → merge into one centroid.
        scan, mz_idx, intens, mz_v, im_v = _two_blobs()
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=500, attach_mz_idx_half_width=2000,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
        )
        assert out.shape[0] == 1


# --------------------------------------------------------------------------
# Box-smoothing helper
# --------------------------------------------------------------------------


class TestBoxSmoothIntensities:
    def test_preserves_array_length(self):
        scan = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        mz = np.array([100, 100, 100, 100, 100], dtype=np.int64)
        intens = np.array([1.0, 10.0, 1.0, 10.0, 1.0])
        out = _box_smooth_intensities_arrays(
            scan, mz, intens,
            smooth_scan_half_width=1, smooth_mz_idx_half_width=0,
        )
        assert out.shape == intens.shape

    def test_constant_input_unchanged(self):
        scan = np.arange(10, dtype=np.int64)
        mz = np.zeros(10, dtype=np.int64)
        intens = np.full(10, 42.0)
        out = _box_smooth_intensities_arrays(
            scan, mz, intens,
            smooth_scan_half_width=2, smooth_mz_idx_half_width=0,
        )
        assert np.allclose(out, 42.0)

    def test_spike_gets_smoothed_to_local_mean(self):
        # Five neighboring points along scan axis, one spike in the middle.
        scan = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        mz = np.zeros(5, dtype=np.int64)
        intens = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
        # smooth_scan_half_width=2 → centre point at scan=2 averages over scans 0..4 (all 5).
        out = _box_smooth_intensities_arrays(
            scan, mz, intens,
            smooth_scan_half_width=2, smooth_mz_idx_half_width=0,
        )
        assert out[2] == pytest.approx((1 + 1 + 100 + 1 + 1) / 5)

    def test_empty(self):
        out = _box_smooth_intensities_arrays(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            smooth_scan_half_width=1, smooth_mz_idx_half_width=1,
        )
        assert out.shape == (0,)
