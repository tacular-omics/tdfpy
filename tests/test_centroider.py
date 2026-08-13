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
    _watershed_kernel,
    box_smooth,
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


class TestWatershedZeroIntensity:
    """A group with no intensity has no weighted mean — use the seed's position.

    Dividing by a "safe" 1.0 instead put the centroid at m/z 0 / IM 0, a
    coordinate no input point occupied. merge_peaks already falls back to the
    seed peak in the same situation.
    """

    @pytest.mark.parametrize("use_numba", [True, False], ids=["numba", "python"])
    def test_single_zero_intensity_point_keeps_its_coordinates(self, use_numba):
        out = _watershed_kernel(
            np.array([7], dtype=np.int64),
            np.array([1000], dtype=np.int64),
            np.array([0.0]),
            np.array([500.25]),
            np.array([1.125]),
            attach_scan_half_width=10, attach_mz_idx_half_width=3,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            use_numba=use_numba,
        )
        assert out.shape == (1, 3)
        assert out[0, 0] == pytest.approx(500.25)
        assert out[0, 1] == pytest.approx(0.0)
        assert out[0, 2] == pytest.approx(1.125)

    @pytest.mark.parametrize("use_numba", [True, False], ids=["numba", "python"])
    def test_zero_group_alongside_a_real_one(self, use_numba):
        # Two zero-intensity points that group together, well away from a
        # normal blob. The zero group must report the seed's coordinates and
        # must not drag the real centroid anywhere.
        scan = np.array([0, 1, 100, 101], dtype=np.int64)
        mz_idx = np.array([10, 11, 900, 901], dtype=np.int64)
        intens = np.array([0.0, 0.0, 300.0, 100.0])
        mz_v = np.array([200.0, 200.1, 700.0, 700.1])
        im_v = np.array([0.60, 0.61, 1.40, 1.41])
        out = _watershed_kernel(
            scan, mz_idx, intens, mz_v, im_v,
            attach_scan_half_width=5, attach_mz_idx_half_width=5,
            min_seed_intensity=0.0, min_centroid_intensity=0.0,
            use_numba=use_numba,
        )
        assert out.shape == (2, 3)
        zero_row = out[out[:, 1] == 0.0]
        assert zero_row.shape == (1, 3)
        # Either zero-intensity point may seed (both tie at 0.0), but the
        # centroid must sit on one of them, never at the origin.
        assert zero_row[0, 0] == pytest.approx(200.0) or zero_row[0, 0] == pytest.approx(200.1)
        assert zero_row[0, 2] == pytest.approx(0.60) or zero_row[0, 2] == pytest.approx(0.61)
        real_row = out[out[:, 1] > 0.0][0]
        assert real_row[1] == pytest.approx(400.0)


@pytest.mark.parametrize(
    "params",
    [
        dict(min_seed_intensity=0.0, min_centroid_intensity=0.0),
        dict(min_seed_intensity=40.0, min_centroid_intensity=0.0),
        dict(min_seed_intensity=0.0, min_centroid_intensity=250.0),
        dict(min_seed_intensity=25.0, min_centroid_intensity=120.0),
    ],
)
def test_watershed_numba_matches_python_on_random_grid(params):
    """Randomised 2-D (scan, TOF) sweep with heavy ties and live thresholds.

    Intensities are drawn from a small integer set on purpose: ties are where
    the two kernels' different neighbourhood traversals could diverge, so the
    tiebreak chain (distance → seed intensity → point index) gets exercised
    rather than avoided.
    """
    rng = np.random.default_rng(2024)
    n = 400
    scan = rng.integers(0, 40, n).astype(np.int64)
    mz_idx = rng.integers(0, 60, n).astype(np.int64)
    # Deduplicate: two points at the same (scan, TOF index) is not a state the
    # reader can produce, and it makes the ordering ambiguous.
    _, unique_idx = np.unique(
        np.column_stack([scan, mz_idx]), axis=0, return_index=True
    )
    scan, mz_idx = scan[unique_idx], mz_idx[unique_idx]
    intens = rng.choice([10.0, 20.0, 30.0, 50.0], size=scan.size)
    mz_v = mz_idx * 0.01 + 300.0
    im_v = scan * 0.01 + 0.8

    common = dict(
        attach_scan_half_width=3, attach_mz_idx_half_width=2,
        max_scan_from_seed=8, max_mz_idx_from_seed=6, **params,
    )
    nb = _watershed_kernel(scan, mz_idx, intens, mz_v, im_v, use_numba=True, **common)
    py = _watershed_kernel(scan, mz_idx, intens, mz_v, im_v, use_numba=False, **common)
    assert nb.shape == py.shape
    canon = lambda a: np.array(sorted(map(tuple, np.round(a, 9))))  # noqa: E731
    np.testing.assert_allclose(canon(nb), canon(py), rtol=0, atol=1e-9)


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
        out = box_smooth(
            scan, mz, intens,
            scan_half_width=1, mz_idx_half_width=0, mode="mean",
        )
        assert out.shape == intens.shape

    def test_constant_input_unchanged(self):
        scan = np.arange(10, dtype=np.int64)
        mz = np.zeros(10, dtype=np.int64)
        intens = np.full(10, 42.0)
        out = box_smooth(
            scan, mz, intens,
            scan_half_width=2, mz_idx_half_width=0, mode="mean",
        )
        assert np.allclose(out, 42.0)

    def test_spike_gets_smoothed_to_local_mean(self):
        # Five neighboring points along scan axis, one spike in the middle.
        scan = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        mz = np.zeros(5, dtype=np.int64)
        intens = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
        # scan_half_width=2 → centre point at scan=2 averages over scans 0..4 (all 5).
        out = box_smooth(
            scan, mz, intens,
            scan_half_width=2, mz_idx_half_width=0, mode="mean",
        )
        assert out[2] == pytest.approx((1 + 1 + 100 + 1 + 1) / 5)

    def test_empty(self):
        out = box_smooth(
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            scan_half_width=1, mz_idx_half_width=1, mode="mean",
        )
        assert out.shape == (0,)

    def test_sum_mode_accumulates_window(self):
        # Five neighbouring points along the scan axis; sum over ±2 = all 5.
        scan = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        mz = np.zeros(5, dtype=np.int64)
        intens = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = box_smooth(
            scan, mz, intens,
            scan_half_width=2, mz_idx_half_width=0, mode="sum",
        )
        assert out[2] == pytest.approx(15.0)  # 1+2+3+4+5
        assert out[0] == pytest.approx(1 + 2 + 3)  # scans 0..2

    def test_both_half_widths_cover_the_full_2d_window(self):
        """Exercise the 2-D window: neither half-width is 0.

        A 3x3 patch of TOF indices 10..12 across scans 0..2, all intensity 1,
        plus one point far away in TOF that must stay outside the box. With
        ``scan_half_width=1`` and ``mz_idx_half_width=1`` the centre point sees
        all 9 patch members; each corner sees only its 4-member quadrant.
        """
        scan, mz, intens = [], [], []
        for s in range(3):
            for m in (10, 11, 12):
                scan.append(s)
                mz.append(m)
                intens.append(1.0)
        scan.append(1)   # far in TOF: inside the scan window, outside the m/z one
        mz.append(50)
        intens.append(1.0)
        scan_a = np.array(scan, dtype=np.int64)
        mz_a = np.array(mz, dtype=np.int64)
        int_a = np.array(intens)

        out = box_smooth(
            scan_a, mz_a, int_a,
            scan_half_width=1, mz_idx_half_width=1, mode="sum",
        )
        centre = np.flatnonzero((scan_a == 1) & (mz_a == 11))[0]
        corner = np.flatnonzero((scan_a == 0) & (mz_a == 10))[0]
        edge = np.flatnonzero((scan_a == 0) & (mz_a == 11))[0]
        far = np.flatnonzero(mz_a == 50)[0]
        assert out[centre] == pytest.approx(9.0)  # 3 scans x 3 TOF indices
        assert out[corner] == pytest.approx(4.0)  # 2 scans x 2 TOF indices
        assert out[edge] == pytest.approx(6.0)    # 2 scans x 3 TOF indices
        assert out[far] == pytest.approx(1.0)     # only itself

        means = box_smooth(
            scan_a, mz_a, int_a,
            scan_half_width=1, mz_idx_half_width=1, mode="mean",
        )
        # Every window is all-ones, so each mean is 1.0 regardless of size —
        # which also proves the count is gathered over the same 2-D window.
        np.testing.assert_allclose(means, 1.0)

    def test_both_half_widths_with_varying_intensities(self):
        # Same 3x3 patch, but intensities 1..9 so the window sum is sensitive
        # to which members it gathers, not just how many.
        scan = np.repeat(np.arange(3), 3).astype(np.int64)
        mz = np.tile(np.array([10, 11, 12]), 3).astype(np.int64)
        intens = np.arange(1.0, 10.0)
        out = box_smooth(
            scan, mz, intens, scan_half_width=1, mz_idx_half_width=1, mode="sum"
        )
        assert out[4] == pytest.approx(intens.sum())          # centre: all 9
        assert out[0] == pytest.approx(1 + 2 + 4 + 5)         # top-left quadrant
        assert out[8] == pytest.approx(5 + 6 + 8 + 9)         # bottom-right quadrant

    def test_invalid_mode_rejected(self):
        scan = np.zeros(3, dtype=np.int64)
        mz = np.arange(3, dtype=np.int64)
        intens = np.ones(3)
        with pytest.raises(ValueError, match="must be 'sum' or 'mean'"):
            box_smooth(
                scan, mz, intens,
                scan_half_width=1, mz_idx_half_width=1,
                mode="median",  # type: ignore[arg-type]
            )

    def test_sum_mode_amplifies_streak_over_isolated_noise(self):
        # A vertical streak (same mz across scans) sums up; a lone hit doesn't.
        scan = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)
        mz = np.array([10, 10, 10, 10, 10, 50], dtype=np.int64)
        intens = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
        out = box_smooth(
            scan, mz, intens,
            scan_half_width=5, mz_idx_half_width=0, mode="sum",
        )
        assert out[0] == pytest.approx(500.0)  # streak point
        assert out[5] == pytest.approx(100.0)  # isolated noise unchanged


class TestSmoothOp:
    def test_smooth_returns_new_spectrum_same_positions(self):
        from tdfpy.pipeline import RawSpectrum, smooth

        spec = RawSpectrum(
            scan_indices=np.array([0, 1, 2], dtype=np.int64),
            mz_indices=np.array([10, 10, 10], dtype=np.int64),
            intensities=np.array([1.0, 1.0, 1.0]),
            num_scans=3,
        )
        out = smooth(spec, scan_half_width=5, mz_idx_half_width=0, mode="sum")
        np.testing.assert_array_equal(out.scan_indices, spec.scan_indices)
        np.testing.assert_array_equal(out.mz_indices, spec.mz_indices)
        assert out.intensities.tolist() == [3.0, 3.0, 3.0]

    def test_smooth_empty(self):
        from tdfpy.pipeline import RawSpectrum, smooth

        spec = RawSpectrum.empty_like(10)
        assert smooth(spec).empty


class TestSmoothConfig:
    def test_defaults(self):
        from tdfpy import Smooth

        s = Smooth()
        assert (s.scan_half_width, s.mz_idx_half_width, s.mode) == (5, 2, "sum")

    def test_is_hashable(self):
        from tdfpy import Smooth

        # Frozen → usable as a dict key / Streamlit cache arg.
        assert hash(Smooth()) == hash(Smooth())

    def test_apply_matches_smooth_op(self):
        from tdfpy import Smooth
        from tdfpy.pipeline import RawSpectrum, smooth

        spec = RawSpectrum(
            scan_indices=np.array([0, 1, 2, 3, 4], dtype=np.int64),
            mz_indices=np.full(5, 10, dtype=np.int64),
            intensities=np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            num_scans=5,
        )
        cfg = Smooth(scan_half_width=2, mz_idx_half_width=0, mode="sum")
        via_cfg = cfg.apply(spec)
        via_op = smooth(spec, scan_half_width=2, mz_idx_half_width=0, mode="sum")
        np.testing.assert_array_equal(via_cfg.intensities, via_op.intensities)

    def test_invalid_mode_rejected_at_construction(self):
        from tdfpy import Smooth

        # Anything but "sum"/"mean" used to fall through to the "sum" branch.
        with pytest.raises(ValueError, match="must be 'sum' or 'mean'"):
            Smooth(mode="median")  # type: ignore[arg-type]

    def test_smooth_op_rejects_invalid_mode(self):
        from tdfpy.pipeline import RawSpectrum, smooth

        spec = RawSpectrum(
            scan_indices=np.array([0, 1], dtype=np.int64),
            mz_indices=np.array([10, 11], dtype=np.int64),
            intensities=np.array([1.0, 2.0]),
            num_scans=2,
        )
        with pytest.raises(ValueError, match="must be 'sum' or 'mean'"):
            smooth(spec, mode="median")  # type: ignore[arg-type]

    def test_smooth_op_rejects_invalid_mode_even_when_empty(self):
        # The empty-spectrum short circuit must not skip validation.
        from tdfpy.pipeline import RawSpectrum, smooth

        with pytest.raises(ValueError, match="must be 'sum' or 'mean'"):
            smooth(RawSpectrum.empty_like(4), mode="median")  # type: ignore[arg-type]
