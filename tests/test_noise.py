"""Tests for tdfpy.noise — intensity-threshold filter classes and coercion."""

import numpy as np
import pytest

from tdfpy.noise import (
    AbsoluteThreshold,
    BaselineThreshold,
    HistogramThreshold,
    HorizontalHaloFilter,
    IntensityThreshold,
    IterativeMedianThreshold,
    MadThreshold,
    NoiseFilter,
    PercentileThreshold,
    VerticalNoiseFilter,
    coerce_filters,
)
from tdfpy.noise import structural as _structural
from tdfpy.noise.structural import (
    _HAS_NUMBA,
    _single_pass_filter,
    _single_pass_filter_python,
)

RNG = np.random.default_rng(42)
# A realistic-ish intensity distribution: mostly low noise, some real peaks.
NOISE_ARRAY = np.concatenate(
    [
        RNG.uniform(100, 500, 900),
        RNG.uniform(5000, 20000, 100),
    ]
).astype(np.float64)


class TestAbsoluteThreshold:
    def test_default_is_zero(self):
        assert AbsoluteThreshold().compute_threshold(NOISE_ARRAY) == 0.0

    def test_passes_value_through(self):
        assert AbsoluteThreshold(value=1234.5).compute_threshold(NOISE_ARRAY) == 1234.5

    def test_keep_mask_uses_threshold(self):
        f = AbsoluteThreshold(value=1000.0)
        mask = f.keep_mask(
            np.zeros_like(NOISE_ARRAY, dtype=np.int64),
            np.zeros_like(NOISE_ARRAY, dtype=np.int64),
            NOISE_ARRAY,
            num_scans=1,
            td=None,  # type: ignore[arg-type]
            frame_id=0,
        )
        assert mask.dtype == bool
        assert mask.sum() == int((NOISE_ARRAY >= 1000.0).sum())


class TestMadThreshold:
    def test_above_median(self):
        result = MadThreshold().compute_threshold(NOISE_ARRAY)
        assert result > float(np.median(NOISE_ARRAY))

    def test_k_scales_threshold(self):
        # Larger k → stricter threshold
        t1 = MadThreshold(k=1.0).compute_threshold(NOISE_ARRAY)
        t3 = MadThreshold(k=3.0).compute_threshold(NOISE_ARRAY)
        t5 = MadThreshold(k=5.0).compute_threshold(NOISE_ARRAY)
        assert t1 < t3 < t5

    def test_reasonable_range(self):
        result = MadThreshold().compute_threshold(NOISE_ARRAY)
        assert 0 < result < float(np.max(NOISE_ARRAY))


class TestPercentileThreshold:
    def test_default_equals_p75(self):
        result = PercentileThreshold().compute_threshold(NOISE_ARRAY)
        assert result == pytest.approx(float(np.percentile(NOISE_ARRAY, 75)))

    def test_custom_q(self):
        result = PercentileThreshold(q=90).compute_threshold(NOISE_ARRAY)
        assert result == pytest.approx(float(np.percentile(NOISE_ARRAY, 90)))


class TestHistogramThreshold:
    def test_positive(self):
        assert HistogramThreshold().compute_threshold(NOISE_ARRAY) > 0

    def test_bins_field(self):
        # Different bin counts should produce different thresholds for non-trivial data
        t_few = HistogramThreshold(bins=10).compute_threshold(NOISE_ARRAY)
        t_many = HistogramThreshold(bins=200).compute_threshold(NOISE_ARRAY)
        assert t_few != t_many


class TestBaselineThreshold:
    def test_positive(self):
        assert BaselineThreshold().compute_threshold(NOISE_ARRAY) > 0

    def test_based_on_bottom_quartile(self):
        result = BaselineThreshold().compute_threshold(NOISE_ARRAY)
        assert result < float(np.percentile(NOISE_ARRAY, 75))


class TestIterativeMedianThreshold:
    def test_positive(self):
        assert IterativeMedianThreshold().compute_threshold(NOISE_ARRAY) > 0


class TestCoerceFilters:
    def test_none(self):
        assert coerce_filters(None) == ()

    def test_string_mad(self):
        result = coerce_filters("mad")
        assert len(result) == 1
        assert isinstance(result[0], MadThreshold)

    def test_string_each_alias(self):
        for name, cls in [
            ("mad", MadThreshold),
            ("percentile", PercentileThreshold),
            ("histogram", HistogramThreshold),
            ("baseline", BaselineThreshold),
            ("iterative_median", IterativeMedianThreshold),
        ]:
            (filt,) = coerce_filters(name)
            assert isinstance(filt, cls)

    def test_numeric_becomes_absolute(self):
        (filt,) = coerce_filters(500.0)
        assert isinstance(filt, AbsoluteThreshold)
        assert filt.value == 500.0

    def test_int_becomes_absolute(self):
        (filt,) = coerce_filters(0)
        assert isinstance(filt, AbsoluteThreshold)
        assert filt.value == 0.0

    def test_instance_passthrough(self):
        f = MadThreshold(k=5.0)
        (out,) = coerce_filters(f)
        assert out is f

    def test_list_flattened(self):
        result = coerce_filters([MadThreshold(), 100, "percentile"])
        assert len(result) == 3
        assert isinstance(result[0], MadThreshold)
        assert isinstance(result[1], AbsoluteThreshold)
        assert isinstance(result[2], PercentileThreshold)

    def test_nested_list_flattened(self):
        result = coerce_filters([MadThreshold(), ["percentile", 50.0]])
        assert len(result) == 3

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="Unknown noise filter name"):
            coerce_filters("bogus")

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="Cannot coerce"):
            coerce_filters({"oops": True})  # type: ignore[arg-type]


class TestFrozenAndHashable:
    """Frozen dataclasses must be hashable so they can act as cache keys."""

    def test_intensity_filters_hashable(self):
        hash((MadThreshold(k=3.0), AbsoluteThreshold(value=10), PercentileThreshold()))

    def test_vertical_noise_filter_hashable(self):
        hash(VerticalNoiseFilter(min_streak_scans=5, num_iterations=2))

    def test_equality(self):
        assert MadThreshold(k=3) == MadThreshold(k=3)
        assert MadThreshold(k=3) != MadThreshold(k=5)


class TestSubclassRelationships:
    def test_intensity_threshold_is_noise_filter(self):
        assert issubclass(IntensityThreshold, NoiseFilter)

    def test_all_intensity_subclasses(self):
        for cls in (
            AbsoluteThreshold,
            MadThreshold,
            PercentileThreshold,
            HistogramThreshold,
            BaselineThreshold,
            IterativeMedianThreshold,
        ):
            assert issubclass(cls, IntensityThreshold)
            assert issubclass(cls, NoiseFilter)

    def test_vertical_noise_filter_is_noise_filter(self):
        assert issubclass(VerticalNoiseFilter, NoiseFilter)


def _synthetic_frame(rng, n_scans=200):
    """Synthetic (scan, mz_idx, intensity) frame: real vertical streaks +
    neighbour-column spill + scattered single-hit noise.

    Intensities are integers, as in real Bruker data — the kernel's
    incremental profile is then exact (float counts would accrue round-off).
    """
    scans, mz, inten = [], [], []
    for col in range(12):  # real streaks
        m = 1000 + col * 7
        start = int(rng.integers(0, n_scans - 30))
        length = int(rng.integers(6, 25))
        for s in range(start, start + length):
            scans.append(s)
            mz.append(m)
            inten.append(int(rng.integers(200, 2000)))
            if rng.random() < 0.3:  # neighbour-column spill
                scans.append(s)
                mz.append(m + int(rng.integers(-2, 3)))
                inten.append(int(rng.integers(100, 500)))
    for _ in range(2000):  # scattered noise
        scans.append(int(rng.integers(0, n_scans)))
        mz.append(int(rng.integers(900, 1100)))
        inten.append(int(rng.integers(50, 300)))
    return (
        np.asarray(scans, dtype=np.int64),
        np.asarray(mz, dtype=np.int64),
        np.asarray(inten, dtype=np.float64),
        n_scans,
    )


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
class TestVerticalNumbaEquivalence:
    """The Numba single-pass kernel must match the pure-Python reference."""

    @pytest.mark.parametrize(
        "params",
        [
            dict(mz_idx_half_width=3, min_streak_scans=5, max_gap_scans=1, min_streak_intensity=50.0),
            dict(mz_idx_half_width=0, min_streak_scans=3, max_gap_scans=0, min_streak_intensity=0.0),
            dict(mz_idx_half_width=5, min_streak_scans=8, max_gap_scans=3, min_streak_intensity=500.0),
            dict(mz_idx_half_width=2, min_streak_scans=1, max_gap_scans=2, min_streak_intensity=1000.0),
        ],
    )
    def test_single_pass_matches_python(self, params):
        rng = np.random.default_rng(7)
        scan, mz, inten, ns = _synthetic_frame(rng)
        k_nb, nc_nb, nck_nb, _ = _single_pass_filter(scan, mz, inten, ns, **params)
        k_py, nc_py, nck_py, _ = _single_pass_filter_python(scan, mz, inten, ns, **params)
        np.testing.assert_array_equal(k_nb, k_py)
        assert (nc_nb, nck_nb) == (nc_py, nck_py)

    def test_full_filter_iterations_match(self):
        rng = np.random.default_rng(11)
        scan, mz, inten, ns = _synthetic_frame(rng)
        filt = VerticalNoiseFilter(
            mz_idx_half_width=3, min_streak_scans=5, max_gap_scans=1,
            min_streak_intensity=50.0, num_iterations=3,
        )
        numba_mask = filt.run(scan, mz, inten, num_scans=ns, diagnostics=False)
        original = _structural._HAS_NUMBA
        _structural._HAS_NUMBA = False  # force the Python fallback path
        try:
            py_mask = filt.run(scan, mz, inten, num_scans=ns, diagnostics=False)
        finally:
            _structural._HAS_NUMBA = original
        np.testing.assert_array_equal(numba_mask, py_mask)

    def test_empty_input(self):
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        keep, n_cols, n_kept, spans = _single_pass_filter(
            empty_i, empty_i, empty_f, 100,
            mz_idx_half_width=3, min_streak_scans=5, max_gap_scans=1, min_streak_intensity=50.0,
        )
        assert keep.size == 0 and n_cols == 0 and n_kept == 0 and spans.size == 0


# --------------------------------------------------------------------------
# Horizontal m/z-halo filter
# --------------------------------------------------------------------------


def _halo_mask(filt, scan, mz_idx, inten, num_scans=200):
    """Run HorizontalHaloFilter.keep_mask on integer-index arrays.

    The filter works purely in (scan, TOF index) space, so ``td``/``frame_id``
    are unused and passed as None/0.
    """
    return filt.keep_mask(
        np.asarray(scan, dtype=np.int64),
        np.asarray(mz_idx, dtype=np.int64),
        np.asarray(inten, dtype=np.float64),
        num_scans=num_scans, td=None, frame_id=0,
    )


class TestHorizontalHaloFilter:
    def test_is_noise_filter(self):
        assert issubclass(HorizontalHaloFilter, NoiseFilter)

    def test_empty(self):
        keep = _halo_mask(
            HorizontalHaloFilter(),
            np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0),
        )
        assert keep.shape == (0,)

    def test_keeps_lone_peak(self):
        # No off-column neighbours -> reference 0 -> always kept.
        keep = _halo_mask(HorizontalHaloFilter(), [10], [1000], [50.0])
        assert keep.tolist() == [True]

    def test_removes_weak_left_right_neighbour(self):
        # Weak peak in the SAME scan row, offset in m/z from a bright peak.
        filt = HorizontalHaloFilter(
            peak_fraction=0.1, mz_idx_half_width=20, scan_half_width=2
        )
        keep = _halo_mask(filt, [10, 10], [1000, 1010], [10000.0, 50.0])
        assert keep.tolist() == [True, False]

    def test_keeps_weak_vertical_neighbour_same_mz(self):
        # Same m/z column, offset in mobility = the vertical streak: always kept,
        # even though it is far weaker than the bright peak directly above it.
        filt = HorizontalHaloFilter(
            peak_fraction=0.1, mz_idx_half_width=20, scan_half_width=10
        )
        keep = _halo_mask(filt, [10, 15], [1000, 1000], [10000.0, 50.0])
        assert keep.tolist() == [True, True]

    def test_keeps_diagonal_weak_peak_when_no_same_row_bright(self):
        # A weak peak below-and-beside a bright one, but whose own row has no
        # bright neighbour, is kept (the bright peak is not in its scan range).
        filt = HorizontalHaloFilter(
            peak_fraction=0.1, mz_idx_half_width=20, scan_half_width=2
        )
        # bright at (scan 10, 1000); weak at (scan 50, 1010) — 40 scans away.
        keep = _halo_mask(filt, [10, 50], [1000, 1010], [10000.0, 50.0])
        assert keep.tolist() == [True, True]

    def test_keeps_comparable_neighbour(self):
        # A neighbour nearly as intense as its surroundings is not halo.
        filt = HorizontalHaloFilter(
            peak_fraction=0.1, mz_idx_half_width=20, scan_half_width=2
        )
        keep = _halo_mask(filt, [10, 10], [1000, 1010], [10000.0, 9000.0])
        assert keep.tolist() == [True, True]

    def test_reference_is_box_max_not_mean(self):
        # Same row: bright (10000) and weak (50) flank a mid peak (600).
        # Threshold = peak_fraction * MAX off-column = 0.1*10000 = 1000, so the
        # mid peak is dropped. (A mean reference (~5025) would have kept it.)
        filt = HorizontalHaloFilter(
            peak_fraction=0.1, mz_idx_half_width=25, scan_half_width=2
        )
        keep = _halo_mask(
            filt, [10, 10, 10], [1000, 1010, 1020], [10000.0, 600.0, 50.0]
        )
        assert keep.tolist() == [True, False, False]


class TestHorizontalHaloFilterLive:
    """Integration against the live DDA fixture (needs the Bruker library)."""

    TDF_PATH = "tests/data/example_dda.d"

    def test_returns_subset_mask(self):
        from tdfpy import timsdata
        from tdfpy.pipeline import read_spectrum

        with timsdata.timsdata_connect(self.TDF_PATH) as td:
            cursor = td.conn.cursor()
            cursor.execute(
                "SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1"
            )
            frame_id = cursor.fetchone()[0]
            spec = read_spectrum(td, frame_id)
            mask = HorizontalHaloFilter().keep_mask(
                spec.scan_indices, spec.mz_indices, spec.intensities,
                num_scans=spec.num_scans, td=td, frame_id=frame_id,
            )
        assert mask.dtype == bool
        assert mask.size == spec.num_peaks
        # The filter only removes points; the brightest peak always survives.
        assert mask.sum() <= spec.num_peaks
        assert mask[int(np.argmax(spec.intensities))]
