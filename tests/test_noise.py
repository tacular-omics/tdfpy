"""Tests for tdfpy.noise — intensity-threshold filter classes and coercion."""

import numpy as np
import pytest

from tdfpy.noise import (
    AbsoluteThreshold,
    BaselineThreshold,
    HistogramThreshold,
    IntensityThreshold,
    IterativeMedianThreshold,
    MadThreshold,
    NoiseFilter,
    PercentileThreshold,
    VerticalNoiseFilter,
    coerce_filters,
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
