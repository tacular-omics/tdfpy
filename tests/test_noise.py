import numpy as np
import pytest

from tdfpy.noise import estimate_noise_level


RNG = np.random.default_rng(42)
# A realistic-ish intensity distribution: mostly low noise, some real peaks
NOISE_ARRAY = np.concatenate([
    RNG.uniform(100, 500, 900),   # noise floor
    RNG.uniform(5000, 20000, 100),  # real peaks
]).astype(np.float64)


class TestEstimateNoiseLevelNumericPassthrough:
    def test_float_value(self):
        result = estimate_noise_level(NOISE_ARRAY, method=1000.0)
        assert result == pytest.approx(1000.0)

    def test_int_value(self):
        result = estimate_noise_level(NOISE_ARRAY, method=500)
        assert result == pytest.approx(500.0)

    def test_zero(self):
        result = estimate_noise_level(NOISE_ARRAY, method=0)
        assert result == pytest.approx(0.0)


class TestEstimateNoiseLevelMad:
    def test_returns_float(self):
        result = estimate_noise_level(NOISE_ARRAY, method="mad")
        assert isinstance(result, float)

    def test_above_median(self):
        result = estimate_noise_level(NOISE_ARRAY, method="mad")
        assert result > float(np.median(NOISE_ARRAY))

    def test_reasonable_range(self):
        result = estimate_noise_level(NOISE_ARRAY, method="mad")
        # Should be somewhere in the noise floor, not up near the signal peaks
        assert 0 < result < float(np.max(NOISE_ARRAY))


class TestEstimateNoiseLevelPercentile:
    def test_returns_float(self):
        result = estimate_noise_level(NOISE_ARRAY, method="percentile")
        assert isinstance(result, float)

    def test_equals_75th_percentile(self):
        expected = float(np.percentile(NOISE_ARRAY, 75))
        result = estimate_noise_level(NOISE_ARRAY, method="percentile")
        assert result == pytest.approx(expected)


class TestEstimateNoiseLevelHistogram:
    def test_returns_float(self):
        result = estimate_noise_level(NOISE_ARRAY, method="histogram")
        assert isinstance(result, float)

    def test_positive(self):
        result = estimate_noise_level(NOISE_ARRAY, method="histogram")
        assert result > 0


class TestEstimateNoiseLevelBaseline:
    def test_returns_float(self):
        result = estimate_noise_level(NOISE_ARRAY, method="baseline")
        assert isinstance(result, float)

    def test_positive(self):
        result = estimate_noise_level(NOISE_ARRAY, method="baseline")
        assert result > 0

    def test_based_on_bottom_quartile(self):
        # Threshold should be below the 75th percentile (it's derived from lower 25%)
        result = estimate_noise_level(NOISE_ARRAY, method="baseline")
        assert result < float(np.percentile(NOISE_ARRAY, 75))


class TestEstimateNoiseLevelIterativeMedian:
    def test_returns_float(self):
        result = estimate_noise_level(NOISE_ARRAY, method="iterative_median")
        assert isinstance(result, float)

    def test_positive(self):
        result = estimate_noise_level(NOISE_ARRAY, method="iterative_median")
        assert result > 0


class TestEstimateNoiseLevelErrors:
    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown noise estimation method"):
            estimate_noise_level(NOISE_ARRAY, method="bogus")  # type: ignore[arg-type]
