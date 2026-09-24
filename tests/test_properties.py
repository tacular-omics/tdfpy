"""Property-based tests for the centroiding kernels and the CCS conversion.

Profiles: ``default`` (fast, used in CI) and ``thorough``. Pick one with
``HYPOTHESIS_PROFILE=thorough uv run pytest tests/test_properties.py``.
"""

import os

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from tdfpy.calibration import ccs_to_ook0, ook0_to_ccs
from tdfpy.centroiding import _HAS_NUMBA, _sum_by_tof_index, merge_peaks

settings.register_profile("default", max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile("thorough", max_examples=2000, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "default"))

KERNELS = [pytest.param(False, id="python"), pytest.param(True, id="numba", marks=pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed"))]


@st.composite
def peak_lists(draw, max_size: int = 60):
    """Raw peaks with clustered m/z and mobility, so merges actually happen.

    Values come from a small grid around a few centres, which produces exact
    ties, zero intensities and identical points: the degenerate cases.
    """
    n = draw(st.integers(0, max_size))
    centres = draw(st.lists(st.floats(100.0, 2000.0), min_size=1, max_size=4))
    mz = np.array([draw(st.sampled_from(centres)) + draw(st.integers(-20, 20)) * 1e-4 for _ in range(n)], dtype=np.float64)
    im = np.array([0.6 + draw(st.integers(0, 30)) * 0.01 for _ in range(n)], dtype=np.float64)
    intensity = draw(arrays(np.float64, n, elements=st.one_of(st.just(0.0), st.floats(0.0, 1e6, allow_subnormal=False))))
    return mz, intensity, im


merge_options = st.fixed_dictionaries(
    {
        "mz_tolerance": st.sampled_from([0.0, 1.0, 10.0, 50.0]),
        "im_tolerance": st.sampled_from([0.0, 0.01, 0.05, 0.2]),
        "im_tolerance_unit": st.sampled_from(["relative", "absolute"]),
    }
)


@pytest.mark.parametrize("use_numba", KERNELS)
@given(peaks=peak_lists(), opts=merge_options)
def test_merge_peaks_conserves_intensity(use_numba, peaks, opts):
    """With no filtering every raw point lands in exactly one centroid."""
    mz, intensity, im = peaks
    out = merge_peaks(mz, intensity, im, min_peaks=1, use_numba=use_numba, **opts)
    assert out.shape[1] == 3
    assert len(out) <= len(mz)
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out[:, 1].sum(), intensity.sum(), rtol=1e-9, atol=1e-6)
    if len(mz):
        # Weighted means stay inside the input range.
        eps = 1e-9 * max(1.0, float(mz.max()))
        assert (out[:, 0] >= mz.min() - eps).all() and (out[:, 0] <= mz.max() + eps).all()
        assert (out[:, 2] >= im.min() - 1e-12).all() and (out[:, 2] <= im.max() + 1e-12).all()


@pytest.mark.parametrize("use_numba", KERNELS)
@given(peaks=peak_lists(), opts=merge_options, min_peaks=st.integers(2, 5), max_peaks=st.one_of(st.none(), st.integers(-1, 10)))
def test_merge_peaks_filters_never_add_intensity(use_numba, peaks, opts, min_peaks, max_peaks):
    mz, intensity, im = peaks
    out = merge_peaks(mz, intensity, im, min_peaks=min_peaks, max_peaks=max_peaks, use_numba=use_numba, **opts)
    assert np.isfinite(out).all()
    assert out[:, 1].sum() <= intensity.sum() * (1 + 1e-9) + 1e-6
    if max_peaks is not None and max_peaks > 0:
        assert len(out) <= max_peaks


@pytest.mark.skipif(not _HAS_NUMBA, reason="numba not installed")
@given(peaks=peak_lists(), opts=merge_options, min_peaks=st.integers(0, 4), noise=st.booleans())
def test_merge_peaks_kernels_agree(peaks, opts, min_peaks, noise):
    mz, intensity, im = peaks
    kwargs = dict(min_peaks=min_peaks, peak_noise_filter=noise, **opts)
    a = merge_peaks(mz, intensity, im, use_numba=False, **kwargs)
    b = merge_peaks(mz, intensity, im, use_numba=True, **kwargs)
    assert a.shape == b.shape
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("use_numba", KERNELS)
@pytest.mark.parametrize("n", [0, 1, 5])
def test_merge_peaks_degenerate_frames(use_numba, n):
    """Empty frames, all-zero intensities and identical points must not crash or emit NaN."""
    mz = np.full(n, 500.0)
    im = np.full(n, 1.0)
    for intensity in (np.zeros(n), np.full(n, 7.0)):
        out = merge_peaks(mz, intensity, im, min_peaks=1, use_numba=use_numba)
        assert out.shape == ((1 if n else 0), 3)
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out[:, 1].sum(), intensity.sum())


@given(
    tof=arrays(np.int64, st.integers(1, 200), elements=st.integers(0, 400_000)),
    data=st.data(),
)
def test_sum_by_tof_index_sorted_and_conserving(tof, data):
    intensity = data.draw(arrays(np.float64, tof.size, elements=st.floats(0.0, 1e6, allow_subnormal=False)))
    keys, sums = _sum_by_tof_index(tof, intensity)
    assert (np.diff(keys) > 0).all()
    assert (sums > 0).all()
    np.testing.assert_allclose(sums.sum(), intensity.sum(), rtol=1e-9)
    # Brute-force reference.
    expected = {}
    for t, v in zip(tof.tolist(), intensity.tolist(), strict=True):
        expected[t] = expected.get(t, 0.0) + v
    expected = {k: v for k, v in expected.items() if v != 0.0}
    assert keys.tolist() == sorted(expected)
    np.testing.assert_allclose(sums, [expected[k] for k in keys.tolist()], rtol=1e-12)


@given(
    ook0=st.floats(0.3, 2.5),
    charge=st.integers(1, 10),
    mz=st.floats(50.0, 5000.0),
)
def test_ccs_round_trip_and_monotone(ook0, charge, mz):
    ccs = ook0_to_ccs(ook0, charge, mz)
    assert ccs > 0
    assert ccs_to_ook0(ccs, charge, mz) == pytest.approx(ook0, rel=1e-12)
    assert ook0_to_ccs(ook0 * 1.01, charge, mz) > ccs
