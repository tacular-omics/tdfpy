import unittest

import numpy as np

from tdfpy import (
    get_centroided_spectrum,
    merge_peaks,
    timsdata,
)
from tdfpy.centroiding import _HAS_NUMBA, _merge_peaks_numba, _merge_peaks_python  # type: ignore[import]

TDF_PATH = r"tests/data/example_dda.d"


class TestSpectra(unittest.TestCase):
    """Test the higher-level spectra API."""

    def test_get_centroided_ms1_spectrum(self):
        """Test extracting a single centroided MS1 spectrum."""
        with timsdata.timsdata_connect(TDF_PATH) as td:
            # Get the first MS1 frame
            cursor = td.conn.cursor()
            cursor.execute(
                "SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1"
            )
            frame_id = cursor.fetchone()[0]

            # Extract spectrum
            spectrum = get_centroided_spectrum(td, frame_id)

            # Verify structure
            self.assertIsInstance(spectrum, np.ndarray)
            self.assertEqual(spectrum.ndim, 2)
            self.assertEqual(spectrum.shape[1], 3)

            # Verify peaks if any exist
            if len(spectrum) > 0:
                first_peak = spectrum[0]
                mz, intensity, mobility = first_peak
                self.assertGreater(mz, 0)
                self.assertGreater(intensity, 0)
                self.assertGreater(mobility, 0)

    def test_get_centroided_ms1_spectra_subset(self):
        """Test extracting specific MS1 spectra (limited to 2)."""
        with timsdata.timsdata_connect(TDF_PATH) as td:
            # Get first 2 MS1 frame IDs
            cursor = td.conn.cursor()
            cursor.execute(
                "SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 2"
            )
            frame_ids = [row[0] for row in cursor.fetchall()]

            if len(frame_ids) >= 2:  # Only test if we have at least 2 frames
                # Get spectra generator and convert to list
                for frame_id in frame_ids:
                    spectrum = get_centroided_spectrum(td, frame_id=frame_id)
                    self.assertIsInstance(spectrum, np.ndarray)
                    self.assertEqual(spectrum.ndim, 2)
                    self.assertEqual(spectrum.shape[1], 3)

    def test_smooth_kwarg_threads_through_convenience_api(self):
        """`smooth=Smooth(...)` reaches get_raw_peaks / get_centroided_spectrum."""
        from tdfpy import Smooth, get_raw_peaks

        with timsdata.timsdata_connect(TDF_PATH) as td:
            cursor = td.conn.cursor()
            cursor.execute(
                "SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1"
            )
            frame_id = cursor.fetchone()[0]

            base = get_raw_peaks(td, frame_id)
            smoothed = get_raw_peaks(
                td, frame_id, smooth=Smooth(scan_half_width=5, mz_idx_half_width=2)
            )
            # Position-preserving: same point count, but intensities differ.
            self.assertEqual(base.shape[0], smoothed.shape[0])
            self.assertFalse(np.allclose(base[:, 1], smoothed[:, 1]))

            centroids = get_centroided_spectrum(td, frame_id, smooth=Smooth())
            self.assertEqual(centroids.shape[1], 3)

    def test_merge_peaks_basic(self):
        """Test basic peak merging functionality."""
        # Create test data with peaks that should merge
        mz_array = np.array([100.0, 100.0008, 200.0, 200.0005])
        intensity_array = np.array([1000.0, 500.0, 2000.0, 800.0])
        ion_mobility_array = np.array([0.8, 0.8, 0.9, 0.9])

        # Merge with 10 ppm tolerance, min_peaks=1 to keep all
        peaks = merge_peaks(
            mz_array,
            intensity_array,
            ion_mobility_array,
            mz_tolerance=10,
            mz_tolerance_type="ppm",
            im_tolerance=0.05,
            im_tolerance_type="relative",
            min_peaks=1,
        )

        # Should merge into 2 peaks
        self.assertEqual(len(peaks), 2)
        self.assertIsInstance(peaks, np.ndarray)
        self.assertEqual(peaks.shape, (2, 3))

    @unittest.skipIf(not _HAS_NUMBA, "Numba not available")
    def test_numba_python_equivalence(self):
        """Test that Numba and Python implementations produce equivalent results."""
        test_cases = [
            {
                "mz": np.array([100.0, 100.0008, 100.0016, 200.0, 200.0005]),
                "intensity": np.array([1000.0, 800.0, 600.0, 2000.0, 1500.0]),
                "im": np.array([0.8, 0.8, 0.8, 0.9, 0.9]),
                "params": {
                    "mz_tolerance": 10,
                    "mz_tolerance_type": "ppm",
                    "min_peaks": 1,
                },
            },
            {
                "mz": np.array([]),
                "intensity": np.array([]),
                "im": np.array([]),
                "params": {
                    "mz_tolerance": 8,
                    "mz_tolerance_type": "ppm",
                    "min_peaks": 3,
                },
            },
            {
                "mz": np.array([100.0]),
                "intensity": np.array([1000.0]),
                "im": np.array([0.8]),
                "params": {
                    "mz_tolerance": 8,
                    "mz_tolerance_type": "ppm",
                    "min_peaks": 1,
                },
            },
            {
                "mz": np.array([100.0, 100.005, 100.01, 200.0]),
                "intensity": np.array([1000.0, 800.0, 600.0, 2000.0]),
                "im": np.array([0.8, 0.8, 0.8, 0.9]),
                "params": {
                    "mz_tolerance": 0.01,
                    "mz_tolerance_type": "da",
                    "min_peaks": 1,
                },
            },
            {
                "mz": np.array([100.0, 100.0008, 200.0, 200.0005]),
                "intensity": np.array([1000.0, 800.0, 2000.0, 1500.0]),
                "im": np.array([0.8, 0.82, 0.9, 0.95]),
                "params": {
                    "mz_tolerance": 10,
                    "mz_tolerance_type": "ppm",
                    "im_tolerance": 0.03,
                    "im_tolerance_type": "absolute",
                    "min_peaks": 1,
                },
            },
            {
                "mz": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
                "intensity": np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
                "im": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
                "params": {
                    "mz_tolerance": 10,
                    "mz_tolerance_type": "ppm",
                    "min_peaks": 1,
                    "max_peaks": 3,
                },
            },
            {
                # Regression: max_peaks=0 must mean "no limit" in BOTH kernels.
                # Previously numba treated 0 as a real cap (→ 1 peak) while
                # Python treated it as falsy/unlimited (→ 5 peaks).
                "mz": np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
                "intensity": np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
                "im": np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
                "params": {
                    "mz_tolerance": 10,
                    "mz_tolerance_type": "ppm",
                    "min_peaks": 1,
                    "max_peaks": 0,
                },
            },
        ]

        for test in test_cases:
            with self.subTest(params=test["params"]):
                py_peaks = _merge_peaks_python(
                    test["mz"], test["intensity"], test["im"], **test["params"]
                )
                numba_peaks = _merge_peaks_numba(
                    test["mz"], test["intensity"], test["im"], **test["params"]
                )

                self.assertEqual(
                    len(py_peaks),
                    len(numba_peaks),
                    f"Different number of peaks: Python={len(py_peaks)}, Numba={len(numba_peaks)}",
                )

                if len(py_peaks) > 0:
                    np.testing.assert_allclose(py_peaks[:, 0], numba_peaks[:, 0], rtol=1e-6)
                    np.testing.assert_allclose(py_peaks[:, 1], numba_peaks[:, 1], rtol=1e-6)
                    np.testing.assert_allclose(py_peaks[:, 2], numba_peaks[:, 2], rtol=1e-6)

    def test_peak_noise_filter_off_is_noop(self):
        """peak_noise_filter=False should produce identical output to omitting it."""
        rng = np.random.default_rng(42)
        mz = np.sort(rng.uniform(100.0, 105.0, size=200))
        intensity = rng.uniform(10.0, 1000.0, size=200)
        im = rng.uniform(0.8, 0.9, size=200)

        baseline = merge_peaks(
            mz, intensity, im,
            mz_tolerance=20, mz_tolerance_type="ppm",
            im_tolerance=0.02, im_tolerance_type="absolute",
            min_peaks=1,
        )
        with_flag_off = merge_peaks(
            mz, intensity, im,
            mz_tolerance=20, mz_tolerance_type="ppm",
            im_tolerance=0.02, im_tolerance_type="absolute",
            min_peaks=1,
            peak_noise_filter=False,
        )
        np.testing.assert_array_equal(baseline, with_flag_off)

    def test_peak_noise_filter_suppresses_satellites(self):
        """Low-intensity raw points within the noise window should get filtered."""
        # One tall anchor at 500.0 (intensity 10000), surrounded by 6 low-intensity
        # satellite points at 0.02 Da spacing on each side. With end_fraction=0.1
        # over a 0.1 Da window, threshold at d=0.02 is 10000 * (1 - 0.2*0.9) = 8200,
        # so all 100-intensity satellites (well below) get suppressed.
        mz = np.array([
            500.00 - 0.06, 500.00 - 0.04, 500.00 - 0.02,
            500.00,
            500.00 + 0.02, 500.00 + 0.04, 500.00 + 0.06,
        ])
        intensity = np.array([100.0, 100.0, 100.0, 10000.0, 100.0, 100.0, 100.0])
        im = np.full(7, 0.85)

        # Use a tight centroid tolerance so satellites are NOT part of the centroid.
        without_filter = merge_peaks(
            mz, intensity, im,
            mz_tolerance=5, mz_tolerance_type="ppm",
            im_tolerance=0.001, im_tolerance_type="absolute",
            min_peaks=1,
        )
        with_filter = merge_peaks(
            mz, intensity, im,
            mz_tolerance=5, mz_tolerance_type="ppm",
            im_tolerance=0.001, im_tolerance_type="absolute",
            min_peaks=1,
            peak_noise_filter=True,
            peak_noise_window=0.1,
            peak_noise_end_fraction=0.1,
        )

        # Without filter: 7 centroids (one per point). With filter: only the anchor.
        self.assertEqual(len(without_filter), 7)
        self.assertEqual(len(with_filter), 1)
        np.testing.assert_allclose(with_filter[0, 0], 500.0, atol=1e-9)
        np.testing.assert_allclose(with_filter[0, 1], 10000.0)

    def test_peak_noise_filter_preserves_real_neighbor(self):
        """A second tall peak inside the noise window should NOT be filtered."""
        # Tall anchor at 500.0 (10000), real peak at 500.05 (5000), noise at 500.02 (100).
        # Threshold at d=0.05: 10000 * (1 - 0.5*0.9) = 5500 → 5000 < 5500, so the
        # "real" neighbor at 5000 is below threshold here. Use a brighter neighbor:
        # 6000 → 6000 > 5500, survives.
        mz = np.array([500.00, 500.02, 500.05])
        intensity = np.array([10000.0, 100.0, 6000.0])
        im = np.full(3, 0.85)

        with_filter = merge_peaks(
            mz, intensity, im,
            mz_tolerance=5, mz_tolerance_type="ppm",
            im_tolerance=0.001, im_tolerance_type="absolute",
            min_peaks=1,
            peak_noise_filter=True,
            peak_noise_window=0.1,
            peak_noise_end_fraction=0.1,
        )

        # The 100-intensity satellite gets killed; the 6000 neighbor survives.
        self.assertEqual(len(with_filter), 2)
        mzs = np.sort(with_filter[:, 0])
        np.testing.assert_allclose(mzs[0], 500.0, atol=1e-9)
        np.testing.assert_allclose(mzs[1], 500.05, atol=1e-9)

    def test_peak_noise_filter_respects_im_window(self):
        """Satellites at a different ion mobility should NOT be suppressed."""
        # Tall anchor at (500.0, 0.85, 10000); a low point at (500.02, 0.95, 100).
        # The 0.10 IM gap is way outside the centroid's IM window (0.001 abs),
        # so it should survive the noise filter.
        mz = np.array([500.00, 500.02])
        intensity = np.array([10000.0, 100.0])
        im = np.array([0.85, 0.95])

        with_filter = merge_peaks(
            mz, intensity, im,
            mz_tolerance=5, mz_tolerance_type="ppm",
            im_tolerance=0.001, im_tolerance_type="absolute",
            min_peaks=1,
            peak_noise_filter=True,
            peak_noise_window=0.1,
            peak_noise_end_fraction=0.1,
        )
        self.assertEqual(len(with_filter), 2)

    def test_all_zero_intensity_cluster_no_nan(self):
        """A zero-intensity cluster must not divide-by-zero into NaN m/z.

        Regression: the Python kernel previously divided by total_intensity with
        no guard; it now falls back to the seed peak like the numba kernel.
        """
        mz = np.array([100.0, 100.0002, 100.0004])
        intensity = np.zeros(3)
        im = np.full(3, 0.8)
        peaks = merge_peaks(
            mz, intensity, im,
            mz_tolerance=10, mz_tolerance_type="ppm", min_peaks=1,
        )
        self.assertGreaterEqual(peaks.shape[0], 1)
        self.assertTrue(np.all(np.isfinite(peaks)))

    @unittest.skipIf(not _HAS_NUMBA, "Numba not available")
    def test_all_zero_intensity_numba_python_equivalence(self):
        """numba and Python agree on an all-zero-intensity cluster (both finite)."""
        mz = np.array([100.0, 100.0002, 100.0004])
        intensity = np.zeros(3)
        im = np.full(3, 0.8)
        params = dict(mz_tolerance=10, mz_tolerance_type="ppm", min_peaks=1)
        py = _merge_peaks_python(mz, intensity, im, **params)
        nb = _merge_peaks_numba(mz, intensity, im, **params)
        self.assertEqual(len(py), len(nb))
        self.assertTrue(np.all(np.isfinite(py)))
        self.assertTrue(np.all(np.isfinite(nb)))
        np.testing.assert_allclose(py, nb, rtol=1e-9)

    def test_max_peaks_zero_means_unlimited(self):
        """max_peaks=0 is falsy → no cap; all distinct peaks are returned."""
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        im = np.full(5, 0.8)
        peaks = merge_peaks(
            mz, intensity, im,
            mz_tolerance=10, mz_tolerance_type="ppm",
            min_peaks=1, max_peaks=0,
        )
        self.assertEqual(len(peaks), 5)

    def test_max_peaks_positive_caps_output(self):
        """A positive max_peaks caps the number of returned centroids."""
        mz = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        intensity = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        im = np.full(5, 0.8)
        peaks = merge_peaks(
            mz, intensity, im,
            mz_tolerance=10, mz_tolerance_type="ppm",
            min_peaks=1, max_peaks=2,
        )
        self.assertEqual(len(peaks), 2)

    @unittest.skipIf(not _HAS_NUMBA, "Numba not available")
    def test_peak_noise_filter_numba_python_equivalence(self):
        """Numba and Python paths should agree when the peak-noise filter is on."""
        rng = np.random.default_rng(7)
        # Build a synthetic spectrum: a handful of tall anchors plus dense
        # low-intensity satellites around each, so the filter has real work to do.
        anchors_mz = np.array([300.0, 500.0, 750.0, 1000.0])
        anchors_int = np.array([20000.0, 15000.0, 30000.0, 8000.0])
        anchors_im = np.array([0.80, 0.85, 0.90, 0.95])

        sat_mz = []
        sat_int = []
        sat_im = []
        for a_mz, a_im in zip(anchors_mz, anchors_im):
            offsets = rng.uniform(-0.09, 0.09, size=30)
            sat_mz.extend((a_mz + offsets).tolist())
            sat_int.extend(rng.uniform(50.0, 500.0, size=30).tolist())
            sat_im.extend(np.full(30, a_im).tolist())

        mz = np.concatenate([anchors_mz, np.asarray(sat_mz)])
        intensity = np.concatenate([anchors_int, np.asarray(sat_int)])
        im = np.concatenate([anchors_im, np.asarray(sat_im)])

        params = dict(
            mz_tolerance=10, mz_tolerance_type="ppm",
            im_tolerance=0.001, im_tolerance_type="absolute",
            min_peaks=1,
            peak_noise_filter=True,
            peak_noise_window=0.1,
            peak_noise_end_fraction=0.1,
        )
        py_peaks = _merge_peaks_python(mz, intensity, im, **params)
        nb_peaks = _merge_peaks_numba(mz, intensity, im, **params)

        self.assertEqual(len(py_peaks), len(nb_peaks))
        order_py = np.argsort(py_peaks[:, 0])
        order_nb = np.argsort(nb_peaks[:, 0])
        np.testing.assert_allclose(py_peaks[order_py, 0], nb_peaks[order_nb, 0], rtol=1e-9)
        np.testing.assert_allclose(py_peaks[order_py, 1], nb_peaks[order_nb, 1], rtol=1e-9)
        np.testing.assert_allclose(py_peaks[order_py, 2], nb_peaks[order_nb, 2], rtol=1e-9)


class TestWatershedCentroiderCall(unittest.TestCase):
    """``WatershedCentroider.__call__`` against a real MS1 frame.

    The kernel is covered synthetically in test_centroider.py; what only shows
    up here is how ``__call__`` wires smoothing into the kernel.
    """

    @staticmethod
    def _ms1_frame(td) -> int:
        cursor = td.conn.cursor()
        cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 1")
        return int(cursor.fetchone()[0])

    def test_output_intensity_is_the_raw_sum_when_smoothing(self):
        """Smoothing reorders the growth; it must not rewrite the intensities.

        With the box-mean weights doubling as the summed output, a centroided
        frame reported the smoothed total instead of the raw one — silently
        rescaling everyone's intensities whenever the (default) smoothing
        half-widths were nonzero.
        """
        from tdfpy.pipeline import WatershedCentroider, read_spectrum

        with timsdata.timsdata_connect(TDF_PATH) as td:
            frame_id = self._ms1_frame(td)
            spectrum = read_spectrum(td, frame_id)
            if spectrum.empty:
                self.skipTest("First MS1 frame has no peaks")

            centroider = WatershedCentroider(
                smooth_scan_half_width=5, smooth_mz_idx_half_width=3
            )
            self.assertGreater(centroider.smooth_scan_half_width, 0)
            centroids = centroider(spectrum, td, frame_id)

        self.assertGreater(len(centroids), 0)
        # Every raw point lands in exactly one group (both thresholds are 0),
        # so the centroid intensities must total the raw input intensity.
        self.assertAlmostEqual(
            float(centroids[:, 1].sum()),
            float(spectrum.intensities.sum()),
            delta=1e-6 * float(spectrum.intensities.sum()),
        )

    def test_smoothing_changes_grouping_but_not_the_total(self):
        """Smoothed and unsmoothed runs differ in grouping, agree on the sum."""
        from tdfpy.pipeline import WatershedCentroider, read_spectrum

        with timsdata.timsdata_connect(TDF_PATH) as td:
            frame_id = self._ms1_frame(td)
            spectrum = read_spectrum(td, frame_id)
            if spectrum.empty:
                self.skipTest("First MS1 frame has no peaks")
            smoothed = WatershedCentroider()(spectrum, td, frame_id)
            unsmoothed = WatershedCentroider(
                smooth_scan_half_width=0, smooth_mz_idx_half_width=0
            )(spectrum, td, frame_id)

        raw_total = float(spectrum.intensities.sum())
        self.assertNotEqual(len(smoothed), len(unsmoothed))
        for out in (smoothed, unsmoothed):
            self.assertAlmostEqual(
                float(out[:, 1].sum()), raw_total, delta=1e-6 * raw_total
            )

    def test_single_nonzero_half_width_still_smooths(self):
        """Skip smoothing only when *both* half-widths are 0.

        The guard used ``and``, so a zero on either axis skipped smoothing
        entirely — contradicting the documented "set both to 0 to skip" and
        making one-axis smoothing silently unreachable.
        """
        from tdfpy.pipeline import WatershedCentroider, read_spectrum

        with timsdata.timsdata_connect(TDF_PATH) as td:
            frame_id = self._ms1_frame(td)
            spectrum = read_spectrum(td, frame_id)
            if spectrum.empty:
                self.skipTest("First MS1 frame has no peaks")
            mz_only = WatershedCentroider(
                smooth_scan_half_width=0, smooth_mz_idx_half_width=3
            )(spectrum, td, frame_id)
            scan_only = WatershedCentroider(
                smooth_scan_half_width=5, smooth_mz_idx_half_width=0
            )(spectrum, td, frame_id)
            none = WatershedCentroider(
                smooth_scan_half_width=0, smooth_mz_idx_half_width=0
            )(spectrum, td, frame_id)

        # Each single-axis config must smooth along its own axis, so neither
        # may reproduce the no-smoothing grouping.
        self.assertNotEqual(len(mz_only), len(none))
        self.assertNotEqual(len(scan_only), len(none))
        self.assertNotEqual(len(mz_only), len(scan_only))

        raw_total = float(spectrum.intensities.sum())
        for out in (mz_only, scan_only, none):
            self.assertAlmostEqual(
                float(out[:, 1].sum()), raw_total, delta=1e-6 * raw_total
            )

    def test_empty_spectrum_returns_empty(self):
        from tdfpy.pipeline import RawSpectrum, WatershedCentroider

        with timsdata.timsdata_connect(TDF_PATH) as td:
            frame_id = self._ms1_frame(td)
            out = WatershedCentroider()(RawSpectrum.empty_like(10), td, frame_id)
        self.assertEqual(out.shape, (0, 3))


if __name__ == "__main__":
    unittest.main()
