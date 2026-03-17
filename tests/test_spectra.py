import unittest

import numpy as np

from tdfpy import (
    get_centroided_spectrum,
    merge_peaks,
    timsdata,
)
from tdfpy.centroiding import _HAS_NUMBA, _merge_peaks_numba, _merge_peaks_python  # type: ignore[import]

TDF_PATH = r"tests/data/200ngHeLaPASEF_1min.d"


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


if __name__ == "__main__":
    unittest.main()
