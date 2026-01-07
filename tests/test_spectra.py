import unittest
import numpy as np

from tdfpy import (
    timsdata,
    Peak,
    Ms1Spectrum,
    merge_peaks,
    get_centroided_ms1_spectrum,
    get_centroided_ms1_spectra,
)
from tdfpy.spectra import _merge_peaks_python, _HAS_RUST # type: ignore[import]

# Try to import Rust extension for comparison tests
if _HAS_RUST:
    from tdfpy._tdfpy_rust import merge_peaks as _merge_peaks_rust # type: ignore[import]

TDF_PATH = r"tests/data/200ngHeLaPASEF_1min.d"


class TestSpectra(unittest.TestCase):
    """Test the higher-level spectra API."""

    def test_peak_namedtuple(self):
        """Test Peak NamedTuple creation and access."""
        peak = Peak(mz=123.456, intensity=1000.0, ion_mobility=0.8)

        self.assertAlmostEqual(peak.mz, 123.456)
        self.assertAlmostEqual(peak.intensity, 1000.0)
        self.assertAlmostEqual(peak.ion_mobility, 0.8)

        # Test that it's a tuple (immutable)
        self.assertIsInstance(peak, tuple)

    def test_ms1_spectrum_namedtuple(self):
        """Test Ms1Spectrum NamedTuple creation and access."""
        peaks = [
            Peak(mz=100.0, intensity=500.0, ion_mobility=0.5),
            Peak(mz=200.0, intensity=1000.0, ion_mobility=0.5),
        ]

        spectrum = Ms1Spectrum(
            spectrum_index=0,
            frame_id=1,
            retention_time=1.5,
            peaks=peaks,
            ion_mobility_type="ook0",
        )

        self.assertEqual(spectrum.spectrum_index, 0)
        self.assertEqual(spectrum.frame_id, 1)
        self.assertAlmostEqual(spectrum.retention_time, 1.5)
        self.assertEqual(spectrum.num_peaks, 2)
        self.assertEqual(len(spectrum.peaks), 2)
        self.assertEqual(spectrum.ion_mobility_type, "ook0")
        self.assertIsInstance(spectrum, tuple)

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
            spectrum = get_centroided_ms1_spectrum(td, frame_id)

            # Verify structure
            self.assertIsInstance(spectrum, Ms1Spectrum)
            self.assertEqual(spectrum.frame_id, frame_id)
            self.assertGreaterEqual(spectrum.retention_time, 0)
            self.assertEqual(spectrum.num_peaks, len(spectrum.peaks))

            # Verify peaks if any exist
            if spectrum.num_peaks > 0:
                first_peak = spectrum.peaks[0]
                self.assertIsInstance(first_peak, Peak)
                self.assertGreater(first_peak.mz, 0)
                self.assertGreater(first_peak.intensity, 0)
                # Ion mobility should be set
                self.assertIsNotNone(first_peak.ion_mobility)

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
                spectra_gen = get_centroided_ms1_spectra(td, frame_ids=frame_ids)
                spectra = list(spectra_gen)

                self.assertEqual(len(spectra), 2)
                self.assertEqual(spectra[0].frame_id, frame_ids[0])
                self.assertEqual(spectra[1].frame_id, frame_ids[1])

                # Verify sequential indexing
                for idx, spectrum in enumerate(spectra):
                    self.assertEqual(spectrum.spectrum_index, idx)

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

        # Check that peaks are Peak objects
        for peak in peaks:
            self.assertIsInstance(peak, Peak)

    @unittest.skipIf(not _HAS_RUST, "Rust extension not available")
    def test_rust_python_equivalence(self):
        """Test that Rust and Python implementations produce identical results."""
        test_cases = [
            # Basic test
            {
                'mz': np.array([100.0, 100.0008, 100.0016, 200.0, 200.0005]),
                'intensity': np.array([1000.0, 800.0, 600.0, 2000.0, 1500.0]),
                'im': np.array([0.8, 0.8, 0.8, 0.9, 0.9]),
                'params': {'mz_tolerance': 10, 'mz_tolerance_type': 'ppm', 'min_peaks': 1}
            },
            # Empty arrays
            {
                'mz': np.array([]),
                'intensity': np.array([]),
                'im': np.array([]),
                'params': {'mz_tolerance': 8, 'mz_tolerance_type': 'ppm', 'min_peaks': 3}
            },
            # Single peak
            {
                'mz': np.array([100.0]),
                'intensity': np.array([1000.0]),
                'im': np.array([0.8]),
                'params': {'mz_tolerance': 8, 'mz_tolerance_type': 'ppm', 'min_peaks': 1}
            },
            # Dalton tolerance
            {
                'mz': np.array([100.0, 100.005, 100.01, 200.0]),
                'intensity': np.array([1000.0, 800.0, 600.0, 2000.0]),
                'im': np.array([0.8, 0.8, 0.8, 0.9]),
                'params': {'mz_tolerance': 0.01, 'mz_tolerance_type': 'da', 'min_peaks': 1}
            },
            # Absolute ion mobility tolerance
            {
                'mz': np.array([100.0, 100.0008, 200.0, 200.0005]),
                'intensity': np.array([1000.0, 800.0, 2000.0, 1500.0]),
                'im': np.array([0.8, 0.82, 0.9, 0.95]),
                'params': {'mz_tolerance': 10, 'mz_tolerance_type': 'ppm', 'im_tolerance': 0.03, 
                          'im_tolerance_type': 'absolute', 'min_peaks': 1}
            },
            # With max_peaks limit
            {
                'mz': np.array([100.0, 200.0, 300.0, 400.0, 500.0]),
                'intensity': np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0]),
                'im': np.array([0.8, 0.8, 0.8, 0.8, 0.8]),
                'params': {'mz_tolerance': 10, 'mz_tolerance_type': 'ppm', 'min_peaks': 1, 'max_peaks': 3}
            },
        ]

        tolerance = 1e-9
        
        for test in test_cases:
            with self.subTest(params=test['params']):
                # Get Python results
                py_peaks = _merge_peaks_python(test['mz'], test['intensity'], test['im'], **test['params'])
                
                # Get Rust results (raw arrays)
                rust_mz, rust_int, rust_im = _merge_peaks_rust(
                    test['mz'], test['intensity'], test['im'], **test['params']
                )
                
                # Compare lengths
                self.assertEqual(len(py_peaks), len(rust_mz),
                               f"Different number of peaks: Python={len(py_peaks)}, Rust={len(rust_mz)}")
                
                # Compare each peak
                for j, py_peak in enumerate(py_peaks):
                    self.assertAlmostEqual(py_peak.mz, rust_mz[j], delta=tolerance,
                                         msg=f"Peak {j} m/z mismatch")
                    self.assertAlmostEqual(py_peak.intensity, rust_int[j], delta=tolerance,
                                         msg=f"Peak {j} intensity mismatch")
                    self.assertAlmostEqual(py_peak.ion_mobility, rust_im[j], delta=tolerance,
                                         msg=f"Peak {j} ion mobility mismatch")


if __name__ == "__main__":
    unittest.main()
