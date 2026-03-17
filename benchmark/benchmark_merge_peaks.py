"""Benchmark Numba vs Python implementation of merge_peaks."""
import time
import numpy as np
import tdfpy as tp
from tdfpy.centroiding import _merge_peaks_python, _HAS_NUMBA

if not _HAS_NUMBA:
    print("ERROR: Numba not available")
    exit(1)

from tdfpy.centroiding import _merge_peaks_numba

TDF_PATH = "tests/data/200ngHeLaPASEF_1min.d"


def benchmark_merge_peaks():
    """Benchmark the merge_peaks implementations."""
    print("="*80)
    print("Benchmarking Numba vs Python merge_peaks Implementation")
    print("="*80)

    with tp.timsdata_connect(TDF_PATH) as td:
        # Get first 5 MS1 frames
        cursor = td.conn.cursor()
        cursor.execute("SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT 5")
        frame_ids = [row[0] for row in cursor.fetchall()]

        print(f"\nCollecting data from {len(frame_ids)} frames...")

        # Collect all peaks from frames
        all_mz = []
        all_intensity = []
        all_im = []

        for frame_id in frame_ids:
            result = cursor.execute(
                "SELECT Time, NumScans FROM Frames WHERE Id = ?", (frame_id,)
            ).fetchone()
            num_scans = result[1]

            if num_scans == 0:
                continue

            ion_mobility = td.scanNumToOneOverK0(frame_id, np.arange(0, num_scans))
            results = td.readScans(frame_id, 0, num_scans)

            for scan_index, (index_array, intensity_scan) in enumerate(results):
                if len(index_array) == 0:
                    continue

                mz_values = td.indexToMz(frame_id, index_array)
                all_mz.extend(mz_values)
                all_intensity.extend(intensity_scan)
                all_im.extend([ion_mobility[scan_index]] * len(index_array))

        mz_array = np.array(all_mz, dtype=np.float64)
        intensity_array = np.array(all_intensity, dtype=np.float64)
        im_array = np.array(all_im, dtype=np.float64)

        print(f"Total peaks collected: {len(mz_array):,}")

        # Benchmark parameters
        params = {
            'mz_tolerance': 8.0,
            'mz_tolerance_type': 'ppm',
            'im_tolerance': 0.05,
            'im_tolerance_type': 'relative',
            'min_peaks': 3,
            'max_peaks': None
        }

        print(f"\nBenchmark parameters:")
        for k, v in params.items():
            print(f"  {k}: {v}")

        # Warm up (important for Numba JIT compilation)
        print("\nWarming up (triggers Numba JIT compilation)...")
        _ = _merge_peaks_python(mz_array, intensity_array, im_array, **params)
        _ = _merge_peaks_numba(mz_array, intensity_array, im_array, **params)

        # Benchmark Python implementation
        print("\nBenchmarking Python implementation...")
        n_iterations = 10
        python_times = []

        for i in range(n_iterations):
            start = time.perf_counter()
            py_result = _merge_peaks_python(mz_array, intensity_array, im_array, **params)
            end = time.perf_counter()
            python_times.append(end - start)

        py_avg = np.mean(python_times)
        py_std = np.std(python_times)
        py_num_peaks = len(py_result)

        print(f"  Average time: {py_avg*1000:.2f} ± {py_std*1000:.2f} ms")
        print(f"  Peaks merged: {len(mz_array):,} → {py_num_peaks:,}")

        # Benchmark Numba implementation
        print("\nBenchmarking Numba implementation...")
        numba_times = []

        for i in range(n_iterations):
            start = time.perf_counter()
            numba_result = _merge_peaks_numba(mz_array, intensity_array, im_array, **params)
            end = time.perf_counter()
            numba_times.append(end - start)

        numba_avg = np.mean(numba_times)
        numba_std = np.std(numba_times)
        numba_num_peaks = len(numba_result)

        print(f"  Average time: {numba_avg*1000:.2f} ± {numba_std*1000:.2f} ms")
        print(f"  Peaks merged: {len(mz_array):,} → {numba_num_peaks:,}")

        # Calculate speedup
        speedup = py_avg / numba_avg
        peak_diff = abs(py_num_peaks - numba_num_peaks)
        peak_diff_pct = (peak_diff / py_num_peaks) * 100 if py_num_peaks > 0 else 0

        print("\n" + "="*80)
        print("RESULTS")
        print("="*80)
        print(f"Python: {py_avg*1000:.2f} ms")
        print(f"Numba:  {numba_avg*1000:.2f} ms")
        print(f"Speedup: {speedup:.2f}x faster")
        print(f"Peak count difference: {peak_diff} ({peak_diff_pct:.3f}%)")
        print(f"  Python: {py_num_peaks:,} peaks")
        print(f"  Numba:  {numba_num_peaks:,} peaks")
        if peak_diff_pct < 0.1:
            print("  Results effectively identical (< 0.1% difference)")
        print("="*80)


if __name__ == "__main__":
    benchmark_merge_peaks()
