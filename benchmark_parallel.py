#!/usr/bin/env python3
"""
Benchmark script to compare sequential vs parallel centroiding performance.

Tests the performance of get_centroided_ms1_spectra with different numbers
of worker threads to measure the speedup from parallelization.
"""

import time
import sys
from pathlib import Path

from tdfpy import timsdata_connect, get_centroided_ms1_spectra


def benchmark_centroiding(data_path: str, num_frames: int = 50):
    """Benchmark centroiding with different worker counts.
    
    Args:
        data_path: Path to .d directory
        num_frames: Number of frames to process (default: 50)
    """
    print(f"Benchmarking centroiding performance on: {data_path}")
    print(f"Processing {num_frames} MS1 frames\n")
    
    # Get frame IDs
    with timsdata_connect(data_path) as td:
        cursor = td.conn.cursor()
        cursor.execute(
            f"SELECT Id FROM Frames WHERE MsMsType = 0 ORDER BY Id LIMIT {num_frames}"
        )
        frame_ids = [row[0] for row in cursor.fetchall()]
    
    if len(frame_ids) < num_frames:
        print(f"Warning: Only {len(frame_ids)} MS1 frames available")
        num_frames = len(frame_ids)
    
    results = []
    
    # Test different worker counts
    worker_counts = [1, 2, 4, 8, 16]
    
    for num_workers in worker_counts:
        print(f"Testing with {num_workers} worker(s)...", end=" ", flush=True)
        
        with timsdata_connect(data_path) as td:
            start = time.time()
            spectra = list(
                get_centroided_ms1_spectra(td, frame_ids=frame_ids, num_workers=num_workers)
            )
            elapsed = time.time() - start
        
        total_peaks = sum(s.num_peaks for s in spectra)
        throughput = num_frames / elapsed
        
        results.append({
            'workers': num_workers,
            'time': elapsed,
            'spectra': len(spectra),
            'peaks': total_peaks,
            'throughput': throughput
        })
        
        print(f"{elapsed:.2f}s ({throughput:.1f} spectra/s)")
    
    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS")
    print("=" * 70)
    print(f"{'Workers':<10} {'Time (s)':<12} {'Throughput':<18} {'Speedup':<10}")
    print("-" * 70)
    
    baseline_time = results[0]['time']
    
    for r in results:
        speedup = baseline_time / r['time']
        print(
            f"{r['workers']:<10} {r['time']:<12.2f} "
            f"{r['throughput']:<18.1f} {speedup:<10.2f}x"
        )
    
    print("=" * 70)
    print(f"\nTotal spectra processed: {results[0]['spectra']}")
    print(f"Total peaks extracted: {results[0]['peaks']:,}")
    print(f"Best speedup: {max(r['time'] / baseline_time for r in results[1:]):.2f}x "
          f"with {max(results[1:], key=lambda r: baseline_time / r['time'])['workers']} workers")


if __name__ == "__main__":
    benchmark_centroiding("tests/data/200ngHeLaPASEF_1min.d", 20)
